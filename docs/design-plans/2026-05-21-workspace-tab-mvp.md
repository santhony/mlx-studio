# Workspace Tab MVP Design

## Summary

The Workspace tab consolidates the existing Notebook and Agent tabs into a single durable surface: a per-workspace chat with the model, scoped to a sandboxed directory under `data/workspaces/<id>/`, with model-callable tools for file I/O, Python execution, and cross-tab capabilities (RAG retrieval and image generation). Architecturally, it is a thin FastAPI router on top of a reusable `workspace_runner.py` module that extracts the existing agent dispatcher loop into a backend-agnostic component talking to mlx-studio's text-server. UI is HTMX + SSE streaming, matching the patterns already used by the chat and RAG tabs.

The rationale for the shape is twofold. First, **reuse over reinvention**: the design deliberately leans on existing infrastructure (the agents dispatcher pattern, `_proxy_sse` from chat, `indexer.retrieve_chunks`, the `/image/generate` endpoint, skill injection at prompt-build time) so the new code is mostly glue. Second, **auto-execute + revert instead of approval gates**: rather than gating each tool call, every assistant turn snapshots the workspace directory (via `shutil.copytree` to `.checkpoints/<seq>/`), letting the user undo bad model behavior cheaply. This trades disk usage for UX fluidity and is the design's pragmatic answer to DS4's known weak decline-calibration.

## Definition of Done

The MVP is done when a user can:

1. Create a new Workspace from the top-nav, give it a name, and land on its detail page.
2. Chat with the model inside that workspace; the model can read, edit, write, and list files in `data/workspaces/<id>/` and run Python via `run_python` — all without per-call approval prompts.
3. Invoke at least two cross-tab tools the workspace exposes: `query_rag(corpus_id, q)` returning chunks the model can cite, and `generate_image(prompt)` writing a PNG into the workspace directory.
4. See, inline in the chat: rendered markdown blocks, embedded images (the ones `generate_image` wrote), and collapsible tool-call/tool-result blocks.
5. Click "Revert" to restore the workspace directory to the state before the previous assistant turn. Checkpoints are taken per assistant turn (one checkpoint per user→assistant exchange), stored at `data/workspaces/<id>/.checkpoints/<seq>/`.
6. Workspaces persist across server restarts: returning to the list shows prior workspaces with name, last-active timestamp, and a one-line summary.
7. The old Notebook and Agent tabs (and their database tables) no longer exist in the codebase.

Operational acceptance: `python -m pytest -q` passes; `./start.sh` brings up the app; the existing offline test suite is not regressed; manual smoke test of the compose-multi-modal-artifact use case succeeds (open workspace → query RAG → ask model to draft markdown using results → ask for a header image → see image embedded in chat → save markdown file → revert → re-do).

Out of scope for this MVP (deferred): per-tool-call checkpoints, multiple concurrent chat sessions per workspace, LSP/debugger integration, file-tree sidebar, skills as model-callable tools, workspace sharing or export, fine-tune integration as a tool.

## Glossary

- **mlx-studio**: This repo — a local-first FastAPI web app for image generation, chat, notebooks, agents, RAG, and fine-tuning on Apple Silicon.
- **text-server**: The `qwen-text-server` FastAPI service on `127.0.0.1:8766` exposing `/chat`, `/complete`, `/embed`. Speaks SSE.
- **ds4-server**: A sibling local-inference backend; mlx-studio's text-server can proxy to it via the local-inference-mcp wiring.
- **HTMX**: Library for HTML-over-the-wire interactivity; the web-app swaps server-rendered partials into the DOM instead of running a SPA.
- **SSE (Server-Sent Events)**: One-way streaming protocol (`data: ...\n\n`) used by the text-server and re-emitted by `chat.py:_proxy_sse` to stream tokens to the browser.
- **FastAPI**: Python async web framework underlying all three servers.
- **RAG (Retrieval-Augmented Generation)**: Existing tab that indexes corpora and retrieves chunks by cosine similarity for grounded responses; exposed here as the `query_rag` tool.
- **Dispatcher pattern**: Tool-routing shape in `routers/agents.py:_dispatch_tool` — a dict from tool-name to callable. The runner loops prompt → model → parse tool call → dispatch → feed result back until the model returns a tool-call-free message.
- **`run_python` sandbox**: A tool that runs Python in a subprocess with `cwd` set to the workspace root and a 30s timeout.
- **Hash-anchored edits**: `edit_file(path, old_str, new_str)` replaces a substring only if `old_str` is found verbatim — protects against blind overwrites.
- **Checkpoint semantics**: One snapshot per user→assistant turn (not per tool call), stored as a full `copytree` at `.checkpoints/<seq>/`; revert restores the directory and truncates message history after that checkpoint.
- **DS4 / ds4-coding-eval**: A separate eval (`ds4-coding-eval/VERDICT-final.md`) that found DS4 over-eager and weak at declining ambiguous prompts; informs the auto-execute + revert UX choice.
- **MLX**: Apple's array framework; backs the in-process text models on Apple Silicon.

## Architecture

The Workspace tab replaces the vestigial Notebook and Agent tabs with a single durable surface: a chat with the model scoped to a sandboxed directory, with model-callable tools for both local file/code ops and mlx-studio's cross-tab capabilities (RAG retrieval, image generation).

Three new components live under `web-app/`:

- `routers/workspace.py` — FastAPI router (list, create, delete, detail, send-message, revert endpoints). Owns HTMX-friendly response rendering for the workspace list and detail templates.
- `workspace_runner.py` — Reusable runner module extracted from `routers/agents.py`'s `_run_agent` loop. Encapsulates the prompt → text-server call → parse-tool-call → dispatch → feed-result-back loop. Backend-agnostic (talks to mlx-studio's text-server at `127.0.0.1:8766`, which already proxies to ds4-server per the local-inference-mcp wiring).
- `templates/workspace_list.html`, `templates/workspace.html`, `templates/_workspace_message.html` — workspace listing, single-workspace detail, and per-message partial for HTMX swap-in.

Three new database tables in `web-app/db.py`:

- `workspaces` — durable workspace identity (id, name, root_dir, summary, created_at, last_active_at).
- `workspace_messages` — full conversation transcript (id, workspace_id, role, content, tool_calls_json, created_at).
- `workspace_checkpoints` — checkpoint metadata (id, workspace_id, seq, message_id, created_at). The actual checkpoint contents live as full directory snapshots at `data/workspaces/<id>/.checkpoints/<seq>/`.

Three tables and three routers are removed: `notebooks`, `cells`, `agent_jobs`, `agent_steps`; `routers/notebook.py`, `routers/agents.py`, `web-app/agent_tools.py` plus their templates (`notebook.html`, `notebook_list.html`, `cell.html`, `agents.html`, `agent_job.html`).

Cross-tab tools call into existing mlx-studio code, not new infrastructure:

- `query_rag(corpus_id, q)` calls `indexer.retrieve_chunks(conn, corpus_id, query, top_k=5)`, returning chunks with citations.
- `generate_image(prompt)` POSTs to the existing `/image/generate` endpoint and writes the resulting PNG into the workspace directory.

The auto-execute + checkpoint pattern works as follows: on each user message, the runner snapshots the workspace directory tree (via `shutil.copytree` to `.checkpoints/<seq>/`), runs the dispatcher loop until the model emits a tool-call-free message, persists everything to the database. The Revert button on a message restores the corresponding checkpoint's directory contents back to the workspace root.

## Existing Patterns

Investigation (via codebase-investigator on 2026-05-21) found three patterns this design follows directly.

**Tool dispatcher pattern** from `web-app/routers/agents.py:283-315`: a `_dispatch_tool(name, args, ...)` function maps tool names to call-out functions. `workspace_runner.py` reuses this exact shape with the tool registry expanded to include `query_rag` and `generate_image` and the file-ops scoped to the workspace root rather than the project root.

**SSE streaming proxy** from `web-app/routers/chat.py:_proxy_sse` (also used by RAG and Notebook): text-server response via `httpx.stream` → `aiter_lines` → re-emit as SSE deltas → persist on `[DONE]`. The workspace router uses this same pattern for streaming model responses to the browser.

**Skill injection at prompt-build time** from `web-app/routers/agents.py:176-177` and `web-app/routers/chat.py:95`: `retrieve_skills(conn, prompt, top_k=3)` returns top-3 semantic matches, formatted via `format_skills_for_context()` and prepended to the system prompt. Workspace prompts use the same call site.

The design diverges from `agents.py` in two ways. First, the approval-gate UX (the `_pending_approval` event and approve/reject buttons in `agent_job.html`) is removed — workspaces are auto-execute with rollback rather than approve-before-act. Second, the data model treats a workspace as a long-lived row with a directory, not a one-shot job — `agent_jobs` was meant for transient execution; `workspaces` are durable projects.

The cross-tab tool function signatures already exist and are imported as-is: `indexer.retrieve_chunks` (line 519 of `web-app/indexer.py`), the `/image/generate` POST endpoint in `web-app/routers/image.py`. No new infrastructure for these.

## Implementation Phases

<!-- START_PHASE_1 -->
### Phase 1: Workspace scaffolding (data model + minimal router)
**Goal:** A user can create, list, and delete workspaces from the top-nav. Workspaces persist; directories exist on disk. No chat yet.

**Components:**
- `web-app/db.py` — add `workspaces`, `workspace_messages`, `workspace_checkpoints` table definitions and a migration that creates them if absent.
- `web-app/routers/workspace.py` — new router with endpoints: `GET /workspace/`, `POST /workspace/`, `GET /workspace/{id}`, `DELETE /workspace/{id}`. The detail endpoint returns a placeholder template for now.
- `web-app/templates/workspace_list.html` — list view with "New Workspace" form.
- `web-app/templates/workspace.html` — minimal detail page (just the workspace name and a "(chat coming)" placeholder).
- `web-app/main.py` — register the new router; add a "Workspaces" link to the nav in `templates/base.html`.
- Workspace creation creates `data/workspaces/<id>/` on disk; deletion removes it.

**Dependencies:** None (first phase).

**Done when:** `./start.sh` starts cleanly, you can click "New Workspace" in the nav, name it, see it appear in the list, navigate to its detail page, and delete it. A pytest covers create/list/delete via the FastAPI TestClient.
<!-- END_PHASE_1 -->

<!-- START_PHASE_2 -->
### Phase 2: Workspace runner module (file ops + run_python, no cross-tab yet)
**Goal:** Extract the dispatcher loop from `agents.py` into a reusable module. Wire a minimal tool set (file ops + run_python) scoped to the workspace root. Confirm a workspace can run a single round-trip turn end-to-end via a test, without UI.

**Components:**
- `web-app/workspace_runner.py` — new module with:
  - `WorkspaceRunner` class taking workspace root, db connection, and a tool registry.
  - `run_turn(user_message)` coroutine that loops: build prompt → call text-server `/chat` → parse tool calls → dispatch → feed result back → persist messages. Stops on tool-call-free response.
  - Tool registry: `read_file`, `edit_file(path, old_str, new_str)` (hash-anchored substring replacement), `write_file`, `list_dir`, `run_python` (subprocess with workspace cwd, 30s timeout). All file ops reject paths that escape the workspace root.
- `web-app/routers/workspace.py` — add `POST /workspace/{id}/messages` endpoint that calls `WorkspaceRunner.run_turn` and persists messages. No streaming yet; returns final state.

**Dependencies:** Phase 1.

**Done when:** Pytest invokes the endpoint with a prompt like "Create a file `hello.txt` with the word 'hi' and then read it back." Assertion: response references "hi", `data/workspaces/<id>/hello.txt` exists and contains "hi", and messages are persisted with role + tool_calls fields populated. No browser interaction required for this phase.
<!-- END_PHASE_2 -->

<!-- START_PHASE_3 -->
### Phase 3: Workspace chat UI (HTMX, streaming, no rendering yet)
**Goal:** A working chat surface in the browser. User types a message, sees the model's streaming response, sees raw tool-call/result text. No markdown rendering or image embedding yet — those come in Phase 5.

**Components:**
- `web-app/routers/workspace.py` — add streaming variant of the message endpoint using `httpx.stream` to the text-server and SSE re-emission to the browser, mirroring `chat.py:_proxy_sse`.
- `web-app/templates/workspace.html` — replace the placeholder with a chat surface: message list, input form (HTMX submit), SSE listener that appends streaming tokens to the latest message div.
- `web-app/templates/_workspace_message.html` — per-message partial: shows role, content (raw at this phase), tool calls as plain JSON blocks.
- Reuse `web-app/static/css/main.css` styles for message classes.

**Dependencies:** Phase 2.

**Done when:** Manually visit a workspace, type "Write a Python program that prints 1..5 and run it", watch streaming response in real time, see the tool calls and final answer appear in the transcript. Refreshing the page replays the persisted conversation. Pytest verifies the SSE endpoint produces well-formed events.
<!-- END_PHASE_3 -->

<!-- START_PHASE_4 -->
### Phase 4: Cross-tab tools (query_rag + generate_image)
**Goal:** The model can call into mlx-studio's RAG corpora and image generation as tools. Tool results appear in the conversation; generated images are saved into the workspace directory.

**Components:**
- `web-app/workspace_runner.py` — extend the tool registry:
  - `query_rag(corpus_id: int, q: str)` calls `indexer.retrieve_chunks(conn, corpus_id, q, top_k=5)`; returns the list of chunks (source, content, score) for the model to cite.
  - `generate_image(prompt: str, filename: str = None)` POSTs to `/image/generate`, polls the SSE stream until done, decodes the base64 PNG, writes it to `data/workspaces/<id>/<filename or generated-name>.png`, returns the filename.
- Tool descriptions in the system prompt explicitly list available corpora (by querying `corpora` table at prompt-build time).

**Dependencies:** Phase 2 (runner).

**Done when:** Pytest invokes a workspace with a prompt like "Query corpus 1 for 'introduction' and tell me what you found", the runner dispatches `query_rag`, the model's response cites chunks. Separate test: "Generate an image of a cat" produces a PNG at `data/workspaces/<id>/cat-*.png`. Both tools fail gracefully (model sees an error string, not a 500) when the corpus doesn't exist or the image server is down.
<!-- END_PHASE_4 -->

<!-- START_PHASE_5 -->
### Phase 5: Inline rendering (markdown, images, collapsible tool blocks)
**Goal:** The chat UI renders content beautifully — markdown is parsed and rendered, images saved to the workspace are embedded inline, tool-call/tool-result blocks are collapsed by default.

**Components:**
- `web-app/templates/_workspace_message.html` — render message content via a server-side markdown filter (using existing `markdown` dependency if present, or `markdown-it-py` added to requirements). Detect image references (`![alt](path.png)`) where path is workspace-relative and rewrite to `/workspace/<id>/file/<path>` URL.
- `web-app/routers/workspace.py` — add `GET /workspace/{id}/file/{filename}` endpoint that serves files from the workspace directory (with path-escape rejection).
- Tool calls + results: wrap each in `<details><summary>tool: name</summary>...</details>` blocks.
- `web-app/static/css/main.css` — small additions for inline image sizing and `.workspace-message` containers.

**Dependencies:** Phase 4 (so generate_image produces images we can embed).

**Done when:** A workspace conversation where the model uses `generate_image` shows the resulting PNG embedded inline. Markdown-formatted assistant responses render with headings/lists/code blocks. Tool calls show as collapsible blocks. Manual smoke test of the multi-modal-artifact use case (chat → RAG → markdown draft → image gen → embedded result) works end-to-end.
<!-- END_PHASE_5 -->

<!-- START_PHASE_6 -->
### Phase 6: Checkpoint + revert
**Goal:** Every assistant turn snapshots the workspace directory. The Revert button on any message restores the workspace to the state before that turn.

**Components:**
- `web-app/workspace_runner.py` — before invoking tools for a turn, call a new `_snapshot(workspace_id, seq)` helper that `shutil.copytree`s the workspace dir (excluding `.checkpoints/`) into `.checkpoints/<seq>/`. Record the seq in `workspace_checkpoints`.
- `web-app/routers/workspace.py` — `POST /workspace/{id}/revert/{seq}` endpoint that removes the workspace dir contents (except `.checkpoints/`) and copies the snapshot back. Truncates the message history after the checkpoint's `message_id`.
- `web-app/templates/_workspace_message.html` — Revert button on each assistant message that has an associated checkpoint.

**Dependencies:** Phase 2 (runner) — checkpoint hook lives in the runner.

**Done when:** Pytest creates a workspace, sends a message that writes file A, sends another that writes file B, reverts to the first checkpoint, confirms file B no longer exists and file A does. Manual: revert button in UI works as expected on a real conversation.
<!-- END_PHASE_6 -->

<!-- START_PHASE_7 -->
### Phase 7: Cleanup — delete Notebook + Agent tabs and tables
**Goal:** The replaced tabs are gone from the codebase. Database migrations drop their tables. Nav is clean. Tests still pass.

**Components:**
- Delete: `web-app/routers/notebook.py`, `web-app/routers/agents.py`, `web-app/agent_tools.py`, `web-app/templates/notebook.html`, `web-app/templates/notebook_list.html`, `web-app/templates/cell.html`, `web-app/templates/agents.html`, `web-app/templates/agent_job.html`.
- `web-app/db.py` — migration block that drops `notebooks`, `cells`, `agent_jobs`, `agent_steps` if they exist.
- `web-app/main.py` — remove notebook and agents router registrations; remove their nav links from `templates/base.html`.
- `web-app/tests/` — remove `test_agent_tools.py` and any notebook/agent-specific tests; update any cross-references.
- `web-app/CLAUDE.md` — update the tabs list, freshness date, and "Status" line to reflect the consolidation.

**Dependencies:** Phases 1-6 (workspace must be functional before deleting predecessors).

**Done when:** `git status` shows the deletions are clean (no orphan imports). `python -m pytest -q` passes. `./start.sh` starts without errors. The nav shows "Workspaces" but not "Notebooks" or "Agents". The DB no longer has the four dropped tables.
<!-- END_PHASE_7 -->

## Additional Considerations

**Error handling for cross-tab tools:** `query_rag` and `generate_image` are the most failure-prone tools (depend on external services). Both return an error string to the model on failure rather than raising — the model then has the chance to retry, give up, or surface the failure to the user in its own words. This matches the prior agent dispatcher's approach.

**Checkpoint disk usage:** Per-turn snapshots are full directory copies. For a workspace with a 100 MB asset directory, a 20-turn conversation produces ~2 GB of checkpoints. Acceptable for MVP (single-user, single-machine). Iteration 2 candidates: hard-link snapshots (instant + space-cheap), then content-addressed storage; or just a fixed-window retention policy (keep last N checkpoints).

**Path-escape defense:** All workspace file operations and the `/workspace/{id}/file/{filename}` endpoint resolve paths via `pathlib.Path.resolve()` and reject any result outside the workspace root. This is the same approach `ds4-coding-eval/tools.py:Toolkit._resolve` used — known-good pattern.

**DS4 behavioral guardrails:** The eval (`ds4-coding-eval/VERDICT-final.md`) established DS4 is over-eager on ambiguous prompts and weak on decline calibration. The auto-execute + revert UX is the design's answer to this: the model's bad choices are cheap to undo via Revert. No per-tool-call confirmation gate is built; iteration 2 may add a scope-tier hard gate (e.g., "more than 3 files written" → confirm) if Revert proves insufficient in practice.

**Future extensibility:** The `workspace_runner.py` tool registry is just a Python dict mapping tool name to a callable. Adding skills-as-tools (`invoke_skill(name)`), fine-tune-job triggers, or other cross-tab capabilities later is a matter of registering new entries — no architecture change needed.
