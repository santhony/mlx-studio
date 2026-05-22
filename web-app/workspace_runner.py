"""Workspace runner — the prompt → tool → result → repeat loop.

# pattern: Imperative Shell
# This module coordinates I/O between the text-server, the workspace
# tools (subprocess/file I/O), and the database (message persistence).
# The parsing of model output and the tool registry shape live in
# workspace_runner_parser.py and workspace_tools.py respectively, both
# of which are pure-er surfaces under unit test.
#
# Adapted from routers/agents.py:_run_agent + _dispatch_tool, with the
# approval-gate logic removed (workspaces are auto-execute by design —
# safety lives in Phase 6's checkpoint+revert pattern).
"""
from __future__ import annotations

import base64
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx

from indexer import retrieve_chunks
from skills import retrieve_skills, format_skills_for_context
from workspace_checkpoint import snapshot_workspace, next_seq
from workspace_runner_parser import parse_tool_calls
from workspace_tools import (
    PathEscapeError,
    _resolve,
    edit_file,
    list_dir,
    read_file,
    run_python,
    write_file,
)


TEXT_SERVER_URL = "http://127.0.0.1:8766"
WEB_APP_URL = "http://127.0.0.1:8080"  # /image/generate is mounted on the web-app
DEFAULT_MAX_ROUNDS = 12  # Soft cap per the eval-derived guardrail stack.
DEFAULT_SYSTEM_PROMPT = (
    "You are an assistant working inside a sandboxed workspace directory. "
    "You can read/edit/write files and run Python via the tools listed below. "
    "All file paths you provide are interpreted relative to the workspace "
    "root. To call a tool, emit a single <tool>{\"name\": \"TOOL\", \"args\": {...}}</tool> "
    "block. When you have finished the task, reply with a brief plain-text "
    "summary (no tool calls).\n\n"
    "Tools available:\n"
    "- read_file(path) -> file contents\n"
    "- edit_file(path, old_str, new_str) -> replaces old_str with new_str; "
    "old_str must occur exactly once in the file (include context to make "
    "it unique).\n"
    "- write_file(path, content) -> overwrites or creates the file.\n"
    "- list_dir(path) -> list of {name, type} entries; use '.' for workspace root.\n"
    "- run_python(code) -> runs the code as a Python script; returns stdout "
    "(last ~4000 chars), stderr (last ~4000 chars), and exit_code. "
    "Output is truncated to avoid extremely large responses.\n"
    "- query_rag(corpus_id, q, top_k=5) -> {chunks: [{id, source_file, chunk_index, content, score}, ...]}\n"
    "  Searches a RAG corpus by semantic similarity. Use to ground responses in source text.\n"
    "- generate_image(prompt, filename=None) -> {filename, bytes}\n"
    "  Generate an image via diffusion; PNG is written to the workspace directory.\n"
)


ModelFn = Callable[[list[dict[str, Any]]], Awaitable[str]]


async def _call_image_server(
    prompt: str,
    filename: str | None,
    *,
    workspace_root: Path,
) -> dict[str, Any]:
    """POST to /image/generate, poll SSE, decode + write the PNG.

    Uses the existing mlx-studio /image/generate endpoint (verified at
    web-app/routers/image.py:130). The endpoint accepts {prompt, width,
    height} and returns {job_id}; the SSE stream emits JSON status
    updates ending with {status:"done", image: <base64>}.

    Note: this self-calls the web-app over HTTP (same process). It works
    and matches the design plan's "POSTs to the existing endpoint"
    decision, but a future iteration may refactor to call the
    underlying image-server helper directly (in-process) to skip a hop.
    """
    target_name = filename or f"image-{time.time_ns()}.png"
    target = _resolve(workspace_root, target_name)
    target.parent.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=5.0, read=300.0, write=10.0, pool=5.0)
    ) as client:
        # Kick off the job
        kickoff = await client.post(
            f"{WEB_APP_URL}/image/generate",
            data={"prompt": prompt, "width": 1024, "height": 1024},
        )
        kickoff.raise_for_status()
        job_id = kickoff.json()["job_id"]

        # Stream progress until done
        async with client.stream(
            "GET", f"{WEB_APP_URL}/image/generate/{job_id}/stream"
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = json.loads(line[len("data: "):])
                if payload.get("status") == "done" and "image" in payload:
                    image_bytes = base64.b64decode(payload["image"])
                    target.write_bytes(image_bytes)
                    return {"filename": target_name, "bytes": len(image_bytes)}
                if payload.get("status") in ("failed", "cancelled"):
                    raise RuntimeError(
                        f"image generation {payload['status']}: {payload}"
                    )
    raise RuntimeError("image stream closed without producing a result")


class WorkspaceRunner:
    """Runs a single user→assistant turn against a workspace.

    Constructor takes the workspace id and a `model_fn` callable. The
    default `model_fn` calls the local text-server; tests inject a stub.
    """

    def __init__(
        self,
        *,
        conn: sqlite3.Connection,
        workspace_id: int,
        model_fn: ModelFn | None = None,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self._conn = conn
        self._ws_id = workspace_id
        self._model_fn = model_fn or self._default_model_fn
        self._max_rounds = max_rounds
        self._system_prompt = system_prompt
        self._on_token: Callable[[str], None] | None = None

    async def run_turn(
        self,
        user_message: str,
        *,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        """Execute one user→assistant turn. Returns the final content.

        If `on_token` is provided, the runner forwards each model-output token
        to it before persisting the message. Callers (the streaming endpoint)
        use this to live-emit SSE deltas.

        Note: Messages are committed incrementally as they are dispatched.
        A crash mid-turn will leave partial state in the database (e.g., an
        assistant message with tool calls but no tool results). This is
        acceptable for Phase 2; Phase 6 will introduce checkpoints and
        rollback/revert functionality to handle recovery.
        """
        ws = self._conn.execute(
            "SELECT root_dir FROM workspaces WHERE id = ?", (self._ws_id,)
        ).fetchone()
        if ws is None:
            raise ValueError(f"workspace {self._ws_id} not found")
        root = Path(ws["root_dir"])
        if not root.is_dir():
            raise FileNotFoundError(f"workspace root missing: {root}")

        self._on_token = on_token
        try:
            # Take a snapshot before the turn runs (BEFORE persisting the user message)
            # NOTE: Snapshot is outside the transaction below; concurrent tool writes
            # between snapshot and DB commit could leak into the next checkpoint.
            # Acceptable for single-user MVP.
            seq = next_seq(self._conn, workspace_id=self._ws_id)
            snapshot_workspace(root, seq=seq)

            # Persist the user message and record the checkpoint in a transaction.
            # NOTE: A crash between snapshot (above) and the transaction (below) leaves
            # an orphaned directory at .checkpoints/<seq>/. Acceptable for MVP;
            # Phase 7 cleanup can add recovery.
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._persist_message("user", user_message, tool_calls=[])
                user_msg_id = self._conn.execute(
                    "SELECT MAX(id) AS m FROM workspace_messages WHERE workspace_id=?",
                    (self._ws_id,),
                ).fetchone()["m"]

                # Record the checkpoint with the user message ID
                self._conn.execute(
                    "INSERT INTO workspace_checkpoints (workspace_id, seq, message_id) "
                    "VALUES (?, ?, ?)",
                    (self._ws_id, seq, user_msg_id),
                )
                self._conn.commit()
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

            messages = self._build_messages_for_model()

            for round_idx in range(self._max_rounds):
                raw = await self._model_fn(messages)
                parsed = parse_tool_calls(raw)
                # Persist the raw response (including tool blocks) so that replays from
                # the database include the full context. The chat UI can strip tool blocks
                # for display in Phase 3/5 (rendering).
                self._persist_message(
                    "assistant", raw, tool_calls=parsed.tool_calls
                )

                if not parsed.tool_calls:
                    return parsed.content

                # Append the assistant's raw response (with tool blocks) once before
                # dispatching its tool calls. This ensures the model sees the full
                # context on the next round without duplication.
                messages.append(
                    {"role": "assistant", "content": raw}
                )

                for call in parsed.tool_calls:
                    result_text = await self._dispatch_tool(root, call)
                    self._persist_message("tool", result_text, tool_calls=[])
                    messages.append(
                        {"role": "tool", "content": result_text}
                    )

            cap_msg = (
                f"(workspace runner stopped at max rounds = {self._max_rounds}; "
                "the model kept emitting tool calls without a final answer.)"
            )
            self._persist_message("assistant", cap_msg, tool_calls=[])
            return cap_msg
        finally:
            self._on_token = None

    async def _dispatch_tool(self, root: Path, call: dict[str, Any]) -> str:
        """Run a single tool call; return its result serialized as text."""
        if call.get("_parse_error"):
            return f"ERROR parsing tool block: {call['_parse_error']}"
        name = call.get("name", "?")
        args = call.get("args") or {}
        try:
            if name == "read_file":
                return read_file(root, args["path"])
            if name == "write_file":
                result = write_file(root, args["path"], args.get("content", ""))
                return json.dumps(result)
            if name == "edit_file":
                result = edit_file(
                    root, args["path"], args["old_str"], args["new_str"]
                )
                return json.dumps(result)
            if name == "list_dir":
                entries = list_dir(root, args.get("path", "."))
                return json.dumps({"entries": entries})
            if name == "run_python":
                result = run_python(root, args.get("code", ""))
                return json.dumps(result)
            if name == "query_rag":
                corpus_id = int(args["corpus_id"])
                q = args["q"]
                top_k = int(args.get("top_k", 5))
                chunks = retrieve_chunks(self._conn, corpus_id, q, top_k=top_k)
                return json.dumps({"chunks": chunks})
            if name == "generate_image":
                # Validate filename before calling image server.
                filename = args.get("filename")
                if filename:
                    # Validates path doesn't escape root; raises PathEscapeError if it does.
                    _resolve(root, filename)
                result = await _call_image_server(
                    args["prompt"], filename, workspace_root=root,
                )
                return json.dumps(result)
            return f"ERROR unknown tool: {name!r}"
        except PathEscapeError as exc:
            return f"ERROR path escapes workspace root: {exc}"
        except FileNotFoundError as exc:
            return f"ERROR file not found: {exc}"
        except ValueError as exc:
            return f"ERROR invalid argument: {exc}"
        except KeyError as exc:
            return f"ERROR missing required arg: {exc}"
        except Exception as exc:  # noqa: BLE001
            return f"ERROR {type(exc).__name__}: {exc}"

    def _build_messages_for_model(self) -> list[dict[str, Any]]:
        """Construct the message list to send to the text-server.

        Includes (in order):
        1. System prompt + tool documentation + available corpora.
        2. Skills retrieved by semantic similarity on the latest user
           message — same pattern as routers/chat.py:94-98.
        3. Replay of persisted workspace history.
        """
        rows = self._conn.execute(
            "SELECT role, content FROM workspace_messages "
            "WHERE workspace_id = ? ORDER BY id",
            (self._ws_id,),
        ).fetchall()

        # Skills injection — same call shape as routers/chat.py:94-98.
        last_user = next(
            (r["content"] for r in reversed(rows) if r["role"] == "user"),
            None,
        )
        skills_ctx = ""
        if last_user:
            try:
                skills = retrieve_skills(self._conn, last_user, top_k=3)
                skills_ctx = format_skills_for_context(skills) or ""
            except Exception as exc:  # noqa: BLE001
                # Skills retrieval is best-effort; never fatal.
                logging.getLogger(__name__).debug(
                    "skills retrieval failed: %s", exc
                )
                skills_ctx = ""

        # Enumerate corpora so the model knows valid corpus_id values.
        corpora_list = "  (none indexed yet)"
        try:
            corpora_rows = self._conn.execute(
                "SELECT id, name FROM corpora ORDER BY id LIMIT 50"
            ).fetchall()
            corpora_list = "\n".join(
                f"  - id={r['id']}: {r['name']}" for r in corpora_rows
            ) or "  (none indexed yet)"
        except Exception as exc:  # noqa: BLE001
            # Corpora table may not exist in early workspace states.
            logging.getLogger(__name__).debug(
                "corpora enumeration failed: %s", exc
            )
            corpora_list = "  (none indexed yet)"
        system = self._system_prompt + (
            f"\n\nAvailable RAG corpora for query_rag:\n{corpora_list}\n"
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system}
        ]
        if skills_ctx:
            messages.append({"role": "system", "content": skills_ctx})
        for row in rows:
            messages.append({"role": row["role"], "content": row["content"]})
        return messages

    def _persist_message(
        self, role: str, content: str, *, tool_calls: list[dict[str, Any]]
    ) -> None:
        """Insert one row into workspace_messages.

        Also bumps `last_active_at` on the workspace row. If this is the
        first user message in the workspace (i.e. the workspace's
        `summary` field is still empty), it's populated with the first
        80 chars of `content` — that's the design's "one-line summary"
        shown in the workspace list (design DoD item 6). No model call
        is needed; the user's first prompt is a reasonable hint.
        """
        self._conn.execute(
            "INSERT INTO workspace_messages "
            "(workspace_id, role, content, tool_calls_json) "
            "VALUES (?, ?, ?, ?)",
            (self._ws_id, role, content, json.dumps(tool_calls)),
        )
        if role == "user":
            self._conn.execute(
                "UPDATE workspaces "
                "SET summary = CASE WHEN summary = '' THEN ? ELSE summary END, "
                "    last_active_at = datetime('now') "
                "WHERE id = ?",
                (content[:80], self._ws_id),
            )
        else:
            self._conn.execute(
                "UPDATE workspaces SET last_active_at = datetime('now') WHERE id = ?",
                (self._ws_id,),
            )
        self._conn.commit()

    async def _default_model_fn(
        self, messages: list[dict[str, Any]]
    ) -> str:
        """Stream from the text-server /chat endpoint, forwarding tokens
        to self._on_token (if set) and returning the full accumulated text.
        """
        accumulated: list[str] = []
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream(
                "POST",
                f"{TEXT_SERVER_URL}/chat",
                json={"messages": messages, "stream": True},
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    token = line[len("data: "):]
                    if token == "[DONE]":
                        break
                    # Unescape \\n → \n (the text-server inherits this from
                    # the chat.py SSE convention — verified file:line in
                    # chat.py:121).
                    # NOTE: ambiguous if token literally contains "\\n" (two chars). Inherited from chat.py SSE convention; revisit in Phase 5 with JSON encoding.
                    token = token.replace("\\n", "\n")
                    accumulated.append(token)
                    if self._on_token is not None:
                        self._on_token(token)
        return "".join(accumulated)
