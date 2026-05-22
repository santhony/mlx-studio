# Workspace Tab MVP — Phase 4: Cross-Tab Tools

**Goal:** The model can call `query_rag(corpus_id, q)` and `generate_image(prompt, filename)` as tools. RAG returns scored chunks the model can cite; image generation writes a PNG into the workspace directory.

**Architecture:** Extend the tool registry in `workspace_runner.py:_dispatch_tool`. `query_rag` calls `indexer.retrieve_chunks(conn, corpus_id, query, top_k=5)` directly (no HTTP — same process). `generate_image` POSTs to the existing `/image/generate` endpoint, polls its SSE stream (`image.py:130`), decodes the final base64 PNG, writes it to the workspace.

**Tech Stack:** httpx (already used), base64 (stdlib). No new dependencies.

**Scope:** Phase 4 of 7.

**Codebase verified:** 2026-05-21. `indexer.retrieve_chunks` at `web-app/indexer.py:519` — returns `list[dict]` with `{id, source_file, chunk_index, content, score}`. Image gen at `web-app/routers/image.py:130` — POST `/image/generate` returns `{job_id: ...}`, then SSE `/image/generate/{job_id}/stream` yields JSON `{status, step, total, elapsed, image (base64 on done)}`.

---

<!-- START_TASK_1 -->
### Task 1: query_rag tool

**Files:**
- Modify: `web-app/workspace_runner.py:_dispatch_tool` — add `query_rag` branch.
- Modify: `web-app/workspace_runner.py:DEFAULT_SYSTEM_PROMPT` — document the new tool.
- Create: `web-app/tests/test_workspace_runner_query_rag.py`

**Step 1: Write the failing test**

```python
"""Tests for the query_rag tool integration."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from workspace_runner import WorkspaceRunner


@pytest.fixture
def ws_id(conn: sqlite3.Connection, tmp_path: Path) -> int:
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    cur = conn.execute(
        "INSERT INTO workspaces (name, root_dir) VALUES (?, ?)",
        ("t", str(ws_dir)),
    )
    conn.commit()
    return cur.lastrowid


@pytest.mark.asyncio
async def test_query_rag_dispatches_to_indexer(
    conn: sqlite3.Connection, ws_id: int
) -> None:
    """query_rag tool dispatch calls indexer.retrieve_chunks and serializes result."""
    sample_chunks = [
        {"id": 1, "source_file": "a.md", "chunk_index": 0, "content": "alpha", "score": 0.9},
        {"id": 2, "source_file": "b.md", "chunk_index": 3, "content": "beta", "score": 0.7},
    ]

    async def model(messages):
        rounds = sum(1 for m in messages if m["role"] == "tool")
        if rounds == 0:
            return '<tool>{"name":"query_rag","args":{"corpus_id":1,"q":"alpha"}}</tool>'
        return "I found 2 chunks."

    with patch("workspace_runner.retrieve_chunks", return_value=sample_chunks) as mock:
        runner = WorkspaceRunner(conn=conn, workspace_id=ws_id, model_fn=model)
        final = await runner.run_turn("query the corpus")
        assert "found 2 chunks" in final
        mock.assert_called_once()
        # Verify it was called with the right args
        args = mock.call_args[1] if mock.call_args[1] else mock.call_args[0]
        # retrieve_chunks signature: conn, corpus_id, query, top_k

    tool_msg = conn.execute(
        "SELECT content FROM workspace_messages WHERE workspace_id=? AND role='tool'",
        (ws_id,),
    ).fetchone()
    body = json.loads(tool_msg["content"])
    assert "chunks" in body
    assert len(body["chunks"]) == 2
    assert body["chunks"][0]["content"] == "alpha"


@pytest.mark.asyncio
async def test_query_rag_handles_missing_corpus(
    conn: sqlite3.Connection, ws_id: int
) -> None:
    """If retrieve_chunks raises, the tool returns an ERROR string the model sees."""
    async def model(messages):
        if sum(1 for m in messages if m["role"] == "tool") == 0:
            return '<tool>{"name":"query_rag","args":{"corpus_id":99999,"q":"x"}}</tool>'
        return "Corpus doesn't exist, I see."

    def raising_retrieve(*args, **kwargs):
        raise ValueError("corpus 99999 not found")

    with patch("workspace_runner.retrieve_chunks", side_effect=raising_retrieve):
        runner = WorkspaceRunner(conn=conn, workspace_id=ws_id, model_fn=model)
        final = await runner.run_turn("query missing corpus")
        assert "doesn't exist" in final
```

**Step 2: Run to verify failure**

Run: `cd web-app && python -m pytest tests/test_workspace_runner_query_rag.py -v`
Expected: FAIL — `query_rag` returns "unknown tool" or import fails.

**Step 3: Add the query_rag dispatch + import**

In `workspace_runner.py`:

```python
# Add to imports at the top:
from indexer import retrieve_chunks
```

In `_dispatch_tool`, add a new branch after `run_python`:

```python
if name == "query_rag":
    corpus_id = int(args["corpus_id"])
    q = args["q"]
    top_k = int(args.get("top_k", 5))
    chunks = retrieve_chunks(self._conn, corpus_id, q, top_k=top_k)
    return json.dumps({"chunks": chunks})
```

Update `DEFAULT_SYSTEM_PROMPT` to document the tool:

```python
DEFAULT_SYSTEM_PROMPT = (
    # ... existing text ...
    "- query_rag(corpus_id, q, top_k=5) -> {chunks: [{id, source_file, chunk_index, content, score}, ...]}\n"
    "  Searches a RAG corpus by semantic similarity. Use to ground responses in source text.\n"
    # ...
)
```

The prompt should also enumerate available corpora at runtime so the model can pass valid `corpus_id` values. **Important:** this is an EXTENSION of the Phase 2 `_build_messages_for_model` — preserve the skills-retrieval block from Phase 2. The change is just appending the corpora list to the system prompt:

```python
def _build_messages_for_model(self) -> list[dict[str, Any]]:
    """... (same docstring as Phase 2, extended with corpora list) ..."""
    rows = self._conn.execute(
        "SELECT role, content FROM workspace_messages "
        "WHERE workspace_id = ? ORDER BY id",
        (self._ws_id,),
    ).fetchall()

    # Skills injection — UNCHANGED from Phase 2. Do not remove.
    last_user = next(
        (r["content"] for r in reversed(rows) if r["role"] == "user"),
        None,
    )
    skills_ctx = ""
    if last_user:
        try:
            skills = retrieve_skills(self._conn, last_user, top_k=3)
            skills_ctx = format_skills_for_context(skills) or ""
        except Exception:
            skills_ctx = ""

    # NEW: enumerate corpora so the model knows valid corpus_id values.
    corpora_rows = self._conn.execute(
        "SELECT id, name FROM corpora ORDER BY id"
    ).fetchall()
    corpora_list = "\n".join(
        f"  - id={r['id']}: {r['name']}" for r in corpora_rows
    ) or "  (none indexed yet)"
    system = self._system_prompt + (
        f"\n\nAvailable RAG corpora for query_rag:\n{corpora_list}\n"
    )

    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    if skills_ctx:
        messages.append({"role": "system", "content": skills_ctx})
    for row in rows:
        messages.append({"role": row["role"], "content": row["content"]})
    return messages
```

**Step 4: Run tests**

Run: `cd web-app && python -m pytest tests/test_workspace_runner_query_rag.py -v`
Expected: 2 tests pass.

**Step 5: Commit**

```bash
cd /Users/santhony/Documents/dev_claude
git add mlx-studio/web-app/workspace_runner.py mlx-studio/web-app/tests/test_workspace_runner_query_rag.py
git commit -m "feat(workspace): query_rag tool — search corpus chunks via indexer"
```
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: generate_image tool

**Files:**
- Modify: `web-app/workspace_runner.py:_dispatch_tool` — add `generate_image` branch + helper.
- Modify: `web-app/workspace_runner.py:DEFAULT_SYSTEM_PROMPT` — document the tool.
- Create: `web-app/tests/test_workspace_runner_generate_image.py`

**Step 1: Write the failing test**

```python
"""Tests for the generate_image tool."""
from __future__ import annotations

import base64
import json
import sqlite3
from pathlib import Path
from unittest.mock import patch, AsyncMock

import pytest

from workspace_runner import WorkspaceRunner


@pytest.fixture
def ws_id(conn: sqlite3.Connection, tmp_path: Path) -> int:
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    cur = conn.execute(
        "INSERT INTO workspaces (name, root_dir) VALUES (?, ?)",
        ("t", str(ws_dir)),
    )
    conn.commit()
    return cur.lastrowid


@pytest.mark.asyncio
async def test_generate_image_writes_png_to_workspace(
    conn: sqlite3.Connection, ws_id: int, tmp_path: Path
) -> None:
    """generate_image returns a PNG written to the workspace dir."""
    ws_dir = tmp_path / "ws"
    fake_png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    fake_png_b64 = base64.b64encode(fake_png_bytes).decode()

    async def model(messages):
        if sum(1 for m in messages if m["role"] == "tool") == 0:
            return '<tool>{"name":"generate_image","args":{"prompt":"a cat","filename":"cat.png"}}</tool>'
        return "Generated cat.png"

    async def stub_generate(prompt: str, filename: str | None, *, workspace_root: Path) -> dict:
        target = workspace_root / "cat.png"
        target.write_bytes(fake_png_bytes)
        return {"filename": "cat.png", "bytes": len(fake_png_bytes)}

    with patch("workspace_runner._call_image_server", new=stub_generate):
        runner = WorkspaceRunner(conn=conn, workspace_id=ws_id, model_fn=model)
        final = await runner.run_turn("draw a cat")
        assert "Generated cat.png" in final

    assert (ws_dir / "cat.png").read_bytes() == fake_png_bytes
```

**Step 2: Run to verify failure**

Run: `cd web-app && python -m pytest tests/test_workspace_runner_generate_image.py -v`
Expected: FAIL.

**Step 3: Implement generate_image**

In `workspace_runner.py`, add the helper at module level:

```python
import base64
import time

WEB_APP_URL = "http://127.0.0.1:8080"  # /image/generate is mounted on the web-app


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
    target_name = filename or f"image-{int(time.time())}.png"
    target = workspace_root / target_name

    async with httpx.AsyncClient(timeout=300.0) as client:
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
```

**Make `_dispatch_tool` async (single-step refactor):**

Phase 2 wrote `_dispatch_tool` as a synchronous method. The `generate_image` tool requires an `await` on `_call_image_server`, so convert the dispatcher to async. This is the smallest refactor:

In `workspace_runner.py`, change the method signature:

```python
# BEFORE (Phase 2):
def _dispatch_tool(self, root: Path, call: dict[str, Any]) -> str:
    # ... existing body ...

# AFTER (Phase 4):
async def _dispatch_tool(self, root: Path, call: dict[str, Any]) -> str:
    # ... existing body unchanged for sync tools ...
    # plus the new branch:
    if name == "generate_image":
        result = await _call_image_server(
            args["prompt"], args.get("filename"), workspace_root=root,
        )
        return json.dumps(result)
    # ...existing branches for read_file/write_file/edit_file/list_dir/run_python/query_rag...
```

Update `run_turn` — the single call site — to `await`:

```python
# In run_turn's dispatcher loop:
for call in parsed.tool_calls:
    result_text = await self._dispatch_tool(root, call)  # ADD await
    # ... rest unchanged
```

No test changes required from Phase 2: those tests never called `_dispatch_tool` directly — they exercised `run_turn`, which is already async. The async-conversion of an internal method is invisible to callers using the public surface.

Verify with: `cd web-app && python -m pytest tests/test_workspace_runner.py tests/test_workspace_tools.py -v` — all should still pass.

**Step 4: Run tests**

Run: `cd web-app && python -m pytest tests/test_workspace_runner_generate_image.py tests/test_workspace_runner.py -v`
Expected: All pass.

**Step 5: Commit**

```bash
cd /Users/santhony/Documents/dev_claude
git add mlx-studio/web-app/workspace_runner.py mlx-studio/web-app/tests/test_workspace_runner_generate_image.py
git commit -m "feat(workspace): generate_image tool calls /image/generate + writes PNG"
```
<!-- END_TASK_2 -->

---

**Phase 4 done when:**
- `query_rag` test passes — model calls tool, indexer is invoked, chunks come back as JSON to the model.
- `generate_image` test passes — model calls tool, fake PNG bytes land on disk inside the workspace dir.
- Manual smoke (optional): with image server up, ask a real model to generate a sample image; PNG appears in `data/workspaces/<id>/`.

**Phase 4 leaves these for later phases:** inline image rendering (Phase 5), markdown rendering of RAG citations (Phase 5).
