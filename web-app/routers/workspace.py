"""Workspace router — CRUD + detail endpoints.

# pattern: Imperative Shell
# This module is HTTP-edge code that wraps the pure workspace_store
# helpers (Functional Core) with FastAPI plumbing and filesystem side
# effects. Keep business logic out of this file — push it into helpers.
#
# SQLite connection is accessed via request.app.state.db, matching the
# existing convention in routers/chat.py and elsewhere in the codebase.
"""
from __future__ import annotations

import asyncio
import shutil
import sqlite3
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.templating import Jinja2Templates

import workspace_store
from workspace_checkpoint import restore_checkpoint
from workspace_runner import WorkspaceRunner
from workspace_tools import PathEscapeError, _resolve
from workspace_render import render_message

router = APIRouter(prefix="/workspace", tags=["workspace"])
# Note: This router has its own Jinja2Templates instance (Phase 1 pattern).
# The render_message global must be registered here even though it's also
# registered in main.py, because each templates instance has its own
# environment.
templates = Jinja2Templates(directory="templates")
templates.env.globals["render_message"] = render_message


def _data_root(request: Request) -> Path:
    """Resolve the workspaces data root.

    Production: web-app/data/workspaces/. Tests override via
    app.state.data_root to a tmp_path so each test has an isolated tree.
    """
    override = getattr(request.app.state, "data_root", None)
    if override is not None:
        return Path(override) / "workspaces"
    return Path("data") / "workspaces"


@router.get("/", response_class=HTMLResponse)
async def list_workspaces_view(request: Request) -> HTMLResponse:
    """List all workspaces with a 'New Workspace' form."""
    conn: sqlite3.Connection = request.app.state.db
    rows = workspace_store.list_workspaces(conn)
    return templates.TemplateResponse(
        request=request,
        name="workspace_list.html",
        context={"workspaces": rows},
    )


@router.post("/")
async def create_workspace_view(
    request: Request, name: str = Form("")
) -> Response:
    """Create a workspace + its on-disk directory, then redirect to it.

    Uses a two-step insert: create the row (so we can get the id), then
    set root_dir to data/workspaces/<id>/ once known. If the on-disk
    mkdir fails, we delete the row to keep state consistent.
    """
    conn: sqlite3.Connection = request.app.state.db
    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")

    root = _data_root(request)
    root.mkdir(parents=True, exist_ok=True)

    # Wrap INSERT + mkdir + UPDATE in a transaction to prevent orphaned rows
    # with empty root_dir if mkdir fails. We do this in the router rather than
    # calling workspace_store.create_workspace so we can control the transaction.
    try:
        conn.execute("BEGIN IMMEDIATE")
        cur = conn.execute(
            "INSERT INTO workspaces (name, root_dir) VALUES (?, ?)",
            (name, ""),
        )
        ws_id = cur.lastrowid
        ws_dir = root / str(ws_id)
        try:
            ws_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            conn.execute("ROLLBACK")
            raise HTTPException(
                status_code=500, detail=f"failed to create workspace dir: {exc}"
            ) from exc
        conn.execute(
            "UPDATE workspaces SET root_dir = ? WHERE id = ?",
            (str(ws_dir), ws_id),
        )
        conn.commit()
        # Fetch the complete row to match create_workspace's return value
        ws = conn.execute(
            "SELECT * FROM workspaces WHERE id = ?", (ws_id,)
        ).fetchone()
        ws = dict(ws)
    except HTTPException:
        raise
    except Exception as exc:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        raise HTTPException(
            status_code=500, detail=f"failed to create workspace: {exc}"
        ) from exc
    return RedirectResponse(url=f"/workspace/{ws['id']}", status_code=303)


@router.get("/{workspace_id}", response_class=HTMLResponse)
async def workspace_detail(
    workspace_id: int, request: Request
) -> HTMLResponse:
    """Workspace detail page with chat surface."""
    conn: sqlite3.Connection = request.app.state.db
    ws = workspace_store.get_workspace(conn, workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail="workspace not found")
    workspace_store.update_last_active(conn, workspace_id)
    rows = conn.execute(
        """
        SELECT m.id, m.role, m.content, m.tool_calls_json, c.seq AS checkpoint_seq
        FROM workspace_messages m
        LEFT JOIN workspace_checkpoints c
          ON c.workspace_id = m.workspace_id AND c.message_id = m.id
        WHERE m.workspace_id = ? ORDER BY m.id
        """,
        (workspace_id,),
    ).fetchall()
    return templates.TemplateResponse(
        request=request,
        name="workspace.html",
        context={"workspace": ws, "messages": [dict(r) for r in rows]},
    )


@router.delete("/{workspace_id}")
async def delete_workspace_view(
    workspace_id: int, request: Request
) -> Response:
    """Delete workspace row and its on-disk directory."""
    conn: sqlite3.Connection = request.app.state.db
    ws = workspace_store.get_workspace(conn, workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail="workspace not found")
    workspace_store.delete_workspace(conn, workspace_id)
    ws_dir = Path(ws["root_dir"])
    if ws_dir.is_dir():
        shutil.rmtree(ws_dir)
    return Response(status_code=204)


@router.post("/{workspace_id}/messages")
async def send_message(
    workspace_id: int,
    request: Request,
    content: str = Form(...),
) -> JSONResponse:
    """Run one user→assistant turn synchronously.

    Phase 2 is non-streaming; Phase 3 adds the streaming SSE variant.
    Returns the final assistant content as JSON. Connection is read
    from request.app.state.db per the codebase-wide convention.
    """
    conn: sqlite3.Connection = request.app.state.db
    ws = workspace_store.get_workspace(conn, workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail="workspace not found")

    runner = WorkspaceRunner(conn=conn, workspace_id=workspace_id)
    final = await runner.run_turn(content)
    return JSONResponse({"content": final})


@router.post("/{workspace_id}/messages/stream")
async def send_message_stream(
    workspace_id: int,
    request: Request,
    content: str = Form(...),
) -> StreamingResponse:
    """Stream a user→assistant turn as SSE.

    The stream yields one `data: <token>\\n\\n` event per model token,
    then a single `data: [DONE]\\n\\n` event when the turn completes.
    The browser listens via inline EventSource (see workspace.html).
    Connection read via request.app.state.db per codebase convention.
    """
    conn: sqlite3.Connection = request.app.state.db
    # NOTE: connection is shared across requests. Concurrent turns on the same workspace will interleave statements at await points; sqlite3 tolerates this in-process but mid-turn partial commits can interleave. Single-user MVP — revisit in Phase 6 with per-turn transactions.
    ws = workspace_store.get_workspace(conn, workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail="workspace not found")

    # asyncio.Queue is the bridge between the runner's on_token callback
    # (sync, called from inside an async loop) and the StreamingResponse
    # generator that the browser reads from.
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    # emit is called from _default_model_fn, which runs on the same event loop as event_stream. put_nowait on an unbounded queue cannot raise QueueFull. If this is ever moved to a thread, switch to loop.call_soon_threadsafe(queue.put_nowait, token).
    def emit(token: str) -> None:
        queue.put_nowait(token)

    async def run_and_finish() -> None:
        try:
            runner = WorkspaceRunner(conn=conn, workspace_id=workspace_id)
            await runner.run_turn(content, on_token=emit)
        finally:
            # Sentinel emitted only after run_turn returns; by then the final
            # assistant message is committed, so the client's follow-up GET
            # /messages-html will see it.
            queue.put_nowait(None)

    async def event_stream():
        # Kick off the runner in a background task; consume the queue
        # in the foreground and yield SSE events.
        task = asyncio.create_task(run_and_finish())
        try:
            while True:
                token = await queue.get()
                if token is None:
                    yield "data: [DONE]\n\n"
                    break
                # NOTE: ambiguous if token literally contains "\\n" (two chars). Inherited from chat.py SSE convention; revisit in Phase 5 with JSON encoding.
                escaped = token.replace("\n", "\\n")
                yield f"data: {escaped}\n\n"
        finally:
            if not task.done():
                # Best-effort cancellation: Starlette only fires the generator's finally on the next yield attempt after the client disconnects. If the model is slow, the runner may continue briefly. Acceptable for local single-user use.
                task.cancel()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/{workspace_id}/revert/{seq}")
async def revert_to_checkpoint(
    workspace_id: int, seq: int, request: Request
) -> Response:
    """Restore workspace state to checkpoint `seq` and truncate messages.

    Semantics: the user message that triggered the snapshot is itself
    removed (because the checkpoint records the state IMMEDIATELY BEFORE
    that message — reverting means undoing the user's prompt too, so
    they can re-edit and re-submit). Everything from that message
    forward (later assistants, tool results, later user turns) is also
    removed. Later checkpoints become unreachable and are dropped.

    The checkpoint row at this seq is preserved, so revert is itself
    idempotent (re-reverting to seq N from a state still after N just
    works).
    """
    conn: sqlite3.Connection = request.app.state.db
    ws = workspace_store.get_workspace(conn, workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail="workspace not found")
    ckpt = conn.execute(
        "SELECT message_id FROM workspace_checkpoints "
        "WHERE workspace_id = ? AND seq = ?",
        (workspace_id, seq),
    ).fetchone()
    if ckpt is None:
        raise HTTPException(status_code=404, detail="checkpoint not found")

    root = Path(ws["root_dir"])
    restore_checkpoint(root, seq=seq)

    # Truncate messages from the checkpoint's user message onward (inclusive).
    if ckpt["message_id"] is not None:
        conn.execute(
            "DELETE FROM workspace_messages "
            "WHERE workspace_id = ? AND id >= ?",
            (workspace_id, ckpt["message_id"]),
        )
        # Drop later checkpoints (they're no longer reachable). The
        # checkpoint at `seq` itself stays — it still describes a valid
        # reachable state.
        conn.execute(
            "DELETE FROM workspace_checkpoints "
            "WHERE workspace_id = ? AND seq > ?",
            (workspace_id, seq),
        )

        # Remove orphaned snapshot directories for reverted checkpoints.
        # Only delete subdirectories of .checkpoints/ whose names parse as
        # integers > seq (defensive against stray files).
        checkpoints_dir = root / ".checkpoints"
        if checkpoints_dir.is_dir():
            for child in checkpoints_dir.iterdir():
                if not child.is_dir():
                    # Skip stray files; only delete named checkpoints
                    continue
                try:
                    child_seq = int(child.name)
                    if child_seq > seq:
                        shutil.rmtree(child)
                except ValueError:
                    # Skip directories whose names don't parse as integers
                    pass

        # Recompute summary from the earliest surviving user message.
        # If the revert removes the original user message, the stale summary
        # must be updated to reflect the new first user message.
        earliest_user = conn.execute(
            "SELECT content FROM workspace_messages "
            "WHERE workspace_id = ? AND role = 'user' ORDER BY id LIMIT 1",
            (workspace_id,),
        ).fetchone()
        new_summary = (earliest_user["content"][:80] if earliest_user else "")
        conn.execute(
            "UPDATE workspaces SET summary = ?, last_active_at = datetime('now') "
            "WHERE id = ?",
            (new_summary, workspace_id),
        )
    conn.commit()

    return Response(status_code=204)


@router.get("/{workspace_id}/file/{filename:path}")
async def serve_workspace_file(
    workspace_id: int,
    filename: str,
    request: Request,
) -> FileResponse:
    """Serve a file from the workspace directory.

    Path-escape rejection mirrors workspace_tools._resolve. The endpoint
    is used by inline-image rewrites from the markdown renderer.
    """
    conn: sqlite3.Connection = request.app.state.db
    ws = workspace_store.get_workspace(conn, workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail="workspace not found")
    root = Path(ws["root_dir"])
    try:
        target = _resolve(root, filename)
    except PathEscapeError:
        raise HTTPException(status_code=400, detail="path escapes workspace root")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(target)


@router.get("/{workspace_id}/messages-html", response_class=HTMLResponse)
async def messages_html(
    workspace_id: int, request: Request
) -> HTMLResponse:
    """Return rendered HTML for all workspace messages.

    Used by the frontend to fetch and display messages after the [DONE]
    signal in streaming. The render_message function must be registered
    as a Jinja global (see main.py).

    For Phase 5 scope (small message counts) per-row rendering is fine.
    Consider a single-template loop or threadpool if message counts grow
    large in future phases.
    """
    conn: sqlite3.Connection = request.app.state.db
    ws = workspace_store.get_workspace(conn, workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail="workspace not found")
    rows = conn.execute(
        """
        SELECT m.id, m.role, m.content, m.tool_calls_json, c.seq AS checkpoint_seq
        FROM workspace_messages m
        LEFT JOIN workspace_checkpoints c
          ON c.workspace_id = m.workspace_id AND c.message_id = m.id
        WHERE m.workspace_id = ? ORDER BY m.id
        """,
        (workspace_id,),
    ).fetchall()
    parts: list[str] = []
    template = templates.env.get_template("_workspace_message.html")
    for row in rows:
        parts.append(template.render(workspace=ws, message=dict(row)))
    return HTMLResponse("\n".join(parts))
