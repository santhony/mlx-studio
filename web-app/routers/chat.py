"""
chat.py — Chat router.

Provides named sessions with persistent multi-turn history (SQLite).
SSE from the text server is proxied to the browser via httpx async streaming.

Routes:
  GET  /chat/                            → session list + chat UI
  POST /chat/sessions                    → create new session, redirect to it
  GET  /chat/{session_id}                → chat view for a session
  POST /chat/{session_id}/send           → send message, start streaming response
  GET  /chat/{session_id}/stream         → SSE proxy from text server to browser
  POST /chat/sessions/{session_id}/delete → delete session
"""

import sqlite3
from typing import AsyncGenerator

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

router = APIRouter(prefix="/chat")
templates = Jinja2Templates(directory="templates")

TEXT_SERVER = "http://127.0.0.1:8766"


# ── Session helpers ────────────────────────────────────────────────────────────

def _get_sessions(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT id, name, updated_at FROM sessions ORDER BY updated_at DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def _get_session(conn: sqlite3.Connection, session_id: int) -> dict | None:
    row = conn.execute(
        "SELECT id, name, created_at, updated_at FROM sessions WHERE id = ?",
        (session_id,),
    ).fetchone()
    return dict(row) if row else None


def _get_messages(conn: sqlite3.Connection, session_id: int) -> list[dict]:
    rows = conn.execute(
        "SELECT id, role, content, created_at FROM messages "
        "WHERE session_id = ? ORDER BY id",
        (session_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def _append_message(conn: sqlite3.Connection, session_id: int, role: str, content: str) -> int:
    cur = conn.execute(
        "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
        (session_id, role, content),
    )
    conn.execute(
        "UPDATE sessions SET updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE id = ?",
        (session_id,),
    )
    conn.commit()
    return cur.lastrowid


# ── SSE proxy ─────────────────────────────────────────────────────────────────

async def _proxy_sse(
    client: httpx.AsyncClient,
    session_id: int,
    conn: sqlite3.Connection,
) -> AsyncGenerator[str, None]:
    """
    Fetch the pending assistant message for session_id from the text server,
    stream it as SSE to the browser, and persist the complete response to SQLite.
    """
    # Retrieve all messages for context
    messages = _get_messages(conn, session_id)
    payload = {
        "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
        "max_tokens": 2048,
    }

    full_response: list[str] = []

    try:
        async with client.stream(
            "POST",
            f"{TEXT_SERVER}/chat",
            json=payload,
            timeout=300.0,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                token = line[len("data: "):]
                if token == "[DONE]":
                    break
                if token.startswith("ERROR:"):
                    yield f"data: {token}\n\n"
                    return
                # Unescape newlines (escaped in text server for SSE safety)
                decoded = token.replace("\\n", "\n")
                full_response.append(decoded)
                # Re-escape for SSE wire format
                yield f"data: {token}\n\n"
    except httpx.HTTPStatusError as exc:
        yield f"data: ERROR: text server returned {exc.response.status_code}\n\n"
        return
    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        yield f"data: ERROR: text server unavailable: {exc}\n\n"
        return

    # Persist completed assistant response
    complete_text = "".join(full_response)
    if complete_text.strip():
        _append_message(conn, session_id, "assistant", complete_text)

    yield "data: [DONE]\n\n"


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
async def chat_list(request: Request):
    conn: sqlite3.Connection = request.app.state.db
    sessions = _get_sessions(conn)
    return templates.TemplateResponse(
        request=request,
        name="chat_list.html",
        context={"sessions": sessions},
    )


@router.post("/sessions")
async def create_session(request: Request):
    form = await request.form()
    name = (form.get("name") or "").strip() or "New chat"
    conn: sqlite3.Connection = request.app.state.db
    cur = conn.execute("INSERT INTO sessions (name) VALUES (?)", (name,))
    conn.commit()
    return RedirectResponse(url=f"/chat/{cur.lastrowid}", status_code=303)


@router.get("/{session_id}", response_class=HTMLResponse)
async def chat_view(session_id: int, request: Request):
    conn: sqlite3.Connection = request.app.state.db
    session = _get_session(conn, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session not found")
    messages = _get_messages(conn, session_id)
    sessions = _get_sessions(conn)
    return templates.TemplateResponse(
        request=request,
        name="chat.html",
        context={
            "session": session,
            "messages": messages,
            "sessions": sessions,
        },
    )


@router.post("/{session_id}/send", response_class=HTMLResponse)
async def send_message(session_id: int, request: Request):
    """
    Persist the user message, then return an HTMX fragment that contains:
    1. The rendered user message bubble
    2. An empty assistant bubble with an SSE listener that streams the response
    """
    conn: sqlite3.Connection = request.app.state.db
    session = _get_session(conn, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session not found")

    form = await request.form()
    content = (form.get("content") or "").strip()
    if not content:
        return HTMLResponse('<p class="error">Message cannot be empty.</p>', status_code=400)

    _append_message(conn, session_id, "user", content)

    # Return HTMX fragment: user bubble + streaming assistant bubble
    escaped_content = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return HTMLResponse(f"""
<div class="msg msg-user">
    <div class="msg-content">{escaped_content}</div>
</div>
<div class="msg msg-assistant"
     hx-ext="sse"
     sse-connect="/chat/{session_id}/stream"
     sse-swap="message"
     hx-swap="beforeend">
    <div class="msg-content" id="streaming-{session_id}">
        <span class="spinner"></span>
    </div>
</div>
""")


@router.get("/{session_id}/stream")
async def stream_response(session_id: int, request: Request):
    """SSE endpoint: proxies text server tokens to the browser."""
    conn: sqlite3.Connection = request.app.state.db
    session = _get_session(conn, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session not found")

    client: httpx.AsyncClient = request.app.state.http_client

    async def _sse_with_event_wrap() -> AsyncGenerator[str, None]:
        """Wrap each token as a named SSE 'message' event for HTMX sse-swap."""
        async for chunk in _proxy_sse(client, session_id, conn):
            if chunk.startswith("data: "):
                token = chunk[len("data: "):]
                if token.strip() == "[DONE]":
                    # Send a final event that clears the spinner
                    yield "event: message\ndata: \n\n"
                    return
                # Unescape for HTML display
                decoded = token.replace("\\n", "\n")
                # HTMX sse-swap appends the data payload as HTML
                html_token = decoded.replace("\n", "<br>")
                yield f"event: message\ndata: {html_token}\n\n"

    return StreamingResponse(
        _sse_with_event_wrap(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/sessions/{session_id}/delete")
async def delete_session(session_id: int, request: Request):
    conn: sqlite3.Connection = request.app.state.db
    conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    return RedirectResponse(url="/chat/", status_code=303)
