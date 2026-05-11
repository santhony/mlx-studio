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

import json
import sqlite3
from typing import AsyncGenerator

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from skills import retrieve_skills, format_skills_for_context
from chat_tools import TOOL_SCHEMAS, dispatch_tool, get_allowed_dirs

router = APIRouter(prefix="/chat")
templates = Jinja2Templates(directory="templates")

TEXT_SERVER = "http://127.0.0.1:8766"

# Maximum tool-call rounds per user turn. Prevents runaway loops if the model
# never decides to stop calling tools.
MAX_TOOL_ROUNDS = 6


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
    app_state=None,
) -> AsyncGenerator[str, None]:
    """
    Drive the chat turn: call the text server with tool schemas, parse any
    tool calls the model emits, execute them, and loop until the model
    returns a tool-call-free reply. Forward text and reasoning tokens to the
    browser as plain SSE; render tool invocations and results inline using
    `<tool>` / `<tool_result>` sentinels.
    """
    # Retrieve history + assemble messages list, with optional skills prompt.
    history = _get_messages(conn, session_id)
    msg_list = [{"role": m["role"], "content": m["content"]} for m in history]
    if msg_list:
        last_user = next(
            (m["content"] for m in reversed(history) if m["role"] == "user"),
            None,
        )
        if last_user:
            skills = retrieve_skills(conn, last_user, top_k=3)
            skills_ctx = format_skills_for_context(skills)
            if skills_ctx:
                msg_list = [{"role": "system", "content": skills_ctx}] + msg_list
            if app_state is not None:
                app_state.last_injected_skills = [s["name"] for s in skills]

    # Tool-use system prompt. Without this, DS4's training prior makes the
    # model refuse URL fetching ("I cannot browse the web") even when fetch_url
    # is in the tool list. Explicit endorsement overrides the safety default.
    msg_list.insert(0, {
        "role": "system",
        "content": (
            "You run inside a local Qwen Studio session. The tools listed in this "
            "request are real, executable, and authorized by the user. When a task "
            "would benefit from web access, code execution, or filesystem access, "
            "CALL the corresponding tool — do not refuse. Filesystem tools are "
            "restricted to an allowlist enforced server-side; refusal on the basis "
            "of supposed sandbox limits is incorrect. Do not invent rules about "
            "URL fetching or HTTP requests; if a tool exists for it, use it."
        ),
    })

    allowed_dirs = get_allowed_dirs(conn)
    # Accumulates the full visible content (including sentinels) for DB persistence.
    persisted_chunks: list[str] = []

    for round_idx in range(MAX_TOOL_ROUNDS + 1):
        payload = {
            "messages": msg_list,
            "max_tokens": 2048,
            "tools": TOOL_SCHEMAS,
        }
        # Tool calls collected during this round, to be executed after the stream ends.
        round_tool_calls: list[dict] = []

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
                    # Intercept tool_call sentinels — they go to the dispatcher,
                    # not to the browser.
                    if token.startswith("<tool_call>") and token.endswith("</tool_call>"):
                        body = token[len("<tool_call>"):-len("</tool_call>")]
                        try:
                            call = json.loads(body)
                        except json.JSONDecodeError:
                            continue
                        round_tool_calls.append(call)
                        # Render a visible "calling tool" block.
                        visible_call = (
                            f'<tool name="{_safe(call.get("name", ""))}">'
                            f'{_safe(call.get("arguments", ""))}'
                            f'</tool>'
                        )
                        persisted_chunks.append(visible_call)
                        yield f"data: {visible_call.replace(chr(10), chr(92) + 'n')}\n\n"
                        continue

                    decoded = token.replace("\\n", "\n")
                    persisted_chunks.append(decoded)
                    yield f"data: {decoded.replace(chr(10), chr(92) + 'n')}\n\n"
        except httpx.HTTPStatusError as exc:
            yield f"data: ERROR: text server returned {exc.response.status_code}\n\n"
            return
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            yield f"data: ERROR: text server unavailable: {exc}\n\n"
            return

        if not round_tool_calls:
            # Model gave a tool-call-free reply. Done.
            break

        if round_idx == MAX_TOOL_ROUNDS:
            warning = f"\n\n[stopped: reached max tool-call rounds ({MAX_TOOL_ROUNDS})]"
            persisted_chunks.append(warning)
            yield f"data: {warning.replace(chr(10), chr(92) + 'n')}\n\n"
            break

        # Append the assistant turn (with tool_calls) and each tool result so
        # the next iteration carries proper conversational state to the model.
        msg_list.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": c.get("id", f"call_{i}"),
                    "type": "function",
                    "function": {
                        "name": c.get("name", ""),
                        "arguments": c.get("arguments", ""),
                    },
                }
                for i, c in enumerate(round_tool_calls)
            ],
        })
        for call in round_tool_calls:
            result = await dispatch_tool(
                call.get("name", ""),
                call.get("arguments", ""),
                allowed_dirs,
            )
            visible_result = f"<tool_result>{_safe(result)}</tool_result>"
            persisted_chunks.append(visible_result)
            yield f"data: {visible_result.replace(chr(10), chr(92) + 'n')}\n\n"
            msg_list.append({
                "role": "tool",
                "content": result,
                "tool_call_id": call.get("id", ""),
            })

    complete_text = "".join(persisted_chunks)
    if complete_text.strip():
        _append_message(conn, session_id, "assistant", complete_text)

    yield "data: [DONE]\n\n"


def _safe(s: str) -> str:
    """Replace closing-tag sequences inside payloads so sentinels stay parseable."""
    return s.replace("</tool>", "</tool >").replace("</tool_result>", "</tool_result >")


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

    msg_id = _append_message(conn, session_id, "user", content)

    # Return HTMX fragment: user bubble + streaming assistant bubble
    # Use msg_id (unique per message) so concurrent responses don't collide
    escaped_content = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return HTMLResponse(f"""
<div class="msg msg-user">
    <div class="msg-content">{escaped_content}</div>
</div>
<div class="msg msg-assistant">
    <div class="msg-content" id="streaming-{msg_id}"><span class="spinner"></span></div>
</div>
<script>
(function() {{
    const el = document.getElementById("streaming-{msg_id}");
    const es = new EventSource("/chat/{session_id}/stream");
    let text = "";
    es.onmessage = function(e) {{
        if (e.data === "[DONE]") {{ es.close(); return; }}
        text += e.data.replace(/\\\\n/g, "\\n");
        el.innerHTML = window.renderChatContent(text);
        el.scrollIntoView({{block: "end"}});
    }};
    es.onerror = function() {{ es.close(); }};
}})();
</script>
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
        """Stream tokens as plain SSE messages for the browser EventSource."""
        async for chunk in _proxy_sse(client, session_id, conn, app_state=request.app.state):
            if chunk.startswith("data: "):
                token = chunk[len("data: "):]
                if token.strip() == "[DONE]":
                    yield "data: [DONE]\n\n"
                    return
                yield f"data: {token}\n\n"

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
