"""
rag.py — RAG corpus management and chat.

RAG chat provides retrieval-augmented generation with SSE streaming.
Reuses the SSE pattern from chat.py.
"""
import asyncio
import json
import logging
import sqlite3
from pathlib import Path
from typing import AsyncGenerator

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from indexer import index_source, retrieve_chunks

log = logging.getLogger("rag")
router = APIRouter(prefix="/rag")
templates = Jinja2Templates(directory="templates")

TEXT_SERVER = "http://127.0.0.1:8766"


# ── DB helpers ─────────────────────────────────────────────────────────────────


def _get_corpora(conn: sqlite3.Connection) -> list[dict]:
    """Fetch all corpora ordered by updated_at DESC."""
    rows = conn.execute(
        "SELECT id, name, description, created_at, updated_at FROM corpora ORDER BY updated_at DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def _get_corpus(conn: sqlite3.Connection, corpus_id: int) -> dict | None:
    """Fetch single corpus by id."""
    row = conn.execute(
        "SELECT id, name, description, created_at, updated_at FROM corpora WHERE id = ?",
        (corpus_id,),
    ).fetchone()
    return dict(row) if row else None


def _get_sources(conn: sqlite3.Connection, corpus_id: int) -> list[dict]:
    """Fetch sources for a corpus with chunk counts."""
    rows = conn.execute(
        """
        SELECT
            s.id,
            s.corpus_id,
            s.source_type,
            s.path,
            s.treat_as_text,
            s.last_indexed_at,
            s.created_at,
            COUNT(c.id) as chunk_count
        FROM corpus_sources s
        LEFT JOIN corpus_chunks c ON s.id = c.source_id
        WHERE s.corpus_id = ?
        GROUP BY s.id
        ORDER BY s.created_at DESC
        """,
        (corpus_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def _get_source(conn: sqlite3.Connection, source_id: int) -> dict | None:
    """Fetch single source."""
    row = conn.execute(
        "SELECT id, corpus_id, source_type, path, treat_as_text, last_indexed_at, created_at FROM corpus_sources WHERE id = ?",
        (source_id,),
    ).fetchone()
    return dict(row) if row else None


def _get_corpora_with_counts(conn: sqlite3.Connection) -> list[dict]:
    """Fetch all corpora with source and chunk counts via single aggregate query."""
    rows = conn.execute(
        """
        SELECT
            c.id,
            c.name,
            c.description,
            c.created_at,
            c.updated_at,
            COUNT(DISTINCT s.id) as source_count,
            COUNT(ch.id) as chunk_count
        FROM corpora c
        LEFT JOIN corpus_sources s ON c.id = s.corpus_id
        LEFT JOIN corpus_chunks ch ON s.id = ch.source_id
        GROUP BY c.id
        ORDER BY c.updated_at DESC
        """
    ).fetchall()
    return [dict(r) for r in rows]


# ── RAG Chat helpers ──────────────────────────────────────────────────────────────


def _build_rag_system_prompt(chunks: list[dict]) -> str:
    """Build a system prompt with numbered document references."""
    if not chunks:
        return (
            "You are a helpful assistant. However, no relevant documents were found "
            "to answer your question. Please let the user know that the documents don't contain relevant information."
        )

    # Build document sections with numbered references
    doc_sections = []
    for i, chunk in enumerate(chunks, start=1):
        source_file = chunk.get("source_file", "unknown")
        chunk_index = chunk.get("chunk_index", 0)
        content = chunk.get("content", "")
        doc_sections.append(
            f"[{i}] (source: {source_file}, chunk {chunk_index})\n{content}"
        )

    docs_text = "\n\n".join(doc_sections)

    return f"""You are a helpful assistant. Answer the user's question using ONLY the document excerpts provided below. Cite your sources using [1], [2], etc. markers. If the documents don't contain relevant information, say so.

Document excerpts:

{docs_text}"""


def _get_rag_sessions(conn: sqlite3.Connection, corpus_id: int) -> list[dict]:
    """Fetch RAG sessions for a corpus, ordered by updated_at DESC."""
    rows = conn.execute(
        "SELECT id, name, updated_at FROM rag_sessions WHERE corpus_id = ? ORDER BY updated_at DESC",
        (corpus_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def _get_rag_session(conn: sqlite3.Connection, session_id: int) -> dict | None:
    """Fetch single RAG session by id."""
    row = conn.execute(
        "SELECT id, corpus_id, name, created_at, updated_at FROM rag_sessions WHERE id = ?",
        (session_id,),
    ).fetchone()
    return dict(row) if row else None


def _get_rag_messages(conn: sqlite3.Connection, session_id: int) -> list[dict]:
    """Fetch messages for a RAG session ordered by id."""
    rows = conn.execute(
        "SELECT id, role, content, citations_json, created_at FROM rag_messages WHERE session_id = ? ORDER BY id",
        (session_id,),
    ).fetchall()
    messages = []
    for r in rows:
        msg = dict(r)
        # Parse citations_json if present, store as 'citations' for template
        if msg.get("citations_json"):
            msg["citations"] = json.loads(msg["citations_json"])
        else:
            msg["citations"] = None
        messages.append(msg)
    return messages


def _append_rag_message(
    conn: sqlite3.Connection,
    session_id: int,
    role: str,
    content: str,
    citations_json: str | None = None,
) -> int:
    """Insert a RAG message and update session updated_at. Follow chat.py:_append_message() pattern."""
    cur = conn.execute(
        "INSERT INTO rag_messages (session_id, role, content, citations_json) VALUES (?, ?, ?, ?)",
        (session_id, role, content, citations_json),
    )
    conn.execute(
        "UPDATE rag_sessions SET updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE id = ?",
        (session_id,),
    )
    conn.commit()
    return cur.lastrowid


async def _rag_proxy_sse(
    client: httpx.AsyncClient,
    conn: sqlite3.Connection,
    session_id: int,
    corpus_id: int,
) -> AsyncGenerator[str, None]:
    """
    RAG SSE proxy: retrieve chunks, build system prompt, stream response.
    Follows chat.py:_proxy_sse() pattern.

    Important: retrieve_chunks() calls synchronous httpx.post() for embedding.
    Wrap it with asyncio.to_thread() to avoid blocking the event loop.
    """
    # Retrieve all messages for context
    messages = _get_rag_messages(conn, session_id)

    # Get the last user message to use as retrieval query
    last_user = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"),
        None,
    )

    chunks = []
    if last_user:
        # Wrap synchronous retrieve_chunks in asyncio.to_thread to avoid blocking
        chunks = await asyncio.to_thread(retrieve_chunks, conn, corpus_id, last_user, 5)

    # Build system prompt with retrieved chunks
    system_prompt = _build_rag_system_prompt(chunks)

    # Construct message list: system prompt + prior messages
    msg_list = [{"role": "system", "content": system_prompt}]
    msg_list.extend([{"role": m["role"], "content": m["content"]} for m in messages])

    payload = {
        "messages": msg_list,
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
                # Re-escape for SSE wire format before sending to browser
                reescaped = decoded.replace("\n", "\\n")
                yield f"data: {reescaped}\n\n"
    except httpx.HTTPStatusError as exc:
        yield f"data: ERROR: text server returned {exc.response.status_code}\n\n"
        return
    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        yield f"data: ERROR: text server unavailable: {exc}\n\n"
        return

    # Build citations JSON from retrieved chunks
    citations = [
        {
            "ref": i + 1,
            "source_file": c["source_file"],
            "chunk_index": c["chunk_index"],
            "excerpt": c["content"][:200],
        }
        for i, c in enumerate(chunks)
    ]
    citations_json = json.dumps(citations) if citations else None

    # Persist completed assistant response
    complete_text = "".join(full_response)
    if complete_text.strip():
        _append_rag_message(conn, session_id, "assistant", complete_text, citations_json)

    # Send citations as a separate SSE event for live rendering BEFORE [DONE]
    # so the client's citations listener fires before the [DONE] handler closes the EventSource
    if citations:
        yield f"event: citations\ndata: {json.dumps(citations)}\n\n"
    yield "data: [DONE]\n\n"


# ── Routes ─────────────────────────────────────────────────────────────────────────


@router.get("/", response_class=HTMLResponse)
async def corpus_list(request: Request):
    """Corpus list page."""
    conn: sqlite3.Connection = request.app.state.db
    corpora = _get_corpora_with_counts(conn)

    return templates.TemplateResponse(
        request=request,
        name="rag.html",
        context={"corpora": corpora},
    )


@router.post("/", response_class=RedirectResponse)
async def create_corpus(request: Request):
    """Create corpus."""
    conn: sqlite3.Connection = request.app.state.db
    form = await request.form()
    name = form.get("name", "").strip()
    description = form.get("description", "").strip()

    if not name:
        raise HTTPException(status_code=400, detail="Corpus name is required")

    cursor = conn.execute(
        "INSERT INTO corpora (name, description) VALUES (?, ?)",
        (name, description),
    )
    conn.commit()
    corpus_id = cursor.lastrowid

    return RedirectResponse(url=f"/rag/{corpus_id}", status_code=303)


@router.get("/{corpus_id}", response_class=HTMLResponse)
async def corpus_detail(request: Request, corpus_id: int):
    """Corpus detail page."""
    conn: sqlite3.Connection = request.app.state.db
    corpus = _get_corpus(conn, corpus_id)

    if not corpus:
        raise HTTPException(status_code=404, detail="Corpus not found")

    sources = _get_sources(conn, corpus_id)

    # Count total chunks for this corpus
    chunk_row = conn.execute(
        "SELECT COUNT(*) as total FROM corpus_chunks WHERE corpus_id = ?",
        (corpus_id,),
    ).fetchone()
    total_chunks = chunk_row["total"] if chunk_row else 0

    return templates.TemplateResponse(
        request=request,
        name="corpus_detail.html",
        context={"corpus": corpus, "corpus_id": corpus_id, "sources": sources, "total_chunks": total_chunks},
    )


@router.post("/{corpus_id}/sources", response_class=RedirectResponse)
async def add_source(request: Request, corpus_id: int):
    """Add source to corpus."""
    conn: sqlite3.Connection = request.app.state.db
    corpus = _get_corpus(conn, corpus_id)

    if not corpus:
        raise HTTPException(status_code=404, detail="Corpus not found")

    form = await request.form()
    source_type = form.get("source_type", "").strip()
    path = form.get("path", "").strip()
    # HTML checkboxes only appear in the form data when checked; presence
    # of the field (any value) means the user opted in.
    treat_as_text = 1 if form.get("treat_as_text") else 0

    if not source_type or not path:
        raise HTTPException(status_code=400, detail="Source type and path are required")

    if source_type not in ("directory", "url", "url_spider"):
        raise HTTPException(status_code=400, detail="Invalid source type")

    # Validate source based on type
    if source_type == "directory":
        if not Path(path).is_dir():
            return RedirectResponse(
                url=f"/rag/{corpus_id}?error=Directory+not+found",
                status_code=303,
            )
    elif source_type == "url":
        if not (path.startswith("http://") or path.startswith("https://")):
            return RedirectResponse(
                url=f"/rag/{corpus_id}?error=URL+must+start+with+http://+or+https://",
                status_code=303,
            )
    elif source_type == "url_spider":
        if not (path.startswith("http://") or path.startswith("https://")):
            return RedirectResponse(
                url=f"/rag/{corpus_id}?error=URL+must+start+with+http://+or+https://",
                status_code=303,
            )

    conn.execute(
        "INSERT INTO corpus_sources (corpus_id, source_type, path, treat_as_text) VALUES (?, ?, ?, ?)",
        (corpus_id, source_type, path, treat_as_text),
    )
    conn.commit()

    return RedirectResponse(url=f"/rag/{corpus_id}", status_code=303)


@router.post("/{corpus_id}/sources/{source_id}/delete", response_class=RedirectResponse)
async def delete_source(request: Request, corpus_id: int, source_id: int):
    """Delete source."""
    conn: sqlite3.Connection = request.app.state.db
    source = _get_source(conn, source_id)

    if not source:
        raise HTTPException(status_code=404, detail="Source not found")

    if source["corpus_id"] != corpus_id:
        raise HTTPException(status_code=400, detail="Source does not belong to corpus")

    conn.execute("DELETE FROM corpus_sources WHERE id = ?", (source_id,))
    conn.commit()

    return RedirectResponse(url=f"/rag/{corpus_id}", status_code=303)


@router.post("/{corpus_id}/sources/{source_id}/index", response_class=RedirectResponse)
async def trigger_index(request: Request, corpus_id: int, source_id: int):
    """Trigger indexing of a source."""
    conn: sqlite3.Connection = request.app.state.db
    corpus = _get_corpus(conn, corpus_id)

    if not corpus:
        raise HTTPException(status_code=404, detail="Corpus not found")

    source = _get_source(conn, source_id)

    if not source:
        raise HTTPException(status_code=404, detail="Source not found")

    if source["corpus_id"] != corpus_id:
        raise HTTPException(status_code=400, detail="Source does not belong to corpus")

    result = index_source(conn, source, corpus_id)

    log.info(
        "indexed source %d: %d files, %d chunks, %d errors",
        source_id,
        result["files_processed"],
        result["chunks_created"],
        len(result["errors"]),
    )

    # Redirect with status query params
    url = f"/rag/{corpus_id}?indexed={source_id}&files={result['files_processed']}&chunks={result['chunks_created']}&errors={len(result['errors'])}"
    return RedirectResponse(url=url, status_code=303)


@router.post("/{corpus_id}/delete", response_class=RedirectResponse)
async def delete_corpus(request: Request, corpus_id: int):
    """Delete entire corpus."""
    conn: sqlite3.Connection = request.app.state.db
    corpus = _get_corpus(conn, corpus_id)

    if not corpus:
        raise HTTPException(status_code=404, detail="Corpus not found")

    conn.execute("DELETE FROM corpora WHERE id = ?", (corpus_id,))
    conn.commit()

    return RedirectResponse(url="/rag/", status_code=303)


# ── RAG Chat Routes ───────────────────────────────────────────────────────────────


@router.get("/{corpus_id}/chat", response_class=HTMLResponse)
async def rag_session_list(request: Request, corpus_id: int):
    """RAG session list page."""
    conn: sqlite3.Connection = request.app.state.db
    corpus = _get_corpus(conn, corpus_id)

    if not corpus:
        raise HTTPException(status_code=404, detail="Corpus not found")

    sessions = _get_rag_sessions(conn, corpus_id)

    return templates.TemplateResponse(
        request=request,
        name="rag_sessions.html",
        context={"corpus": corpus, "corpus_id": corpus_id, "sessions": sessions},
    )


@router.post("/{corpus_id}/chat/sessions", response_class=RedirectResponse)
async def create_rag_session(request: Request, corpus_id: int):
    """Create a new RAG session."""
    conn: sqlite3.Connection = request.app.state.db
    corpus = _get_corpus(conn, corpus_id)

    if not corpus:
        raise HTTPException(status_code=404, detail="Corpus not found")

    form = await request.form()
    name = (form.get("name") or "").strip() or "New RAG chat"

    cur = conn.execute(
        "INSERT INTO rag_sessions (corpus_id, name) VALUES (?, ?)",
        (corpus_id, name),
    )
    conn.commit()

    return RedirectResponse(url=f"/rag/{corpus_id}/chat/{cur.lastrowid}", status_code=303)


@router.post("/{corpus_id}/chat/sessions/{session_id}/delete", response_class=RedirectResponse)
async def delete_rag_session(request: Request, corpus_id: int, session_id: int):
    """Delete a RAG session."""
    conn: sqlite3.Connection = request.app.state.db
    session = _get_rag_session(conn, session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session["corpus_id"] != corpus_id:
        raise HTTPException(status_code=400, detail="Session does not belong to corpus")

    conn.execute("DELETE FROM rag_sessions WHERE id = ?", (session_id,))
    conn.commit()

    return RedirectResponse(url=f"/rag/{corpus_id}/chat", status_code=303)


@router.get("/{corpus_id}/chat/{session_id}", response_class=HTMLResponse)
async def rag_chat_view(request: Request, corpus_id: int, session_id: int):
    """RAG chat page."""
    conn: sqlite3.Connection = request.app.state.db
    corpus = _get_corpus(conn, corpus_id)

    if not corpus:
        raise HTTPException(status_code=404, detail="Corpus not found")

    session = _get_rag_session(conn, session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session["corpus_id"] != corpus_id:
        raise HTTPException(status_code=400, detail="Session does not belong to corpus")

    messages = _get_rag_messages(conn, session_id)
    sessions = _get_rag_sessions(conn, corpus_id)

    # Count chunks and sources for info line
    chunk_rows = conn.execute(
        "SELECT COUNT(*) as chunk_count FROM corpus_chunks WHERE corpus_id = ?",
        (corpus_id,),
    ).fetchone()
    chunk_count = chunk_rows["chunk_count"] if chunk_rows else 0

    source_rows = conn.execute(
        "SELECT COUNT(*) as source_count FROM corpus_sources WHERE corpus_id = ?",
        (corpus_id,),
    ).fetchone()
    source_count = source_rows["source_count"] if source_rows else 0

    return templates.TemplateResponse(
        request=request,
        name="rag_chat.html",
        context={
            "corpus": corpus,
            "corpus_id": corpus_id,
            "session": session,
            "session_id": session_id,
            "messages": messages,
            "sessions": sessions,
            "chunk_count": chunk_count,
            "source_count": source_count,
        },
    )


@router.post("/{corpus_id}/chat/{session_id}/send", response_class=HTMLResponse)
async def send_rag_message(request: Request, corpus_id: int, session_id: int):
    """
    Persist user message and return HTMX fragment with SSE listener.
    Follow chat.py:send_message() pattern.
    """
    conn: sqlite3.Connection = request.app.state.db
    session = _get_rag_session(conn, session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session["corpus_id"] != corpus_id:
        raise HTTPException(status_code=400, detail="Session does not belong to corpus")

    form = await request.form()
    content = (form.get("content") or "").strip()
    if not content:
        return HTMLResponse('<p class="error">Message cannot be empty.</p>', status_code=400)

    msg_id = _append_rag_message(conn, session_id, "user", content)

    # Return HTMX fragment: user bubble + streaming assistant bubble
    escaped_content = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return HTMLResponse(f"""
<div class="msg msg-user">
    <div class="msg-content">{escaped_content}</div>
</div>
<div class="msg msg-assistant">
    <div class="msg-content" id="streaming-{msg_id}"><span class="spinner"></span></div>
    <div id="citations-{msg_id}"></div>
</div>
<script>
(function() {{
    const contentEl = document.getElementById("streaming-{msg_id}");
    const citationsEl = document.getElementById("citations-{msg_id}");
    const es = new EventSource("/rag/{corpus_id}/chat/{session_id}/stream");
    let text = "";

    // Helper to escape HTML special characters
    function esc(s) {{
        const d = document.createElement('div');
        d.textContent = s;
        return d.innerHTML;
    }}

    es.onmessage = function(e) {{
        if (e.data === "[DONE]") {{ es.close(); return; }}
        text += e.data.replace(/\\\\n/g, "\\n");
        // Use the shared renderer so <think>...</think> sentinels are
        // rendered as muted italic blocks, matching the Chat tab.
        contentEl.innerHTML = window.renderChatContent(text);
        contentEl.scrollIntoView({{block: "end"}});
    }};

    es.addEventListener("citations", function(e) {{
        const citations = JSON.parse(e.data);
        if (citations && citations.length > 0) {{
            let html = '<div class="msg-citations"><details><summary>Sources (' + citations.length + ')</summary><ol>';
            for (let i = 0; i < citations.length; i++) {{
                const c = citations[i];
                const excerpt = c.excerpt.slice(0, 150);
                html += '<li><code>' + esc(c.source_file) + '</code> (chunk ' + c.chunk_index + '): <em>"' + esc(excerpt) + '..."</em></li>';
            }}
            html += '</ol></details></div>';
            citationsEl.innerHTML = html;
            citationsEl.scrollIntoView({{block: "end"}});
        }}
        es.close();
    }});

    es.onerror = function() {{ es.close(); }};
}})();
</script>
""")


@router.get("/{corpus_id}/chat/{session_id}/stream")
async def stream_rag_response(request: Request, corpus_id: int, session_id: int):
    """SSE endpoint for RAG chat streaming."""
    conn: sqlite3.Connection = request.app.state.db
    session = _get_rag_session(conn, session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session["corpus_id"] != corpus_id:
        raise HTTPException(status_code=400, detail="Session does not belong to corpus")

    client: httpx.AsyncClient = request.app.state.http_client

    return StreamingResponse(
        _rag_proxy_sse(client, conn, session_id, corpus_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
