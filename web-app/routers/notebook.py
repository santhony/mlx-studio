"""
notebook.py — Jupyter-style notebook router.

Each notebook contains ordered cells. Each cell has:
  - A prompt (user input textarea)
  - Generated code (streamed from text server /complete)
  - Execution output (stdout + stderr from subprocess)

Routes:
  GET  /notebook/                                → notebook list
  POST /notebook/                                → create notebook, redirect to it
  GET  /notebook/{notebook_id}                   → notebook view
  POST /notebook/{notebook_id}/cells             → add new cell
  DELETE /notebook/{notebook_id}/cells/{cell_id} → delete cell
  POST /notebook/{notebook_id}/cells/{cell_id}/generate  → stream code generation
  GET  /notebook/{notebook_id}/cells/{cell_id}/stream    → SSE: stream code tokens
  POST /notebook/{notebook_id}/cells/{cell_id}/run       → execute code, return output
  POST /notebook/{notebook_id}/cells/{cell_id}/save      → save prompt/code edits
"""

import sqlite3
import subprocess
import sys
from typing import AsyncGenerator

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from skills import retrieve_skills, format_skills_for_context

router = APIRouter(prefix="/notebook")
templates = Jinja2Templates(directory="templates")

TEXT_SERVER = "http://127.0.0.1:8766"
DEFAULT_TIMEOUT = 300  # seconds for code execution


# ── DB helpers ─────────────────────────────────────────────────────────────────

def _get_notebooks(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT id, name, updated_at FROM notebooks ORDER BY updated_at DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def _get_notebook(conn: sqlite3.Connection, notebook_id: int) -> dict | None:
    row = conn.execute(
        "SELECT id, name, created_at, updated_at FROM notebooks WHERE id = ?",
        (notebook_id,),
    ).fetchone()
    return dict(row) if row else None


def _get_cells(conn: sqlite3.Connection, notebook_id: int) -> list[dict]:
    rows = conn.execute(
        "SELECT id, notebook_id, position, prompt, code, output, created_at "
        "FROM cells WHERE notebook_id = ? ORDER BY position, id",
        (notebook_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def _get_cell(conn: sqlite3.Connection, cell_id: int) -> dict | None:
    row = conn.execute(
        "SELECT id, notebook_id, position, prompt, code, output FROM cells WHERE id = ?",
        (cell_id,),
    ).fetchone()
    return dict(row) if row else None


def _touch_notebook(conn: sqlite3.Connection, notebook_id: int) -> None:
    conn.execute(
        "UPDATE notebooks SET updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE id = ?",
        (notebook_id,),
    )


# ── Code execution ─────────────────────────────────────────────────────────────

def _execute_code(code: str, timeout: float = DEFAULT_TIMEOUT) -> dict:
    """
    Run Python code in a subprocess with timeout.
    Returns {"output": str, "success": bool, "timed_out": bool}.

    Uses sys.executable so the same Python binary runs the code.
    stdout and stderr are merged into output (stderr appended after stdout).

    Security Model: This tool executes arbitrary user-provided code without
    additional sandboxing. It is designed for LOCAL-ONLY use by a TRUSTED USER
    in a development environment. Do not expose this endpoint over the network
    to untrusted clients or use in production without additional security controls
    (containerization, seccomp, network isolation, etc.).
    """
    if not code.strip():
        return {"output": "", "success": True, "timed_out": False}
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        output = result.stdout
        if result.stderr:
            output = output + ("\n" if output else "") + result.stderr
        return {
            "output": output,
            "success": result.returncode == 0,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired:
        return {
            "output": f"Execution timed out after {timeout:.0f} seconds",
            "success": False,
            "timed_out": True,
        }
    except Exception as exc:
        return {
            "output": f"failed to execute: {exc}",
            "success": False,
            "timed_out": False,
        }


# ── SSE proxy (same pattern as chat router) ───────────────────────────────────

async def _stream_code(
    client: httpx.AsyncClient,
    cell_id: int,
    prompt: str,
    conn: sqlite3.Connection,
    app_state=None,
) -> AsyncGenerator[str, None]:
    """
    Request code generation from text server /complete.
    Streams tokens as SSE events and persists the complete code to the cell.
    """
    skills = retrieve_skills(conn, prompt, top_k=3)
    skills_ctx = format_skills_for_context(skills)
    if app_state is not None:
        app_state.last_injected_skills = [s["name"] for s in skills]

    system_prefix = ""
    if skills_ctx:
        system_prefix = skills_ctx + "\n\n"

    payload = {
        "prompt": (
            system_prefix
            + "Write Python code that does the following. "
            "Return only the code, no explanation, no markdown fences:\n\n"
            + prompt
        ),
        "max_tokens": 1024,
    }
    full_code: list[str] = []

    try:
        async with client.stream(
            "POST",
            f"{TEXT_SERVER}/complete",
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
                    yield f"event: message\ndata: {token}\n\n"
                    return
                decoded = token.replace("\\n", "\n")
                full_code.append(decoded)
                # Escape for SSE and yield as named 'message' event for HTMX
                sse_token = decoded.replace("\n", "\\n")
                yield f"event: message\ndata: {sse_token}\n\n"
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
        yield f"event: message\ndata: ERROR: text server unavailable: {exc}\n\n"
        return

    # Persist generated code to cell
    complete_code = "".join(full_code)
    if complete_code.strip():
        conn.execute(
            "UPDATE cells SET code = ? WHERE id = ?",
            (complete_code.strip(), cell_id),
        )
        conn.commit()

    yield "event: done\ndata: [DONE]\n\n"


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
async def notebook_list(request: Request):
    conn: sqlite3.Connection = request.app.state.db
    notebooks = _get_notebooks(conn)
    return templates.TemplateResponse(
        request=request,
        name="notebook_list.html",
        context={"notebooks": notebooks},
    )


@router.post("/")
async def create_notebook(request: Request):
    form = await request.form()
    name = (form.get("name") or "").strip() or "Untitled Notebook"
    conn: sqlite3.Connection = request.app.state.db
    cur = conn.execute("INSERT INTO notebooks (name) VALUES (?)", (name,))
    conn.commit()
    return RedirectResponse(url=f"/notebook/{cur.lastrowid}", status_code=303)


@router.get("/{notebook_id}", response_class=HTMLResponse)
async def notebook_view(notebook_id: int, request: Request):
    conn: sqlite3.Connection = request.app.state.db
    notebook = _get_notebook(conn, notebook_id)
    if not notebook:
        raise HTTPException(status_code=404, detail="notebook not found")
    cells = _get_cells(conn, notebook_id)
    return templates.TemplateResponse(
        request=request,
        name="notebook.html",
        context={"notebook": notebook, "cells": cells, "notebook_id": notebook_id},
    )


@router.post("/{notebook_id}/cells", response_class=HTMLResponse)
async def add_cell(notebook_id: int, request: Request):
    conn: sqlite3.Connection = request.app.state.db
    if not _get_notebook(conn, notebook_id):
        raise HTTPException(status_code=404, detail="notebook not found")
    # Position = max existing position + 1
    row = conn.execute(
        "SELECT COALESCE(MAX(position), -1) FROM cells WHERE notebook_id = ?",
        (notebook_id,),
    ).fetchone()
    next_pos = (row[0] or -1) + 1
    cur = conn.execute(
        "INSERT INTO cells (notebook_id, position, prompt, code, output) VALUES (?, ?, '', '', '')",
        (notebook_id, next_pos),
    )
    _touch_notebook(conn, notebook_id)
    conn.commit()
    cell = _get_cell(conn, cur.lastrowid)
    return templates.TemplateResponse(
        request=request,
        name="cell.html",
        context={"cell": cell, "notebook_id": notebook_id},
    )


@router.delete("/{notebook_id}/cells/{cell_id}", response_class=HTMLResponse)
async def delete_cell(notebook_id: int, cell_id: int, request: Request):
    conn: sqlite3.Connection = request.app.state.db
    conn.execute(
        "DELETE FROM cells WHERE id = ? AND notebook_id = ?",
        (cell_id, notebook_id),
    )
    _touch_notebook(conn, notebook_id)
    conn.commit()
    return HTMLResponse("")  # Empty response removes the cell from DOM


@router.post("/{notebook_id}/cells/{cell_id}/generate", response_class=HTMLResponse)
async def generate_code(notebook_id: int, cell_id: int, request: Request):
    """Persist the prompt and return an SSE-connected code block."""
    conn: sqlite3.Connection = request.app.state.db
    cell = _get_cell(conn, cell_id)
    if not cell or cell["notebook_id"] != notebook_id:
        raise HTTPException(status_code=404, detail="cell not found")

    form = await request.form()
    prompt = (form.get("prompt") or "").strip()
    if not prompt:
        return HTMLResponse('<p class="error">Prompt cannot be empty.</p>', status_code=400)

    # Save prompt
    conn.execute("UPDATE cells SET prompt = ?, code = '', output = '' WHERE id = ?", (prompt, cell_id))
    _touch_notebook(conn, notebook_id)
    conn.commit()

    # Return a fragment with a JS EventSource that streams tokens into the code block
    return HTMLResponse(f"""
<div id="cell-code-{cell_id}" class="cell-code-block">
    <pre><code id="code-content-{cell_id}" class="language-python"></code></pre>
</div>
<div id="cell-output-{cell_id}" class="cell-output"></div>
<script>
(function() {{
    const el = document.getElementById("code-content-{cell_id}");
    const es = new EventSource("/notebook/{notebook_id}/cells/{cell_id}/stream");
    let code = "";
    es.onmessage = function(e) {{
        if (e.data === "[DONE]") {{
            es.close();
            if (typeof Prism !== "undefined") Prism.highlightElement(el);
            return;
        }}
        code += e.data.replace(/\\\\n/g, "\\n");
        el.textContent = code;
    }};
    es.onerror = function() {{ es.close(); }};
}})();
</script>
""")


@router.get("/{notebook_id}/cells/{cell_id}/stream")
async def stream_code(notebook_id: int, cell_id: int, request: Request):
    """SSE endpoint: proxies /complete tokens to the browser."""
    conn: sqlite3.Connection = request.app.state.db
    cell = _get_cell(conn, cell_id)
    if not cell or cell["notebook_id"] != notebook_id:
        raise HTTPException(status_code=404, detail="cell not found")

    client: httpx.AsyncClient = request.app.state.http_client

    async def _wrapped() -> AsyncGenerator[str, None]:
        async for chunk in _stream_code(client, cell_id, cell["prompt"], conn, app_state=request.app.state):
            if chunk.startswith("event: done"):
                yield "data: [DONE]\n\n"
                return
            # _stream_code yields "event: message\ndata: <token>\n\n" — extract just the token
            if "data: " in chunk:
                token = chunk.split("data: ", 1)[1].rstrip("\n")
                yield f"data: {token}\n\n"

    return StreamingResponse(
        _wrapped(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/{notebook_id}/cells/{cell_id}/run", response_class=HTMLResponse)
async def run_cell(notebook_id: int, cell_id: int, request: Request):
    """Execute the cell's code and return an output fragment."""
    conn: sqlite3.Connection = request.app.state.db
    cell = _get_cell(conn, cell_id)
    if not cell or cell["notebook_id"] != notebook_id:
        raise HTTPException(status_code=404, detail="cell not found")

    code = cell["code"]
    if not code.strip():
        return HTMLResponse(
            f'<div id="cell-output-{cell_id}" class="cell-output cell-output-empty">No code to run.</div>',
        )

    result = _execute_code(code)
    output = result["output"] or "(no output)"
    css_class = "cell-output"
    if result["timed_out"]:
        css_class += " cell-output-error"
    elif not result["success"]:
        css_class += " cell-output-error"

    # Persist output
    conn.execute("UPDATE cells SET output = ? WHERE id = ?", (output, cell_id))
    _touch_notebook(conn, notebook_id)
    conn.commit()

    escaped = output.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return HTMLResponse(
        f'<div id="cell-output-{cell_id}" class="{css_class}"><pre>{escaped}</pre></div>'
    )


@router.post("/{notebook_id}/cells/{cell_id}/save", response_class=HTMLResponse)
async def save_cell(notebook_id: int, cell_id: int, request: Request):
    """Save manual edits to prompt and/or code."""
    conn: sqlite3.Connection = request.app.state.db
    cell = _get_cell(conn, cell_id)
    if not cell or cell["notebook_id"] != notebook_id:
        raise HTTPException(status_code=404, detail="cell not found")

    form = await request.form()
    prompt = form.get("prompt", cell["prompt"])
    code = form.get("code", cell["code"])
    conn.execute(
        "UPDATE cells SET prompt = ?, code = ? WHERE id = ?",
        (prompt, code, cell_id),
    )
    _touch_notebook(conn, notebook_id)
    conn.commit()
    return HTMLResponse('<span class="saved-indicator">Saved</span>')
