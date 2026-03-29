"""
agents.py — Human-in-the-loop agent router.

WebSocket channel at /agents/{job_id}/ws delivers live transcript HTML.
Approval is via POST to /agents/{job_id}/approve or /agents/{job_id}/deny.

asyncio.Event coordinates approval: the agent loop awaits event.wait(),
the approval endpoint calls event.set().
"""

import asyncio
import json
import re
import sqlite3
from typing import Any, Optional

import httpx
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

import agent_tools
from agent_tools import TOOL_DESCRIPTIONS
from skills import retrieve_skills, format_skills_for_context

router = APIRouter(prefix="/agents")
templates = Jinja2Templates(directory="templates")

TEXT_SERVER = "http://127.0.0.1:8766"
MAX_STEPS = 20

# Per-job state: {job_id: {"event": asyncio.Event, "approved": bool, "session_tools": set}}
_job_state: dict[int, dict] = {}
# Per-job WebSocket connection: {job_id: WebSocket}
_job_ws: dict[int, WebSocket] = {}


# ── Tool call parsing ─────────────────────────────────────────────────────────

def _parse_tool_call(text: str) -> Optional[dict[str, Any]]:
    """Extract first <tool>...</tool> JSON block from LLM output."""
    match = re.search(r"<tool>(.*?)</tool>", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(1).strip())
        if "tool" in data and "args" in data:
            return data
        return None
    except json.JSONDecodeError:
        return None


# ── DB helpers ─────────────────────────────────────────────────────────────────

def _create_job(conn: sqlite3.Connection, task: str) -> int:
    cur = conn.execute(
        "INSERT INTO agent_jobs (task, status) VALUES (?, 'running')",
        (task,),
    )
    conn.commit()
    return cur.lastrowid


def _update_job_status(conn: sqlite3.Connection, job_id: int, status: str) -> None:
    conn.execute("UPDATE agent_jobs SET status = ? WHERE id = ?", (status, job_id))
    conn.commit()


def _append_step(
    conn: sqlite3.Connection,
    job_id: int,
    step_type: str,
    content: str,
    approved: Optional[bool] = None,
) -> None:
    conn.execute(
        "INSERT INTO agent_steps (job_id, type, content, approved) VALUES (?, ?, ?, ?)",
        (job_id, step_type, content, approved),
    )
    conn.commit()


def _get_jobs(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT id, task, status, created_at FROM agent_jobs ORDER BY id DESC LIMIT 50"
    ).fetchall()
    return [dict(r) for r in rows]


def _get_steps(conn: sqlite3.Connection, job_id: int) -> list[dict]:
    rows = conn.execute(
        "SELECT id, type, content, approved, created_at FROM agent_steps "
        "WHERE job_id = ? ORDER BY id",
        (job_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def _get_allowed_dirs(conn: sqlite3.Connection) -> list[str]:
    """Return filesystem allowlist from settings table."""
    rows = conn.execute(
        "SELECT value FROM settings WHERE key LIKE 'allowed_dir_%'"
    ).fetchall()
    return [r["value"] for r in rows]


# ── WebSocket helpers ─────────────────────────────────────────────────────────

async def _ws_send(job_id: int, html: str) -> None:
    """Send HTML fragment to connected WebSocket client (if any)."""
    ws = _job_ws.get(job_id)
    if ws:
        try:
            await ws.send_text(html)
        except Exception:
            pass


def _step_html(step_type: str, content: str, approved: Optional[bool] = None) -> str:
    """Render a transcript step as an HTML fragment for OOB swap."""
    css = {
        "reasoning": "step-reasoning",
        "tool_call": "step-tool",
        "tool_result": "step-result",
        "approval_request": "step-approval",
        "denial": "step-denial",
        "error": "step-error",
        "complete": "step-complete",
    }.get(step_type, "step-default")

    escaped = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    label = step_type.replace("_", " ").title()

    return f"""<div id="transcript" hx-swap-oob="beforeend">
<div class="step {css}">
    <span class="step-label">{label}</span>
    <pre class="step-content">{escaped}</pre>
</div>
</div>"""


def _approval_card_html(job_id: int, tool_name: str, args_json: str) -> str:
    """Render approval card HTML for a tool call."""
    escaped_args = args_json.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"""<div id="transcript" hx-swap-oob="beforeend">
<div class="step step-approval" id="approval-card-{job_id}">
    <span class="step-label">Approval Required</span>
    <p>Tool: <strong>{tool_name}</strong></p>
    <pre class="step-content">{escaped_args}</pre>
    <div class="approval-buttons">
        <form method="post" action="/agents/{job_id}/approve" style="display:inline">
            <button type="submit" class="btn-allow">Allow once</button>
        </form>
        <form method="post" action="/agents/{job_id}/deny" style="display:inline">
            <button type="submit" class="secondary btn-deny">Deny</button>
        </form>
    </div>
</div>
</div>"""


# ── Agent loop ────────────────────────────────────────────────────────────────

async def _run_agent(
    job_id: int,
    task: str,
    conn: sqlite3.Connection,
    http_client: httpx.AsyncClient,
) -> None:
    """
    Main agent reasoning loop. Runs as an asyncio background task.
    Sends HTML fragments to the browser via WebSocket.
    """
    allowed_dirs = _get_allowed_dirs(conn)
    skills = retrieve_skills(conn, task, top_k=3)
    skills_ctx = format_skills_for_context(skills)

    system_prompt = TOOL_DESCRIPTIONS
    if skills_ctx:
        system_prompt = skills_ctx + "\n\n" + system_prompt

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]

    _job_state[job_id] = {
        "event": None,
        "approved": False,
        "session_tools": set(),
    }

    exit_status = "completed"
    try:
        for step in range(MAX_STEPS):
            # Call text server for reasoning
            try:
                async with http_client.stream(
                    "POST",
                    f"{TEXT_SERVER}/chat",
                    json={"messages": messages, "max_tokens": 1024},
                    timeout=300.0,
                ) as resp:
                    resp.raise_for_status()
                    tokens: list[str] = []
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            t = line[len("data: "):]
                            if t == "[DONE]":
                                break
                            tokens.append(t.replace("\\n", "\n"))
                    reasoning = "".join(tokens)
            except Exception as exc:
                error_msg = f"text server error: {exc}"
                _append_step(conn, job_id, "error", error_msg)
                await _ws_send(job_id, _step_html("error", error_msg))
                exit_status = "failed"
                break

            if not reasoning.strip():
                break

            messages.append({"role": "assistant", "content": reasoning})
            _append_step(conn, job_id, "reasoning", reasoning)
            await _ws_send(job_id, _step_html("reasoning", reasoning))

            # Parse for tool call
            tool_call = _parse_tool_call(reasoning)
            if tool_call is None:
                # No tool call → task complete
                _append_step(conn, job_id, "complete", "Task completed")
                await _ws_send(job_id, _step_html("complete", "Task completed"))
                break

            tool_name = tool_call["tool"]
            tool_args = tool_call["args"]
            args_json = json.dumps(tool_args, indent=2)

            # Check session-level approval
            if tool_name not in _job_state[job_id]["session_tools"]:
                # Request approval
                event = asyncio.Event()
                _job_state[job_id]["event"] = event
                _job_state[job_id]["approved"] = False
                _job_state[job_id]["current_tool"] = tool_name

                _append_step(conn, job_id, "approval_request", f"{tool_name}\n{args_json}")
                await _ws_send(job_id, _approval_card_html(job_id, tool_name, args_json))

                # Wait for user decision (timeout after 5 min)
                try:
                    await asyncio.wait_for(event.wait(), timeout=300.0)
                except asyncio.TimeoutError:
                    _append_step(conn, job_id, "denial", "Approval timed out — tool not executed")
                    await _ws_send(job_id, _step_html("denial", "Approval timed out"))
                    exit_status = "timed_out"
                    break

                if not _job_state[job_id]["approved"]:
                    denial_msg = f"Tool '{tool_name}' was denied by user"
                    messages.append({"role": "user", "content": f"Tool call denied: {tool_name}"})
                    _append_step(conn, job_id, "denial", denial_msg)
                    await _ws_send(job_id, _step_html("denial", denial_msg))
                    continue

            # Execute tool
            result = await _dispatch_tool(tool_name, tool_args, allowed_dirs, conn, http_client)
            tool_result = f"Tool '{tool_name}' result:\n{result}"
            messages.append({"role": "user", "content": tool_result})
            _append_step(conn, job_id, "tool_result", tool_result)
            await _ws_send(job_id, _step_html("tool_result", tool_result))
    except Exception as exc:
        error_msg = f"unhandled error in agent loop: {exc}"
        _append_step(conn, job_id, "error", error_msg)
        await _ws_send(job_id, _step_html("error", error_msg))
        exit_status = "failed"

    _update_job_status(conn, job_id, exit_status)
    _job_state.pop(job_id, None)


async def _dispatch_tool(
    tool_name: str,
    args: dict,
    allowed_dirs: list[str],
    conn: sqlite3.Connection,
    http_client: httpx.AsyncClient,
) -> str:
    """Route tool_name to its implementation."""
    try:
        if tool_name == "filesystem_read":
            return await agent_tools.filesystem_read(args["path"], allowed_dirs)
        elif tool_name == "filesystem_write":
            return await agent_tools.filesystem_write(args["path"], args["content"], allowed_dirs)
        elif tool_name == "filesystem_list":
            return await agent_tools.filesystem_list(args["path"], allowed_dirs)
        elif tool_name == "shell":
            return await agent_tools.shell(args["command"])
        elif tool_name == "python_exec":
            return await agent_tools.python_exec(args["code"])
        elif tool_name == "web_fetch":
            return await agent_tools.web_fetch(args["url"])
        elif tool_name == "web_search":
            return await agent_tools.web_search(args["query"])
        elif tool_name == "call_model":
            return await agent_tools.call_model(args["prompt"], http_client)
        elif tool_name == "load_skill":
            return await agent_tools.load_skill(args["name"], conn)
        else:
            return f"ERROR: unknown tool '{tool_name}'"
    except KeyError as exc:
        return f"ERROR: missing required argument {exc} for tool '{tool_name}'"
    except Exception as exc:
        return f"ERROR: tool '{tool_name}' raised: {exc}"


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
async def agents_list(request: Request):
    conn: sqlite3.Connection = request.app.state.db
    jobs = _get_jobs(conn)
    return templates.TemplateResponse(
        request=request, name="agents.html", context={"jobs": jobs}
    )


@router.post("/")
async def create_job(request: Request):
    form = await request.form()
    task = (form.get("task") or "").strip()
    if not task:
        raise HTTPException(status_code=400, detail="task cannot be empty")

    conn: sqlite3.Connection = request.app.state.db
    http_client: httpx.AsyncClient = request.app.state.http_client
    job_id = _create_job(conn, task)

    # Run agent in background
    asyncio.create_task(_run_agent(job_id, task, conn, http_client))

    return RedirectResponse(url=f"/agents/{job_id}", status_code=303)


@router.get("/{job_id}", response_class=HTMLResponse)
async def job_view(job_id: int, request: Request):
    conn: sqlite3.Connection = request.app.state.db
    row = conn.execute("SELECT id, task, status FROM agent_jobs WHERE id = ?", (job_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="job not found")
    steps = _get_steps(conn, job_id)
    jobs = _get_jobs(conn)
    return templates.TemplateResponse(
        request=request,
        name="agent_job.html",
        context={"job": dict(row), "steps": steps, "jobs": jobs},
    )


@router.websocket("/{job_id}/ws")
async def job_ws(job_id: int, websocket: WebSocket):
    """WebSocket for live transcript. Stores connection; closed on disconnect."""
    await websocket.accept()
    _job_ws[job_id] = websocket
    try:
        # Keep connection open until client disconnects
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _job_ws.pop(job_id, None)


@router.post("/{job_id}/approve")
async def approve_tool(job_id: int, request: Request):
    form = await request.form()
    session = form.get("session", "false") == "true"

    state = _job_state.get(job_id)
    if not state or state.get("event") is None:
        raise HTTPException(status_code=404, detail="no pending approval")

    state["approved"] = True
    if session and state.get("current_tool"):
        state["session_tools"].add(state["current_tool"])
    state["event"].set()

    return RedirectResponse(url=f"/agents/{job_id}", status_code=303)


@router.post("/{job_id}/deny")
async def deny_tool(job_id: int):
    state = _job_state.get(job_id)
    if not state or state.get("event") is None:
        raise HTTPException(status_code=404, detail="no pending approval")

    state["approved"] = False
    state["event"].set()

    return RedirectResponse(url=f"/agents/{job_id}", status_code=303)
