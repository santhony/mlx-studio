"""
bridge.py — Router that exposes the Bridge as a Qwen Studio endpoint.

Allows you to feed DeepSeek messages from the chat into Qwen Studio's sandbox,
get tool execution results, and feed them back.

Endpoint:
  POST /bridge/process
    body: {"message": "DeepSeek text with <tool> blocks", "mode": "auto|supervised|interactive"}
    response: {"reasoning":..., "tool_call":..., "result":..., "needs_approval":..., "tool_name":...}

Approval:
  POST /bridge/approve
    body: {"tool_name": "shell", "allowed": true, "session": true}
"""

import asyncio
import json
import logging
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from bridge import Bridge, CATEGORY_SAFE, CATEGORY_UNSAFE

log = logging.getLogger("bridge-router")

router = APIRouter(prefix="/bridge")

# Per-session state: stores a Bridge instance keyed by a session_id
_sessions: dict[str, Bridge] = {}

# Pending approval state: {tool_name: asyncio.Event, etc}
_pending_approvals: dict[str, dict] = {}


def _get_allowed_dirs(request: Request) -> list[str]:
    """Get allowed dirs from the DB."""
    conn = request.app.state.db
    rows = conn.execute(
        "SELECT value FROM settings WHERE key LIKE 'allowed_dir_%' ORDER BY key"
    ).fetchall()
    return [r["value"] for r in rows]


def _get_or_create_bridge(session_id: str, request: Request, mode: str = "supervised") -> Bridge:
    """Get or create a Bridge for a session."""
    if session_id not in _sessions:
        allowed_dirs = _get_allowed_dirs(request)
        http_client = request.app.state.http_client
        _sessions[session_id] = Bridge(
            allowed_dirs=allowed_dirs,
            http_client=http_client,
            mode=mode,
        )
    return _sessions[session_id]


@router.post("/process")
async def process_message(request: Request):
    """
    Process a DeepSeek message through the bridge.

    Request body:
      {
        "message": "The DeepSeek reasoning text with <tool> blocks",
        "session_id": "optional-session-key",
        "mode": "auto|supervised|interactive",
        "approval_grant": true | false | null  (whether to approve a pending tool)
      }
    """
    body = await request.json()
    message = body.get("message", "")
    if not message:
        raise HTTPException(status_code=400, detail="missing 'message'")

    session_id = body.get("session_id", "default")
    mode = body.get("mode", "supervised")
    approval_grant = body.get("approval_grant", None)

    bridge = _get_or_create_bridge(session_id, request, mode)

    result = await bridge.process_message(message, approval_grant=approval_grant)

    return JSONResponse(result)


@router.post("/reset")
async def reset_session(request: Request):
    """Reset a bridge session (clear session allowlist)."""
    body = await request.json()
    session_id = body.get("session_id", "default")
    _sessions.pop(session_id, None)
    return JSONResponse({"status": f"session '{session_id}' reset"})


@router.get("/status")
async def bridge_status(request: Request):
    """Check bridge status and active sessions."""
    return JSONResponse({
        "active_sessions": list(_sessions.keys()),
        "session_details": {
            sid: {
                "mode": b.mode,
                "session_allowed_tools": list(b.session_allowed_tools),
            }
            for sid, b in _sessions.items()
        },
    })
