"""
skills.py router — Read-only view of all skills and their DB status.

Shows which skills were injected in the most recent chat or workspace request.
Does not allow editing — the filesystem is the interface.
"""

import sqlite3
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter(prefix="/skills")
templates = Jinja2Templates(directory="templates")


@router.get("/", response_class=HTMLResponse)
async def skills_page(request: Request):
    conn: sqlite3.Connection = request.app.state.db
    rows = conn.execute(
        "SELECT filepath, name, mtime FROM skill_embeddings ORDER BY name"
    ).fetchall()
    skills = [dict(r) for r in rows]

    # Add relative path for display
    studio_root = request.app.state.studio_root
    for s in skills:
        try:
            s["rel_path"] = str(Path(s["filepath"]).relative_to(studio_root))
        except ValueError:
            s["rel_path"] = s["filepath"]

    last_injected = getattr(request.app.state, "last_injected_skills", [])
    last_injected_set = set(last_injected)

    skills_dir = studio_root / "data" / "skills"
    return templates.TemplateResponse(
        request=request,
        name="skills.html",
        context={
            "skills": skills,
            "skills_dir": str(skills_dir),
            "last_injected": last_injected,
            "last_injected_set": last_injected_set,
        },
    )
