"""
settings.py — Settings page and allowlist management.

Allowed filesystem directories are stored in the `settings` table with
keys like `allowed_dir_0`, `allowed_dir_1`, etc.
"""

import sqlite3
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

router = APIRouter(prefix="/settings")
templates = Jinja2Templates(directory="templates")


def _get_allowed_dirs(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT key, value FROM settings WHERE key LIKE 'allowed_dir_%' ORDER BY key"
    ).fetchall()
    return [r["value"] for r in rows]


def _set_allowed_dirs(conn: sqlite3.Connection, dirs: list[str]) -> None:
    """Replace all allowed_dir_* settings with new list."""
    conn.execute("DELETE FROM settings WHERE key LIKE 'allowed_dir_%'")
    for i, d in enumerate(dirs):
        conn.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?)",
            (f"allowed_dir_{i}", d),
        )
    conn.commit()


def init_default_allowlist(conn: sqlite3.Connection, studio_root: Path) -> None:
    """
    Set default allowlist if none is configured.
    Called from main.py lifespan after schema init.
    """
    existing = _get_allowed_dirs(conn)
    if not existing:
        defaults = [
            str(studio_root / "data" / "skills"),
            str(studio_root / "data" / "workspace"),
        ]
        _set_allowed_dirs(conn, defaults)


@router.get("/", response_class=HTMLResponse)
async def settings_page(request: Request):
    conn: sqlite3.Connection = request.app.state.db
    dirs = _get_allowed_dirs(conn)
    return templates.TemplateResponse(
        request=request,
        name="settings.html",
        context={"allowed_dirs": dirs},
    )


@router.post("/allowlist/add")
async def add_allowed_dir(request: Request):
    form = await request.form()
    new_dir = (form.get("directory") or "").strip()
    if not new_dir:
        return RedirectResponse(url="/settings/", status_code=303)
    conn: sqlite3.Connection = request.app.state.db
    dirs = _get_allowed_dirs(conn)
    resolved = str(Path(new_dir).resolve())
    if resolved not in dirs:
        dirs.append(resolved)
        _set_allowed_dirs(conn, dirs)
    return RedirectResponse(url="/settings/", status_code=303)


@router.post("/allowlist/remove")
async def remove_allowed_dir(request: Request):
    form = await request.form()
    remove_dir = (form.get("directory") or "").strip()
    conn: sqlite3.Connection = request.app.state.db
    dirs = _get_allowed_dirs(conn)
    dirs = [d for d in dirs if d != remove_dir]
    _set_allowed_dirs(conn, dirs)
    return RedirectResponse(url="/settings/", status_code=303)
