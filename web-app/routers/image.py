"""
image.py — Image generation router.

Proxies POST /image/generate to the qwen-image-server at 127.0.0.1:8765.
Saves returned PNG to data/images/ with a timestamp filename.
Returns HTMX-swappable HTML fragments.
"""

import base64
import sqlite3
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import httpx
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates

router = APIRouter(prefix="/image")
templates = Jinja2Templates(directory="templates")

IMAGE_SERVER = "http://127.0.0.1:8765"
IMAGE_DIR = Path("../data/images")  # relative to web-app working dir


def _images_dir(base: Path) -> Path:
    """Return absolute images directory, creating it if needed."""
    d = base / "data" / "images"
    d.mkdir(parents=True, exist_ok=True)
    return d


@router.get("/", response_class=HTMLResponse)
async def image_page(request: Request):
    conn: sqlite3.Connection = request.app.state.db
    rows = conn.execute(
        "SELECT id, prompt, filename, width, height, created_at "
        "FROM images ORDER BY id DESC LIMIT 50"
    ).fetchall()
    return templates.TemplateResponse(
        request=request,
        name="image.html",
        context={"images": [dict(r) for r in rows]},
    )


@router.post("/generate", response_class=HTMLResponse)
async def generate(request: Request):
    form = await request.form()
    prompt = (form.get("prompt") or "").strip()
    width = int(form.get("width") or 768)
    height = int(form.get("height") or 768)

    if not prompt:
        return HTMLResponse(
            '<p class="error">Prompt cannot be empty.</p>',
            status_code=400,
        )

    client: httpx.AsyncClient = request.app.state.http_client
    try:
        resp = await client.post(
            f"{IMAGE_SERVER}/generate",
            json={"prompt": prompt, "width": width, "height": height},
            timeout=300.0,
        )
        resp.raise_for_status()
    except (httpx.HTTPError, httpx.TimeoutException) as exc:
        return HTMLResponse(
            f'<p class="error">Image server error: {exc}</p>',
            status_code=502,
        )

    data = resp.json()
    b64 = data["image"]

    # Save to disk
    studio_root = request.app.state.studio_root
    images_dir = _images_dir(studio_root)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{ts}.png"
    (images_dir / filename).write_bytes(base64.b64decode(b64))

    # Persist to DB
    conn: sqlite3.Connection = request.app.state.db
    conn.execute(
        "INSERT INTO images (prompt, filename, width, height) VALUES (?, ?, ?, ?)",
        (prompt, filename, width, height),
    )
    conn.commit()

    # Return a gallery card fragment to be prepended by HTMX
    escaped_prompt = escape(prompt)
    return HTMLResponse(f"""
<div class="gallery-item">
    <img src="data:image/png;base64,{b64}" alt="{escaped_prompt}">
    <div class="caption">{escaped_prompt}</div>
</div>
""")


@router.get("/file/{filename}")
async def serve_image(filename: str, request: Request):
    """Serve a saved image by filename."""
    # Reject any path traversal attempts
    if "/" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="invalid filename")
    studio_root = request.app.state.studio_root
    path = _images_dir(studio_root) / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="image not found")
    return FileResponse(str(path), media_type="image/png")
