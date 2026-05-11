"""
image.py — Image generation router.

Proxies POST /image/generate to the qwen-image-server job system.
Saves returned PNG to data/images/ on completion.
Returns HTMX-swappable HTML fragments with SSE progress tracking.
"""

import asyncio
import base64
import json
import sqlite3
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import httpx
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

router = APIRouter(prefix="/image")
templates = Jinja2Templates(directory="templates")

IMAGE_SERVER = "http://127.0.0.1:8765"


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
            timeout=60.0,
        )
        resp.raise_for_status()
    except (httpx.HTTPError, httpx.TimeoutException) as exc:
        return HTMLResponse(
            f'<p class="error">Image server error: {escape(str(exc))}</p>',
            status_code=502,
        )

    job_id = resp.json()["job_id"]
    escaped_prompt = escape(prompt)

    # Return a progress card that streams status via SSE
    return HTMLResponse(f"""
<div class="gallery-item" id="job-{job_id}">
    <div class="generation-progress" style="padding: 1rem;">
        <div style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 0.5rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{escaped_prompt}</div>
        <div style="background: var(--border); border-radius: 4px; height: 6px; margin-bottom: 0.5rem;">
            <div id="bar-{job_id}" style="background: var(--accent); height: 6px; border-radius: 4px; width: 0%; transition: width 0.3s;"></div>
        </div>
        <div id="status-{job_id}" style="font-size: 0.8rem; color: var(--text-muted);">Starting…</div>
        <button id="cancel-{job_id}" onclick="cancelJob('{job_id}')"
            class="secondary" style="margin-top: 0.5rem; padding: 0.2rem 0.6rem; font-size: 0.8rem; color: var(--red);">
            Cancel
        </button>
    </div>
</div>
<script>
(function() {{
    const es = new EventSource("/image/progress/{job_id}");
    es.onmessage = function(e) {{
        const d = JSON.parse(e.data);
        if (d.status === "done") {{
            es.close();
            document.getElementById("job-{job_id}").outerHTML =
                '<div class="gallery-item"><img src="/image/file/' + d.filename + '" alt="{escaped_prompt}"><div class="caption">{escaped_prompt}</div></div>';
            return;
        }}
        if (d.status === "cancelled" || d.status === "failed") {{
            es.close();
            document.getElementById("job-{job_id}").outerHTML =
                '<div class="gallery-item" style="padding:1rem;color:var(--red);font-size:0.85rem;">' + d.status + (d.error ? ': ' + d.error : '') + '</div>';
            return;
        }}
        const pct = d.total > 0 ? Math.round(d.step / d.total * 100) : 0;
        const secs = d.step > 0 && d.elapsed > 0 ? Math.round((d.total - d.step) * (d.elapsed / d.step)) : null;
        document.getElementById("bar-{job_id}").style.width = pct + "%";
        document.getElementById("status-{job_id}").textContent =
            "Step " + d.step + " / " + d.total + " (" + pct + "%)" +
            (secs !== null ? " — ~" + secs + "s remaining" : "");
    }};
    es.onerror = function() {{ es.close(); }};
}})();

function cancelJob(jobId) {{
    fetch("/image/cancel/" + jobId, {{method: "POST"}});
    document.getElementById("cancel-" + jobId).disabled = true;
    document.getElementById("cancel-" + jobId).textContent = "Cancelling…";
}}
</script>
""")


@router.get("/progress/{job_id}")
async def generation_progress(job_id: str, request: Request):
    """SSE endpoint: polls image server for job status and streams progress to browser."""
    client: httpx.AsyncClient = request.app.state.http_client
    conn: sqlite3.Connection = request.app.state.db
    studio_root = request.app.state.studio_root

    async def _stream():
        import time
        start = time.monotonic()
        while True:
            try:
                resp = await client.get(
                    f"{IMAGE_SERVER}/status/{job_id}",
                    timeout=5.0,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                yield f"data: {json.dumps({'status': 'failed', 'error': str(exc)})}\n\n"
                return

            elapsed = time.monotonic() - start
            data["elapsed"] = round(elapsed, 1)

            if data["status"] == "done":
                # Save image to disk and DB
                b64 = data.pop("image")
                images_dir = _images_dir(studio_root)
                ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                filename = f"{ts}.png"
                (images_dir / filename).write_bytes(base64.b64decode(b64))

                conn.execute(
                    "INSERT INTO images (prompt, filename, width, height) VALUES (?, ?, ?, ?)",
                    (data.get("prompt", ""), filename, data.get("width", 0), data.get("height", 0)),
                )
                conn.commit()
                data["filename"] = filename
                yield f"data: {json.dumps(data)}\n\n"
                return

            if data["status"] in ("cancelled", "failed"):
                yield f"data: {json.dumps(data)}\n\n"
                return

            yield f"data: {json.dumps(data)}\n\n"
            await asyncio.sleep(1.0)

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/cancel/{job_id}")
async def cancel_generation(job_id: str, request: Request):
    client: httpx.AsyncClient = request.app.state.http_client
    try:
        await client.post(f"{IMAGE_SERVER}/cancel/{job_id}", timeout=5.0)
    except Exception:
        pass
    return HTMLResponse("", status_code=204)


@router.get("/file/{filename}")
async def serve_image(filename: str, request: Request):
    """Serve a saved image by filename."""
    if "/" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="invalid filename")
    studio_root = request.app.state.studio_root
    path = _images_dir(studio_root) / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="image not found")
    return FileResponse(str(path), media_type="image/png")
