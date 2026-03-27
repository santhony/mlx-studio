"""
main.py — Qwen Studio web application.

FastAPI app running on 127.0.0.1:8080. Proxies to:
  - qwen-image-server: 127.0.0.1:8765
  - qwen-text-server:  127.0.0.1:8766

All persistence in data/studio.db (SQLite).
"""

import logging
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from db import get_connection, init_schema
from routers import image as image_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("web-app")

HOST = "127.0.0.1"
PORT = 8080
IMAGE_SERVER = "http://127.0.0.1:8765"
TEXT_SERVER = "http://127.0.0.1:8766"

# Path to qwen-studio/ directory (parent of web-app/)
STUDIO_ROOT = Path(__file__).parent.parent
DB_PATH = STUDIO_ROOT / "data" / "studio.db"

templates = Jinja2Templates(directory="templates")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Qwen Studio web-app starting on %s:%d", HOST, PORT)

    # SQLite
    conn = get_connection(DB_PATH)
    init_schema(conn)
    app.state.db = conn
    app.state.studio_root = STUDIO_ROOT

    # httpx client (shared across requests)
    async with httpx.AsyncClient() as client:
        app.state.http_client = client
        yield

    conn.close()
    log.info("Qwen Studio web-app shut down")


app = FastAPI(title="Qwen Studio", lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(image_router.router)


# ── Health / status endpoints ─────────────────────────────────────────────────

@app.get("/status/image", response_class=HTMLResponse)
async def status_image(request: Request):
    client: httpx.AsyncClient = request.app.state.http_client
    try:
        resp = await client.get(f"{IMAGE_SERVER}/health", timeout=2.0)
        data = resp.json()
        status = data.get("status", "offline")
    except Exception:
        status = "offline"

    css = {
        "ready": "status-ready",
        "loading": "status-loading",
        "offline": "status-offline",
    }.get(status, "status-offline")
    label = {"ready": "✓ image", "loading": "⏳ image", "offline": "✗ image"}.get(status, "✗ image")

    return HTMLResponse(
        f'<span id="image-server-status" class="{css}" '
        f'hx-get="/status/image" hx-trigger="every 5s" hx-swap="outerHTML">{label}</span>'
    )


@app.get("/status/text", response_class=HTMLResponse)
async def status_text(request: Request):
    client: httpx.AsyncClient = request.app.state.http_client
    try:
        resp = await client.get(f"{TEXT_SERVER}/health", timeout=2.0)
        data = resp.json()
        status = data.get("status", "offline")
    except Exception:
        status = "offline"

    css = {
        "ready": "status-ready",
        "loading": "status-loading",
        "offline": "status-offline",
    }.get(status, "status-offline")
    label = {"ready": "✓ text", "loading": "⏳ text", "offline": "✗ text"}.get(status, "✗ text")

    return HTMLResponse(
        f'<span id="text-server-status" class="{css}" '
        f'hx-get="/status/text" hx-trigger="every 5s" hx-swap="outerHTML">{label}</span>'
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
