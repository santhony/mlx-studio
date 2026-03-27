"""
qwen-text-server stub — Phase 1 placeholder.

Returns 503 for all inference endpoints. Real MLX implementation in Phase 2.
Runs on port 8766.
"""

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("qwen-text-server")

HOST = "127.0.0.1"
PORT = 8766


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("qwen-text-server (stub) starting on %s:%d", HOST, PORT)
    yield
    log.info("qwen-text-server shutting down")


app = FastAPI(title="qwen-text-server", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "offline", "model": "not yet implemented"}


@app.post("/chat")
async def chat():
    return JSONResponse(
        status_code=503,
        content={"status": "not yet implemented"},
    )


@app.post("/complete")
async def complete():
    return JSONResponse(
        status_code=503,
        content={"status": "not yet implemented"},
    )


@app.post("/embed")
async def embed():
    return JSONResponse(
        status_code=503,
        content={"status": "not yet implemented"},
    )


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
