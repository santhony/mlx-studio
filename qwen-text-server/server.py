"""
qwen-text-server — MLX inference server for chat, code completion, and embeddings.

Runs on port 8766. Internal use only (127.0.0.1).

Model: mlx-community/Qwen2.5-Coder-32B-Instruct (BF16, ~65 GB)
Downloads to ~/.cache/huggingface on first request.

Endpoints:
  POST /chat      { messages: [{role, content}], max_tokens?: int } → SSE tokens
  POST /complete  { prompt: str, max_tokens?: int }                 → SSE tokens
  POST /embed     { text: str }                                     → { embedding: [float] }
  GET  /health    → { status: "ready"|"loading"|"offline", model: str }
"""

import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("qwen-text-server")

MODEL_ID = "mlx-community/Qwen2.5-Coder-32B-Instruct"
HOST = "127.0.0.1"
PORT = 8766
DEFAULT_MAX_TOKENS = 2048

_model = None
_tokenizer = None
_load_lock = threading.Lock()
_is_loading = False


def _load_model() -> None:
    """
    Load the MLX model and tokenizer. Thread-safe via _load_lock.
    Called lazily from the first inference request.
    """
    global _model, _tokenizer, _is_loading
    with _load_lock:
        if _model is not None:
            return
        _is_loading = True
        log.info("Loading %s ...", MODEL_ID)
        try:
            from mlx_lm import load
            _model, _tokenizer = load(MODEL_ID)
            log.info("Model loaded successfully")
        except Exception:
            log.exception("failed to load model")
            raise
        finally:
            _is_loading = False


async def _token_stream(prompt: str, max_tokens: int) -> AsyncGenerator[str, None]:
    """
    Bridge sync mlx_lm.stream_generate() to an async SSE generator.

    Runs stream_generate in a daemon thread and communicates results via an
    asyncio.Queue. Each queue item is either a token string, None (sentinel
    meaning done), or an Exception.

    Yields: SSE-formatted lines, e.g. "data: token\\n\\n", "data: [DONE]\\n\\n"
    """
    if _model is None:
        await asyncio.get_running_loop().run_in_executor(None, _load_model)

    q: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _run() -> None:
        try:
            from mlx_lm import stream_generate
            for response in stream_generate(_model, _tokenizer, prompt, max_tokens=max_tokens):
                # Escape newlines in token text so SSE stays well-formed
                token = response.text.replace("\n", "\\n")
                asyncio.run_coroutine_threadsafe(q.put(token), loop)
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(q.put(exc), loop)
        finally:
            asyncio.run_coroutine_threadsafe(q.put(None), loop)

    threading.Thread(target=_run, daemon=True).start()

    while True:
        item = await q.get()
        if item is None:
            yield "data: [DONE]\n\n"
            return
        if isinstance(item, Exception):
            yield f"data: ERROR: {item}\n\n"
            return
        yield f"data: {item}\n\n"


# ── Pydantic models ────────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    max_tokens: int = DEFAULT_MAX_TOKENS


class CompleteRequest(BaseModel):
    prompt: str
    max_tokens: int = DEFAULT_MAX_TOKENS


class EmbedRequest(BaseModel):
    text: str


# ── FastAPI app ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("qwen-text-server starting on %s:%d", HOST, PORT)
    yield
    log.info("qwen-text-server shutting down")


app = FastAPI(title="qwen-text-server", lifespan=lifespan)


@app.get("/health")
async def health():
    if _model is not None:
        status = "ready"
    elif _is_loading:
        status = "loading"
    else:
        status = "offline"
    return {"status": status, "model": MODEL_ID}


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Stream a chat completion as SSE.
    Messages are formatted via the model's chat template before inference.
    """
    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    # Ensure model is loaded (run_in_executor so we don't block the event loop)
    if _model is None:
        await asyncio.get_running_loop().run_in_executor(None, _load_model)

    try:
        prompt = _tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"failed to apply chat template: {exc}",
        ) from exc

    return StreamingResponse(
        _token_stream(prompt, req.max_tokens),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/complete")
async def complete(req: CompleteRequest):
    """Stream a raw completion (no chat template) as SSE."""
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt must not be empty")

    if _model is None:
        await asyncio.get_running_loop().run_in_executor(None, _load_model)

    return StreamingResponse(
        _token_stream(req.prompt, req.max_tokens),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/embed")
async def embed(req: EmbedRequest):
    """
    Embed text using nomic-embed-text (Phase 4).
    Returns 503 until Phase 4 is implemented.
    """
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=503,
        content={"status": "not yet implemented — coming in Phase 4"},
    )


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
