"""
qwen-text-server — inference server for chat, code completion, and embeddings.

Runs on port 8766. Internal use only (127.0.0.1).

Supports three backends:
  - MLX (default): loads HuggingFace MLX models in-process
  - Ollama: proxies to a local Ollama server (localhost:11434)
  - DS4: proxies to ds4-server (DeepSeek V4 Flash via OpenAI-compatible API)

Backend is selected via TEXT_BACKEND env var ("mlx", "ollama", or "ds4").
The MLX embedding model is always available regardless of backend, since
Ollama/DS4 don't ship MiniLM and RAG depends on the 384-dim contract.

Endpoints:
  POST /chat      { messages: [{role, content}], max_tokens?: int } → SSE tokens
  POST /complete  { prompt: str, max_tokens?: int }                 → SSE tokens
  POST /embed     { text: str }                                     → { embedding: [float] }
  GET  /health    → { status: "ready"|"loading"|"offline", model: str, backend: str }
"""

import asyncio
import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("qwen-text-server")


class _NoHealthFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "GET /health" not in record.getMessage()


logging.getLogger("uvicorn.access").addFilter(_NoHealthFilter())

# ── Backend configuration ─────────────────────────────────────────────────────
BACKEND = os.environ.get("TEXT_BACKEND", "mlx")  # "mlx" | "ollama" | "ds4"
MODEL_ID = os.environ.get("QWEN_TEXT_MODEL", "mlx-community/Qwen2.5-Coder-32B-Instruct-8bit")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:26b")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
DS4_HOST = os.environ.get("DS4_HOST", "http://127.0.0.1:8767")
DS4_MODEL = os.environ.get("DS4_MODEL", "deepseek-v4-flash")

HOST = "127.0.0.1"
PORT = int(os.environ.get("TEXT_SERVER_PORT", "8766"))
DEFAULT_MAX_TOKENS = 2048

# ── MLX state (only used when BACKEND == "mlx") ──────────────────────────────
_model = None
_tokenizer = None
_load_lock = threading.Lock()
_is_loading = False

EMBED_MODEL_ID = "mlx-community/all-MiniLM-L6-v2-4bit"
_embed_model = None
_embed_processor = None
_embed_lock = threading.Lock()
_embed_loading = False

# Serializes all MLX inference calls — Metal cannot handle concurrent generation
_inference_lock = threading.Lock()

# ── Ollama HTTP client (only used when BACKEND == "ollama") ───────────────────
_ollama_client: httpx.AsyncClient | None = None


def _get_ollama_client() -> httpx.AsyncClient:
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = httpx.AsyncClient(base_url=OLLAMA_HOST, timeout=300.0)
    return _ollama_client


# ── DS4 HTTP client (only used when BACKEND == "ds4") ─────────────────────────
_ds4_client: httpx.AsyncClient | None = None


def _get_ds4_client() -> httpx.AsyncClient:
    global _ds4_client
    if _ds4_client is None:
        _ds4_client = httpx.AsyncClient(base_url=DS4_HOST, timeout=600.0)
    return _ds4_client


def _load_embed_model() -> None:
    """Load the MLX embedding model. Thread-safe via _embed_lock."""
    global _embed_model, _embed_processor, _embed_loading
    with _embed_lock:
        if _embed_model is not None:
            return
        _embed_loading = True
        log.info("Loading embedding model %s ...", EMBED_MODEL_ID)
        try:
            from mlx_embeddings import load as embed_load
            _embed_model, _embed_processor = embed_load(EMBED_MODEL_ID)
            log.info("Embedding model loaded")
        except Exception:
            log.exception("failed to load embedding model")
            raise
        finally:
            _embed_loading = False


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


async def _mlx_token_stream(prompt: str, max_tokens: int) -> AsyncGenerator[str, None]:
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
        with _inference_lock:
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


async def _ollama_chat_stream(messages: list[dict], max_tokens: int) -> AsyncGenerator[str, None]:
    """
    Stream chat completion from Ollama, translating NDJSON to SSE.

    Uses a background task + asyncio.Queue because httpx streaming inside
    a FastAPI async generator doesn't yield tokens to the client properly.
    """
    q: asyncio.Queue = asyncio.Queue()

    async def _read_stream():
        client = _get_ollama_client()
        payload = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": True,
            "think": False,
            "options": {"num_predict": max_tokens},
        }
        try:
            async with client.stream("POST", "/api/chat", json=payload) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    await q.put(f"data: ERROR: Ollama returned {resp.status_code}: {body.decode()}\n\n")
                    return
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    if chunk.get("done"):
                        break
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        await q.put(f"data: {content.replace(chr(10), chr(92) + 'n')}\n\n")
        except httpx.ConnectError:
            await q.put("data: ERROR: cannot connect to Ollama — is it running?\n\n")
        finally:
            await q.put(None)  # sentinel

    asyncio.create_task(_read_stream())

    while True:
        item = await q.get()
        if item is None:
            yield "data: [DONE]\n\n"
            return
        yield item


async def _ds4_chat_stream(messages: list[dict], max_tokens: int) -> AsyncGenerator[str, None]:
    """
    Stream chat completion from ds4-server, translating OpenAI SSE → our
    `data: <token>\\n\\n` contract. DS4 emits standard OpenAI chunks:
        data: {"choices":[{"delta":{"content":"...","reasoning_content":"..."}}]}
        data: [DONE]
    Reasoning-mode tokens (`delta.reasoning_content`) are wrapped with
    `<think>` / `</think>` sentinel tokens so the UI can style them
    differently from the final answer.
    """
    q: asyncio.Queue = asyncio.Queue()

    def _escape(s: str) -> str:
        return s.replace(chr(10), chr(92) + 'n')

    async def _read_stream():
        client = _get_ds4_client()
        payload = {
            "model": DS4_MODEL,
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens,
        }
        in_thinking = False
        try:
            async with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    await q.put(f"data: ERROR: DS4 returned {resp.status_code}: {body.decode()}\n\n")
                    return
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload_str = line[5:].strip()
                    if not payload_str or payload_str == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(payload_str)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {}) or {}
                    reasoning = delta.get("reasoning_content") or ""
                    content = delta.get("content") or ""
                    if reasoning:
                        if not in_thinking:
                            await q.put("data: <think>\n\n")
                            in_thinking = True
                        await q.put(f"data: {_escape(reasoning)}\n\n")
                    if content:
                        if in_thinking:
                            await q.put("data: </think>\n\n")
                            in_thinking = False
                        await q.put(f"data: {_escape(content)}\n\n")
        except httpx.ConnectError:
            await q.put("data: ERROR: cannot connect to ds4-server — is it running on " + DS4_HOST + "?\n\n")
        except Exception as exc:  # noqa: BLE001
            await q.put(f"data: ERROR: DS4 stream failed: {exc}\n\n")
        finally:
            if in_thinking:
                await q.put("data: </think>\n\n")
            await q.put(None)

    asyncio.create_task(_read_stream())

    while True:
        item = await q.get()
        if item is None:
            yield "data: [DONE]\n\n"
            return
        yield item


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
    if BACKEND == "ollama":
        active_model = OLLAMA_MODEL
    elif BACKEND == "ds4":
        active_model = DS4_MODEL
    else:
        active_model = MODEL_ID

    if BACKEND == "ollama":
        try:
            client = _get_ollama_client()
            resp = await client.get("/api/tags")
            models = [m["name"] for m in resp.json().get("models", [])]
            available = any(
                OLLAMA_MODEL == m or OLLAMA_MODEL == m.split(":")[0]
                for m in models
            )
            status = "ready" if available else "offline"
        except httpx.ConnectError:
            status = "offline"
    elif BACKEND == "ds4":
        try:
            client = _get_ds4_client()
            resp = await client.get("/v1/models", timeout=5.0)
            status = "ready" if resp.status_code == 200 else "offline"
        except httpx.ConnectError:
            status = "offline"
        except Exception:  # noqa: BLE001
            status = "offline"
    else:
        if _model is not None:
            status = "ready"
        elif _is_loading:
            status = "loading"
        else:
            status = "offline"

    # Embedding model: Ollama proxies; MLX/DS4 use the local MLX MiniLM.
    if BACKEND == "ollama":
        embed_status = "ready"
        embed_model = OLLAMA_EMBED_MODEL
    else:
        embed_status = "ready" if _embed_model is not None else ("loading" if _embed_loading else "offline")
        embed_model = EMBED_MODEL_ID

    return {
        "status": status,
        "model": active_model,
        "backend": BACKEND,
        "embed_status": embed_status,
        "embed_model": embed_model,
    }


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Stream a chat completion as SSE.
    Dispatches to MLX or Ollama based on BACKEND config.
    """
    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    if BACKEND == "ollama":
        gen = _ollama_chat_stream(messages, req.max_tokens)
    elif BACKEND == "ds4":
        gen = _ds4_chat_stream(messages, req.max_tokens)
    else:
        # MLX: apply chat template and stream
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
        gen = _mlx_token_stream(prompt, req.max_tokens)

    return StreamingResponse(
        gen,
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

    if BACKEND in ("ollama", "ds4"):
        # Route through chat API so the model's chat template is applied.
        # Raw completion causes degeneration with instruction-tuned models.
        messages = [
            {"role": "system", "content": "Output only the requested content. Do not add explanations, commentary, or follow-up suggestions. Stop immediately when the content is complete."},
            {"role": "user", "content": req.prompt},
        ]
        if BACKEND == "ollama":
            gen = _ollama_chat_stream(messages, req.max_tokens)
        else:
            gen = _ds4_chat_stream(messages, req.max_tokens)
    else:
        if _model is None:
            await asyncio.get_running_loop().run_in_executor(None, _load_model)
        gen = _mlx_token_stream(req.prompt, req.max_tokens)

    return StreamingResponse(
        gen,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/embed")
async def embed(req: EmbedRequest):
    """
    Embed text. Uses Ollama or MLX depending on backend.
    Returns { "embedding": [float, ...] }
    """
    if BACKEND == "ollama":
        try:
            client = _get_ollama_client()
            resp = await client.post("/api/embed", json={
                "model": OLLAMA_EMBED_MODEL,
                "input": req.text,
            })
            if resp.status_code != 200:
                raise HTTPException(status_code=500, detail=f"Ollama embed failed: {resp.text}")
            data = resp.json()
            embedding = data["embeddings"][0]
            return {"embedding": embedding}
        except httpx.ConnectError as exc:
            raise HTTPException(status_code=503, detail="cannot connect to Ollama") from exc
    # MLX and DS4 both use the local MLX MiniLM model — DS4 has no embed endpoint.
    else:
        if _embed_model is None:
            await asyncio.get_running_loop().run_in_executor(None, _load_embed_model)
        try:
            output = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: _embed_processor([req.text]),
            )
            embedding = output.text_embeds[0].tolist()
            return {"embedding": embedding}
        except Exception as exc:
            log.exception("embedding failed")
            raise HTTPException(status_code=500, detail=f"embedding failed: {exc}") from exc


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
