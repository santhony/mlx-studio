"""
qwen-image-server — FastAPI server for Qwen-Image-2512 inference.

Runs on port 8765. Internal use only (127.0.0.1).

Inherits the MPS SDPA monkey-patch and GQA head-expansion fix from
bluesky-alt-reimagine/local-server/qwen_server.py, rewritten for FastAPI.
"""

import base64
import io
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Must be set before torch import
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("qwen-image-server")

MODEL_ID = "Qwen/Qwen-Image-2512"
HOST = "127.0.0.1"
PORT = 8765

_pipe = None
_device = None


def _patch_sdpa_for_mps() -> None:
    """
    Replace torch.nn.functional.scaled_dot_product_attention with a manual
    matmul+softmax implementation that bypasses the crashing MPS fused SDPA
    kernel (_scaled_dot_product_attention_math_mps in MetalPerformanceShadersGraph).

    PYTORCH_ENABLE_MPS_FALLBACK=1 does not prevent the crash because the kernel
    calls C-level abort() rather than raising a Python exception.

    Also handles Grouped Query Attention (GQA) by expanding KV heads to match
    query head count via repeat_interleave before the attention computation.
    """
    import torch
    import torch.nn.functional as F

    _orig = F.scaled_dot_product_attention

    def _mps_safe_sdpa(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale=None,
        **kwargs,
    ):
        if query.device.type != "mps":
            return _orig(
                query, key, value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                **kwargs,
            )

        L, S = query.size(-2), key.size(-2)
        scale_factor = scale if scale is not None else query.size(-1) ** -0.5

        # GQA: expand KV heads to match Q heads (e.g. 28 query / 4 KV)
        if key.size(1) != query.size(1):
            n_rep = query.size(1) // key.size(1)
            key = key.repeat_interleave(n_rep, dim=1)
            value = value.repeat_interleave(n_rep, dim=1)

        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        if is_causal:
            causal_mask = torch.ones(
                L, S, dtype=torch.bool, device=query.device
            ).tril()
            attn_bias = attn_bias.masked_fill(~causal_mask, float("-inf"))
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias = attn_bias.masked_fill(~attn_mask, float("-inf"))
            else:
                attn_bias = attn_bias + attn_mask

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight = attn_weight + attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        if dropout_p > 0.0:
            attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        return attn_weight @ value

    F.scaled_dot_product_attention = _mps_safe_sdpa
    log.info("MPS SDPA patch applied — using manual attention on MPS")


def _load_pipeline():
    """Load Qwen-Image-2512 via diffusers. Called lazily on first /generate request."""
    global _pipe, _device
    import torch
    from diffusers import DiffusionPipeline

    _device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info("Loading %s on %s ...", MODEL_ID, _device)

    if _device == "mps":
        _patch_sdpa_for_mps()

    _pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    _pipe = _pipe.to(_device)
    _pipe.enable_attention_slicing()
    log.info("Model loaded on %s", _device)


# ── Pydantic models ────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str
    width: int = 768
    height: int = 768


# ── FastAPI app ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("qwen-image-server starting on %s:%d", HOST, PORT)
    yield
    log.info("qwen-image-server shutting down")


app = FastAPI(title="qwen-image-server", lifespan=lifespan)


@app.get("/health")
async def health():
    status = "ready" if _pipe is not None else "loading"
    return {"status": status, "model": MODEL_ID}


@app.post("/generate")
async def generate(req: GenerateRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt must not be empty")

    # Lazy-load model on first request
    if _pipe is None:
        _load_pipeline()

    # Clamp and snap dimensions to multiples of 64
    width = min(max(req.width, 64), 1024)
    height = min(max(req.height, 64), 1024)
    width = (width // 64) * 64
    height = (height // 64) * 64

    try:
        result = _pipe(
            prompt=req.prompt,
            negative_prompt="blurry, low quality, distorted, watermark",
            width=width,
            height=height,
            num_inference_steps=30,
        )
        image = result.images[0]
    except Exception as exc:
        log.exception("generation failed")
        raise HTTPException(status_code=500, detail=f"generation failed: {exc}") from exc

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"image": b64}


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
