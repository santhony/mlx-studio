"""
finetune.py — Fine-tuning router.

Manages LoRA/QLoRA fine-tuning jobs via mlx_lm.lora subprocess.
Uses the venv-text Python binary so mlx-lm is available.

Routes:
  GET  /finetune/                           → job list + new job form
  POST /finetune/upload                     → upload JSONL dataset
  POST /finetune/                           → create + start job
  GET  /finetune/{job_id}                   → job view + live metrics
  POST /finetune/{job_id}/stop              → stop running job
  POST /finetune/{job_id}/export            → copy adapter to checkpoints/
  WS   /finetune/{job_id}/ws               → live metrics WebSocket
"""

import json
import re
import shutil
import sqlite3
import subprocess
import threading
from pathlib import Path
from typing import Optional

import psutil
from fastapi import APIRouter, HTTPException, Request, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

router = APIRouter(prefix="/finetune")
templates = Jinja2Templates(directory="templates")

# Path to the venv-text Python binary — mlx-lm is only installed there
VENV_TEXT_PYTHON = str(
    Path(__file__).parent.parent.parent / "qwen-text-server" / "venv-text" / "bin" / "python3"
)

# Regex for parsing mlx_lm.lora output
_TRAIN_RE = re.compile(
    r"Iter\s+(\d+):\s+Train loss\s+([\d.]+)"
    r"(?:,\s+Learning Rate\s+([\d.e\-]+))?"
    r"(?:,\s+It/sec\s+([\d.]+))?"
    r"(?:,\s+Tokens/sec\s+([\d.]+))?"
    r"(?:,\s+Trained Tokens\s+(\d+))?"
    r"(?:,\s+Peak mem\s+([\d.]+)\s+GB)?"
)
_VAL_RE = re.compile(r"Iter\s+(\d+):\s+Val loss\s+([\d.]+)")

# Per-job live state: {job_id: {"process": Popen, "metrics": [...], "ws": WebSocket}}
_job_live: dict[int, dict] = {}


# ── DB helpers ─────────────────────────────────────────────────────────────────

def _get_jobs(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT id, base_model, dataset_path, config_json, status, created_at "
        "FROM finetune_jobs ORDER BY id DESC LIMIT 50"
    ).fetchall()
    return [dict(r) for r in rows]


def _get_job(conn: sqlite3.Connection, job_id: int) -> Optional[dict]:
    row = conn.execute(
        "SELECT id, base_model, dataset_path, config_json, status, created_at "
        "FROM finetune_jobs WHERE id = ?",
        (job_id,),
    ).fetchone()
    return dict(row) if row else None


def _update_job_status(conn: sqlite3.Connection, job_id: int, status: str) -> None:
    conn.execute("UPDATE finetune_jobs SET status = ? WHERE id = ?", (status, job_id))
    conn.commit()


# ── Dataset upload + validation ───────────────────────────────────────────────

def _validate_jsonl(content: bytes) -> tuple[bool, str, int]:
    """
    Validate JSONL content.
    Returns (is_valid, error_message, line_count).
    """
    lines = content.decode("utf-8", errors="replace").strip().split("\n")
    valid_count = 0
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            return False, f"invalid JSON on line {i}: {exc}", 0
        # Accept any of the supported formats
        if not any(k in obj for k in ("text", "prompt", "messages")):
            return (
                False,
                f"line {i}: each record must have 'text', 'prompt', or 'messages' key",
                0,
            )
        valid_count += 1
    if valid_count == 0:
        return False, "file contains no valid records", 0
    return True, "", valid_count


# ── WebSocket helpers ─────────────────────────────────────────────────────────

async def _ws_send(job_id: int, html: str) -> None:
    state = _job_live.get(job_id)
    if state:
        ws = state.get("ws")
        if ws:
            try:
                await ws.send_text(html)
            except Exception:
                pass


def _metric_row_html(metric: dict) -> str:
    """Render a single metrics update as an OOB HTMX HTML fragment."""
    iteration = metric.get("iteration", "?")
    loss = f"{metric['loss']:.4f}" if metric.get("loss") is not None else "—"
    tokens_sec = f"{metric['tokens_sec']:.1f}" if metric.get("tokens_sec") else "—"
    peak_mem = f"{metric['peak_mem']:.2f} GB" if metric.get("peak_mem") else "—"
    mem_avail = metric.get("mem_available_gb", "?")
    mem_avail_str = f"{mem_avail:.1f} GB" if isinstance(mem_avail, (int, float)) else "—"

    return f"""<div id="metrics-table" hx-swap-oob="beforeend">
<tr>
    <td>{iteration}</td>
    <td class="loss-cell">{loss}</td>
    <td>{tokens_sec}</td>
    <td>{peak_mem}</td>
    <td>{mem_avail_str}</td>
</tr>
</div>
<div id="latest-loss" hx-swap-oob="innerHTML">{loss}</div>"""


# ── Training subprocess management ───────────────────────────────────────────

def _parse_metric_line(line: str) -> Optional[dict]:
    """Parse a training output line into a metric dict. Returns None if not a metric."""
    m = _TRAIN_RE.search(line)
    if m:
        iteration, loss, lr, it_sec, tokens_sec, trained_tokens, peak_mem = m.groups()
        return {
            "type": "train",
            "iteration": int(iteration),
            "loss": float(loss),
            "lr": float(lr) if lr else None,
            "tokens_sec": float(tokens_sec) if tokens_sec else None,
            "peak_mem": float(peak_mem) if peak_mem else None,
        }
    m = _VAL_RE.search(line)
    if m:
        iteration, loss = m.groups()
        return {"type": "val", "iteration": int(iteration), "loss": float(loss)}
    return None


def _monitor_training(
    job_id: int,
    process: subprocess.Popen,
    conn: sqlite3.Connection,
    loop,
) -> None:
    """
    Background thread: reads mlx_lm.lora stdout, parses metrics,
    sends WebSocket updates, and marks job complete/failed on exit.
    """
    import asyncio

    state = _job_live.get(job_id, {})

    for raw_line in process.stdout:
        line = raw_line.rstrip()
        metric = _parse_metric_line(line)
        if metric:
            # Add system memory reading
            metric["mem_available_gb"] = psutil.virtual_memory().available / 1024**3
            state.setdefault("metrics", []).append(metric)
            html = _metric_row_html(metric)
            asyncio.run_coroutine_threadsafe(
                _ws_send(job_id, html), loop
            )

    # Process finished
    process.wait()
    status = "completed" if process.returncode == 0 else "failed"
    _update_job_status(conn, job_id, status)

    done_html = f'<div id="job-status" hx-swap-oob="innerHTML"><span class="status-badge status-{status}">{status}</span></div>'
    asyncio.run_coroutine_threadsafe(_ws_send(job_id, done_html), loop)
    _job_live.pop(job_id, None)


def _start_training(
    job_id: int,
    config: dict,
    studio_root: Path,
    conn: sqlite3.Connection,
    loop,
) -> subprocess.Popen:
    """
    Build mlx_lm.lora command and start subprocess.
    The adapter is saved to data/checkpoints/job_{job_id}/ during training.
    """
    adapter_dir = studio_root / "data" / "checkpoints" / f"job_{job_id}"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        VENV_TEXT_PYTHON, "-m", "mlx_lm.lora",
        "--model", config["base_model"],
        "--data", config["dataset_path"],
        "--train",
        "--adapter-path", str(adapter_dir),
        "--iters", str(config.get("iters", 600)),
        "--learning-rate", str(config.get("learning_rate", "2e-4")),
        "--lora-r", str(config.get("lora_r", 16)),
        "--lora-alpha", str(config.get("lora_alpha", 32)),
        "--batch-size", str(config.get("batch_size", 4)),
        "--num-layers", str(config.get("num_layers", 16)),
        "--steps-per-report", "10",
        "--save-every", "100",
        "--mask-prompt",
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # Line-buffered
    )

    _job_live[job_id] = {"process": process, "metrics": [], "ws": None}

    # Start monitoring thread
    t = threading.Thread(
        target=_monitor_training,
        args=(job_id, process, conn, loop),
        daemon=True,
    )
    t.start()

    return process


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
async def finetune_list(request: Request):
    conn: sqlite3.Connection = request.app.state.db
    jobs = _get_jobs(conn)
    return templates.TemplateResponse(
        request=request, name="finetune.html", context={"jobs": jobs}
    )


@router.post("/upload")
async def upload_dataset(request: Request, file: UploadFile = File(...)):
    """Upload and validate a JSONL dataset file."""
    if not file.filename.endswith(".jsonl"):
        raise HTTPException(status_code=400, detail="file must be a .jsonl file")

    content = await file.read()
    valid, error, line_count = _validate_jsonl(content)
    if not valid:
        raise HTTPException(status_code=400, detail=error)

    studio_root: Path = request.app.state.studio_root
    datasets_dir = studio_root / "data" / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(file.filename).name
    dest = datasets_dir / safe_name
    dest.write_bytes(content)

    # Create train.jsonl structure expected by mlx_lm.lora
    job_data_dir = datasets_dir / dest.stem
    job_data_dir.mkdir(exist_ok=True)
    shutil.copy(dest, job_data_dir / "train.jsonl")

    return {"path": str(job_data_dir), "lines": line_count, "filename": file.filename}


@router.post("/")
async def create_job(request: Request):
    import asyncio
    form = await request.form()

    base_model = (form.get("base_model") or "").strip()
    dataset_path = (form.get("dataset_path") or "").strip()
    if not base_model or not dataset_path:
        raise HTTPException(status_code=400, detail="base_model and dataset_path are required")

    config = {
        "base_model": base_model,
        "dataset_path": dataset_path,
        "iters": int(form.get("iters") or 600),
        "learning_rate": form.get("learning_rate") or "2e-4",
        "lora_r": int(form.get("lora_r") or 16),
        "lora_alpha": int(form.get("lora_alpha") or 32),
        "batch_size": int(form.get("batch_size") or 4),
        "num_layers": int(form.get("num_layers") or 16),
    }

    conn: sqlite3.Connection = request.app.state.db
    cur = conn.execute(
        "INSERT INTO finetune_jobs (base_model, dataset_path, config_json, status) VALUES (?, ?, ?, 'running')",
        (base_model, dataset_path, json.dumps(config)),
    )
    conn.commit()
    job_id = cur.lastrowid

    studio_root: Path = request.app.state.studio_root
    loop = asyncio.get_running_loop()
    _start_training(job_id, config, studio_root, conn, loop)

    return RedirectResponse(url=f"/finetune/{job_id}", status_code=303)


@router.get("/{job_id}", response_class=HTMLResponse)
async def job_view(job_id: int, request: Request):
    conn: sqlite3.Connection = request.app.state.db
    job = _get_job(conn, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    config = json.loads(job["config_json"] or "{}")
    metrics = _job_live.get(job_id, {}).get("metrics", [])
    adapter_dir = request.app.state.studio_root / "data" / "checkpoints" / f"job_{job_id}"
    adapter_exists = (adapter_dir / "adapters.safetensors").exists()

    return templates.TemplateResponse(
        request=request,
        name="finetune_job.html",
        context={
            "job": job,
            "config": config,
            "metrics": metrics,
            "adapter_exists": adapter_exists,
        },
    )


@router.post("/{job_id}/stop")
async def stop_job(job_id: int, request: Request):
    state = _job_live.get(job_id)
    if state and state.get("process"):
        state["process"].terminate()
    conn: sqlite3.Connection = request.app.state.db
    _update_job_status(conn, job_id, "stopped")
    return RedirectResponse(url=f"/finetune/{job_id}", status_code=303)


@router.post("/{job_id}/export")
async def export_adapter(job_id: int, request: Request):
    """Copy final adapter safetensors to a named checkpoint."""
    conn: sqlite3.Connection = request.app.state.db
    job = _get_job(conn, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    studio_root: Path = request.app.state.studio_root
    adapter_src = studio_root / "data" / "checkpoints" / f"job_{job_id}" / "adapters.safetensors"
    if not adapter_src.exists():
        raise HTTPException(status_code=404, detail="adapter not found — training may not be complete")

    # Copy to named export in checkpoints/
    form = await request.form()
    export_name = Path(form.get("name") or f"job_{job_id}_adapter").name
    export_name = export_name.strip().replace(" ", "_")
    export_path = studio_root / "data" / "checkpoints" / f"{export_name}.safetensors"
    shutil.copy(adapter_src, export_path)

    return RedirectResponse(url=f"/finetune/{job_id}", status_code=303)


@router.websocket("/{job_id}/ws")
async def job_ws(job_id: int, websocket: WebSocket):
    await websocket.accept()
    state = _job_live.get(job_id)
    if state:
        state["ws"] = websocket
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if _job_live.get(job_id):
            _job_live[job_id]["ws"] = None
