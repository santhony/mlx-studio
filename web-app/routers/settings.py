"""
settings.py — Settings page and allowlist management.

Allowed filesystem directories are stored in the `settings` table with
keys like `allowed_dir_0`, `allowed_dir_1`, etc.

Text model selection is stored in data/config.env (sourced by start.sh).
"""

import asyncio
import os
import signal
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

router = APIRouter(prefix="/settings")
templates = Jinja2Templates(directory="templates")

# (backend, model_id, display_label)
TEXT_MODELS = [
    ("mlx", "mlx-community/Qwen2.5-Coder-32B-Instruct-8bit",  "Qwen2.5-Coder 32B 8-bit  (~34 GB) [MLX]"),
    ("mlx", "mlx-community/Qwen2.5-Coder-32B-Instruct-4bit",  "Qwen2.5-Coder 32B 4-bit  (~20 GB) [MLX]"),
    ("mlx", "mlx-community/Qwen2.5-Coder-32B-Instruct-bf16",  "Qwen2.5-Coder 32B bf16   (~65 GB) [MLX]"),
    ("mlx", "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",   "Qwen2.5-Coder 7B  4-bit  (~4 GB) [MLX]"),
    ("mlx", "mlx-community/Qwen2.5-Coder-7B-Instruct-8bit",   "Qwen2.5-Coder 7B  8-bit  (~8 GB) [MLX]"),
    ("ollama", "gemma4:26b",                                    "Gemma 4 26B MoE  (~18 GB) [Ollama]"),
    ("ollama", "gemma4:31b",                                    "Gemma 4 31B Dense (~20 GB) [Ollama]"),
    ("ds4",    "deepseek-v4-flash",                             "DeepSeek V4 Flash 284B q2 (~81 GB) [DS4]"),
]
DEFAULT_TEXT_MODEL = "mlx-community/Qwen2.5-Coder-32B-Instruct-8bit"
DEFAULT_BACKEND = "mlx"

DS4_HEALTH_URL = "http://127.0.0.1:8767/v1/models"
DS4_STARTUP_TIMEOUT_S = 180


def _get_config_path(request: Request) -> Path:
    return request.app.state.studio_root / "data" / "config.env"


def _read_config(request: Request) -> dict[str, str]:
    """Read all key=value pairs from config.env."""
    config = _get_config_path(request)
    result = {}
    if config.exists():
        for line in config.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                key, val = line.split("=", 1)
                result[key.strip()] = val.strip().strip('"')
    return result


def _read_text_model(request: Request) -> str:
    cfg = _read_config(request)
    backend = cfg.get("TEXT_BACKEND", DEFAULT_BACKEND)
    if backend == "ollama":
        return cfg.get("OLLAMA_MODEL", DEFAULT_TEXT_MODEL)
    if backend == "ds4":
        return cfg.get("DS4_MODEL", "deepseek-v4-flash")
    return cfg.get("QWEN_TEXT_MODEL", DEFAULT_TEXT_MODEL)


def _read_backend(request: Request) -> str:
    return _read_config(request).get("TEXT_BACKEND", DEFAULT_BACKEND)


def _write_model_config(request: Request, backend: str, model_id: str) -> None:
    """Write backend + model selection to config.env."""
    config = _get_config_path(request)
    config.parent.mkdir(parents=True, exist_ok=True)
    lines = config.read_text().splitlines() if config.exists() else []
    keys_to_remove = {"QWEN_TEXT_MODEL", "TEXT_BACKEND", "OLLAMA_MODEL", "DS4_MODEL"}
    lines = [l for l in lines if not any(l.startswith(k + "=") for k in keys_to_remove)]
    lines.append(f'TEXT_BACKEND="{backend}"')
    if backend == "ollama":
        lines.append(f'OLLAMA_MODEL="{model_id}"')
    elif backend == "ds4":
        lines.append(f'DS4_MODEL="{model_id}"')
    else:
        lines.append(f'QWEN_TEXT_MODEL="{model_id}"')
    config.write_text("\n".join(lines) + "\n")


SERVERS = {
    "image": {
        "pid_file": "data/logs/image-server.pid",
        "venv":     "qwen-image-server/venv-image",
        "script":   "qwen-image-server/server.py",
        "log":      "data/logs/image-server.log",
        "env":      {"PYTORCH_ENABLE_MPS_FALLBACK": "1"},
        "port":     8765,
    },
    "text": {
        "pid_file": "data/logs/text-server.pid",
        "venv":     "qwen-text-server/venv-text",
        "script":   "qwen-text-server/server.py",
        "log":      "data/logs/text-server.log",
        "env":      {},
        "port":     8766,
    },
    "ds4": {
        "port":       8767,
        "pid_file":   "data/logs/ds4-server.pid",
        "binary":     "../ds4/ds4-server",
        "cwd":        "../ds4",
        "args":       [
            "--host", "127.0.0.1",
            "--port", "8767",
            "--ctx", "100000",
            "--kv-disk-dir", "data/ds4-kv",
            "--kv-disk-space-mb", "8192",
        ],
        "log":        "data/logs/ds4-server.log",
        "env":        {},
        "health_url": DS4_HEALTH_URL,
    },
}


def _server_pid(studio_root: Path, name: str) -> int | None:
    """Return PID from pid file if the process is actually running, else None."""
    pid_path = studio_root / SERVERS[name]["pid_file"]
    if not pid_path.exists():
        return None
    try:
        pid = int(pid_path.read_text().strip())
        os.kill(pid, 0)  # signal 0 just checks existence
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        return None


def _server_status(studio_root: Path, name: str) -> str:
    return "running" if _server_pid(studio_root, name) is not None else "stopped"


def _stop_server(studio_root: Path, name: str) -> None:
    pid = _server_pid(studio_root, name)
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    pid_path = studio_root / SERVERS[name]["pid_file"]
    pid_path.unlink(missing_ok=True)
    # Also free the port: an out-of-band process (started by start.sh on a prior
    # session, or by hand) won't be in our pid file but will block rebind.
    port = SERVERS[name].get("port")
    if port:
        try:
            out = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            out = ""
        for line in out.splitlines():
            try:
                stray = int(line)
            except ValueError:
                continue
            if stray == os.getpid():
                continue  # never kill the web-app itself
            try:
                os.kill(stray, signal.SIGTERM)
            except ProcessLookupError:
                pass
        # Brief wait, then SIGKILL anything still on the port.
        time.sleep(0.5)
        try:
            out = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            out = ""
        for line in out.splitlines():
            try:
                stray = int(line)
            except ValueError:
                continue
            if stray == os.getpid():
                continue
            try:
                os.kill(stray, signal.SIGKILL)
            except ProcessLookupError:
                pass


def _start_server(studio_root: Path, name: str, extra_env: dict | None = None) -> None:
    cfg = SERVERS[name]
    log_path = studio_root / cfg["log"]
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(cfg["env"])
    if extra_env:
        env.update(extra_env)

    if "binary" in cfg:
        # Native binary (e.g. ds4-server). cwd is relative to studio_root.
        cmd = [str((studio_root / cfg["binary"]).resolve()), *cfg.get("args", [])]
        cwd = str((studio_root / cfg["cwd"]).resolve())
    else:
        # Python server inside its venv.
        python = studio_root / cfg["venv"] / "bin" / "python3"
        script = studio_root / cfg["script"]
        cmd = [str(python), str(script)]
        cwd = str(studio_root / Path(cfg["script"]).parent)

    with open(log_path, "a") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=log_file,
            stderr=log_file,
            env=env,
        )

    pid_path = studio_root / cfg["pid_file"]
    pid_path.write_text(str(proc.pid))


def _wait_for_health(url: str, timeout_s: int) -> bool:
    """Poll a URL until it returns 2xx or timeout. Returns True on success."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if 200 <= resp.status < 300:
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(2)
    return False


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
    current_model = _read_text_model(request)
    current_backend = _read_backend(request)
    studio_root = request.app.state.studio_root
    # Build composite key for the dropdown: "backend:model_id"
    current_selection = f"{current_backend}:{current_model}"
    return templates.TemplateResponse(
        request=request,
        name="settings.html",
        context={
            "allowed_dirs": dirs,
            "text_models": TEXT_MODELS,
            "current_selection": current_selection,
            "current_text_model": current_model,
            "current_backend": current_backend,
            "image_status": _server_status(studio_root, "image"),
            "text_status": _server_status(studio_root, "text"),
            "ds4_status": _server_status(studio_root, "ds4"),
        },
    )


@router.get("/server/{name}/status")
async def server_status(name: str, request: Request):
    if name not in SERVERS:
        return JSONResponse({"error": "unknown server"}, status_code=400)
    status = _server_status(request.app.state.studio_root, name)
    return templates.TemplateResponse(
        request=request,
        name="server_status_badge.html",
        context={"name": name, "status": status},
    )


@router.post("/server/{name}/start")
async def start_server(name: str, request: Request):
    if name not in SERVERS:
        return JSONResponse({"error": "unknown server"}, status_code=400)
    studio_root = request.app.state.studio_root
    extra_env: dict[str, str] = {}

    if name == "text":
        backend = _read_backend(request)
        model = _read_text_model(request)
        extra_env["TEXT_BACKEND"] = backend
        if backend == "ollama":
            extra_env["OLLAMA_MODEL"] = model
        elif backend == "ds4":
            extra_env["DS4_MODEL"] = model
            # DS4 backend needs ds4-server up first. Start it and wait for /v1/models
            # before launching the text-server (else text-server thinks DS4 is offline).
            if _server_status(studio_root, "ds4") != "running":
                _stop_server(studio_root, "ds4")  # clear stale pid
                _start_server(studio_root, "ds4")
            ready = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: _wait_for_health(DS4_HEALTH_URL, DS4_STARTUP_TIMEOUT_S),
            )
            if not ready:
                return JSONResponse(
                    {"error": f"ds4-server did not respond at {DS4_HEALTH_URL} within {DS4_STARTUP_TIMEOUT_S}s. Check data/logs/ds4-server.log."},
                    status_code=503,
                )
        else:
            extra_env["QWEN_TEXT_MODEL"] = model

    _stop_server(studio_root, name)  # clean up any stale pid first
    _start_server(studio_root, name, extra_env)
    return templates.TemplateResponse(
        request=request,
        name="server_status_badge.html",
        context={"name": name, "status": "running"},
    )


@router.post("/server/{name}/stop")
async def stop_server(name: str, request: Request):
    if name not in SERVERS:
        return JSONResponse({"error": "unknown server"}, status_code=400)
    studio_root = request.app.state.studio_root
    _stop_server(studio_root, name)
    # Stopping the text-server while DS4 backend is active orphans the 81GB
    # ds4-server process; tear it down too unless another backend will use it later.
    if name == "text" and _read_backend(request) == "ds4":
        _stop_server(studio_root, "ds4")
    return templates.TemplateResponse(
        request=request,
        name="server_status_badge.html",
        context={"name": name, "status": "stopped"},
    )


@router.post("/webapp/restart")
async def restart_webapp(request: Request):
    """
    Restart the web-app process. Launches a new process then schedules
    a delayed self-kill so the HTTP response can be sent first.
    The browser is redirected to a 'restarting' page that polls until back up.
    """
    studio_root = request.app.state.studio_root
    web_app_dir = studio_root / "web-app"
    python = web_app_dir / "venv-web" / "bin" / "python3"
    log_path = studio_root / "data" / "logs" / "web-app.log"
    pid_path = studio_root / "data" / "logs" / "web-app.pid"

    # Kill current process after a short delay (so the response is sent),
    # then launch the new process. Order matters: the old process must release
    # port 8080 before the new one tries to bind it.
    async def _restart():
        await asyncio.sleep(0.5)
        # Launch new process with a small startup delay so the port is free
        with open(log_path, "a") as log_file:
            proc = subprocess.Popen(
                [str(python), "-c",
                 "import time; time.sleep(1); import runpy; runpy.run_path('main.py', run_name='__main__')"],
                cwd=str(web_app_dir),
                stdout=log_file,
                stderr=log_file,
            )
        pid_path.write_text(str(proc.pid))
        os.kill(os.getpid(), signal.SIGTERM)

    asyncio.create_task(_restart())

    return HTMLResponse("""
<html><head>
<meta http-equiv="refresh" content="3;url=/settings/">
<style>body{font-family:sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;background:#1a1a1a;color:#ccc;}</style>
</head><body><p>Restarting web-app… you'll be redirected in a moment.</p></body></html>
""")


@router.post("/model/text")
async def set_text_model(request: Request):
    form = await request.form()
    selection = (form.get("model_selection") or "").strip()
    # selection is "backend:model_id"
    if ":" not in selection:
        return RedirectResponse(url="/settings/", status_code=303)
    backend, model_id = selection.split(":", 1)
    known = {(b, m) for b, m, _ in TEXT_MODELS}
    if (backend, model_id) in known:
        _write_model_config(request, backend, model_id)
    return RedirectResponse(url="/settings/", status_code=303)


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
