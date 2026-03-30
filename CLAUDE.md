# Qwen Studio - Local AI Development Environment

Freshness: 2026-03-29

## Purpose

Qwen Studio is a local-first web application for AI-powered image generation, chat, code notebooks, agents, and fine-tuning. It runs three Python servers on localhost, using Qwen models via diffusers (image) and MLX (text) on Apple Silicon.

## Architecture

- **Three-server design:** All servers bind to 127.0.0.1 (local only)
  - `qwen-image-server` (port 8765): FastAPI + diffusers, Qwen-Image-2512 on MPS
  - `qwen-text-server` (port 8766): FastAPI + mlx-lm, Qwen2.5-Coder-32B-Instruct
  - `web-app` (port 8080): FastAPI + Jinja2 + HTMX, proxies to image/text servers
- **Data layer:** SQLite at `data/studio.db`, schema managed by `web-app/db.py` (idempotent `init_schema`)
- **Lazy model loading:** Both inference servers load models on first request, not at startup
- **Embedding model:** all-MiniLM-L6-v2 (MLX, 384-dim) for skill similarity search, loaded lazily by text server

## Project Structure

```
qwen-studio/
  start.sh                  # Start all three servers (background, PID files in data/logs/)
  stop.sh                   # Stop all servers via PID files
  qwen-image-server/
    server.py               # FastAPI: /health, /generate (returns base64 PNG)
    setup.sh                # Creates venv-image, installs requirements
    requirements.txt
  qwen-text-server/
    server.py               # FastAPI: /health, /chat (SSE), /complete (SSE), /embed
    setup.sh                # Creates venv-text, installs requirements
    requirements.txt
  web-app/
    main.py                 # FastAPI app entry point, lifespan, status endpoints
    db.py                   # SQLite connection + schema init
    skills.py               # Skill embedding + filesystem watcher
    agent_tools.py          # Sandboxed tool execution for agents
    routers/                # FastAPI routers: image, chat, notebook, skills, agents, settings, finetune
    templates/              # Jinja2 HTML templates (HTMX-driven)
    static/css/             # Stylesheets
    setup.sh                # Creates venv-web, installs requirements
    requirements.txt
  data/
    studio.db               # SQLite database (gitignored)
    images/                 # Generated images (gitignored)
    skills/                 # Markdown skill files (tracked via .gitkeep)
    workspace/              # Agent sandbox directory (gitignored)
    checkpoints/            # Fine-tuning checkpoints (gitignored)
    datasets/               # Fine-tuning datasets (gitignored)
    logs/                   # Server logs and PID files (created by start.sh)
```

## Contracts

- **Server ports are fixed:** image=8765, text=8766, web=8080. The web-app hardcodes these as `IMAGE_SERVER` and `TEXT_SERVER` constants.
- **Health endpoint contract:** All three servers expose `GET /health` returning `{"status": "ready"|"loading"|"offline", ...}`. The web-app polls these via HTMX every 5s.
- **Text server SSE format:** `/chat` and `/complete` stream `data: <token>\n\n` lines, terminated by `data: [DONE]\n\n`. Newlines in tokens are escaped as `\n`.
- **Image server response:** `/generate` returns `{"image": "<base64-png>"}`. Dimensions clamped to 64-1024 and snapped to multiples of 64.
- **Schema is idempotent:** `db.init_schema()` uses `CREATE TABLE IF NOT EXISTS` for all tables. Safe to call on every startup.
- **MPS SDPA patch:** The image server monkey-patches `torch.nn.functional.scaled_dot_product_attention` to avoid MPS kernel crashes. This patch must remain for Apple Silicon compatibility.
- **Separate venvs:** Each server has its own virtual environment (venv-image, venv-text, venv-web). Never share venvs between servers.

## Database Tables

images, sessions, messages, notebooks, cells, agent_jobs, agent_steps, finetune_jobs, skill_embeddings, settings

## Setup and Run

```bash
# First-time setup (run each once)
cd qwen-studio/qwen-image-server && ./setup.sh
cd qwen-studio/qwen-text-server && ./setup.sh
cd qwen-studio/web-app && ./setup.sh

# Start all servers
cd qwen-studio && ./start.sh

# Stop all servers
cd qwen-studio && ./stop.sh

# Logs
tail -f qwen-studio/data/logs/web-app.log
tail -f qwen-studio/data/logs/image-server.log
tail -f qwen-studio/data/logs/text-server.log
```

## Dependencies

- **qwen-image-server:** torch, diffusers, transformers, accelerate, fastapi, uvicorn
- **qwen-text-server:** mlx, mlx-lm, mlx-embeddings, fastapi, uvicorn
- **web-app:** fastapi, uvicorn, jinja2, httpx, python-multipart, watchdog
