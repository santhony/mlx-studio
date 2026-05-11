# Qwen Studio - Local AI Development Environment

Freshness: 2026-05-11

## Purpose

Qwen Studio is a local-first web application for AI-powered image generation, chat, code notebooks, agents, and fine-tuning. It runs three Python servers on localhost, using Qwen models via diffusers (image) and MLX (text) on Apple Silicon.

## Architecture

- **Server design:** All servers bind to 127.0.0.1 (local only)
  - `qwen-image-server` (port 8765): FastAPI + diffusers, Qwen-Image-2512 on MPS
  - `qwen-text-server` (port 8766): FastAPI, triple backend: MLX (in-process), Ollama (proxy), or DS4 (proxy)
  - `ds4-server` (port 8767, optional): native C/Metal inference engine from `antirez/ds4`, launched by `start.sh` only when `TEXT_BACKEND=ds4`. Sibling repo at `../ds4/` (not vendored).
  - `web-app` (port 8080): FastAPI + Jinja2 + HTMX, proxies to image/text servers
- **Text backends:** Configured via `TEXT_BACKEND` env var in `data/config.env`
  - `mlx` (default): MLX models (Qwen2.5-Coder variants) loaded in-process on Apple Silicon
  - `ollama`: Proxies to local Ollama server for Gemma 4 (26B MoE, 31B Dense) or other Ollama models
  - `ds4`: Proxies to `ds4-server` at `DS4_HOST` (default `http://127.0.0.1:8767`) running DeepSeek V4 Flash with 1M context, on-disk KV cache, OpenAI-compatible `/v1/chat/completions`. Embeddings still served by the MLX MiniLM model.
- **Data layer:** SQLite at `data/studio.db`, schema managed by `web-app/db.py` (idempotent `init_schema`)
- **Lazy model loading:** Both inference servers load models on first request, not at startup
- **Embedding model:** all-MiniLM-L6-v2 (MLX, 384-dim) for skill similarity search and RAG corpus indexing, loaded lazily by text server
- **RAG (Retrieval-Augmented Generation):** Web-app includes a dedicated RAG tab for corpus management and RAG chat
  - **Corpus management:** Users create corpora, add sources (local directories, single URLs, or URL spiders), and index documents into chunks
  - **URL spider:** `url_spider` source type fetches a seed URL, discovers same-domain links, and indexes all discovered pages as separate chunks (uses BeautifulSoup for HTML parsing)
  - **RAG chat:** Queries are embedded via the text server, documents are retrieved using cosine similarity, and citations are rendered in the response with source file and chunk index

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
    indexer.py              # RAG corpus indexing: walk directories, extract PDFs, spider URLs, create chunks
    routers/                # FastAPI routers: image, chat, notebook, skills, agents, settings, finetune, rag
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

- **Server ports are fixed:** image=8765, text=8766, web=8080, ds4=8767 (when enabled). The web-app hardcodes image and text as `IMAGE_SERVER` and `TEXT_SERVER` constants; ds4-server is internal to qwen-text-server and not addressed directly by the web-app.
- **Health endpoint contract:** All three servers expose `GET /health` returning `{"status": "ready"|"loading"|"offline", ...}`. The web-app polls these via HTMX every 5s.
- **Text server SSE format:** `/chat` and `/complete` stream `data: <token>\n\n` lines, terminated by `data: [DONE]\n\n`. Newlines in tokens are escaped as `\n`. This contract is the same regardless of backend (MLX, Ollama, or DS4) — the text server translates Ollama's NDJSON and DS4's OpenAI-style SSE into the same downstream format. DS4 thinking-mode `reasoning_content` deltas are surfaced as tokens too.
- **Image server response:** `/generate` returns `{"image": "<base64-png>"}`. Dimensions clamped to 64-1024 and snapped to multiples of 64.
- **Schema is idempotent:** `db.init_schema()` uses `CREATE TABLE IF NOT EXISTS` for all tables. Safe to call on every startup.
- **MPS SDPA patch:** The image server monkey-patches `torch.nn.functional.scaled_dot_product_attention` to avoid MPS kernel crashes. This patch must remain for Apple Silicon compatibility.
- **Separate venvs:** Each server has its own virtual environment (venv-image, venv-text, venv-web). Never share venvs between servers.
- **RAG embedding contract:** Corpus chunks are embedded via text server `/embed` endpoint (returns embedding vectors). Retrieval uses cosine similarity in Python (numpy).
- **RAG citation format:** Citations are stored as JSON array in `rag_messages.citations_json`. Each citation has `source_file` (string), `chunk_index` (int), and `excerpt` (string). The system prompt instructs the model to use `[N]` citation markers in responses, which are parsed post-generation.
- **RAG chunk truncation:** System prompt tells the model "For each fact you cite, include [N] where N is the 1-indexed citation number." The web-app stores citations in `citations_json` during post-processing and renders them in `<details>` collapsible sections.

## Database Tables

images, sessions, messages, notebooks, cells, agent_jobs, agent_steps, finetune_jobs, skill_embeddings, settings, corpora, corpus_sources, corpus_chunks, rag_sessions, rag_messages

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
- **qwen-text-server:** mlx, mlx-lm, mlx-embeddings, httpx, fastapi, uvicorn
- **web-app:** fastapi, uvicorn, jinja2, httpx, python-multipart, watchdog, PyMuPDF (for PDF extraction in RAG), beautifulsoup4 (for HTML parsing in URL spider)
