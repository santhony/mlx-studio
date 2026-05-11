# Qwen Studio

A local-first AI development environment for Apple Silicon. Image generation,
chat, code notebooks, sandboxed agents, fine-tuning, and a RAG corpus pipeline
— all served from a small set of FastAPI processes that bind only to 127.0.0.1.

> This project was implemented end-to-end by [Claude Code][cc] (Anthropic's CLI
> coding agent) working from natural-language instructions. The human role has
> been requirements, review, and runtime testing; the code itself — Python,
> HTML, CSS, the inference servers, the RAG indexer, the DS4 integration — was
> written by Claude. The same applies to this README.

[cc]: https://claude.com/claude-code

## What it does

- **Image generation** — Qwen-Image-2512 via `diffusers` on MPS.
- **Chat** — pluggable text backend: MLX (in-process), Ollama (proxy), or
  DS4 (DeepSeek V4 Flash via [antirez/ds4][ds4]). Backend is switchable from
  the Settings page.
- **Code notebooks** — Jupyter-style cells with code execution and
  Prism syntax highlighting.
- **Agents** — sandboxed tool execution against an allowlisted filesystem
  (skills, workspace).
- **Fine-tuning** — MLX LoRA fine-tuning jobs with progress streaming and
  checkpoint management.
- **RAG** — create corpora, add sources (local directories, single URLs, or
  URL spiders), index documents into embedded chunks, chat against them with
  citation rendering.

[ds4]: https://github.com/antirez/ds4

## Architecture

Four loopback-only services. The web-app proxies to the inference servers;
nothing binds to a public interface.

| Process            | Port | Purpose                                                      |
| ------------------ | ---- | ------------------------------------------------------------ |
| `web-app`          | 8080 | FastAPI + Jinja2 + HTMX UI, SQLite at `data/studio.db`       |
| `qwen-image-server`| 8765 | Image generation (`diffusers` + Qwen-Image-2512 on MPS)      |
| `qwen-text-server` | 8766 | Chat / completion / embeddings; MLX, Ollama, or DS4 backend  |
| `ds4-server`       | 8767 | (only when `TEXT_BACKEND=ds4`) Native Metal inference engine |

Each Python server has its own venv. The text server's SSE contract
(`data: <token>\n\n`) is the same across backends; backend-specific protocols
(Ollama NDJSON, DS4 OpenAI SSE) are translated internally.

DS4's reasoning-mode output is wrapped with `<think>...</think>` sentinel
tokens so the chat UI can render the chain of thought as a separate muted
block, distinct from the visible answer.

## Requirements

- Apple Silicon Mac (developed on M4 Max, 128 GB). The MLX and MPS paths
  assume Metal.
- macOS with Xcode Command Line Tools (Metal headers).
- Python 3.11.
- Optional: [Ollama](https://ollama.com) if you want the Ollama backend; a
  built [antirez/ds4](https://github.com/antirez/ds4) checkout at `../ds4/`
  if you want the DS4 backend.

## Setup

First-time setup, once per server:

```bash
cd qwen-studio/qwen-image-server && ./setup.sh
cd qwen-studio/qwen-text-server  && ./setup.sh
cd qwen-studio/web-app           && ./setup.sh
```

Then start everything:

```bash
cd qwen-studio
./start.sh           # spawns the three servers (and ds4-server if configured)
open http://127.0.0.1:8080
```

Stop:

```bash
./stop.sh
```

Backend selection lives in `data/config.env` (`TEXT_BACKEND=mlx|ollama|ds4`)
and is editable from the Settings page in the UI.

### Using DS4 (DeepSeek V4 Flash, optional)

1. Clone and build `antirez/ds4` as a sibling directory:

   ```bash
   cd ..    # one level above qwen-studio
   git clone https://github.com/antirez/ds4.git
   cd ds4 && make           # Metal build, produces ./ds4-server
   ./download_model.sh q2   # ~81 GB GGUF for 128 GB Macs
   ```

2. In Qwen Studio's Settings page, pick the **DS4** option in the model
   dropdown, Save, then Stop and Start the text server. The web-app will
   launch `ds4-server` on port 8767 and wait for it to become ready before
   starting the text server.

## Project layout

```
qwen-studio/
  start.sh, stop.sh           # process management (PID files in data/logs/)
  qwen-image-server/          # FastAPI image server (port 8765)
  qwen-text-server/           # FastAPI text server (port 8766)
  web-app/                    # FastAPI web UI (port 8080)
    main.py                   # entry, lifespan, status endpoints
    db.py                     # SQLite connection + schema init
    skills.py                 # skill embeddings + filesystem watcher
    agent_tools.py            # sandboxed tool execution
    indexer.py                # RAG indexing (dirs, URLs, URL spider)
    routers/                  # chat, image, notebook, agents, settings,
                              #   finetune, rag, skills
    templates/                # Jinja2 HTML (HTMX-driven)
    static/                   # CSS + vendored JS (htmx, prism)
  data/                       # runtime state (mostly gitignored)
    studio.db                 # SQLite database
    skills/                   # markdown skill files
    config.env                # backend selection
    logs/                     # server logs and PID files
```

For a more detailed architectural rundown — contracts between servers, the
SSE format, the schema, dependency lists, and the freshness convention — see
[CLAUDE.md](CLAUDE.md). That document is the source of truth that Claude Code
reads at the start of each session.

## Notes on the "Claude wrote it" disclosure

This is a personal project. The disclosure isn't lawyerly fine print; it's
intended to be useful context for anyone reading the code:

- Style conventions (function size, docstring tone, naming) reflect Claude's
  defaults more than any individual house style.
- Architectural decisions — three-server split, FastAPI everywhere, HTMX over
  a JS framework, MLX for text, `<think>` sentinels for the DS4 reasoning
  stream — were made collaboratively in conversation, then implemented by
  Claude.
- Bugs are real bugs. They were also fixed by Claude. If you find one, an
  issue is welcome.

## License

No license file is committed. Treat as all rights reserved unless one is
added.
