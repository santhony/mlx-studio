#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/data/logs"
mkdir -p "$LOG_DIR"

# Load model config if present (written by Settings UI)
CONFIG_FILE="$SCRIPT_DIR/data/config.env"
if [ -f "$CONFIG_FILE" ]; then
    # shellcheck source=/dev/null
    source "$CONFIG_FILE"
fi

echo "=== Starting Qwen Studio ==="

# ── ds4-server (if backend is ds4) ────────────────────────────────────────────
if [ "${TEXT_BACKEND:-mlx}" = "ds4" ]; then
    DS4_DIR="${DS4_DIR:-$SCRIPT_DIR/../ds4}"
    DS4_PORT="${DS4_PORT:-8767}"
    DS4_CTX="${DS4_CTX:-100000}"
    DS4_KV_DIR="${DS4_KV_DIR:-$SCRIPT_DIR/data/ds4-kv}"
    DS4_KV_MB="${DS4_KV_MB:-8192}"
    DS4_URL="http://127.0.0.1:${DS4_PORT}"

    if curl -sf "$DS4_URL/v1/models" > /dev/null 2>&1; then
        echo "ds4-server already running at $DS4_URL"
    elif [ ! -x "$DS4_DIR/ds4-server" ]; then
        echo "ERROR: ds4-server binary not found at $DS4_DIR/ds4-server. Run 'make' in $DS4_DIR."
        exit 1
    elif [ ! -f "$DS4_DIR/ds4flash.gguf" ] && [ -z "${DS4_MODEL_FILE:-}" ]; then
        echo "ERROR: $DS4_DIR/ds4flash.gguf not found. Run $DS4_DIR/download_model.sh q2 first."
        exit 1
    else
        echo "Starting ds4-server on port $DS4_PORT..."
        mkdir -p "$DS4_KV_DIR"
        (
            cd "$DS4_DIR"
            MODEL_ARG=""
            if [ -n "${DS4_MODEL_FILE:-}" ]; then MODEL_ARG="-m $DS4_MODEL_FILE"; fi
            ./ds4-server \
                $MODEL_ARG \
                --host 127.0.0.1 \
                --port "$DS4_PORT" \
                --ctx "$DS4_CTX" \
                --kv-disk-dir "$DS4_KV_DIR" \
                --kv-disk-space-mb "$DS4_KV_MB" \
                > "$LOG_DIR/ds4-server.log" 2>&1 &
            echo $! > "$LOG_DIR/ds4-server.pid"
        )
        # Wait for ds4-server to respond. Initial model load can take a while.
        for i in $(seq 1 60); do
            if curl -sf "$DS4_URL/v1/models" > /dev/null 2>&1; then
                echo "  ✓ ds4-server is responding"
                break
            fi
            sleep 2
        done
    fi
fi

# ── Ollama (if backend is ollama) ─────────────────────────────────────────────
if [ "${TEXT_BACKEND:-mlx}" = "ollama" ]; then
    OLLAMA_URL="${OLLAMA_HOST:-http://localhost:11434}"
    if curl -sf "$OLLAMA_URL/" > /dev/null 2>&1; then
        echo "Ollama already running at $OLLAMA_URL"
    else
        echo "Starting Ollama..."
        ollama serve > "$LOG_DIR/ollama.log" 2>&1 &
        echo $! > "$LOG_DIR/ollama.pid"
        # Wait for Ollama to be ready
        for i in $(seq 1 15); do
            if curl -sf "$OLLAMA_URL/" > /dev/null 2>&1; then
                echo "  ✓ Ollama is responding"
                break
            fi
            sleep 1
        done
    fi
fi

# ── qwen-image-server ──────────────────────────────────────────────────────────
if [ ! -d "qwen-image-server/venv-image" ]; then
    echo "ERROR: qwen-image-server venv not found. Run qwen-image-server/setup.sh first."
    exit 1
fi

echo "Starting qwen-image-server on port 8765..."
(
    cd qwen-image-server
    source venv-image/bin/activate
    PYTORCH_ENABLE_MPS_FALLBACK=1 python3 server.py \
        > "$LOG_DIR/image-server.log" 2>&1 &
    echo $! > "$LOG_DIR/image-server.pid"
    deactivate
)

# ── qwen-text-server ───────────────────────────────────────────────────────────
if [ ! -d "qwen-text-server/venv-text" ]; then
    # venv-text not required for stub — use system python3 or skip
    echo "WARNING: qwen-text-server venv not found. Starting stub with system python3..."
    (
        cd qwen-text-server
        python3 server.py > "$LOG_DIR/text-server.log" 2>&1 &
        echo $! > "$LOG_DIR/text-server.pid"
    )
else
    echo "Starting qwen-text-server on port 8766..."
    (
        cd qwen-text-server
        source venv-text/bin/activate
        python3 server.py > "$LOG_DIR/text-server.log" 2>&1 &
        echo $! > "$LOG_DIR/text-server.pid"
        deactivate
    )
fi

# ── web-app ────────────────────────────────────────────────────────────────────
if [ ! -d "web-app/venv-web" ]; then
    echo "ERROR: web-app venv not found. Run web-app/setup.sh first."
    exit 1
fi

echo "Starting web-app on port 8080..."
(
    cd web-app
    source venv-web/bin/activate
    python3 main.py > "$LOG_DIR/web-app.log" 2>&1 &
    echo $! > "$LOG_DIR/web-app.pid"
    deactivate
)

# ── Wait for health checks ─────────────────────────────────────────────────────
echo "Waiting for servers to start..."
sleep 3

check_health() {
    local name="$1"
    local url="$2"
    local max_retries=10
    local i=0
    while [ $i -lt $max_retries ]; do
        if curl -sf "$url" > /dev/null 2>&1; then
            echo "  ✓ $name is responding"
            return 0
        fi
        sleep 1
        i=$((i + 1))
    done
    echo "  ✗ $name did not respond within ${max_retries}s (check $LOG_DIR/${name}.log)"
    return 1
}

check_health "web-app" "http://127.0.0.1:8080/"
check_health "qwen-image-server" "http://127.0.0.1:8765/health"
check_health "qwen-text-server" "http://127.0.0.1:8766/health"

echo ""
echo "=== Qwen Studio running at http://127.0.0.1:8080 ==="
echo "Logs in $LOG_DIR/"
