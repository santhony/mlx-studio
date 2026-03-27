#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/data/logs"
mkdir -p "$LOG_DIR"

echo "=== Starting Qwen Studio ==="

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
