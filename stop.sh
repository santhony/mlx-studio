#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/data/logs"

echo "=== Stopping MLX Studio ==="

stop_pid() {
    local name="$1"
    local pidfile="$LOG_DIR/${name}.pid"
    if [ -f "$pidfile" ]; then
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping $name (pid $pid)..."
            kill "$pid"
            # Wait up to 5s for process to exit
            for i in 1 2 3 4 5; do
                sleep 1
                kill -0 "$pid" 2>/dev/null || break
            done
            # Force kill if still running
            kill -0 "$pid" 2>/dev/null && kill -9 "$pid"
        else
            echo "$name is not running (stale pid $pid)"
        fi
        rm -f "$pidfile"
    else
        echo "No PID file for $name"
    fi
    # Also kill any orphaned server.py processes for this server
    pkill -f "python3 server.py" 2>/dev/null || true
}

stop_pid "image-server"
stop_pid "text-server"
stop_pid "web-app"
stop_pid "ollama"
stop_pid "ds4-server"

echo "Done."
