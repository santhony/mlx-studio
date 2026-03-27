#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/data/logs"

echo "=== Stopping Qwen Studio ==="

stop_pid() {
    local name="$1"
    local pidfile="$LOG_DIR/${name}.pid"
    if [ -f "$pidfile" ]; then
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping $name (pid $pid)..."
            kill "$pid"
        else
            echo "$name is not running (stale pid $pid)"
        fi
        rm -f "$pidfile"
    else
        echo "No PID file for $name"
    fi
}

stop_pid "image-server"
stop_pid "text-server"
stop_pid "web-app"

echo "Done."
