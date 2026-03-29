#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Setting up qwen-text-server ==="

if [ -d "venv-text" ]; then
    echo "Removing old venv-text..."
    rm -rf venv-text
fi

# MLX requires Python 3.11+ and Apple Silicon (ARM64)
echo "Creating venv-text with Python 3.11..."
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 -m venv venv-text
source venv-text/bin/activate

echo "Installing mlx-lm, mlx-embeddings, and server dependencies..."
pip install "mlx-lm>=0.20.0" "mlx-embeddings>=0.1.0" "fastapi>=0.110.0" "uvicorn>=0.29.0"

echo ""
echo "=== Verifying installation ==="
python3 -c "
import mlx.core as mx
import mlx_lm
import mlx_embeddings
import fastapi
print('mlx_lm:', mlx_lm.__version__)
print('fastapi:', fastapi.__version__)
print('MLX device:', mx.default_device())
"

echo ""
echo "=== Setup complete ==="
echo "Run with: source venv-text/bin/activate && python3 server.py"
echo "Note: First request downloads mlx-community/Qwen2.5-Coder-32B-Instruct (~65 GB)"
