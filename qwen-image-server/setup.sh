#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Setting up qwen-image-server ==="

if [ -d "venv-image" ]; then
    echo "Removing old venv-image..."
    rm -rf venv-image
fi

echo "Creating venv-image with Python 3.11..."
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 -m venv venv-image
source venv-image/bin/activate

echo "Installing numpy<2 first (must precede torch)..."
pip install "numpy<2"

echo "Installing PyTorch 2.6.0..."
pip install torch==2.6.0 torchvision==0.21.0

echo "Installing diffusers (git HEAD)..."
pip install "git+https://github.com/huggingface/diffusers"

echo "Installing remaining dependencies..."
pip install "transformers>=4.51.0" "tokenizers>=0.21.0" accelerate "Pillow>=10.0.0"

echo "Installing FastAPI + uvicorn..."
pip install "fastapi>=0.110.0" "uvicorn>=0.29.0"

echo ""
echo "=== Verifying installation ==="
python3 -c "
import torch
import diffusers
import fastapi
print('torch:', torch.__version__)
print('diffusers:', diffusers.__version__)
print('fastapi:', fastapi.__version__)
print('MPS available:', torch.backends.mps.is_available())
"

echo ""
echo "=== Setup complete ==="
echo "Run with: source venv-image/bin/activate && PYTORCH_ENABLE_MPS_FALLBACK=1 python3 server.py"
