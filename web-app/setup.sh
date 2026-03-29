#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Setting up web-app ==="

if [ -d "venv-web" ]; then
    echo "Removing old venv-web..."
    rm -rf venv-web
fi

echo "Creating venv-web with Python 3.11..."
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 -m venv venv-web
source venv-web/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "=== Verifying installation ==="
python3 -c "
import fastapi, uvicorn, jinja2, httpx, watchdog, frontmatter
print('fastapi:', fastapi.__version__)
print('httpx:', httpx.__version__)
print('jinja2:', jinja2.__version__)
print('watchdog:', watchdog.__version__)
"

echo ""
echo "=== Setup complete ==="
echo "Run with: source venv-web/bin/activate && python3 main.py"
