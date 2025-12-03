#!/bin/bash
# Run Recipe LLM locally on Linux/macOS

set -e

echo "========================================"
echo "   Recipe LLM - Local Setup & Run"
echo "========================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

echo "Python version:"
python3 --version

if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo ""
echo "Activating virtual environment..."
source venv/bin/activate

echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

if [ ! -f "data/recipes_small.json" ]; then
    echo ""
    echo "Error: Recipe data not found at data/recipes_small.json"
    exit 1
fi

echo ""
echo "Recipe data found: $(wc -l < data/recipes_small.json) lines"

export RECIPE_MODE="${RECIPE_MODE:-minimal}"
export RECIPE_USE_MOCK="${RECIPE_USE_MOCK:-true}"
export RECIPE_DATA_DIR="${RECIPE_DATA_DIR:-data}"

echo ""
echo "========================================"
echo "   Starting Recipe LLM Server"
echo "========================================"
echo ""
echo "Mode: $RECIPE_MODE"
echo "Mock Model: $RECIPE_USE_MOCK"
echo ""
echo "API will be available at: http://0.0.0.0:5000"
echo "Health check: http://0.0.0.0:5000/health"
echo "Web UI: http://0.0.0.0:5000/"
echo ""
echo "Press Ctrl+C to stop"
echo "========================================"
echo ""

python3 -m uvicorn app.main:app --host 0.0.0.0 --port 5000 --reload
