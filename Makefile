# Recipe LLM Makefile
# Common operations for the project

.PHONY: setup install run test clean lint download-model finetune prepare-data help

PYTHON := python3
PIP := pip
UVICORN := uvicorn

# Default target
help:
	@echo "Recipe LLM - Available Commands"
	@echo "================================"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make setup          - Create venv and install dependencies"
	@echo "  make install        - Install dependencies only"
	@echo ""
	@echo "Running:"
	@echo "  make run            - Start the API server (minimal mode)"
	@echo "  make run-full       - Start with full model (requires GPU)"
	@echo ""
	@echo "Data & Model:"
	@echo "  make prepare-data   - Prepare and generate training data"
	@echo "  make download-model - Download a quantized model"
	@echo "  make finetune       - Run LoRA fine-tuning"
	@echo "  make finetune-full  - Run full fine-tuning (GPU required)"
	@echo ""
	@echo "Testing & Quality:"
	@echo "  make test           - Run all tests"
	@echo "  make test-unit      - Run unit tests only"
	@echo "  make test-api       - Run API integration tests"
	@echo "  make lint           - Run linting checks"
	@echo ""
	@echo "Utilities:"
	@echo "  make query          - Run CLI query tool"
	@echo "  make clean          - Remove generated files"
	@echo ""

# Setup virtual environment and install dependencies
setup:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv venv
	@echo "Installing dependencies..."
	. venv/bin/activate && $(PIP) install --upgrade pip && $(PIP) install -r requirements.txt
	@echo ""
	@echo "Setup complete! Activate with: source venv/bin/activate"

# Install dependencies only
install:
	$(PIP) install -r requirements.txt

# Run the server in minimal mode
run:
	RECIPE_MODE=minimal RECIPE_USE_MOCK=true $(PYTHON) -m $(UVICORN) app.main:app --host 0.0.0.0 --port 5000 --reload

# Run with full model (requires model download and GPU)
run-full:
	RECIPE_MODE=full RECIPE_USE_MOCK=false $(PYTHON) -m $(UVICORN) app.main:app --host 0.0.0.0 --port 5000 --reload

# Prepare training data
prepare-data:
	$(PYTHON) scripts/prepare_data.py

# Download model (interactive)
download-model:
	$(PYTHON) scripts/download_model.py --list
	@echo ""
	@echo "To download a specific model, run:"
	@echo "  python scripts/download_model.py --model tinyllama-1b"

# LoRA fine-tuning
finetune:
	$(PYTHON) scripts/finetune_lora.py --dry-run
	@echo ""
	@echo "This was a dry run. To actually train, run:"
	@echo "  python scripts/finetune_lora.py --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Full fine-tuning (GPU required)
finetune-full:
	$(PYTHON) scripts/finetune_full.py --dry-run

# Run all tests
test:
	$(PYTHON) -m pytest tests/ -v

# Run unit tests only
test-unit:
	$(PYTHON) -m pytest tests/test_ranking.py tests/test_data_loader.py -v

# Run API tests
test-api:
	$(PYTHON) -m pytest tests/test_api.py -v

# Run linting
lint:
	$(PYTHON) -m flake8 app/ tests/ scripts/ --max-line-length=100 || true
	$(PYTHON) -m mypy app/ --ignore-missing-imports || true

# CLI query tool
query:
	@echo "Usage: python tools/query_cli.py --ingredients \"egg,onion\""
	@echo ""
	$(PYTHON) tools/query_cli.py --help

# Clean generated files
clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache
	rm -rf app/__pycache__ tests/__pycache__ scripts/__pycache__
	rm -rf app/core/__pycache__ app/models/__pycache__
	rm -rf data/training_data.json data/negative_examples.json data/few_shot_examples.json
	rm -rf models/lora_adapters models/finetuned
	find . -name "*.pyc" -delete
	find . -name ".DS_Store" -delete
	@echo "Cleaned generated files"

# Quick health check
health:
	curl -s http://localhost:5000/health | python -m json.tool

# Example prediction
predict:
	curl -X POST http://localhost:5000/api/v1/predict \
		-H "Content-Type: application/json" \
		-d '{"ingredients": ["egg", "onion"], "top_k": 3, "mode": "hybrid"}' \
		| python -m json.tool
