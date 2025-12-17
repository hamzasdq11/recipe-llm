# Recipe LLM - Local-First Recipe Recommendation System

## Overview
A production-quality local-first recipe recommendation system that runs on standard Windows/Linux laptops without requiring cloud services. The system uses a small LLM (quantized Llama-2-7B or Mistral-7B via llama.cpp) with LoRA fine-tuning capability, exposes recipes via FastAPI, and provides both a web UI and chatbot interface.

## Project Goals
- Enable users to input ingredients (e.g., "egg, onion") and receive ranked recipe suggestions
- Support CPU-only minimal mode (8GB RAM) and optional GPU full mode
- Provide hybrid ranking combining exact, fuzzy, and semantic matching
- Run completely locally without cloud dependencies

## Current State
**Status**: Fully functional with 39 passing tests (25 unit + 14 integration)

### Working Features
- FastAPI backend with /api/v1/predict, /health, /api/v1/chat endpoints
- Web UI with ingredient search, recipe display, and chat interface
- Hybrid ranking system (exact + fuzzy matching via rapidfuzz)
- Mock model for CPU-only development (real llama.cpp integration ready)
- 200 recipes with 283 unique ingredients
- CLI tool for command-line queries
- LoRA fine-tuning scripts for model adaptation

## Recent Changes
- 2024-12-17: Added FINE_TUNING_GUIDE.md with local training instructions
- 2024-12-17: Fixed chat prompt formatting and response handling
- 2024-12-17: Implemented background model loading for faster server startup
- 2024-12-17: Switched from llama-cpp-python to transformers library (CPU compatible)
- 2024-12-17: Integrated TinyLlama-1.1B real LLM for chat responses
- 2024-12-03: Fixed static file serving for CSS/JS
- 2024-12-03: Fixed API integration tests with proper lifespan handling
- 2024-12-03: Created 200 diverse recipes with normalized ingredients
- 2024-12-03: Implemented hybrid ranking with configurable weights

## Project Architecture

```
recipe-llm/
├── app/                    # FastAPI application
│   ├── main.py            # Application entry point
│   ├── core/              # Business logic
│   │   ├── data_loader.py # Recipe data management
│   │   ├── ranking.py     # Hybrid ranking system
│   │   └── model_interface.py # LLM integration
│   └── models/            # Pydantic schemas
├── data/                  # Recipe datasets
│   ├── recipes_small.json # 200 curated recipes
│   └── training/          # Fine-tuning data
├── scripts/               # Utility scripts
│   ├── finetune_lora.py   # LoRA fine-tuning
│   ├── finetune_full.py   # Full fine-tuning
│   ├── prepare_data.py    # Dataset preparation
│   └── download_model.py  # Model download helper
├── ui/                    # Web frontend
│   ├── index.html         # Main page
│   ├── style.css          # Styling
│   └── app.js             # Client logic
├── tests/                 # Test suite
│   ├── test_ranking.py    # Ranking tests
│   ├── test_data_loader.py # Data loader tests
│   └── test_api.py        # API integration tests
└── tools/                 # CLI tools
    └── query_cli.py       # Command-line interface
```

## Key Technical Decisions

### Ranking System
- **Exact Match Score**: Weighted at 0.5 (50%)
- **Fuzzy Match Score**: Weighted at 0.3 (30%), threshold 80
- **Semantic Score**: Weighted at 0.2 (20%), placeholder for LLM integration

### Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| RECIPE_DATA_DIR | data | Directory for recipe data |
| RECIPE_MODE | minimal | Mode: minimal (CPU) or full (GPU) |
| RECIPE_USE_MOCK | true | Use mock model for development |
| RECIPE_MODEL_PATH | - | Path to GGUF model file |
| RECIPE_WEIGHT_EXACT | 0.5 | Weight for exact matching |
| RECIPE_WEIGHT_FUZZY | 0.3 | Weight for fuzzy matching |
| RECIPE_WEIGHT_SEMANTIC | 0.2 | Weight for semantic matching |
| RECIPE_FUZZY_THRESHOLD | 80 | Fuzzy match threshold (0-100) |

## User Preferences
- Focus on local-first, privacy-preserving architecture
- Prefer CPU-only operation by default
- Mock model for development, real LLM for production

## How to Run

### Development (Default - Mock Model)
```bash
python -m app.main
```

### With Real LLM (requires GGUF model)
```bash
export RECIPE_MODEL_PATH=/path/to/model.gguf
export RECIPE_USE_MOCK=false
python -m app.main
```

### Run Tests
```bash
pytest tests/ -v
```

## API Endpoints
- `GET /health` - Health check with system status
- `POST /api/v1/predict` - Get recipe recommendations
- `POST /api/v1/chat` - Chat with recipe assistant
- `GET /api/v1/recipes` - List recipes
- `GET /api/v1/ingredients` - List available ingredients
