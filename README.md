# Recipe LLM

A production-quality, local-first recipe recommendation system with LLM integration. Given ingredients you have on hand, this system suggests recipes with intelligent matching and optional AI-powered responses.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Recipe LLM                                │
│         Local Recipe Recommendation System                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌──────────────┐    ┌────────────────────┐     │
│  │  Web UI  │───▶│  FastAPI     │───▶│  Recipe Database   │     │
│  │  (HTML)  │    │  Backend     │    │  (200+ recipes)    │     │
│  └──────────┘    └──────────────┘    └────────────────────┘     │
│                         │                                        │
│                         ▼                                        │
│  ┌──────────┐    ┌──────────────┐    ┌────────────────────┐     │
│  │  CLI     │───▶│  Ranking     │───▶│  LLM Model         │     │
│  │  Tool    │    │  Engine      │    │  (Optional)        │     │
│  └──────────┘    └──────────────┘    └────────────────────┘     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Features

- **Hybrid Matching**: Combines exact, fuzzy, and semantic ingredient matching
- **200+ Recipes**: Curated recipe database with diverse cuisines
- **Two Modes**: Minimal (CPU-only) and Full (GPU with fine-tuning)
- **REST API**: FastAPI endpoints for predictions, chat, and recipe management
- **Web UI**: Simple, responsive interface for recipe search
- **CLI Tool**: Command-line interface for quick queries
- **Fine-tuning**: LoRA and full fine-tuning scripts included

## Hardware Requirements

### Minimal Mode (Recommended for most users)
- **CPU**: Any modern CPU (4+ cores recommended)
- **RAM**: 4-8 GB
- **Disk**: 100 MB (500 MB+ if downloading models)
- **GPU**: Not required

### Full Mode (With LLM)
- **CPU**: 4+ cores
- **RAM**: 8-16 GB
- **Disk**: 5-10 GB (for model weights)
- **GPU**: Optional (NVIDIA GPU with 8GB+ VRAM for fine-tuning)

## Quick Start

### 1. Clone and Install

```bash
# Clone the repository
git clone <repository-url>
cd recipe-llm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Server

```bash
# Start in minimal mode (no GPU required)
python -m uvicorn app.main:app --host 0.0.0.0 --port 5000 --reload
```

### 3. Try It Out

Open your browser to `http://localhost:5000` for the web UI, or use curl:

```bash
# Get recipe suggestions
curl -X POST http://localhost:5000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"ingredients": ["egg", "onion"], "top_k": 3}'

# Check health
curl http://localhost:5000/health
```

## Project Structure

```
recipe-llm/
├── app/                    # FastAPI application
│   ├── main.py            # API endpoints
│   ├── core/              # Core logic
│   │   ├── ranking.py     # Hybrid ranking engine
│   │   ├── data_loader.py # Recipe data management
│   │   └── model_interface.py  # LLM integration
│   └── models/            # Pydantic schemas
├── data/                   # Recipe datasets
│   └── recipes_small.json # 200+ recipes
├── models/                 # Model weights (not committed)
├── scripts/               # Utility scripts
│   ├── run_local.sh       # Linux/macOS startup
│   ├── run_local.bat      # Windows startup
│   ├── download_model.py  # Model downloader
│   ├── prepare_data.py    # Data preparation
│   ├── finetune_lora.py   # LoRA fine-tuning
│   └── finetune_full.py   # Full fine-tuning
├── tests/                  # Unit & integration tests
├── tools/                  # CLI tools
│   └── query_cli.py       # Command-line query tool
├── ui/                     # Web interface
├── examples/               # Example requests
├── requirements.txt        # Python dependencies
├── Makefile               # Common commands
└── README.md              # This file
```

## API Reference

### POST /api/v1/predict

Get recipe recommendations based on ingredients.

**Request:**
```json
{
  "ingredients": ["egg", "onion", "butter"],
  "top_k": 5,
  "mode": "hybrid",
  "include_explanation": false
}
```

**Response:**
```json
{
  "query": ["egg", "onion", "butter"],
  "results": [
    {
      "recipe_id": "r0001",
      "title": "Scrambled Eggs with Onion",
      "score": 0.92,
      "ingredients_matched": ["egg", "onion", "butter"],
      "missing_ingredients": ["salt", "pepper"],
      "instructions": "Beat eggs, sauté onion, add butter, scramble...",
      "tags": ["breakfast", "easy"],
      "prep_time_min": 10,
      "difficulty": "easy"
    }
  ],
  "mode": "hybrid",
  "total_recipes_searched": 200
}
```

### POST /api/v1/chat

Chat with the recipe assistant.

**Request:**
```json
{
  "message": "What can I make with eggs and cheese?",
  "conversation_history": [],
  "max_tokens": 256
}
```

### GET /health

Check service status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "mock-recipe-model",
  "mode": "minimal",
  "recipes_loaded": 200,
  "version": "1.0.0"
}
```

### GET /api/v1/recipes

List recipes with optional filtering.

**Query Parameters:**
- `tag`: Filter by tag (e.g., "breakfast")
- `difficulty`: Filter by difficulty ("easy", "medium", "hard")
- `limit`: Maximum results (default: 20)

### GET /api/v1/ingredients

List all available ingredients in the database.

## Running Modes

### Minimal Mode (Default)

Uses a mock model interface - no LLM download required. Perfect for:
- Testing the API
- Development
- Low-resource machines

```bash
RECIPE_MODE=minimal RECIPE_USE_MOCK=true python -m uvicorn app.main:app --port 5000
```

### Full Mode (With LLM)

Uses a local quantized LLM for chat functionality.

1. Download a model:
```bash
python scripts/download_model.py --model tinyllama-1b  # 669 MB
# or
python scripts/download_model.py --model phi-2         # 1.6 GB
# or
python scripts/download_model.py --model mistral-7b   # 4.4 GB
```

2. Run with the model:
```bash
RECIPE_MODE=minimal RECIPE_USE_MOCK=false \
RECIPE_MODEL_PATH=models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
python -m uvicorn app.main:app --port 5000
```

## Fine-Tuning

### LoRA Fine-Tuning (Recommended)

Lightweight adaptation that works on CPU or GPU:

```bash
# Prepare training data first
python scripts/prepare_data.py

# Run fine-tuning (dry run first)
python scripts/finetune_lora.py --dry-run

# Actual training
python scripts/finetune_lora.py \
  --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --epochs 3
```

### Full Fine-Tuning (GPU Required)

For users with NVIDIA GPU (16GB+ VRAM):

```bash
python scripts/finetune_full.py \
  --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --epochs 3 \
  --gradient-checkpointing
```

## CLI Tool

Query recipes from the command line:

```bash
# Basic query
python tools/query_cli.py --ingredients "egg,onion"

# JSON output
python tools/query_cli.py -i "chicken,garlic,lemon" --json

# List all ingredients
python tools/query_cli.py --list-ingredients

# More options
python tools/query_cli.py --help
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_ranking.py -v

# Run with coverage
pytest tests/ --cov=app
```

## Troubleshooting

### "Module not found" errors

Make sure you've activated the virtual environment:
```bash
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

### "Recipe data not found"

Ensure `data/recipes_small.json` exists. If missing, the application won't start.

### Slow model loading

Quantized models can take 30-60 seconds to load on first startup. Subsequent requests will be fast.

### Out of memory

- Use a smaller model (TinyLlama instead of Mistral-7B)
- Reduce batch size in fine-tuning
- Enable gradient checkpointing
- Use LoRA instead of full fine-tuning

### Port already in use

```bash
# Find process using port 5000
lsof -i :5000  # Linux/macOS
netstat -ano | findstr :5000  # Windows

# Use a different port
python -m uvicorn app.main:app --port 8000
```

## Security Notes

- Rate limiting recommended for production (use FastAPI middleware)
- Never expose API keys in code
- The `/api/v1/chat` endpoint should be rate-limited
- Consider adding authentication for production use

## Model Licensing

- **TinyLlama**: Apache 2.0 - Free for commercial use
- **Phi-2**: MIT License - Free for commercial use
- **Mistral-7B**: Apache 2.0 - Free for commercial use
- **Llama-2**: Meta License - Requires acceptance of terms

Always verify the license of any model you download before commercial use.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/`
5. Submit a pull request

## License

MIT License - See [LICENSE](LICENSE) for details.
