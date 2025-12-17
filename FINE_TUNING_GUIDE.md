# Fine-Tuning Guide for Recipe LLM

This guide explains how to fine-tune the Recipe LLM model on your local machine.

## Prerequisites

- **RAM**: Minimum 16GB (32GB recommended)
- **Storage**: 10GB free space
- **Python**: 3.10+
- **GPU**: Optional but recommended (NVIDIA with 8GB+ VRAM)

## Quick Start

### 1. Install Dependencies
```bash
pip install transformers peft torch datasets accelerate
```

### 2. Run LoRA Fine-Tuning
```bash
# Basic CPU training (requires 16GB+ RAM)
python scripts/finetune_lora.py --model-name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --epochs 3

# With GPU (faster, uses less CPU RAM)
python scripts/finetune_lora.py --model-name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --epochs 3 --quantize

# Custom settings
python scripts/finetune_lora.py \
    --model-name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --epochs 5 \
    --batch-size 4 \
    --learning-rate 1e-4 \
    --lora-r 16 \
    --lora-alpha 32
```

## Training Data

The training data is pre-prepared in `data/training_data.json`:
- **597 training examples** derived from 200 recipes
- Format: instruction/input/output pairs
- Example:
```json
{
  "instruction": "Given these ingredients: egg, onion, suggest a recipe.",
  "input": "",
  "output": "Recipe: Scrambled Eggs with Onion\nIngredients needed: salt, pepper, butter\nInstructions: Beat eggs..."
}
```

## Fine-Tuning Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model-name` | TinyLlama-1.1B | Base model from HuggingFace |
| `--epochs` | 3 | Training epochs |
| `--batch-size` | 4 | Samples per batch |
| `--learning-rate` | 2e-4 | Learning rate |
| `--lora-r` | 16 | LoRA rank (lower = smaller adapter) |
| `--lora-alpha` | 32 | LoRA alpha scaling |
| `--quantize` | False | Use 4-bit quantization (GPU only) |
| `--max-length` | 512 | Max sequence length |

## Output

After training, LoRA adapters are saved to `models/lora_adapters/`:
- `adapter_config.json` - LoRA configuration
- `adapter_model.bin` - Trained weights (~10-50MB)

## Using the Fine-Tuned Model

Update the workflow to use your fine-tuned adapter:
```bash
export RECIPE_LORA_PATH=models/lora_adapters
python -m uvicorn app.main:app --host 0.0.0.0 --port 5000
```

## Memory Tips

If running out of memory:
1. Reduce `--batch-size` to 1 or 2
2. Reduce `--max-length` to 256
3. Use `--quantize` with GPU
4. Close other applications

## Alternative: Full Fine-Tuning

For better results with more resources:
```bash
python scripts/finetune_full.py --model-name "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```
Requires 32GB+ RAM or GPU with 16GB+ VRAM.
