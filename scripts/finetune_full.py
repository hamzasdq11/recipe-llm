#!/usr/bin/env python3
"""
Full Fine-tuning Script for Recipe LLM (GPU Required).

This script performs full weight fine-tuning on a base LLM.
Requires a GPU with sufficient VRAM (16GB+ recommended).

WARNING: This is resource-intensive and may take several hours.
For most use cases, LoRA fine-tuning (finetune_lora.py) is recommended.

Requirements:
- NVIDIA GPU with 16GB+ VRAM
- transformers
- torch
- datasets
- accelerate

Usage:
  python scripts/finetune_full.py --model-name "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "models" / "finetuned"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Full fine-tuning (GPU required)")
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Base model name from Hugging Face"
    )
    
    parser.add_argument(
        "--data-file",
        type=str,
        default="training_data.json",
        help="Training data file name"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Output directory for fine-tuned model"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Training batch size (reduce if OOM)"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Use gradient checkpointing to save memory"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check setup without training"
    )
    
    return parser.parse_args()


def check_gpu():
    """Check GPU availability and memory."""
    try:
        import torch
        
        if not torch.cuda.is_available():
            logger.error("No GPU detected. Full fine-tuning requires a GPU.")
            logger.info("Consider using LoRA fine-tuning instead: python scripts/finetune_lora.py")
            return False
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        logger.info(f"GPU detected: {gpu_name}")
        logger.info(f"GPU memory: {gpu_memory:.1f} GB")
        
        if gpu_memory < 8:
            logger.warning("GPU has less than 8GB VRAM. Training may fail.")
            logger.info("Recommendations:")
            logger.info("  1. Use a smaller model (TinyLlama)")
            logger.info("  2. Reduce batch size")
            logger.info("  3. Enable gradient checkpointing")
            logger.info("  4. Use LoRA instead: python scripts/finetune_lora.py")
        
        return True
        
    except ImportError:
        logger.error("PyTorch not installed")
        return False


def load_training_data(data_file: str) -> list:
    """Load training data from JSON file."""
    filepath = DATA_DIR / data_file
    
    if not filepath.exists():
        logger.error(f"Training data not found: {filepath}")
        logger.info("Run 'python scripts/prepare_data.py' first")
        sys.exit(1)
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} training examples")
    return data


def format_training_example(example: dict) -> str:
    """Format a training example."""
    prompt = f"### Instruction:\n{example['instruction']}\n\n"
    if example.get('input'):
        prompt += f"### Input:\n{example['input']}\n\n"
    prompt += f"### Response:\n{example['output']}"
    return prompt


def main():
    """Main training function."""
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("Recipe LLM Full Fine-tuning")
    logger.info("=" * 60)
    logger.info("WARNING: This is resource-intensive!")
    logger.info("For most cases, LoRA is recommended instead.")
    logger.info("=" * 60)
    
    if not check_gpu():
        sys.exit(1)
    
    training_data = load_training_data(args.data_file)
    
    if args.dry_run:
        logger.info("\n[DRY RUN] Setup check complete.")
        logger.info(f"Would train on {len(training_data)} examples")
        logger.info(f"Model: {args.model_name}")
        logger.info(f"Output: {args.output_dir}")
        return
    
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
            DataCollatorForLanguageModeling
        )
        from datasets import Dataset
    except ImportError as e:
        logger.error(f"Missing package: {e}")
        logger.info("Install: pip install transformers torch datasets accelerate")
        sys.exit(1)
    
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    if args.gradient_checkpointing:
        logger.info("Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable()
    
    logger.info("Preparing dataset...")
    formatted_data = [
        {"text": format_training_example(ex)} for ex in training_data
    ]
    
    dataset = Dataset.from_list(formatted_data)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length"
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"recipe_full_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        fp16=True,
        push_to_hub=False,
        report_to="none",
        gradient_checkpointing=args.gradient_checkpointing,
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    logger.info("Starting full fine-tuning...")
    logger.info("This may take several hours...")
    
    if args.resume:
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()
    
    logger.info("Saving model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    config = {
        "base_model": args.model_name,
        "fine_tuning_type": "full",
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "training_examples": len(training_data),
        "timestamp": timestamp
    }
    
    with open(output_dir / "training_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info("=" * 60)
    logger.info("Full fine-tuning complete!")
    logger.info(f"Model saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
