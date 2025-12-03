#!/usr/bin/env python3
"""
LoRA Fine-tuning Script for Recipe LLM.

This script fine-tunes a base LLM using LoRA (Low-Rank Adaptation)
on the recipe dataset. Designed to run on CPU or GPU.

Requirements:
- transformers
- peft
- torch
- datasets
- bitsandbytes (optional, for quantization)

Usage:
  python scripts/finetune_lora.py --model-name "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  python scripts/finetune_lora.py --model-name "microsoft/phi-2" --epochs 3
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
OUTPUT_DIR = Path(__file__).parent.parent / "models" / "lora_adapters"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune LLM with LoRA")
    
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
        help="Output directory for LoRA adapters"
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
        default=4,
        help="Training batch size"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank"
    )
    
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha"
    )
    
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout"
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Use 4-bit quantization (requires GPU)"
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


def load_training_data(data_file: str) -> list:
    """Load training data from JSON file."""
    filepath = DATA_DIR / data_file
    
    if not filepath.exists():
        logger.error(f"Training data not found: {filepath}")
        logger.info("Run 'python scripts/prepare_data.py' first to generate training data")
        sys.exit(1)
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} training examples")
    return data


def format_training_example(example: dict) -> str:
    """Format a training example as a prompt-response pair."""
    prompt = f"### Instruction:\n{example['instruction']}\n\n"
    if example.get('input'):
        prompt += f"### Input:\n{example['input']}\n\n"
    prompt += f"### Response:\n{example['output']}"
    return prompt


def main():
    """Main training function."""
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("Recipe LLM LoRA Fine-tuning")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"LoRA rank: {args.lora_r}")
    logger.info(f"Quantization: {args.quantize}")
    logger.info("=" * 60)
    
    training_data = load_training_data(args.data_file)
    
    if args.dry_run:
        logger.info("\n[DRY RUN] Setup check complete. Not training.")
        logger.info(f"Would train on {len(training_data)} examples")
        logger.info(f"Output would be saved to: {args.output_dir}")
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
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from datasets import Dataset
    except ImportError as e:
        logger.error(f"Missing required package: {e}")
        logger.info("Install with: pip install transformers peft torch datasets")
        sys.exit(1)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    if device == "cpu" and args.quantize:
        logger.warning("Quantization requires GPU, disabling")
        args.quantize = False
    
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    logger.info("Loading base model...")
    
    if args.quantize:
        try:
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                quantization_config=quantization_config,
                device_map="auto"
            )
            model = prepare_model_for_kbit_training(model)
        except ImportError:
            logger.warning("bitsandbytes not available, loading without quantization")
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
    
    logger.info("Configuring LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
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
    output_dir = Path(args.output_dir) / f"recipe_lora_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        fp16=device == "cuda",
        push_to_hub=False,
        report_to="none",
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
    
    logger.info("Starting training...")
    
    if args.resume:
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()
    
    logger.info("Saving LoRA adapter...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    config = {
        "base_model": args.model_name,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "training_examples": len(training_data),
        "timestamp": timestamp
    }
    
    with open(output_dir / "training_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"LoRA adapter saved to: {output_dir}")
    logger.info("=" * 60)
    
    print(f"\nTo use this adapter, set environment variable:")
    print(f"  RECIPE_LORA_ADAPTER_PATH={output_dir}")


if __name__ == "__main__":
    main()
