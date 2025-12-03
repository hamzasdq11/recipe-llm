#!/usr/bin/env python3
"""
Model download script for Recipe LLM.

Downloads quantized GGUF models for local inference.
Supports multiple model options with size/quality tradeoffs.
"""

import os
import sys
import hashlib
import argparse
from pathlib import Path
from tqdm import tqdm

try:
    import requests
except ImportError:
    print("Please install requests: pip install requests")
    sys.exit(1)

MODELS_DIR = Path(__file__).parent.parent / "models"

MODEL_OPTIONS = {
    "tinyllama-1b": {
        "name": "TinyLlama 1.1B Chat (Recommended for minimal mode)",
        "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "size_mb": 669,
        "ram_required_gb": 2,
        "description": "Smallest option, fast inference, good for testing"
    },
    "phi-2": {
        "name": "Phi-2 2.7B (Good balance)",
        "url": "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf",
        "filename": "phi-2.Q4_K_M.gguf",
        "size_mb": 1600,
        "ram_required_gb": 4,
        "description": "Microsoft's Phi-2, excellent reasoning for its size"
    },
    "mistral-7b": {
        "name": "Mistral 7B Instruct (High quality)",
        "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "size_mb": 4370,
        "ram_required_gb": 8,
        "description": "Best quality, requires more RAM"
    },
    "llama2-7b": {
        "name": "Llama 2 7B Chat",
        "url": "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf",
        "filename": "llama-2-7b-chat.Q4_K_M.gguf",
        "size_mb": 4080,
        "ram_required_gb": 8,
        "description": "Meta's Llama 2, well-tested"
    }
}


def download_file(url: str, filepath: Path, expected_size_mb: int = None):
    """Download a file with progress bar."""
    print(f"Downloading to: {filepath}")
    print(f"URL: {url}")
    
    if expected_size_mb:
        print(f"Expected size: ~{expected_size_mb} MB")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    print(f"Download complete: {filepath}")
    return filepath


def verify_checksum(filepath: Path, expected_hash: str = None) -> bool:
    """Verify file checksum."""
    if not expected_hash:
        print("No checksum provided, skipping verification")
        return True
    
    print("Verifying checksum...")
    sha256_hash = hashlib.sha256()
    
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    
    actual_hash = sha256_hash.hexdigest()
    
    if actual_hash == expected_hash:
        print("Checksum verified!")
        return True
    else:
        print(f"Checksum mismatch!")
        print(f"Expected: {expected_hash}")
        print(f"Got: {actual_hash}")
        return False


def list_models():
    """List available models."""
    print("\nAvailable models:")
    print("=" * 70)
    
    for key, model in MODEL_OPTIONS.items():
        print(f"\n  {key}:")
        print(f"    Name: {model['name']}")
        print(f"    Size: ~{model['size_mb']} MB")
        print(f"    RAM Required: ~{model['ram_required_gb']} GB")
        print(f"    {model['description']}")
    
    print("\n" + "=" * 70)
    print("\nRecommendation:")
    print("  - Low RAM (4-8GB): tinyllama-1b or phi-2")
    print("  - Standard (8-16GB): mistral-7b or llama2-7b")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Download models for Recipe LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/download_model.py --list
  python scripts/download_model.py --model tinyllama-1b
  python scripts/download_model.py --model mistral-7b --force
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODEL_OPTIONS.keys()),
        help="Model to download"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if file exists"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(MODELS_DIR),
        help="Output directory for models"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_models()
        return
    
    if not args.model:
        print("Please specify a model with --model or use --list to see options")
        list_models()
        return
    
    model_info = MODEL_OPTIONS[args.model]
    output_dir = Path(args.output_dir)
    filepath = output_dir / model_info["filename"]
    
    print(f"\nModel: {model_info['name']}")
    print(f"Size: ~{model_info['size_mb']} MB")
    print(f"RAM Required: ~{model_info['ram_required_gb']} GB")
    print()
    
    if filepath.exists() and not args.force:
        print(f"Model already exists at: {filepath}")
        print("Use --force to re-download")
        return
    
    confirm = input(f"Download {model_info['size_mb']} MB file? [y/N]: ")
    if confirm.lower() != 'y':
        print("Download cancelled")
        return
    
    try:
        download_file(
            url=model_info["url"],
            filepath=filepath,
            expected_size_mb=model_info["size_mb"]
        )
        
        env_file = Path(__file__).parent.parent / ".env"
        with open(env_file, 'a') as f:
            f.write(f"\nRECIPE_MODEL_PATH={filepath}\n")
            f.write("RECIPE_USE_MOCK=false\n")
        
        print(f"\nModel downloaded successfully!")
        print(f"Path: {filepath}")
        print(f"\nTo use this model, set environment variable:")
        print(f"  RECIPE_MODEL_PATH={filepath}")
        print(f"  RECIPE_USE_MOCK=false")
        
    except Exception as e:
        print(f"\nError downloading model: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
