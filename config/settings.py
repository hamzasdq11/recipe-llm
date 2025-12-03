"""
Configuration settings for the Recipe LLM application.
Supports both minimal (CPU-only) and full (GPU) modes.
"""

import os
from pydantic_settings import BaseSettings
from typing import Optional, Literal
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application settings
    app_name: str = "Recipe LLM"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 5000
    
    # Mode: "minimal" (CPU-only) or "full" (GPU)
    mode: Literal["minimal", "full"] = "minimal"
    
    # Model settings
    model_path: Optional[str] = None
    model_name: str = "TheBloke/Llama-2-7B-Chat-GGUF"
    model_file: str = "llama-2-7b-chat.Q4_K_M.gguf"
    
    # For transformers-based inference (full mode)
    hf_model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    
    # LoRA adapter path (if fine-tuned)
    lora_adapter_path: Optional[str] = None
    
    # Inference settings
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    context_length: int = 2048
    
    # Ranking weights (for hybrid matching)
    weight_exact_match: float = 0.5
    weight_fuzzy_match: float = 0.3
    weight_semantic: float = 0.2
    
    # Fuzzy matching threshold (0-100)
    fuzzy_threshold: int = 80
    
    # Cache settings
    cache_maxsize: int = 100
    cache_ttl: int = 3600
    
    # Data paths
    data_dir: str = "data"
    recipes_file: str = "recipes_small.json"
    
    # Hardware constraints
    n_threads: int = 4
    n_gpu_layers: int = 0  # 0 for CPU-only
    batch_size: int = 8
    
    class Config:
        env_file = ".env"
        env_prefix = "RECIPE_"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Default configuration for ranking
RANKING_CONFIG = {
    "weights": {
        "exact_match": 0.5,
        "fuzzy_match": 0.3,
        "semantic": 0.2
    },
    "fuzzy_threshold": 80,
    "top_k_default": 5,
    "min_score_threshold": 0.1
}

# Prompt templates for recipe generation
PROMPT_TEMPLATES = {
    "recipe_suggestion": """You are a helpful cooking assistant. Given a list of ingredients, suggest a recipe that uses those ingredients.

Available ingredients: {ingredients}

Please provide a recipe with:
1. Recipe name
2. Required ingredients (mark any additional ingredients needed)
3. Step-by-step instructions
4. Cooking time and difficulty

Recipe:""",

    "recipe_from_match": """Based on the following recipe information, provide a friendly response:

Recipe: {title}
Ingredients: {ingredients}
Steps: {steps}
Prep time: {prep_time} minutes
Difficulty: {difficulty}

User has these ingredients: {user_ingredients}

Provide a brief, helpful response about this recipe:""",

    "chat_system": """You are a friendly cooking assistant specializing in recipe suggestions. 
You help users find recipes based on ingredients they have available.
Be concise, helpful, and suggest practical cooking tips when appropriate."""
}

# Model configuration for different modes
MODEL_CONFIGS = {
    "minimal": {
        "use_llama_cpp": True,
        "quantization": "Q4_K_M",
        "n_gpu_layers": 0,
        "context_length": 2048,
        "description": "CPU-only mode using quantized GGUF model"
    },
    "full": {
        "use_llama_cpp": False,
        "quantization": "4bit",  # bitsandbytes 4-bit
        "n_gpu_layers": -1,  # All layers on GPU
        "context_length": 4096,
        "description": "GPU mode with optional fine-tuning support"
    }
}
