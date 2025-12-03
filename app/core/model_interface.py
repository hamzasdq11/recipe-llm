"""Model interface for LLM integration."""

import os
import logging
from typing import Optional, List, Dict
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseModelInterface(ABC):
    """Abstract base class for model interfaces."""
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get model name."""
        pass


class MockModelInterface(BaseModelInterface):
    """Mock model interface for testing and minimal mode without model."""
    
    def __init__(self):
        self._loaded = True
        self._name = "mock-recipe-model"
    
    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate a mock response based on the prompt."""
        if "recipe" in prompt.lower() or "ingredient" in prompt.lower():
            return self._generate_recipe_response(prompt)
        return "I'm a recipe assistant. Please ask me about cooking or recipes!"
    
    def _generate_recipe_response(self, prompt: str) -> str:
        """Generate a recipe-focused response."""
        ingredients = []
        if "egg" in prompt.lower():
            ingredients.append("eggs")
        if "onion" in prompt.lower():
            ingredients.append("onion")
        if "tomato" in prompt.lower():
            ingredients.append("tomatoes")
        if "chicken" in prompt.lower():
            ingredients.append("chicken")
        if "garlic" in prompt.lower():
            ingredients.append("garlic")
        
        if not ingredients:
            return "I'd be happy to suggest a recipe! What ingredients do you have available?"
        
        ing_str = ", ".join(ingredients)
        return f"""Based on the ingredients you have ({ing_str}), here's a quick suggestion:

A simple dish would work well with these ingredients. Start by preparing your ingredients - chop, dice, or slice as needed. Heat a pan with a bit of oil, then cook your ingredients in order from longest cooking time to shortest. Season with salt and pepper to taste.

Would you like me to suggest a specific recipe from my database?"""
    
    def is_loaded(self) -> bool:
        return self._loaded
    
    def get_model_name(self) -> str:
        return self._name


class LlamaCppInterface(BaseModelInterface):
    """Interface for llama.cpp models (GGUF format)."""
    
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_threads: int = 4,
        n_gpu_layers: int = 0
    ):
        self._model = None
        self._model_path = model_path
        self._n_ctx = n_ctx
        self._n_threads = n_threads
        self._n_gpu_layers = n_gpu_layers
        self._loaded = False
        self._name = Path(model_path).stem if model_path else "llama-cpp-model"
    
    def load(self) -> bool:
        """Load the model."""
        try:
            from llama_cpp import Llama
            
            if not os.path.exists(self._model_path):
                logger.error(f"Model file not found: {self._model_path}")
                return False
            
            logger.info(f"Loading model from {self._model_path}...")
            self._model = Llama(
                model_path=self._model_path,
                n_ctx=self._n_ctx,
                n_threads=self._n_threads,
                n_gpu_layers=self._n_gpu_layers,
                verbose=False
            )
            self._loaded = True
            logger.info("Model loaded successfully")
            return True
            
        except ImportError:
            logger.error("llama-cpp-python not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate text using the loaded model."""
        if not self._loaded or self._model is None:
            return "Model not loaded"
        
        try:
            response = self._model(
                prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                stop=["User:", "\n\n\n"],
                echo=False
            )
            return response["choices"][0]["text"].strip()
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"Error generating response: {str(e)}"
    
    def is_loaded(self) -> bool:
        return self._loaded
    
    def get_model_name(self) -> str:
        return self._name


class TransformersInterface(BaseModelInterface):
    """Interface for Hugging Face transformers models."""
    
    def __init__(
        self,
        model_name: str,
        quantize: bool = True,
        device_map: str = "auto"
    ):
        self._model = None
        self._tokenizer = None
        self._model_name = model_name
        self._quantize = quantize
        self._device_map = device_map
        self._loaded = False
    
    def load(self) -> bool:
        """Load the model."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            logger.info(f"Loading model: {self._model_name}...")
            
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            
            if self._quantize:
                try:
                    from transformers import BitsAndBytesConfig
                    
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16
                    )
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self._model_name,
                        quantization_config=quantization_config,
                        device_map=self._device_map
                    )
                except ImportError:
                    logger.warning("bitsandbytes not available, loading without quantization")
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self._model_name,
                        device_map=self._device_map,
                        torch_dtype=torch.float16
                    )
            else:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self._model_name,
                    device_map=self._device_map,
                    torch_dtype=torch.float16
                )
            
            self._loaded = True
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate text using the loaded model."""
        if not self._loaded:
            return "Model not loaded"
        
        try:
            inputs = self._tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id
            )
            
            response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response[len(prompt):].strip()
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"Error generating response: {str(e)}"
    
    def is_loaded(self) -> bool:
        return self._loaded
    
    def get_model_name(self) -> str:
        return self._model_name


class ModelManager:
    """Manages model loading and selection."""
    
    def __init__(self, mode: str = "minimal"):
        self.mode = mode
        self._interface: Optional[BaseModelInterface] = None
    
    def initialize(
        self,
        model_path: Optional[str] = None,
        model_name: Optional[str] = None,
        use_mock: bool = False
    ) -> BaseModelInterface:
        """Initialize the appropriate model interface."""
        if use_mock:
            logger.info("Using mock model interface")
            self._interface = MockModelInterface()
            return self._interface
        
        if self.mode == "minimal":
            if model_path and os.path.exists(model_path):
                self._interface = LlamaCppInterface(
                    model_path=model_path,
                    n_ctx=2048,
                    n_threads=4,
                    n_gpu_layers=0
                )
                if self._interface.load():
                    return self._interface
            
            logger.info("Falling back to mock model (no model file found)")
            self._interface = MockModelInterface()
            return self._interface
        
        else:
            if model_name:
                self._interface = TransformersInterface(
                    model_name=model_name,
                    quantize=True
                )
                if self._interface.load():
                    return self._interface
            
            logger.warning("Failed to load transformers model, falling back to mock")
            self._interface = MockModelInterface()
            return self._interface
    
    @property
    def interface(self) -> Optional[BaseModelInterface]:
        return self._interface
    
    def is_ready(self) -> bool:
        return self._interface is not None and self._interface.is_loaded()


_model_manager: Optional[ModelManager] = None


def get_model_manager(mode: str = "minimal") -> ModelManager:
    """Get the global model manager."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager(mode=mode)
    return _model_manager
