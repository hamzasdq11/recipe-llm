"""API models for the Recipe LLM application."""

from .schemas import (
    PredictRequest,
    PredictResponse,
    RecipeResult,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    ScoreExplanation
)

__all__ = [
    "PredictRequest",
    "PredictResponse", 
    "RecipeResult",
    "ChatRequest",
    "ChatResponse",
    "HealthResponse",
    "ScoreExplanation"
]
