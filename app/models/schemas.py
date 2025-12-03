"""Pydantic models for API request/response schemas."""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class ScoreExplanation(BaseModel):
    """Explains how the score was calculated."""
    exact_match_score: float = Field(description="Score from exact ingredient matching")
    fuzzy_match_score: float = Field(description="Score from fuzzy ingredient matching")
    semantic_score: float = Field(default=0.0, description="Score from semantic similarity")
    exact_matches: List[str] = Field(default_factory=list, description="Ingredients that matched exactly")
    fuzzy_matches: List[dict] = Field(default_factory=list, description="Ingredients that matched via fuzzy matching")


class RecipeResult(BaseModel):
    """A single recipe result with matching information."""
    recipe_id: str = Field(description="Unique identifier for the recipe")
    title: str = Field(description="Recipe title")
    score: float = Field(ge=0, le=1, description="Match score between 0 and 1")
    ingredients_matched: List[str] = Field(description="Ingredients that matched")
    missing_ingredients: List[str] = Field(description="Ingredients needed but not provided")
    instructions: str = Field(description="Short cooking instructions")
    tags: List[str] = Field(default_factory=list, description="Recipe tags")
    prep_time_min: int = Field(default=0, description="Preparation time in minutes")
    difficulty: str = Field(default="easy", description="Recipe difficulty level")
    explanation: Optional[ScoreExplanation] = Field(default=None, description="Score explanation")


class PredictRequest(BaseModel):
    """Request body for the /predict endpoint."""
    ingredients: List[str] = Field(
        min_length=1,
        description="List of available ingredients"
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Number of recipes to return")
    mode: Literal["exact", "fuzzy", "hybrid"] = Field(
        default="hybrid",
        description="Matching mode: exact, fuzzy, or hybrid"
    )
    include_explanation: bool = Field(
        default=False,
        description="Include score explanation in response"
    )


class PredictResponse(BaseModel):
    """Response body for the /predict endpoint."""
    query: List[str] = Field(description="Original query ingredients")
    results: List[RecipeResult] = Field(description="Matched recipes")
    mode: str = Field(description="Matching mode used")
    total_recipes_searched: int = Field(description="Total recipes in database")


class ChatMessage(BaseModel):
    """A single chat message."""
    role: Literal["user", "assistant", "system"] = Field(description="Message role")
    content: str = Field(description="Message content")


class ChatRequest(BaseModel):
    """Request body for the /chat endpoint."""
    message: str = Field(min_length=1, description="User message")
    conversation_history: List[ChatMessage] = Field(
        default_factory=list,
        description="Previous messages in the conversation"
    )
    max_tokens: int = Field(default=256, ge=50, le=1024, description="Maximum tokens in response")


class ChatResponse(BaseModel):
    """Response body for the /chat endpoint."""
    response: str = Field(description="Assistant response")
    conversation_history: List[ChatMessage] = Field(description="Updated conversation history")


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""
    status: str = Field(description="Service status")
    model_loaded: bool = Field(description="Whether the model is loaded")
    model_name: str = Field(description="Name of the loaded model")
    mode: str = Field(description="Current operating mode")
    recipes_loaded: int = Field(description="Number of recipes in database")
    version: str = Field(description="API version")
