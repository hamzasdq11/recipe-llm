"""FastAPI application for Recipe LLM."""

import os
import logging
import asyncio
from typing import List
from contextlib import asynccontextmanager
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from cachetools import TTLCache

from app.models.schemas import (
    PredictRequest,
    PredictResponse,
    RecipeResult,
    ChatRequest,
    ChatResponse,
    ChatMessage,
    HealthResponse,
    ScoreExplanation
)
from app.core.data_loader import DataLoader, get_data_loader
from app.core.ranking import RecipeRanker
from app.core.model_interface import get_model_manager, ModelManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

prediction_cache: TTLCache = TTLCache(maxsize=100, ttl=3600)

data_loader: DataLoader = None
ranker: RecipeRanker = None
model_manager: ModelManager = None
model_loading: bool = False
executor = ThreadPoolExecutor(max_workers=1)


def _load_model_sync(mode: str, model_path: str, model_name: str, use_mock: bool):
    """Load model synchronously (runs in thread pool)."""
    global model_manager, model_loading
    model_loading = True
    logger.info("Background model loading started...")
    try:
        model_manager = get_model_manager(mode)
        model_manager.initialize(model_path=model_path, model_name=model_name, use_mock=use_mock)
        logger.info(f"Model loaded successfully: {model_manager.interface.get_model_name() if model_manager.interface else 'unknown'}")
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
    finally:
        model_loading = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global data_loader, ranker, model_manager
    
    logger.info("Starting Recipe LLM application...")
    
    data_dir = os.environ.get("RECIPE_DATA_DIR", "data")
    data_loader = DataLoader(data_dir)
    try:
        data_loader.load_recipes()
        logger.info(f"Loaded {data_loader.recipe_count} recipes")
    except FileNotFoundError as e:
        logger.error(f"Failed to load recipes: {e}")
    
    weight_exact = float(os.environ.get("RECIPE_WEIGHT_EXACT", "0.5"))
    weight_fuzzy = float(os.environ.get("RECIPE_WEIGHT_FUZZY", "0.3"))
    weight_semantic = float(os.environ.get("RECIPE_WEIGHT_SEMANTIC", "0.2"))
    fuzzy_threshold = int(os.environ.get("RECIPE_FUZZY_THRESHOLD", "80"))
    
    ranker = RecipeRanker(
        weight_exact=weight_exact,
        weight_fuzzy=weight_fuzzy,
        weight_semantic=weight_semantic,
        fuzzy_threshold=fuzzy_threshold
    )
    logger.info("Ranking system initialized")
    
    mode = os.environ.get("RECIPE_MODE", "minimal")
    model_path = os.environ.get("RECIPE_MODEL_PATH")
    model_name = os.environ.get("RECIPE_MODEL_NAME")
    use_mock = os.environ.get("RECIPE_USE_MOCK", "true").lower() == "true"
    
    if use_mock:
        model_manager = get_model_manager(mode)
        model_manager.initialize(model_path=model_path, model_name=model_name, use_mock=use_mock)
        logger.info("Mock model initialized (instant)")
    else:
        loop = asyncio.get_event_loop()
        loop.run_in_executor(executor, _load_model_sync, mode, model_path, model_name, use_mock)
        logger.info("Model loading started in background...")
    
    yield
    
    logger.info("Shutting down Recipe LLM application...")
    executor.shutdown(wait=False)


app = FastAPI(
    title="Recipe LLM API",
    description="A local-first recipe recommendation system with LLM integration",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check service health and status."""
    model_loaded = model_manager.is_ready() if model_manager else False
    model_name = ""
    if model_manager and model_manager.interface:
        model_name = model_manager.interface.get_model_name()
    elif model_loading:
        model_name = "loading..."
    
    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        model_name=model_name,
        mode=os.environ.get("RECIPE_MODE", "minimal"),
        recipes_loaded=data_loader.recipe_count if data_loader else 0,
        version="1.0.0"
    )


@app.post("/api/v1/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict_recipes(request: PredictRequest):
    """
    Get recipe recommendations based on available ingredients.
    
    - **ingredients**: List of available ingredients
    - **top_k**: Number of recipes to return (1-20)
    - **mode**: Matching mode (exact, fuzzy, or hybrid)
    - **include_explanation**: Include score breakdown
    """
    if not data_loader or not data_loader.recipes:
        raise HTTPException(status_code=503, detail="Recipe data not loaded")
    
    cache_key = (
        tuple(sorted(request.ingredients)),
        request.top_k,
        request.mode
    )
    
    if cache_key in prediction_cache:
        logger.info("Returning cached prediction")
        return prediction_cache[cache_key]
    
    results = ranker.rank_recipes(
        query_ingredients=request.ingredients,
        recipes=data_loader.recipes,
        mode=request.mode,
        top_k=request.top_k
    )
    
    recipe_results = []
    for match in results:
        explanation = None
        if request.include_explanation:
            explanation = ScoreExplanation(
                exact_match_score=match.exact_score,
                fuzzy_match_score=match.fuzzy_score,
                semantic_score=match.semantic_score,
                exact_matches=match.exact_matches,
                fuzzy_matches=match.fuzzy_matches
            )
        
        recipe_results.append(RecipeResult(
            recipe_id=match.recipe.id,
            title=match.recipe.title,
            score=round(match.score, 3),
            ingredients_matched=match.exact_matches + [m["matched"] for m in match.fuzzy_matches],
            missing_ingredients=match.missing_ingredients,
            instructions=match.recipe.instructions_short,
            tags=match.recipe.tags,
            prep_time_min=match.recipe.prep_time_min,
            difficulty=match.recipe.difficulty,
            explanation=explanation
        ))
    
    response = PredictResponse(
        query=request.ingredients,
        results=recipe_results,
        mode=request.mode,
        total_recipes_searched=data_loader.recipe_count
    )
    
    prediction_cache[cache_key] = response
    
    return response


@app.post("/api/v1/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Chat with the recipe assistant.
    
    - **message**: User message
    - **conversation_history**: Previous messages for context
    - **max_tokens**: Maximum response length
    """
    if not model_manager or not model_manager.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    message_lower = request.message.lower()
    response_text = ""
    
    all_ingredients = data_loader.get_all_ingredients() if data_loader else []
    found_ingredients = [ing for ing in all_ingredients if ing.lower() in message_lower]
    
    if found_ingredients and ranker and data_loader:
        results = ranker.rank_recipes(
            query_ingredients=found_ingredients,
            recipes=data_loader.recipes,
            top_k=3,
            mode="hybrid"
        )
        
        if results:
            recipe_suggestions = []
            for i, match in enumerate(results, 1):
                matched_str = ", ".join(match.exact_matches[:3])
                missing_str = ", ".join(match.missing_ingredients[:3])
                recipe_suggestions.append(
                    f"{i}. **{match.recipe.title}** (Score: {match.score:.0%})\n"
                    f"   - Matched: {matched_str}\n"
                    f"   - You'll need: {missing_str if missing_str else 'Nothing else!'}\n"
                    f"   - Prep time: {match.recipe.prep_time_min} min | Difficulty: {match.recipe.difficulty}"
                )
            
            response_text = f"Based on your ingredients ({', '.join(found_ingredients)}), here are my top recipe suggestions:\n\n"
            response_text += "\n\n".join(recipe_suggestions)
            response_text += "\n\nWould you like the full recipe for any of these? Just ask!"
        else:
            response_text = f"I found the ingredients ({', '.join(found_ingredients)}), but couldn't find matching recipes. Try adding more ingredients!"
    
    elif "hello" in message_lower or "hi" in message_lower:
        response_text = "Hello! I'm your recipe assistant. Tell me what ingredients you have, and I'll suggest delicious recipes you can make!"
    
    elif "help" in message_lower:
        response_text = ("I can help you find recipes! Just tell me what ingredients you have. "
                        "For example: 'I have eggs, onion, and cheese' or 'What can I make with chicken and rice?'")
    
    elif any(word in message_lower for word in ["thank", "thanks"]):
        response_text = "You're welcome! Happy cooking! Let me know if you need more recipe ideas."
    
    else:
        response_text = model_manager.interface.generate(
            prompt=f"User asks about cooking: {request.message}\nAssistant:",
            max_tokens=request.max_tokens
        )
    
    updated_history = list(request.conversation_history)
    updated_history.append(ChatMessage(role="user", content=request.message))
    updated_history.append(ChatMessage(role="assistant", content=response_text))
    
    if len(updated_history) > 10:
        updated_history = updated_history[-10:]
    
    return ChatResponse(
        response=response_text,
        conversation_history=updated_history
    )


@app.get("/api/v1/recipes", tags=["Recipes"])
async def list_recipes(
    tag: str = Query(None, description="Filter by tag"),
    difficulty: str = Query(None, description="Filter by difficulty"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results")
):
    """List available recipes with optional filtering."""
    if not data_loader:
        raise HTTPException(status_code=503, detail="Recipe data not loaded")
    
    recipes = data_loader.recipes
    
    if tag:
        recipes = [r for r in recipes if tag.lower() in [t.lower() for t in r.tags]]
    
    if difficulty:
        recipes = [r for r in recipes if r.difficulty.lower() == difficulty.lower()]
    
    return {
        "total": len(recipes),
        "recipes": [r.to_dict() for r in recipes[:limit]]
    }


@app.get("/api/v1/recipes/{recipe_id}", tags=["Recipes"])
async def get_recipe(recipe_id: str):
    """Get a specific recipe by ID."""
    if not data_loader:
        raise HTTPException(status_code=503, detail="Recipe data not loaded")
    
    recipe = data_loader.get_recipe_by_id(recipe_id)
    if not recipe:
        raise HTTPException(status_code=404, detail="Recipe not found")
    
    return recipe.to_dict()


@app.get("/api/v1/ingredients", tags=["Ingredients"])
async def list_ingredients():
    """List all available ingredients in the database."""
    if not data_loader:
        raise HTTPException(status_code=503, detail="Recipe data not loaded")
    
    ingredients = sorted(data_loader.get_all_ingredients())
    return {
        "total": len(ingredients),
        "ingredients": ingredients
    }


if os.path.exists("ui"):
    app.mount("/static", StaticFiles(directory="ui"), name="static")
    
    @app.get("/", tags=["UI"])
    async def serve_ui():
        """Serve the web UI."""
        return FileResponse("ui/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=5000,
        reload=True
    )
