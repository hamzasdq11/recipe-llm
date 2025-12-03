"""Data loading and recipe management."""

import json
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field
from functools import lru_cache


@dataclass
class Recipe:
    """Recipe data class."""
    id: str
    title: str
    ingredients: List[str]
    steps: List[str]
    tags: List[str] = field(default_factory=list)
    prep_time_min: int = 0
    difficulty: str = "easy"
    
    @property
    def instructions_short(self) -> str:
        """Return a short version of the instructions."""
        if not self.steps:
            return ""
        short_steps = self.steps[:3] if len(self.steps) > 3 else self.steps
        return " ".join(short_steps)
    
    @property
    def ingredients_normalized(self) -> List[str]:
        """Return normalized (lowercase) ingredients."""
        return [ing.lower().strip() for ing in self.ingredients]
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "ingredients": self.ingredients,
            "steps": self.steps,
            "tags": self.tags,
            "prep_time_min": self.prep_time_min,
            "difficulty": self.difficulty
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Recipe":
        """Create Recipe from dictionary."""
        return cls(
            id=data.get("id", ""),
            title=data.get("title", ""),
            ingredients=data.get("ingredients", []),
            steps=data.get("steps", []),
            tags=data.get("tags", []),
            prep_time_min=data.get("prep_time_min", 0),
            difficulty=data.get("difficulty", "easy")
        )


class DataLoader:
    """Loads and manages recipe data."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self._recipes: List[Recipe] = []
        self._loaded = False
    
    def load_recipes(self, filename: str = "recipes_small.json") -> List[Recipe]:
        """Load recipes from JSON file."""
        file_path = self.data_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Recipe file not found: {file_path}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self._recipes = [Recipe.from_dict(item) for item in data]
        self._loaded = True
        return self._recipes
    
    @property
    def recipes(self) -> List[Recipe]:
        """Get loaded recipes."""
        if not self._loaded:
            self.load_recipes()
        return self._recipes
    
    @property
    def recipe_count(self) -> int:
        """Get number of loaded recipes."""
        return len(self._recipes)
    
    def get_recipe_by_id(self, recipe_id: str) -> Optional[Recipe]:
        """Get a recipe by its ID."""
        for recipe in self.recipes:
            if recipe.id == recipe_id:
                return recipe
        return None
    
    def get_all_ingredients(self) -> set:
        """Get all unique ingredients across all recipes."""
        ingredients = set()
        for recipe in self.recipes:
            ingredients.update(recipe.ingredients_normalized)
        return ingredients
    
    def get_recipes_by_tag(self, tag: str) -> List[Recipe]:
        """Get recipes with a specific tag."""
        tag_lower = tag.lower()
        return [r for r in self.recipes if tag_lower in [t.lower() for t in r.tags]]
    
    def get_recipes_by_difficulty(self, difficulty: str) -> List[Recipe]:
        """Get recipes with a specific difficulty."""
        diff_lower = difficulty.lower()
        return [r for r in self.recipes if r.difficulty.lower() == diff_lower]
    
    def search_by_title(self, query: str) -> List[Recipe]:
        """Search recipes by title."""
        query_lower = query.lower()
        return [r for r in self.recipes if query_lower in r.title.lower()]


@lru_cache(maxsize=1)
def get_data_loader(data_dir: str = "data") -> DataLoader:
    """Get cached data loader instance."""
    loader = DataLoader(data_dir)
    loader.load_recipes()
    return loader
