"""Unit tests for data loading."""

import pytest
import json
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.data_loader import DataLoader, Recipe


class TestRecipe:
    """Tests for Recipe dataclass."""
    
    def test_recipe_from_dict(self):
        """Test creating Recipe from dictionary."""
        data = {
            "id": "r0001",
            "title": "Test Recipe",
            "ingredients": ["egg", "butter"],
            "steps": ["Step 1", "Step 2"],
            "tags": ["breakfast"],
            "prep_time_min": 10,
            "difficulty": "easy"
        }
        
        recipe = Recipe.from_dict(data)
        
        assert recipe.id == "r0001"
        assert recipe.title == "Test Recipe"
        assert len(recipe.ingredients) == 2
        assert len(recipe.steps) == 2
        assert recipe.prep_time_min == 10
    
    def test_recipe_to_dict(self):
        """Test converting Recipe to dictionary."""
        recipe = Recipe(
            id="r0001",
            title="Test Recipe",
            ingredients=["egg", "butter"],
            steps=["Step 1"],
            tags=["breakfast"],
            prep_time_min=10,
            difficulty="easy"
        )
        
        data = recipe.to_dict()
        
        assert data["id"] == "r0001"
        assert data["title"] == "Test Recipe"
    
    def test_instructions_short(self):
        """Test short instructions property."""
        recipe = Recipe(
            id="r0001",
            title="Test",
            ingredients=["egg"],
            steps=["Step 1.", "Step 2.", "Step 3.", "Step 4.", "Step 5."],
            tags=[],
            prep_time_min=5,
            difficulty="easy"
        )
        
        short = recipe.instructions_short
        
        assert "Step 1." in short
        assert "Step 2." in short
        assert "Step 3." in short
        assert "Step 4." not in short
    
    def test_ingredients_normalized(self):
        """Test normalized ingredients property."""
        recipe = Recipe(
            id="r0001",
            title="Test",
            ingredients=["Egg", "BUTTER", "Salt"],
            steps=[],
            tags=[],
            prep_time_min=5,
            difficulty="easy"
        )
        
        normalized = recipe.ingredients_normalized
        
        assert normalized == ["egg", "butter", "salt"]


class TestDataLoader:
    """Tests for DataLoader class."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory with test recipes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recipes = [
                {
                    "id": "r0001",
                    "title": "Test Recipe 1",
                    "ingredients": ["egg", "butter"],
                    "steps": ["Cook it"],
                    "tags": ["breakfast"],
                    "prep_time_min": 10,
                    "difficulty": "easy"
                },
                {
                    "id": "r0002",
                    "title": "Test Recipe 2",
                    "ingredients": ["chicken", "garlic"],
                    "steps": ["Cook it"],
                    "tags": ["dinner"],
                    "prep_time_min": 30,
                    "difficulty": "medium"
                }
            ]
            
            filepath = Path(tmpdir) / "recipes_small.json"
            with open(filepath, 'w') as f:
                json.dump(recipes, f)
            
            yield tmpdir
    
    def test_load_recipes(self, temp_data_dir):
        """Test loading recipes from file."""
        loader = DataLoader(temp_data_dir)
        recipes = loader.load_recipes()
        
        assert len(recipes) == 2
        assert recipes[0].id == "r0001"
    
    def test_recipe_count(self, temp_data_dir):
        """Test recipe count property."""
        loader = DataLoader(temp_data_dir)
        loader.load_recipes()
        
        assert loader.recipe_count == 2
    
    def test_get_recipe_by_id(self, temp_data_dir):
        """Test getting recipe by ID."""
        loader = DataLoader(temp_data_dir)
        loader.load_recipes()
        
        recipe = loader.get_recipe_by_id("r0001")
        
        assert recipe is not None
        assert recipe.title == "Test Recipe 1"
    
    def test_get_recipe_by_id_not_found(self, temp_data_dir):
        """Test getting non-existent recipe."""
        loader = DataLoader(temp_data_dir)
        loader.load_recipes()
        
        recipe = loader.get_recipe_by_id("r9999")
        
        assert recipe is None
    
    def test_get_all_ingredients(self, temp_data_dir):
        """Test getting all unique ingredients."""
        loader = DataLoader(temp_data_dir)
        loader.load_recipes()
        
        ingredients = loader.get_all_ingredients()
        
        assert "egg" in ingredients
        assert "butter" in ingredients
        assert "chicken" in ingredients
        assert "garlic" in ingredients
    
    def test_get_recipes_by_tag(self, temp_data_dir):
        """Test filtering recipes by tag."""
        loader = DataLoader(temp_data_dir)
        loader.load_recipes()
        
        breakfast_recipes = loader.get_recipes_by_tag("breakfast")
        
        assert len(breakfast_recipes) == 1
        assert breakfast_recipes[0].id == "r0001"
    
    def test_get_recipes_by_difficulty(self, temp_data_dir):
        """Test filtering recipes by difficulty."""
        loader = DataLoader(temp_data_dir)
        loader.load_recipes()
        
        easy_recipes = loader.get_recipes_by_difficulty("easy")
        
        assert len(easy_recipes) == 1
        assert easy_recipes[0].id == "r0001"
    
    def test_file_not_found(self):
        """Test error when file doesn't exist."""
        loader = DataLoader("/nonexistent/path")
        
        with pytest.raises(FileNotFoundError):
            loader.load_recipes()
