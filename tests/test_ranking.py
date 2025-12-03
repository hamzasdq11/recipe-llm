"""Unit tests for the ranking system."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.data_loader import Recipe
from app.core.ranking import RecipeRanker, MatchResult


class TestRecipeRanker:
    """Tests for RecipeRanker class."""
    
    @pytest.fixture
    def ranker(self):
        """Create a ranker instance."""
        return RecipeRanker(
            weight_exact=0.5,
            weight_fuzzy=0.3,
            weight_semantic=0.2,
            fuzzy_threshold=80
        )
    
    @pytest.fixture
    def sample_recipes(self):
        """Create sample recipes for testing."""
        return [
            Recipe(
                id="r0001",
                title="Scrambled Eggs with Onion",
                ingredients=["egg", "onion", "salt", "pepper", "butter"],
                steps=["Beat eggs.", "Cook onion.", "Scramble eggs."],
                tags=["breakfast", "easy"],
                prep_time_min=10,
                difficulty="easy"
            ),
            Recipe(
                id="r0002",
                title="Classic Omelette",
                ingredients=["egg", "butter", "salt", "pepper", "cheese"],
                steps=["Beat eggs.", "Cook in butter.", "Add cheese."],
                tags=["breakfast", "easy"],
                prep_time_min=8,
                difficulty="easy"
            ),
            Recipe(
                id="r0003",
                title="Chicken Stir Fry",
                ingredients=["chicken", "bell pepper", "onion", "garlic", "soy sauce"],
                steps=["Cut chicken.", "Stir fry.", "Add sauce."],
                tags=["dinner", "asian"],
                prep_time_min=25,
                difficulty="medium"
            ),
            Recipe(
                id="r0004",
                title="Tomato Soup",
                ingredients=["tomato", "onion", "garlic", "basil", "vegetable broth"],
                steps=["Sauté onion.", "Add tomatoes.", "Blend."],
                tags=["soup", "vegetarian"],
                prep_time_min=30,
                difficulty="easy"
            )
        ]
    
    def test_exact_match_perfect(self, ranker, sample_recipes):
        """Test exact matching with all ingredients."""
        query = ["egg", "onion", "salt", "pepper", "butter"]
        matches, score = ranker.exact_match(query, sample_recipes[0])
        
        assert len(matches) == 5
        assert score == 1.0
    
    def test_exact_match_partial(self, ranker, sample_recipes):
        """Test exact matching with partial ingredients."""
        query = ["egg", "onion"]
        matches, score = ranker.exact_match(query, sample_recipes[0])
        
        assert len(matches) == 2
        assert 0 < score < 1.0
    
    def test_exact_match_none(self, ranker, sample_recipes):
        """Test exact matching with no matching ingredients."""
        query = ["pasta", "marinara"]
        matches, score = ranker.exact_match(query, sample_recipes[0])
        
        assert len(matches) == 0
        assert score == 0.0
    
    def test_fuzzy_match_singular_plural(self, ranker, sample_recipes):
        """Test fuzzy matching handles singular/plural."""
        query = ["eggs", "onions"]
        
        results = ranker.rank_recipes(
            query_ingredients=query,
            recipes=sample_recipes,
            mode="hybrid",
            top_k=5
        )
        
        assert len(results) > 0
        assert results[0].recipe.title == "Scrambled Eggs with Onion"
    
    def test_rank_recipes_returns_sorted(self, ranker, sample_recipes):
        """Test that results are sorted by score."""
        query = ["egg", "butter"]
        
        results = ranker.rank_recipes(
            query_ingredients=query,
            recipes=sample_recipes,
            mode="hybrid",
            top_k=5
        )
        
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score
    
    def test_rank_recipes_respects_top_k(self, ranker, sample_recipes):
        """Test that top_k limits results."""
        query = ["egg"]
        
        results = ranker.rank_recipes(
            query_ingredients=query,
            recipes=sample_recipes,
            mode="hybrid",
            top_k=2
        )
        
        assert len(results) <= 2
    
    def test_rank_recipes_exact_mode(self, ranker, sample_recipes):
        """Test exact matching mode."""
        query = ["egg", "onion"]
        
        results = ranker.rank_recipes(
            query_ingredients=query,
            recipes=sample_recipes,
            mode="exact",
            top_k=5
        )
        
        assert len(results) > 0
        assert results[0].fuzzy_matches == []
    
    def test_rank_recipes_hybrid_mode(self, ranker, sample_recipes):
        """Test hybrid matching mode."""
        query = ["egg", "onion"]
        
        results = ranker.rank_recipes(
            query_ingredients=query,
            recipes=sample_recipes,
            mode="hybrid",
            top_k=5
        )
        
        assert len(results) > 0
    
    def test_missing_ingredients_calculated(self, ranker, sample_recipes):
        """Test that missing ingredients are correctly identified."""
        query = ["egg", "onion"]
        
        results = ranker.rank_recipes(
            query_ingredients=query,
            recipes=sample_recipes,
            mode="exact",
            top_k=1
        )
        
        assert len(results) == 1
        assert "butter" in results[0].missing_ingredients or "salt" in results[0].missing_ingredients
    
    def test_egg_onion_returns_scrambled_eggs(self, ranker, sample_recipes):
        """Integration test: egg + onion should return scrambled eggs first."""
        query = ["egg", "onion"]
        
        results = ranker.rank_recipes(
            query_ingredients=query,
            recipes=sample_recipes,
            mode="hybrid",
            top_k=1
        )
        
        assert len(results) == 1
        assert "egg" in results[0].recipe.title.lower() or "scrambled" in results[0].recipe.title.lower()


class TestIngredientNormalization:
    """Tests for ingredient normalization."""
    
    @pytest.fixture
    def ranker(self):
        return RecipeRanker()
    
    def test_normalize_lowercase(self, ranker):
        """Test that ingredients are lowercased."""
        result = ranker._normalize_ingredient("EGGS")
        assert result == result.lower()
    
    def test_normalize_strips_whitespace(self, ranker):
        """Test that whitespace is stripped."""
        result = ranker._normalize_ingredient("  egg  ")
        assert result == "egg"
    
    def test_get_ingredient_base_variants(self, ranker):
        """Test that common variants are recognized."""
        assert ranker._get_ingredient_base("eggs") == "egg"
        assert ranker._get_ingredient_base("onions") == "onion"
