"""Integration tests for the API."""

import pytest
import os
from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("RECIPE_DATA_DIR", "data")
os.environ.setdefault("RECIPE_MODE", "minimal")
os.environ.setdefault("RECIPE_USE_MOCK", "true")

from app.main import app


@pytest.fixture
def client():
    """Create test client."""
    with TestClient(app) as client:
        yield client


class TestHealthEndpoint:
    """Tests for /health endpoint."""
    
    def test_health_returns_ok(self, client):
        """Test health endpoint returns 200."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "recipes_loaded" in data


class TestPredictEndpoint:
    """Tests for /api/v1/predict endpoint."""
    
    def test_predict_basic(self, client):
        """Test basic prediction."""
        response = client.post(
            "/api/v1/predict",
            json={
                "ingredients": ["egg", "onion"],
                "top_k": 5,
                "mode": "hybrid"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "query" in data
        assert data["query"] == ["egg", "onion"]
    
    def test_predict_returns_recipes(self, client):
        """Test that prediction returns recipe results."""
        response = client.post(
            "/api/v1/predict",
            json={
                "ingredients": ["egg", "butter"],
                "top_k": 3,
                "mode": "hybrid"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) > 0
        
        recipe = data["results"][0]
        assert "recipe_id" in recipe
        assert "title" in recipe
        assert "score" in recipe
        assert "ingredients_matched" in recipe
        assert "missing_ingredients" in recipe
        assert "instructions" in recipe
    
    def test_predict_egg_onion_returns_scrambled_eggs(self, client):
        """Integration test: egg + onion should return scrambled eggs."""
        response = client.post(
            "/api/v1/predict",
            json={
                "ingredients": ["egg", "onion"],
                "top_k": 5,
                "mode": "hybrid"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        titles = [r["title"].lower() for r in data["results"]]
        assert any("scrambled" in t or "egg" in t for t in titles)
    
    def test_predict_with_explanation(self, client):
        """Test prediction with score explanation."""
        response = client.post(
            "/api/v1/predict",
            json={
                "ingredients": ["egg", "butter"],
                "top_k": 1,
                "mode": "hybrid",
                "include_explanation": True
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) > 0
        
        recipe = data["results"][0]
        assert "explanation" in recipe
        assert recipe["explanation"] is not None
    
    def test_predict_exact_mode(self, client):
        """Test exact matching mode."""
        response = client.post(
            "/api/v1/predict",
            json={
                "ingredients": ["egg"],
                "top_k": 5,
                "mode": "exact"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "exact"
    
    def test_predict_validates_ingredients(self, client):
        """Test that empty ingredients returns error."""
        response = client.post(
            "/api/v1/predict",
            json={
                "ingredients": [],
                "top_k": 5,
                "mode": "hybrid"
            }
        )
        
        assert response.status_code == 422
    
    def test_predict_validates_top_k(self, client):
        """Test that invalid top_k returns error."""
        response = client.post(
            "/api/v1/predict",
            json={
                "ingredients": ["egg"],
                "top_k": 100,
                "mode": "hybrid"
            }
        )
        
        assert response.status_code == 422


class TestRecipesEndpoint:
    """Tests for /api/v1/recipes endpoints."""
    
    def test_list_recipes(self, client):
        """Test listing recipes."""
        response = client.get("/api/v1/recipes")
        
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "recipes" in data
        assert len(data["recipes"]) > 0
    
    def test_list_recipes_with_tag(self, client):
        """Test filtering recipes by tag."""
        response = client.get("/api/v1/recipes?tag=breakfast")
        
        assert response.status_code == 200
        data = response.json()
        for recipe in data["recipes"]:
            assert "breakfast" in [t.lower() for t in recipe["tags"]]
    
    def test_list_recipes_with_difficulty(self, client):
        """Test filtering recipes by difficulty."""
        response = client.get("/api/v1/recipes?difficulty=easy")
        
        assert response.status_code == 200
        data = response.json()
        for recipe in data["recipes"]:
            assert recipe["difficulty"].lower() == "easy"
    
    def test_get_recipe_by_id(self, client):
        """Test getting specific recipe."""
        response = client.get("/api/v1/recipes/r0001")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "r0001"
    
    def test_get_recipe_not_found(self, client):
        """Test getting non-existent recipe."""
        response = client.get("/api/v1/recipes/r9999")
        
        assert response.status_code == 404


class TestIngredientsEndpoint:
    """Tests for /api/v1/ingredients endpoint."""
    
    def test_list_ingredients(self, client):
        """Test listing all ingredients."""
        response = client.get("/api/v1/ingredients")
        
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "ingredients" in data
        assert len(data["ingredients"]) > 0
        assert "egg" in data["ingredients"]
