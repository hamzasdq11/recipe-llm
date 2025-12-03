"""Hybrid ranking system for recipe matching."""

from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from rapidfuzz import fuzz, process
import re

from .data_loader import Recipe


@dataclass
class MatchResult:
    """Result of matching ingredients to a recipe."""
    recipe: Recipe
    score: float
    exact_matches: List[str] = field(default_factory=list)
    fuzzy_matches: List[Dict] = field(default_factory=list)
    missing_ingredients: List[str] = field(default_factory=list)
    exact_score: float = 0.0
    fuzzy_score: float = 0.0
    semantic_score: float = 0.0


class RecipeRanker:
    """Hybrid ranking system combining exact, fuzzy, and semantic matching."""
    
    def __init__(
        self,
        weight_exact: float = 0.5,
        weight_fuzzy: float = 0.3,
        weight_semantic: float = 0.2,
        fuzzy_threshold: int = 80
    ):
        self.weight_exact = weight_exact
        self.weight_fuzzy = weight_fuzzy
        self.weight_semantic = weight_semantic
        self.fuzzy_threshold = fuzzy_threshold
        
        self._ingredient_variants = self._build_ingredient_variants()
    
    def _build_ingredient_variants(self) -> Dict[str, Set[str]]:
        """Build common ingredient variants for better matching."""
        variants = {
            "egg": {"eggs", "egg", "egg yolk", "egg white", "beaten egg"},
            "onion": {"onions", "onion", "red onion", "white onion", "green onion", "spring onion"},
            "garlic": {"garlic", "garlic clove", "garlic cloves", "minced garlic"},
            "tomato": {"tomatoes", "tomato", "cherry tomato", "roma tomato", "tomato sauce"},
            "potato": {"potatoes", "potato", "sweet potato", "russet potato"},
            "chicken": {"chicken", "chicken breast", "chicken thigh", "chicken wing", "chicken leg"},
            "beef": {"beef", "ground beef", "beef sirloin", "beef chuck", "steak"},
            "pork": {"pork", "pork chop", "pork shoulder", "pork belly", "bacon", "ham"},
            "cheese": {"cheese", "cheddar", "mozzarella", "parmesan", "feta", "gruyere", "swiss cheese"},
            "pasta": {"pasta", "spaghetti", "fettuccine", "penne", "linguine", "macaroni", "lasagna noodles"},
            "rice": {"rice", "basmati rice", "arborio rice", "sushi rice", "brown rice"},
            "milk": {"milk", "whole milk", "skim milk", "buttermilk", "coconut milk"},
            "butter": {"butter", "unsalted butter", "salted butter", "ghee"},
            "oil": {"oil", "olive oil", "vegetable oil", "sesame oil", "coconut oil"},
            "pepper": {"pepper", "black pepper", "bell pepper", "chili pepper", "jalapeno"},
            "salt": {"salt", "sea salt", "kosher salt", "table salt"},
            "flour": {"flour", "all-purpose flour", "bread flour", "whole wheat flour"},
            "sugar": {"sugar", "brown sugar", "powdered sugar", "granulated sugar"},
            "cream": {"cream", "heavy cream", "sour cream", "whipping cream"},
            "lemon": {"lemon", "lemon juice", "lemon zest", "lime"},
        }
        return variants
    
    def _normalize_ingredient(self, ingredient: str) -> str:
        """Normalize an ingredient name."""
        ing = ingredient.lower().strip()
        ing = re.sub(r'[^\w\s]', '', ing)
        ing = re.sub(r'\s+', ' ', ing)
        if ing.endswith('s') and len(ing) > 3:
            singular = ing[:-1]
            if singular not in ['s', 'ss']:
                ing = singular if len(singular) > 2 else ing
        return ing
    
    def _get_ingredient_base(self, ingredient: str) -> str:
        """Get the base form of an ingredient."""
        ing = self._normalize_ingredient(ingredient)
        for base, variants in self._ingredient_variants.items():
            if ing in variants or ing == base:
                return base
        return ing
    
    def exact_match(
        self,
        query_ingredients: List[str],
        recipe: Recipe
    ) -> Tuple[List[str], float]:
        """Perform exact ingredient matching."""
        query_set = {self._get_ingredient_base(ing) for ing in query_ingredients}
        recipe_set = {self._get_ingredient_base(ing) for ing in recipe.ingredients}
        
        matches = query_set & recipe_set
        matched_ingredients = [ing for ing in query_ingredients 
                              if self._get_ingredient_base(ing) in matches]
        
        if not recipe_set:
            return [], 0.0
        
        coverage = len(matches) / len(recipe_set)
        return matched_ingredients, coverage
    
    def fuzzy_match(
        self,
        query_ingredients: List[str],
        recipe: Recipe,
        already_matched: Set[str]
    ) -> Tuple[List[Dict], float]:
        """Perform fuzzy ingredient matching."""
        query_normalized = [self._normalize_ingredient(ing) for ing in query_ingredients]
        recipe_normalized = [self._normalize_ingredient(ing) for ing in recipe.ingredients]
        
        already_matched_normalized = {self._normalize_ingredient(m) for m in already_matched}
        remaining_recipe = [ing for ing in recipe_normalized 
                          if ing not in already_matched_normalized]
        
        if not remaining_recipe:
            return [], 0.0
        
        fuzzy_matches = []
        matched_recipe_ingredients = set()
        
        for query_ing in query_normalized:
            if query_ing in already_matched_normalized:
                continue
            
            best_match = process.extractOne(
                query_ing,
                remaining_recipe,
                scorer=fuzz.ratio,
                score_cutoff=self.fuzzy_threshold
            )
            
            if best_match and best_match[0] not in matched_recipe_ingredients:
                fuzzy_matches.append({
                    "query": query_ing,
                    "matched": best_match[0],
                    "score": best_match[1]
                })
                matched_recipe_ingredients.add(best_match[0])
        
        if not remaining_recipe:
            return fuzzy_matches, 0.0
        
        coverage = len(fuzzy_matches) / len(remaining_recipe)
        return fuzzy_matches, coverage
    
    def rank_recipes(
        self,
        query_ingredients: List[str],
        recipes: List[Recipe],
        mode: str = "hybrid",
        top_k: int = 5
    ) -> List[MatchResult]:
        """Rank recipes based on ingredient matching."""
        results = []
        
        for recipe in recipes:
            exact_matches, exact_score = self.exact_match(query_ingredients, recipe)
            
            fuzzy_matches = []
            fuzzy_score = 0.0
            
            if mode in ["fuzzy", "hybrid"]:
                already_matched = set(exact_matches)
                fuzzy_matches, fuzzy_score = self.fuzzy_match(
                    query_ingredients, recipe, already_matched
                )
            
            if mode == "exact":
                total_score = exact_score
            elif mode == "fuzzy":
                total_score = exact_score * 0.3 + fuzzy_score * 0.7
            else:
                total_score = (
                    exact_score * self.weight_exact +
                    fuzzy_score * self.weight_fuzzy
                )
            
            all_matched = set(exact_matches)
            all_matched.update(m["matched"] for m in fuzzy_matches)
            
            recipe_normalized = {self._normalize_ingredient(ing) for ing in recipe.ingredients}
            matched_normalized = {self._normalize_ingredient(m) for m in all_matched}
            missing = [ing for ing in recipe.ingredients 
                      if self._normalize_ingredient(ing) not in matched_normalized]
            
            result = MatchResult(
                recipe=recipe,
                score=total_score,
                exact_matches=exact_matches,
                fuzzy_matches=fuzzy_matches,
                missing_ingredients=missing,
                exact_score=exact_score,
                fuzzy_score=fuzzy_score,
                semantic_score=0.0
            )
            results.append(result)
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    def explain_score(self, result: MatchResult) -> Dict:
        """Generate an explanation for the score."""
        return {
            "exact_match_score": round(result.exact_score, 3),
            "fuzzy_match_score": round(result.fuzzy_score, 3),
            "semantic_score": round(result.semantic_score, 3),
            "exact_matches": result.exact_matches,
            "fuzzy_matches": result.fuzzy_matches,
            "weights": {
                "exact": self.weight_exact,
                "fuzzy": self.weight_fuzzy,
                "semantic": self.weight_semantic
            }
        }
