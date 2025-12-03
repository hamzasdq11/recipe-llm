#!/usr/bin/env python3
"""
Data preparation script for the Recipe LLM project.

This script:
- Normalizes ingredient names (lowercase, singular/plural)
- Generates synthetic recipe variations
- Creates few-shot prompt examples
- Generates negative/contrastive examples
"""

import json
import random
import re
import sys
from pathlib import Path
from typing import List, Dict, Set
from collections import defaultdict

DATA_DIR = Path(__file__).parent.parent / "data"


def normalize_ingredient(ingredient: str) -> str:
    """Normalize an ingredient name."""
    ing = ingredient.lower().strip()
    ing = re.sub(r'[^\w\s]', '', ing)
    ing = re.sub(r'\s+', ' ', ing)
    
    if ing.endswith('ies'):
        ing = ing[:-3] + 'y'
    elif ing.endswith('ves'):
        ing = ing[:-3] + 'f'
    elif ing.endswith('es') and len(ing) > 4:
        if ing[-3] in ['s', 'x', 'z', 'h']:
            ing = ing[:-2]
        else:
            ing = ing[:-1]
    elif ing.endswith('s') and len(ing) > 3 and ing[-2] not in ['s', 'u']:
        ing = ing[:-1]
    
    return ing


def load_recipes(filename: str = "recipes_small.json") -> List[Dict]:
    """Load recipes from JSON file."""
    filepath = DATA_DIR / filename
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_recipes(recipes: List[Dict], filename: str):
    """Save recipes to JSON file."""
    filepath = DATA_DIR / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(recipes, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(recipes)} recipes to {filepath}")


def normalize_all_ingredients(recipes: List[Dict]) -> List[Dict]:
    """Normalize all ingredient names in recipes."""
    for recipe in recipes:
        recipe['ingredients'] = [
            normalize_ingredient(ing) for ing in recipe['ingredients']
        ]
    return recipes


def get_all_ingredients(recipes: List[Dict]) -> Set[str]:
    """Get set of all unique ingredients."""
    ingredients = set()
    for recipe in recipes:
        for ing in recipe['ingredients']:
            ingredients.add(normalize_ingredient(ing))
    return ingredients


def generate_synthetic_variations(recipes: List[Dict], num_variations: int = 50) -> List[Dict]:
    """Generate synthetic recipe variations."""
    all_ingredients = list(get_all_ingredients(recipes))
    
    variations = []
    
    substitutions = {
        'chicken breast': ['chicken thigh', 'turkey breast'],
        'beef': ['ground beef', 'beef sirloin'],
        'butter': ['margarine', 'olive oil'],
        'milk': ['almond milk', 'oat milk'],
        'cheese': ['cheddar', 'mozzarella', 'parmesan'],
        'onion': ['red onion', 'shallot'],
        'garlic': ['garlic powder', 'roasted garlic'],
        'pasta': ['spaghetti', 'penne', 'fettuccine'],
        'rice': ['basmati rice', 'jasmine rice'],
    }
    
    base_id = len(recipes) + 1
    
    for i in range(num_variations):
        base_recipe = random.choice(recipes)
        
        new_recipe = {
            'id': f'r{base_id + i:04d}',
            'title': f"{base_recipe['title']} (Variation {i+1})",
            'ingredients': list(base_recipe['ingredients']),
            'steps': list(base_recipe['steps']),
            'tags': list(base_recipe['tags']) + ['variation'],
            'prep_time_min': base_recipe['prep_time_min'] + random.randint(-5, 10),
            'difficulty': base_recipe['difficulty']
        }
        
        for j, ing in enumerate(new_recipe['ingredients']):
            if ing in substitutions and random.random() < 0.3:
                new_recipe['ingredients'][j] = random.choice(substitutions[ing])
        
        if random.random() < 0.3:
            extra_ingredients = random.sample(all_ingredients, k=min(2, len(all_ingredients)))
            for extra in extra_ingredients:
                if extra not in new_recipe['ingredients']:
                    new_recipe['ingredients'].append(extra)
        
        variations.append(new_recipe)
    
    return variations


def generate_negative_examples(recipes: List[Dict], num_examples: int = 20) -> List[Dict]:
    """Generate negative/contrastive examples."""
    all_ingredients = list(get_all_ingredients(recipes))
    
    negative_examples = []
    
    for i in range(num_examples):
        num_ingredients = random.randint(3, 7)
        random_ingredients = random.sample(all_ingredients, k=min(num_ingredients, len(all_ingredients)))
        
        matching_recipes = []
        for recipe in recipes:
            recipe_ings = set(normalize_ingredient(ing) for ing in recipe['ingredients'])
            query_ings = set(random_ingredients)
            overlap = len(recipe_ings & query_ings) / len(recipe_ings) if recipe_ings else 0
            if overlap < 0.2:
                matching_recipes.append(recipe['title'])
        
        negative_examples.append({
            'ingredients': random_ingredients,
            'should_not_match': matching_recipes[:5] if matching_recipes else [],
            'note': 'These ingredients should not strongly match the listed recipes'
        })
    
    return negative_examples


def generate_few_shot_examples(recipes: List[Dict], num_examples: int = 10) -> List[Dict]:
    """Generate few-shot prompt examples for LLM fine-tuning."""
    examples = []
    
    for i in range(min(num_examples, len(recipes))):
        recipe = recipes[i]
        
        prompt = f"I have these ingredients: {', '.join(recipe['ingredients'][:4])}. What can I make?"
        
        response = f"""Based on your ingredients, I recommend making **{recipe['title']}**!

**Ingredients needed:**
{chr(10).join('- ' + ing for ing in recipe['ingredients'])}

**Instructions:**
{chr(10).join(f'{j+1}. {step}' for j, step in enumerate(recipe['steps']))}

**Prep time:** {recipe['prep_time_min']} minutes
**Difficulty:** {recipe['difficulty']}

Enjoy your meal!"""
        
        examples.append({
            'prompt': prompt,
            'response': response,
            'recipe_id': recipe['id']
        })
    
    return examples


def create_training_data(recipes: List[Dict]) -> List[Dict]:
    """Create training data for fine-tuning."""
    training_data = []
    
    for recipe in recipes:
        num_ingredients = len(recipe['ingredients'])
        for subset_size in range(2, min(num_ingredients + 1, 5)):
            subset = random.sample(recipe['ingredients'], subset_size)
            
            instruction = f"Given these ingredients: {', '.join(subset)}, suggest a recipe."
            
            output = f"Recipe: {recipe['title']}\n"
            output += f"Additional ingredients needed: {', '.join(ing for ing in recipe['ingredients'] if ing not in subset)}\n"
            output += f"Instructions: {' '.join(recipe['steps'][:3])}\n"
            output += f"Prep time: {recipe['prep_time_min']} minutes"
            
            training_data.append({
                'instruction': instruction,
                'input': '',
                'output': output
            })
    
    return training_data


def main():
    """Main function to prepare all data."""
    print("Loading recipes...")
    recipes = load_recipes()
    print(f"Loaded {len(recipes)} recipes")
    
    print("\nNormalizing ingredients...")
    recipes = normalize_all_ingredients(recipes)
    
    all_ingredients = get_all_ingredients(recipes)
    print(f"Found {len(all_ingredients)} unique ingredients")
    
    print("\nGenerating synthetic variations...")
    variations = generate_synthetic_variations(recipes, num_variations=50)
    print(f"Generated {len(variations)} variations")
    
    full_dataset = recipes + variations
    save_recipes(full_dataset, "recipes_full_sample.json")
    
    print("\nGenerating negative examples...")
    negative_examples = generate_negative_examples(recipes)
    save_path = DATA_DIR / "negative_examples.json"
    with open(save_path, 'w') as f:
        json.dump(negative_examples, f, indent=2)
    print(f"Saved {len(negative_examples)} negative examples")
    
    print("\nGenerating few-shot examples...")
    few_shot = generate_few_shot_examples(recipes)
    save_path = DATA_DIR / "few_shot_examples.json"
    with open(save_path, 'w') as f:
        json.dump(few_shot, f, indent=2)
    print(f"Saved {len(few_shot)} few-shot examples")
    
    print("\nCreating training data...")
    training_data = create_training_data(recipes)
    save_path = DATA_DIR / "training_data.json"
    with open(save_path, 'w') as f:
        json.dump(training_data, f, indent=2)
    print(f"Saved {len(training_data)} training examples")
    
    print("\n" + "="*50)
    print("Data preparation complete!")
    print(f"  - Original recipes: {len(recipes)}")
    print(f"  - Full dataset: {len(full_dataset)}")
    print(f"  - Unique ingredients: {len(all_ingredients)}")
    print(f"  - Training examples: {len(training_data)}")


if __name__ == "__main__":
    main()
