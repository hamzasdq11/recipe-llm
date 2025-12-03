#!/usr/bin/env python3
"""
CLI tool for querying recipes from the command line.
Usage: python tools/query_cli.py --ingredients "egg,onion,butter"
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.data_loader import DataLoader
from app.core.ranking import RecipeRanker


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Query recipes based on available ingredients",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/query_cli.py --ingredients "egg,onion"
  python tools/query_cli.py -i "chicken,garlic,lemon" --top-k 10
  python tools/query_cli.py -i "pasta,tomato" --mode exact
  python tools/query_cli.py --list-ingredients
        """
    )
    
    parser.add_argument(
        "-i", "--ingredients",
        type=str,
        help="Comma-separated list of ingredients"
    )
    
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=5,
        help="Number of recipes to return (default: 5)"
    )
    
    parser.add_argument(
        "-m", "--mode",
        type=str,
        choices=["exact", "fuzzy", "hybrid"],
        default="hybrid",
        help="Matching mode (default: hybrid)"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to data directory (default: data)"
    )
    
    parser.add_argument(
        "--list-ingredients",
        action="store_true",
        help="List all available ingredients"
    )
    
    parser.add_argument(
        "--list-tags",
        action="store_true",
        help="List all recipe tags"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed matching information"
    )
    
    return parser.parse_args()


def format_recipe(match, verbose=False):
    """Format a recipe match for display."""
    recipe = match.recipe
    
    output = []
    output.append(f"\n{'='*60}")
    output.append(f"  {recipe.title}")
    output.append(f"  Score: {match.score:.1%} | Prep: {recipe.prep_time_min} min | Difficulty: {recipe.difficulty}")
    output.append(f"{'='*60}")
    
    output.append(f"\n  Matched ingredients: {', '.join(match.exact_matches)}")
    
    if match.fuzzy_matches:
        fuzzy_str = ", ".join(f"{m['query']}~{m['matched']}" for m in match.fuzzy_matches)
        output.append(f"  Fuzzy matches: {fuzzy_str}")
    
    if match.missing_ingredients:
        output.append(f"  Missing ingredients: {', '.join(match.missing_ingredients)}")
    
    output.append(f"\n  Instructions:")
    output.append(f"  {recipe.instructions_short}")
    
    if recipe.tags:
        output.append(f"\n  Tags: {', '.join(recipe.tags)}")
    
    if verbose:
        output.append(f"\n  Score breakdown:")
        output.append(f"    - Exact match: {match.exact_score:.3f}")
        output.append(f"    - Fuzzy match: {match.fuzzy_score:.3f}")
        output.append(f"    - Semantic: {match.semantic_score:.3f}")
    
    return "\n".join(output)


def main():
    """Main CLI entry point."""
    args = parse_args()
    
    try:
        loader = DataLoader(args.data_dir)
        loader.load_recipes()
        print(f"Loaded {loader.recipe_count} recipes\n")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    if args.list_ingredients:
        ingredients = sorted(loader.get_all_ingredients())
        if args.json:
            print(json.dumps({"ingredients": ingredients}, indent=2))
        else:
            print("Available ingredients:")
            for ing in ingredients:
                print(f"  - {ing}")
        return
    
    if args.list_tags:
        tags = set()
        for recipe in loader.recipes:
            tags.update(recipe.tags)
        tags = sorted(tags)
        if args.json:
            print(json.dumps({"tags": tags}, indent=2))
        else:
            print("Available tags:")
            for tag in tags:
                print(f"  - {tag}")
        return
    
    if not args.ingredients:
        print("Error: --ingredients is required (or use --list-ingredients/--list-tags)")
        sys.exit(1)
    
    ingredients = [i.strip() for i in args.ingredients.split(",") if i.strip()]
    
    if not ingredients:
        print("Error: Please provide at least one ingredient")
        sys.exit(1)
    
    ranker = RecipeRanker()
    results = ranker.rank_recipes(
        query_ingredients=ingredients,
        recipes=loader.recipes,
        mode=args.mode,
        top_k=args.top_k
    )
    
    if args.json:
        output = {
            "query": ingredients,
            "mode": args.mode,
            "results": []
        }
        for match in results:
            output["results"].append({
                "recipe_id": match.recipe.id,
                "title": match.recipe.title,
                "score": round(match.score, 3),
                "ingredients_matched": match.exact_matches + [m["matched"] for m in match.fuzzy_matches],
                "missing_ingredients": match.missing_ingredients,
                "instructions": match.recipe.instructions_short,
                "tags": match.recipe.tags,
                "prep_time_min": match.recipe.prep_time_min,
                "difficulty": match.recipe.difficulty
            })
        print(json.dumps(output, indent=2))
    else:
        print(f"Searching for recipes with: {', '.join(ingredients)}")
        print(f"Mode: {args.mode}")
        
        if not results:
            print("\nNo matching recipes found.")
        else:
            print(f"\nFound {len(results)} matching recipes:")
            for match in results:
                print(format_recipe(match, verbose=args.verbose))


if __name__ == "__main__":
    main()
