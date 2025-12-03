"""Vercel serverless function entry point."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("RECIPE_DATA_DIR", "data")
os.environ.setdefault("RECIPE_MODE", "minimal")
os.environ.setdefault("RECIPE_USE_MOCK", "true")

from app.main import app

handler = app
