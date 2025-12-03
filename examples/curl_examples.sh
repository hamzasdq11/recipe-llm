#!/bin/bash
# Example curl commands for Recipe LLM API

BASE_URL="${BASE_URL:-http://localhost:5000}"

echo "Recipe LLM API Examples"
echo "======================="
echo ""

# Health check
echo "1. Health Check"
echo "---------------"
echo "curl $BASE_URL/health"
curl -s "$BASE_URL/health" | python3 -m json.tool
echo ""

# Basic prediction
echo "2. Basic Recipe Prediction"
echo "--------------------------"
echo 'curl -X POST $BASE_URL/api/v1/predict -H "Content-Type: application/json" -d '"'"'{"ingredients": ["egg", "onion"], "top_k": 3}'"'"
curl -s -X POST "$BASE_URL/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{"ingredients": ["egg", "onion"], "top_k": 3}' | python3 -m json.tool
echo ""

# Prediction with explanation
echo "3. Prediction with Score Explanation"
echo "-------------------------------------"
curl -s -X POST "$BASE_URL/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{"ingredients": ["chicken", "garlic", "lemon"], "top_k": 2, "include_explanation": true}' | python3 -m json.tool
echo ""

# Exact matching mode
echo "4. Exact Matching Mode"
echo "----------------------"
curl -s -X POST "$BASE_URL/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{"ingredients": ["pasta", "tomato"], "top_k": 3, "mode": "exact"}' | python3 -m json.tool
echo ""

# List recipes
echo "5. List Recipes"
echo "---------------"
curl -s "$BASE_URL/api/v1/recipes?limit=5" | python3 -m json.tool
echo ""

# Filter by tag
echo "6. Filter Recipes by Tag"
echo "------------------------"
curl -s "$BASE_URL/api/v1/recipes?tag=breakfast&limit=3" | python3 -m json.tool
echo ""

# Get specific recipe
echo "7. Get Recipe by ID"
echo "-------------------"
curl -s "$BASE_URL/api/v1/recipes/r0001" | python3 -m json.tool
echo ""

# List ingredients
echo "8. List All Ingredients"
echo "-----------------------"
curl -s "$BASE_URL/api/v1/ingredients" | python3 -m json.tool
echo ""

# Chat endpoint
echo "9. Chat with Assistant"
echo "----------------------"
curl -s -X POST "$BASE_URL/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What can I make with eggs and cheese?", "max_tokens": 100}' | python3 -m json.tool
echo ""

echo "Done!"
