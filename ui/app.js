const API_BASE = window.location.origin;

const ingredientsInput = document.getElementById('ingredients');
const searchBtn = document.getElementById('searchBtn');
const modeSelect = document.getElementById('mode');
const topKSelect = document.getElementById('topK');
const loadingDiv = document.getElementById('loading');
const resultsDiv = document.getElementById('results');
const recipeList = document.getElementById('recipeList');
const errorDiv = document.getElementById('error');
const errorMessage = document.getElementById('errorMessage');

const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');
const chatMessages = document.getElementById('chatMessages');

let conversationHistory = [];

function showLoading() {
    loadingDiv.classList.remove('hidden');
    resultsDiv.classList.add('hidden');
    errorDiv.classList.add('hidden');
}

function hideLoading() {
    loadingDiv.classList.add('hidden');
}

function showError(message) {
    errorMessage.textContent = message;
    errorDiv.classList.remove('hidden');
    resultsDiv.classList.add('hidden');
}

function showResults() {
    resultsDiv.classList.remove('hidden');
    errorDiv.classList.add('hidden');
}

function getScoreClass(score) {
    if (score >= 0.7) return '';
    if (score >= 0.4) return 'medium';
    return 'low';
}

function createRecipeCard(recipe) {
    const card = document.createElement('div');
    card.className = 'recipe-card';
    
    const matchedIngredients = recipe.ingredients_matched.map(ing => 
        `<span class="ingredient-tag matched">${ing}</span>`
    ).join('');
    
    const missingIngredients = recipe.missing_ingredients.map(ing =>
        `<span class="ingredient-tag missing">${ing}</span>`
    ).join('');
    
    const tags = recipe.tags.map(tag =>
        `<span class="tag">${tag}</span>`
    ).join('');
    
    card.innerHTML = `
        <div class="recipe-header" onclick="toggleRecipeDetails(this)">
            <span class="recipe-title">${recipe.title}</span>
            <div class="recipe-meta">
                <span class="prep-time">${recipe.prep_time_min} min</span>
                <span class="difficulty-badge">${recipe.difficulty}</span>
                <span class="score-badge ${getScoreClass(recipe.score)}">${Math.round(recipe.score * 100)}% match</span>
            </div>
        </div>
        <div class="recipe-details">
            <div class="detail-section">
                <h4>Matched Ingredients</h4>
                <div class="ingredient-list">${matchedIngredients || '<span class="text-muted">None</span>'}</div>
            </div>
            ${recipe.missing_ingredients.length > 0 ? `
            <div class="detail-section">
                <h4>Missing Ingredients</h4>
                <div class="ingredient-list">${missingIngredients}</div>
            </div>
            ` : ''}
            <div class="detail-section">
                <h4>Instructions</h4>
                <p class="instructions-text">${recipe.instructions}</p>
            </div>
            ${tags ? `
            <div class="detail-section">
                <h4>Tags</h4>
                <div class="tag-list">${tags}</div>
            </div>
            ` : ''}
        </div>
    `;
    
    return card;
}

function toggleRecipeDetails(header) {
    const details = header.nextElementSibling;
    details.classList.toggle('expanded');
}

async function searchRecipes() {
    const ingredientsText = ingredientsInput.value.trim();
    
    if (!ingredientsText) {
        showError('Please enter at least one ingredient');
        return;
    }
    
    const ingredients = ingredientsText.split(',').map(i => i.trim()).filter(i => i);
    
    if (ingredients.length === 0) {
        showError('Please enter valid ingredients');
        return;
    }
    
    showLoading();
    searchBtn.disabled = true;
    
    try {
        const response = await fetch(`${API_BASE}/api/v1/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                ingredients: ingredients,
                top_k: parseInt(topKSelect.value),
                mode: modeSelect.value,
                include_explanation: false
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to get recipes');
        }
        
        const data = await response.json();
        
        hideLoading();
        
        if (data.results.length === 0) {
            showError('No recipes found matching your ingredients. Try different ingredients or use fuzzy matching mode.');
            return;
        }
        
        recipeList.innerHTML = '';
        data.results.forEach(recipe => {
            recipeList.appendChild(createRecipeCard(recipe));
        });
        
        showResults();
        
    } catch (error) {
        hideLoading();
        showError(error.message);
    } finally {
        searchBtn.disabled = false;
    }
}

function addChatMessage(content, role) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    messageDiv.innerHTML = `<p>${content}</p>`;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function sendChatMessage() {
    const message = chatInput.value.trim();
    
    if (!message) return;
    
    addChatMessage(message, 'user');
    chatInput.value = '';
    sendBtn.disabled = true;
    
    try {
        const response = await fetch(`${API_BASE}/api/v1/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                conversation_history: conversationHistory,
                max_tokens: 256
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to get response');
        }
        
        const data = await response.json();
        
        addChatMessage(data.response, 'assistant');
        conversationHistory = data.conversation_history;
        
    } catch (error) {
        addChatMessage(`Error: ${error.message}`, 'assistant');
    } finally {
        sendBtn.disabled = false;
    }
}

searchBtn.addEventListener('click', searchRecipes);
ingredientsInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') searchRecipes();
});

sendBtn.addEventListener('click', sendChatMessage);
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendChatMessage();
});

async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        const data = await response.json();
        console.log('API Health:', data);
    } catch (error) {
        console.error('API not available:', error);
    }
}

checkHealth();
