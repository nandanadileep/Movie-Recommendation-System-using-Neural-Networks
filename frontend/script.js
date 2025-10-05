// Configuration - Use your actual deployed backend URL
const API_BASE_URL = 'https://movierecommendationss.onrender.com';

// Add loading state helper functions
function showLoading() {
    const button = document.getElementById("get-recommendations");
    button.disabled = true;
    button.textContent = "Loading...";
}

function hideLoading() {
    const button = document.getElementById("get-recommendations");
    button.disabled = false;
    button.textContent = "Get Recommendations";
}

// Check backend health on page load
async function checkBackendHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        console.log('✅ Backend is healthy:', data);
        return true;
    } catch (error) {
        console.warn('⚠️ Backend is not responding (may be sleeping):', error);
        // Show warning to user
        const warning = document.createElement('div');
        warning.style.cssText = `
            position: fixed;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: #ffc107;
            color: #000;
            padding: 15px 25px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            z-index: 9999;
            font-family: Arial, sans-serif;
        `;
        warning.innerHTML = '⏳ Backend is starting up (first visit may take 30-60 seconds)...';
        document.body.appendChild(warning);
        setTimeout(() => warning.remove(), 60000);
        return false;
    }
}

// Initialize on page load
window.addEventListener('DOMContentLoaded', () => {
    console.log('Using backend URL:', API_BASE_URL);
    checkBackendHealth();
});

// Main recommendation function
document.getElementById("get-recommendations").addEventListener("click", async () => {
    const textarea = document.getElementById("favorite-movies");
    const movieInput = textarea.value;
    const movieList = movieInput.split(",").map(m => m.trim()).filter(m => m);
    
    if (movieList.length === 0) {
        alert("Please enter at least one movie.");
        return;
    }
    
    // Show loading state
    showLoading();
    
    try {
        console.log('Sending request to:', `${API_BASE_URL}/recommend`);
        console.log('With movies:', movieList);
        
        const response = await fetch(`${API_BASE_URL}/recommend`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ favorite_movies: movieList })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Received data:', data);
        
        if (data.error) {
            alert(data.error);
            hideLoading();
            return;
        }
        
        // Display recommendations
        const recommendationsList = document.getElementById("recommendations-list");
        recommendationsList.innerHTML = "";
        
        if (data.recommendations && data.recommendations.length > 0) {
            data.recommendations.forEach(movie => {
                const li = document.createElement("li");
                li.textContent = `${movie.title} (${movie.genres}) — Score: ${movie.predicted_score.toFixed(2)}`;
                recommendationsList.appendChild(li);
            });
            
            // Show which movies were found
            if (data.favorite_movies_found && data.favorite_movies_found.length > 0) {
                console.log('✅ Found favorite movies:', data.favorite_movies_found);
            }
        } else {
            recommendationsList.innerHTML = "<li>No recommendations found. Try different movies!</li>";
        }
        
        hideLoading();
        
    } catch (err) {
        console.error('Error details:', err);
        hideLoading();
        
        // More helpful error message
        let errorMessage = "Error connecting to the backend. ";
        if (err.message.includes('Failed to fetch')) {
            errorMessage += "The backend may be sleeping (Render free tier). Please wait 30-60 seconds and try again.";
        } else if (err.message.includes('CORS')) {
            errorMessage += "CORS error - backend needs to be updated with your domain.";
        } else {
            errorMessage += err.message;
        }
        
        alert(errorMessage);
    }
});