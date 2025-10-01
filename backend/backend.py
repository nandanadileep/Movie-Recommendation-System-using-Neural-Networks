from fastapi import FastAPI
from pydantic import BaseModel
import torch
import pandas as pd
import pickle
from typing import List
from model import MovieRecNet
import numpy as np

# Initialize FastAPI app first
app = FastAPI(title="Movie Recommendation API")

# Global variables for lazy loading
model = None
movies = None
tfidf = None
X_all = None
device = None

def load_resources():
    """Lazy load all resources"""
    global model, movies, tfidf, X_all, device
    
    if model is not None:
        return  # Already loaded
    
    print("Loading resources...")
    
    # Set device
    device = torch.device("cpu")  # Force CPU for Render free tier
    print(f"Using device: {device}")
    
    # Load model
    model = MovieRecNet(input_dim=5022)
    model.load_state_dict(torch.load("movie_rec_model.pth", map_location=device, weights_only=True))
    model.eval()
    model.to(device)
    print("Model loaded")
    
    # Load movies
    movies = pd.read_csv("movies_clean.csv")
    movies['overview'] = movies['overview'].fillna('')
    print(f"Loaded {len(movies)} movies")
    
    # Load TF-IDF vectorizer
    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    print("TF-IDF vectorizer loaded")
    
    # Transform all movie overviews
    X_all = tfidf.transform(movies['overview'])
    print(f"TF-IDF matrix shape: {X_all.shape}")
    
    print("All resources loaded successfully!")

class UserInput(BaseModel):
    favorite_movies: List[str]

@app.get("/")
def read_root():
    return {"message": "Movie Recommendation API is running", "status": "healthy"}

@app.get("/health")
def health_check():
    """Detailed health check"""
    try:
        load_resources()
        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "movies_count": len(movies) if movies is not None else 0,
            "device": str(device)
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/recommend")
def recommend(user_input: UserInput):
    try:
        # Load resources if not already loaded
        load_resources()
        
        # Find valid favorite movies
        fav_titles = [m.strip() for m in user_input.favorite_movies if m.strip() in movies['title'].values]
        
        if not fav_titles:
            return {"error": "No valid movies found in database!"}
        
        # Process in batches to avoid memory issues
        batch_size = 500
        all_scores = []
        
        with torch.no_grad():
            for i in range(0, X_all.shape[0], batch_size):
                end_idx = min(i + batch_size, X_all.shape[0])
                X_batch = X_all[i:end_idx].toarray()
                X_tensor = torch.tensor(X_batch, dtype=torch.float32).to(device)
                
                batch_scores = model(X_tensor).cpu().numpy().flatten()
                all_scores.extend(batch_scores)
        
        # Add scores to movies dataframe
        movies['predicted_score'] = all_scores
        
        # Filter out favorite movies and get top recommendations
        recommendations = movies[~movies['title'].isin(fav_titles)]
        top_k = recommendations.sort_values(by='predicted_score', ascending=False).head(10)
        
        return {
            "recommendations": top_k[['title', 'genres', 'predicted_score']].to_dict(orient='records'),
            "favorite_movies_found": fav_titles
        }
    
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

@app.on_event("startup")
async def startup_event():
    """Load resources on startup"""
    try:
        load_resources()
    except Exception as e:
        print(f"Error loading resources on startup: {e}")
        # Don't crash the app, allow it to try loading on first request