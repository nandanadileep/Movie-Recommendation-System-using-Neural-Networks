from fastapi import FastAPI
from pydantic import BaseModel
import torch
import pandas as pd
import pickle
from typing import List
from model import MovieRecNet
import numpy as np
import os

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Path to the model file in the same folder as this script
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "movie_rec_model.pth")
    
    # Initialize and load model
    model = MovieRecNet(input_dim=5022)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
    model.eval()
    model.to(device)
    
    # Load movies
    CSV_PATH = os.path.join(os.path.dirname(__file__), "movies_clean.csv")
    try:
        movies_df = pd.read_csv(CSV_PATH)
        # Fill NaN overviews with empty strings
        movies_df['overview'] = movies_df['overview'].fillna('')
        print("Movies data loaded")
    except FileNotFoundError:
        print(f"Error loading resources: {CSV_PATH} not found")
        raise
    
    # Load TF-IDF vectorizer
    VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "tfidf_vectorizer.pkl")
    try:
        with open(VECTORIZER_PATH, "rb") as f:
            tfidf = pickle.load(f)
        print("TF-IDF vectorizer loaded")
    except FileNotFoundError:
        print(f"Error loading resources: {VECTORIZER_PATH} not found")
        raise
    
    # Transform all movie overviews
    X_all = tfidf.transform(movies_df['overview'])
    print(f"TF-IDF matrix shape: {X_all.shape}")
    
    # Set global movies variable
    movies = movies_df
    
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
        
        # Add scores to movies dataframe (create a copy to avoid modifying global state)
        movies_copy = movies.copy()
        movies_copy['predicted_score'] = all_scores
        
        # Filter out favorite movies and get top recommendations
        recommendations = movies_copy[~movies_copy['title'].isin(fav_titles)]
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