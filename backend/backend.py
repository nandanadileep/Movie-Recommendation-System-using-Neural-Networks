# backend.py

import os
import torch
import pandas as pd
import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from model import MovieRecNet

# -----------------------------
# Initialize FastAPI app
# -----------------------------
app = FastAPI(title="Movie Recommendation API")

# -----------------------------
# Global variables
# -----------------------------
model = None
movies = None
tfidf = None
X_all = None
device = None

# -----------------------------
# User input model
# -----------------------------
class UserInput(BaseModel):
    favorite_movies: List[str]

# -----------------------------
# Load resources function
# -----------------------------
def load_resources():
    """Lazy load all model and data resources"""
    global model, movies, tfidf, X_all, device

    if model is not None:
        return  # Already loaded

    print("Loading resources...")

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    BASE_DIR = os.path.dirname(__file__)
    MODEL_PATH = os.path.join(BASE_DIR, "movie_rec_model.pth")
    CSV_PATH = os.path.join(BASE_DIR, "movies_clean.csv")
    VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")

    # Load model
    model = MovieRecNet(input_dim=5022)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
    model.eval()
    model.to(device)

    # Load movies CSV
    movies_df = pd.read_csv(CSV_PATH)
    movies_df['overview'] = movies_df['overview'].fillna('')
    movies = movies_df

    # Load TF-IDF vectorizer
    with open(VECTORIZER_PATH, "rb") as f:
        tfidf = pickle.load(f)

    # Transform overviews
    X_all = tfidf.transform(movies['overview'])

    print("All resources loaded successfully!")

# -----------------------------
# Root route
# -----------------------------
@app.get("/")
def read_root():
    return {"message": "Movie Recommendation API is running", "status": "healthy"}

# -----------------------------
# Health check route
# -----------------------------
@app.get("/health")
def health_check():
    global model, movies, device
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

# -----------------------------
# Recommendation route
# -----------------------------
@app.post("/recommend")
def recommend(user_input: UserInput):
    global model, movies, tfidf, X_all, device
    try:
        load_resources()

        # Filter favorite movies present in database
        fav_titles = [m.strip() for m in user_input.favorite_movies if m.strip() in movies['title'].values]

        if not fav_titles:
            return {"error": "No valid movies found in database!"}

        # Compute scores in batches
        batch_size = 500
        all_scores = []

        with torch.no_grad():
            for i in range(0, X_all.shape[0], batch_size):
                end_idx = min(i + batch_size, X_all.shape[0])
                X_batch = X_all[i:end_idx].toarray()
                X_tensor = torch.tensor(X_batch, dtype=torch.float32).to(device)
                batch_scores = model(X_tensor).cpu().numpy().flatten()
                all_scores.extend(batch_scores)

        # Add scores to dataframe
        movies_copy = movies.copy()
        movies_copy['predicted_score'] = all_scores

        # Exclude favorite movies and get top 10
        recommendations = movies_copy[~movies_copy['title'].isin(fav_titles)]
        top_k = recommendations.sort_values(by='predicted_score', ascending=False).head(10)

        return {
            "recommendations": top_k[['title', 'genres', 'predicted_score']].to_dict(orient='records'),
            "favorite_movies_found": fav_titles
        }

    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

# -----------------------------
# Startup event to preload resources
# -----------------------------
@app.on_event("startup")
async def startup_event():
    global model, movies, tfidf, X_all, device
    try:
        load_resources()
    except Exception as e:
        print(f"Error loading resources on startup: {e}")

# -----------------------------
# Main entry point for Render
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("backend:app", host="0.0.0.0", port=port)