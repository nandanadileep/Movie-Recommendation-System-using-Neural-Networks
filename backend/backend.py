# backend.py
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import pandas as pd
import pickle
from typing import List
from model import MovieRecNet  


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MovieRecNet(input_dim=5022)
model.load_state_dict(torch.load("movie_rec_model.pth", map_location=device, weights_only=True))
model.eval()

movies = pd.read_csv("movies_clean.csv")
# Fill NaN values in overview column with empty strings
movies['overview'] = movies['overview'].fillna('')

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

X_all = tfidf.transform(movies['overview'])

app = FastAPI(title="Movie Recommendation API")

class UserInput(BaseModel):
    favorite_movies: List[str]

@app.post("/recommend")
def recommend(user_input: UserInput):
    fav_titles = [m.strip() for m in user_input.favorite_movies if m.strip() in movies['title'].values]
    
    if not fav_titles:
        return {"error": "No valid movies found in database!"}
    
    movie_indices = movies[movies['title'].isin(fav_titles)].index
    user_profile = X_all[movie_indices].mean(axis=0)
    
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_all.toarray(), dtype=torch.float32).to(device)
        scores = model(X_tensor).cpu().numpy().flatten()
    movies['predicted_score'] = scores
    recommendations = movies[~movies['title'].isin(fav_titles)]
    
    top_k = recommendations.sort_values(by='predicted_score', ascending=False).head(10)
    
    return {"recommendations": top_k[['title','genres','predicted_score']].to_dict(orient='records')}

@app.get("/")
def read_root():
    return {"message": "Movie Recommendation API is running", "status": "healthy"}