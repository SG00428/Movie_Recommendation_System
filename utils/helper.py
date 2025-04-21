import pickle
import torch
import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from surprise import SVD
from models.ncf import NCF
from models.ensemble import FusionModel
from models.explainable import AttentionNCF

# Load encoders once
with open("utils/encoders.pkl", "rb") as f:
    user_enc, item_enc = pickle.load(f)

# Load models
svd_model = pickle.load(open("models/svd_model.pkl", "rb"))
ncf_model = NCF(len(user_enc.classes_), len(item_enc.classes_))
ncf_model.load_state_dict(torch.load("models/ncf_model.pth"))
ncf_model.eval()

fusion_model = FusionModel()
fusion_model.load_state_dict(torch.load("models/fusion_model.pth"))
fusion_model.eval()

# Load the AttentionNCF model
attention_model = AttentionNCF(len(user_enc.classes_), len(item_enc.classes_))

# Loading pre-trained weights if available (since you have this model and not the `.pth` weights, we'll skip this part)
# attention_model.load_state_dict(torch.load("models/attention_model.pth"))
attention_model.eval()

# Load enriched movie data (merged with netflix_titles.csv offline)
movie_df = pd.read_csv("data/Netflix_Movie_Enriched.csv")


def hybrid_predict(user_id, movie_id):
    try:
        uid = user_enc.transform([user_id])[0]
        iid = item_enc.transform([movie_id])[0]
    except ValueError:
        return None, None  # ID not in training set

    mf_score = svd_model.predict(uid, iid).est
    with torch.no_grad():
        ncf_score = ncf_model(torch.LongTensor([uid]), torch.LongTensor([iid])).item()
        final_score = fusion_model(torch.tensor([mf_score]), torch.tensor([ncf_score])).item()
        _, attention = attention_model(torch.LongTensor([uid]), torch.LongTensor([iid]))

    return final_score, attention.item()


def recommend_top_n(user_id, top_n=10, genre_filter="All"):
    movie_ids = movie_df['Movie_ID'].tolist()
    scored_movies = []

    for movie_id in movie_ids:
        score, attn = hybrid_predict(user_id, movie_id)
        if score is not None:
            scored_movies.append((movie_id, score, attn))

    # Deduplicate by Movie_ID
    seen = set()
    unique_scored_movies = []
    for movie_id, score, attn in scored_movies:
        if movie_id not in seen:
            seen.add(movie_id)
            unique_scored_movies.append((movie_id, score, attn))

    # Sort by predicted score
    unique_scored_movies.sort(key=lambda x: x[1], reverse=True)

    # Genre filter (if applied)
    if genre_filter != "All":
        filtered_ids = movie_df[movie_df['Genre'].str.contains(genre_filter, na=False)]['Movie_ID'].tolist()
        unique_scored_movies = [m for m in unique_scored_movies if m[0] in filtered_ids]

    return unique_scored_movies[:top_n]


def find_similar_movies(movie_name_input, top_n=5):
    movie_lookup = {row['Name']: row['Movie_ID'] for _, row in movie_df.iterrows()}
    matches = [name for name in movie_lookup if movie_name_input.lower() in name.lower()]
    if not matches:
        return []

    base_name = matches[0]
    base_id = movie_lookup[base_name]
    genre = movie_df[movie_df['Movie_ID'] == base_id]['Genre'].values[0] if 'Genre' in movie_df.columns else None
    if not genre:
        return []

    genre_movies = movie_df[movie_df['Genre'].str.contains(genre.split(",")[0], na=False, regex=False)]
    genre_movies = genre_movies[genre_movies['Movie_ID'] != base_id]

    similar = genre_movies[['Movie_ID']].head(top_n)
    return [(row['Movie_ID'], 0.8) for _, row in similar.iterrows()]


# def get_movie_metadata(movie_id):
#     row = movie_df[movie_df['Movie_ID'] == movie_id]
#     if row.empty:
#         return None
#     row = row.iloc[0]
#     return {
#         "Name": row.get("Name", "Unknown"),
#         "Genre": row.get("Genre", "Unknown"),
#         "Language": row.get("Language", "Unknown"),
#         "IMDb_Rating": row.get("IMDb_Rating", "N/A"),
#         "IMDb_Votes": row.get("IMDb_Votes", "N/A"),
#         "Description": row.get("Description", "N/A"),
#         "Year": row.get("Year", "N/A"),
#         "Runtime": row.get("Runtime", "N/A"),
#         "Tagline": row.get("Tagline", "N/A"),
#     }

import requests

TMDB_API_KEY = "8d7dff184af1bb11e57559d0cdb6fcdc"  # Replace with your actual TMDB API key

def get_movie_metadata(movie_id):
    row = movie_df[movie_df['Movie_ID'] == movie_id]
    if row.empty:
        return None
    row = row.iloc[0]

    movie_title = row.get("Name", "Unknown")
    movie_year = row.get("Year", "N/A")

    metadata = {
        "Name": movie_title,
        "Genre": row.get("Genre", "Unknown"),
        "Language": row.get("Language", "Unknown"),
        "IMDb_Rating": row.get("IMDb_Rating", "N/A"),
        "IMDb_Votes": row.get("IMDb_Votes", "N/A"),
        "Description": row.get("Description", "N/A"),
        "Year": movie_year,
        "Runtime": row.get("Runtime", "N/A"),
        "Tagline": row.get("Tagline", "N/A"),
        "popularity": row.get("popularity", "N/A"),
        "Poster_URL": None
    }

    # Fetch poster URL from TMDB
    try:
        search_url = "https://api.themoviedb.org/3/search/movie"
        params = {
            "api_key": TMDB_API_KEY,
            "query": movie_title,
            "year": movie_year
        }
        response = requests.get(search_url, params=params)
        if response.status_code == 200 and response.json().get("results"):
            tmdb_data = response.json()["results"][0]
            poster_path = tmdb_data.get("poster_path")
            if poster_path:
                metadata["Poster_URL"] = f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception as e:
        print(f"Error fetching poster for '{movie_title}': {e}")

    return metadata



def get_attention_score(user_id, movie_id):
    try:
        uid = user_enc.transform([user_id])[0]
        iid = item_enc.transform([movie_id])[0]
    except ValueError:
        return None  # ID not in training set

    with torch.no_grad():
        _, attention = attention_model(torch.LongTensor([uid]), torch.LongTensor([iid]))
        return attention.item()

import re

def get_new_user_recommendations(selected_movie_ids, movie_df, top_k_per_genre=3):
    import ast
    genre_to_movies = {}

    # Collect genres and language from selected movies
    user_genres = set()
    user_languages = set()

    for movie_id in selected_movie_ids:
        row = movie_df[movie_df['Movie_ID'] == movie_id]
        if not row.empty:
            genres = row.iloc[0]['Genre']
            language = row.iloc[0].get('Language', None)
            
            # Parse genres if in list/dict string form
            try:
                parsed_genres = ast.literal_eval(genres) if isinstance(genres, str) else genres
                for g in parsed_genres:
                    if isinstance(g, dict) and 'name' in g:
                        user_genres.add(g['name'])
                    elif isinstance(g, str):
                        user_genres.add(g.strip())
            except (ValueError, SyntaxError, TypeError):
                pass

            if pd.notna(language):
                user_languages.add(language.strip())

    # Now for each genre, recommend movies of that genre and preferred languages
    recommendations = {}
    for genre in user_genres:
        filtered = movie_df[
            movie_df['Genre'].str.contains(genre, case=False, na=False) &
            movie_df['Language'].isin(user_languages)
        ].drop_duplicates('Movie_ID')

        rec_ids = filtered[~filtered['Movie_ID'].isin(selected_movie_ids)].head(top_k_per_genre)['Movie_ID'].tolist()
        if rec_ids:
            recommendations[genre] = rec_ids

    return recommendations



