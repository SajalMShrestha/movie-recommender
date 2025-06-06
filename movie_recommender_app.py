import streamlit as st
from tmdbv3api import TMDb, Movie
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import concurrent.futures
from datetime import datetime
import time
import requests
import os

# Initialize NLTK
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Initialize TMDb API
tmdb = TMDb()
tmdb.api_key = st.secrets["TMDB_API_KEY"]
tmdb.language = 'en'
movie_api = Movie()

# Initialize app state
if 'favorites' not in st.session_state:
    st.session_state.favorites = []

# UI: Title and instructions
st.title("🎬 Movie AI Recommender")
st.write("Enter up to 5 of your favorite movies, and we'll recommend similar ones.")

# Collect up to 5 favorite movies
favorite_inputs = []
for i in range(5):
    fav = st.text_input(f"Favorite Movie {i+1}", value="" if i >= len(st.session_state.favorites) else st.session_state.favorites[i], key=f"fav_{i}")
    favorite_inputs.append(fav)

# Update state
st.session_state.favorites = [f for f in favorite_inputs if f.strip() != ""]

# Helper: Fetch poster URL
def get_poster_url(movie):
    base_url = "https://image.tmdb.org/t/p/w200"
    return base_url + movie.poster_path if movie.poster_path else ""

# Submit and show recommendations
if st.button("Get Recommendations"):
    favorites = st.session_state.favorites
    if len(favorites) < 3:
        st.warning("Please enter at least 3 movies before submitting.")
    else:
        with st.spinner("Finding recommendations based on your favorites..."):
            recommended = {}
            for title in favorites:
                try:
                    results = movie_api.search(title)
                    if results:
                        movie_id = results[0].id
                        similar = movie_api.similar(movie_id)
                        for m in similar:
                            if m.title not in favorites and m.title not in recommended:
                                recommended[m.title] = {
                                    'score': m.vote_average or 0,
                                    'poster': get_poster_url(m)
                                }
                except:
                    continue

            sorted_recs = sorted(recommended.items(), key=lambda x: x[1]['score'], reverse=True)[:10]

            st.subheader("Top 10 Recommendations Based on Your Favorites:")
            for title, info in sorted_recs:
                col1, col2 = st.columns([0.3, 0.7])
                with col1:
                    if info['poster']:
                        st.image(info['poster'], width=120)
                with col2:
                    st.markdown(f"**{title}** (score: {round(info['score'], 2)})")
