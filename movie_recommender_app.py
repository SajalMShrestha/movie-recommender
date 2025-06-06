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
st.title("🎬 Movie AI Recommender V2")
st.write("Search and select up to 5 of your favorite movies. You must choose at least 3.")

# Function to search TMDb movies
def search_movies(query):
    try:
        return movie_api.search(query)[:10]
    except:
        return []

# Enhanced UI: Autocomplete dropdown using st.selectbox only
all_suggestions = []
query = st.text_input("Type to search and select a movie")
if query:
    suggestions = search_movies(query)
    all_suggestions = [m.title for m in suggestions if m.title not in st.session_state.favorites]

if all_suggestions:
    selected = st.selectbox("Click a title to add to favorites", all_suggestions)
    if selected and selected not in st.session_state.favorites:
        if len(st.session_state.favorites) < 5:
            st.session_state.favorites.append(selected)
            st.experimental_rerun()

# Display selected movies with metadata and delete option
st.subheader("Selected Favorites:")
to_remove = []
for i, title in enumerate(st.session_state.favorites):
    try:
        result = movie_api.search(title)[0]
        poster_url = f"https://image.tmdb.org/t/p/w200{result.poster_path}" if result.poster_path else ""
        year = result.release_date.split("-")[0] if result.release_date else "Unknown"
        rating = result.vote_average or "N/A"
        cast_info = movie_api.credits(result.id)['cast'][:3]
        cast_names = ", ".join([c['name'] for c in cast_info]) or "Unknown"
    except:
        poster_url = ""
        year = "Unknown"
        rating = "N/A"
        cast_names = "Unknown"

    col1, col2, col3 = st.columns([0.2, 0.7, 0.1])
    with col1:
        if poster_url:
            st.image(poster_url, width=80)
    with col2:
        st.markdown(f"**{title}** ({year})\n⭐ {rating} | 🎭 {cast_names}")
    with col3:
        if st.button("❌", key=f"remove_{i}"):
            to_remove.append(title)

for title in to_remove:
    st.session_state.favorites.remove(title)

# Helper: Fetch poster URL
def get_poster_url(movie):
    base_url = "https://image.tmdb.org/t/p/w200"
    return base_url + movie.poster_path if movie.poster_path else ""

# Submit and show recommendations
if st.button("🎯 Submit"):
    favorites = st.session_state.favorites
    if len(favorites) < 3:
        st.warning("Please select at least 3 movies before submitting.")
    else:
        with st.spinner("Finding recommendations..."):
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

            st.subheader("🎥 Top 10 Recommendations:")
            for title, info in sorted_recs:
                col1, col2 = st.columns([0.3, 0.7])
                with col1:
                    if info['poster']:
                        st.image(info['poster'], width=120)
                    else:
                        st.markdown("![No Poster](https://via.placeholder.com/120x180?text=No+Image)")
                with col2:
                    st.markdown(f"**{title}**\n⭐ Score: {round(info['score'], 1)}")
