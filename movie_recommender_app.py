import streamlit as st
from tmdbv3api import TMDb, Movie
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import concurrent.futures
from datetime import datetime
import time
import requests

# Initialize NLTK
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Initialize TMDb API
tmdb = TMDb()
tmdb.api_key = '1ab8629cf19578c0576135e9bd71bb23'
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
        return [m.title for m in movie_api.search(query)][:10]
    except:
        return []

# UI: Movie search and dropdown autocomplete
query = st.text_input("Search for a movie to add")
if query:
    suggestions = search_movies(query)
    selected = st.selectbox("Select from results", suggestions)
    if selected and selected not in st.session_state.favorites:
        if len(st.session_state.favorites) < 5:
            st.session_state.favorites.append(selected)

# Display selected movies with delete option
st.subheader("Selected Favorites:")
to_remove = []
for i, title in enumerate(st.session_state.favorites):
    col1, col2 = st.columns([0.85, 0.15])
    with col1:
        st.markdown(f"- {title}")
    with col2:
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
                with col2:
                    st.markdown(f"**{title}**\n⭐ Score: {round(info['score'], 1)}")
