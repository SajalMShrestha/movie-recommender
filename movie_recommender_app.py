import streamlit as st
from tmdbv3api import TMDb, Movie
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import requests

# NLTK Sentiment Setup
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# TMDb Setup
tmdb = TMDb()
tmdb.api_key = st.secrets["TMDB_API_KEY"]
tmdb.language = 'en'
movie_api = Movie()

st.set_page_config(page_title="Movie Recommender", layout="wide")

# --- App State ---
if 'favorites' not in st.session_state:
    st.session_state.favorites = []
if 'search' not in st.session_state:
    st.session_state['search'] = ""

# --- Helper: Get Poster URL ---
def get_poster(movie):
    return f"https://image.tmdb.org/t/p/w200{movie.poster_path}" if movie.poster_path else "https://via.placeholder.com/120x180?text=No+Image"

# --- UI Title ---
st.title("🎬 Movie Recommender Tool (v1)")
st.markdown("Search for your favorite movies to get personalized recommendations.")

# --- Search Input ---
query = st.text_input("Search for your favorite movie…", key="search")
if query:
    results = movie_api.search(query)
    suggestion_titles = [m.title for m in results if m.title not in st.session_state.favorites]
    if suggestion_titles:
        selected = st.selectbox("Did you mean?", suggestion_titles)
        if selected:
            if selected not in st.session_state.favorites and len(st.session_state.favorites) < 5:
                st.session_state.favorites.append(selected)
                st.session_state["search"] = ""
                st.rerun()

# --- Favorites Section ---
if st.session_state.favorites:
    st.subheader("⭐ Your Favorite Movies")
    fav_cols = st.columns(min(len(st.session_state.favorites), 5))
    for idx, title in enumerate(st.session_state.favorites):
        try:
            movie = movie_api.search(title)[0]
            poster = get_poster(movie)
        except:
            poster = "https://via.placeholder.com/120x180?text=No+Image"
        with fav_cols[idx % 5]:
            st.image(poster, width=120)
            st.caption(title)

# --- Recommendations Section ---
if len(st.session_state.favorites) == 5:
    st.subheader("🎯 Your Recommendations")
    recommended = {}
    for fav in st.session_state.favorites:
        try:
            results = movie_api.search(fav)
            if results:
                movie_id = results[0].id
                similar = movie_api.similar(movie_id)
                for m in similar:
                    if m.title not in st.session_state.favorites and m.title not in recommended:
                        recommended[m.title] = {
                            'score': m.vote_average or 0,
                            'poster': get_poster(m)
                        }
        except:
            continue
    sorted_recs = sorted(recommended.items(), key=lambda x: x[1]['score'], reverse=True)[:10]
    rec_cols = st.columns(5)
    for i, (title, info) in enumerate(sorted_recs):
        with rec_cols[i % 5]:
            st.image(info['poster'], width=120)
            st.caption(f"{title}")
