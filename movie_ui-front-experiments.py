import streamlit as st
from tmdbv3api import TMDb, Movie
from datetime import datetime
import concurrent.futures
import requests
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

# TMDb setup
tmdb = TMDb()
tmdb.api_key = st.secrets["TMDB_API_KEY"]
tmdb.language = 'en'
tmdb.debug = True
movie = Movie()
sia = SentimentIntensityAnalyzer()

# --- Recommendation weights and platform priorities ---
recommendation_weights = {
    "genre_similarity": 0.20,
    "cast_crew": 0.25,
    "release_year": 0.15,
    "ratings": 0.20,
    "streaming_availability": 0.10,
    "mood_tone": 0.05,
    "trending_factor": 0.05,
    "age_alignment": 0.05
}

streaming_platform_priority = {
    "netflix": 1.0,
    "disney_plus": 0.9,
    "hbo_max": 0.85,
    "hulu": 0.8,
    "prime_video": 0.75,
    "apple_tv": 0.7,
    "peacock": 0.6,
    "paramount_plus": 0.5
}

mood_tone_map = {
    "feel_good": {"Comedy", "Romance", "Music", "Adventure"},
    "gritty": {"Crime", "Thriller", "Mystery", "Drama"},
    "cerebral": {"Sci-Fi", "Mystery", "History"},
    "intense": {"Action", "War", "Horror"},
    "melancholic": {"Drama", "History"},
    "classic": {"Western", "Film-Noir"}
}

immature_genres = {"Family", "Animation", "Kids"}

# --- Feature Functions ---
def get_maturity_penalty(genres):
    return 0.15 if any(g['name'] in immature_genres for g in genres) else 0

def get_mood_score(genres, preferred_moods):
    matched_moods = set()
    for g in genres:
        for mood, tags in mood_tone_map.items():
            if g['name'] in tags:
                matched_moods.add(mood)
    overlap = matched_moods & preferred_moods
    return len(overlap) / max(len(preferred_moods), 1)

def infer_mood_from_plot(plot):
    sentiment = sia.polarity_scores(plot)
    if sentiment['compound'] >= 0.4:
        return 'feel_good'
    elif sentiment['compound'] <= -0.3:
        return 'melancholic'
    else:
        return 'cerebral'

def estimate_user_age(years):
    if not years:
        return 30
    median = sorted(years)[len(years)//2]
    return datetime.now().year - median + 18

# --- Caching and Movie Details Fetch ---
cache = {}
def fetch_similar_movie_details(m_id):
    if m_id in cache:
        return m_id, cache[m_id]
    try:
        m_details = movie.details(m_id)
        m_credits = movie.credits(m_id)
        m_details.genres = m_details.genres
        m_details.cast = list(m_credits['cast'])[:3]
        m_details.directors = [d['name'] for d in m_credits['crew'] if d['job'] == 'Director']
        m_details.plot = m_details.overview or ""
        cache[m_id] = m_details
        return m_id, m_details
    except:
        return m_id, None

# --- Recommendation Logic ---
# [Unchanged — omitted for brevity, but is present in your code]

# --- Display Movie Cards ---
# [Unchanged — omitted for brevity, but is present in your code]

# --- Streamlit App UI ---
st.title("🎬 Movie AI Recommender")
st.markdown("Enter up to 5 of your favorite movies, and we'll recommend similar ones.")

# Initialize session state
if "favorite_movies" not in st.session_state:
    st.session_state.favorite_movies = []

new_movie = st.text_input("Search and add your favorite movie")

add_movie_clicked = st.button("➕ Add Movie")

if add_movie_clicked:
    if new_movie:
        if new_movie in st.session_state.favorite_movies:
            st.info("This movie is already in your favorites.")
        elif len(st.session_state.favorite_movies) >= 5:
            st.warning("You can only add up to 5 movies.")
        else:
            st.session_state.favorite_movies.append(new_movie)
            st.success(f"Added: {new_movie}")

if st.session_state.favorite_movies:
    st.subheader("🎥 Your Favorite Movies")
    for i, title in enumerate(st.session_state.favorite_movies, 1):
        st.markdown(f"{i}. {title}")

if st.button("❌ Clear All"):
    st.session_state.favorite_movies = []
    st.experimental_rerun()

if st.button("🎬 Get Recommendations"):
    if len(st.session_state.favorite_movies) < 3:
        st.warning("Please add at least 3 movies to get recommendations.")
    else:
        with st.spinner("Finding recommendations..."):
            recs, candidate_movies = recommend_movies(st.session_state.favorite_movies)
            st.subheader("🎯 Top 10 Recommendations")
            cols = st.columns(4)
            for idx, (title, _) in enumerate(recs):
                movie_obj = next((m for m in candidate_movies.values() if m.title == title), None)
                if movie_obj:
                    with cols[idx % 4]:
                        display_movie_card(movie_obj, idx + 1)
