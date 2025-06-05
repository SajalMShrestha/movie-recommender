# Install necessary library



# Import tools
from tmdbv3api import TMDb, Movie, Person
from collections import defaultdict
from datetime import datetime
import concurrent.futures
import time
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# Setup TMDb API
tmdb = TMDb()
tmdb.api_key = '1ab8629cf19578c0576135e9bd71bb23'

movie = Movie()
person = Person()
sia = SentimentIntensityAnalyzer()

# Updated recommendation weights and platform priorities
recommendation_weights = {
    "genre_similarity": 0.20,
    "cast_crew": 0.25,
    "release_year": 0.15,
    "ratings": 0.20,
    "streaming_availability": 0.10,
    "mood_tone": 0.05,
    "trending_factor": 0.05
}

streaming_platform_priority = {
    "netflix": 1.0,
    "disney_plus": 0.9,
    "hbo_max": 0.85,
    "hulu": 0.8,
    "prime_video": 0.75,
    "apple_tv": 0.7,
    "peacock": 0.6,
    "paramount_plus": 0.5,
}

mood_tone_map = {
    "feel_good": {"Comedy", "Romance", "Music", "Adventure"},
    "gritty": {"Crime", "Thriller", "Mystery", "Drama"},
    "cerebral": {"Sci-Fi", "Mystery", "History"},
    "intense": {"Action", "War", "Horror"},
    "melancholic": {"Drama", "History"},
    "classic": {"Western", "Film-Noir"}
}

# Maturity filter
immature_genres = {"Family", "Animation", "Kids"}

def get_maturity_penalty(genres):
    return 0.15 if any(g['name'] in immature_genres for g in genres) else 0

# Function: Compute recommendation score

def get_mood_score(genres, preferred_moods):
    matched_moods = set()
    for g in genres:
        for mood, tags in mood_tone_map.items():
            if g['name'] in tags:
                matched_moods.add(mood)
    overlap = matched_moods & preferred_moods
    return len(overlap) / len(preferred_moods or [1])

def infer_moods_from_genres(genres):
    inferred = set()
    for g in genres:
        for mood, tags in mood_tone_map.items():
            if g in tags:
                inferred.add(mood)
    return inferred

def infer_mood_from_plot(plot):
    sentiment = sia.polarity_scores(plot)
    if sentiment['compound'] >= 0.4:
        return 'feel_good'
    elif sentiment['compound'] <= -0.3:
        return 'melancholic'
    else:
        return 'cerebral'

def compute_score(candidate, favorite_genres, favorite_actors, user_preferences):
    score = 0
    base_similarity_boost = 0.2
    score += base_similarity_boost

    # Genre match (primary and secondary)
    genre_match = len(set([g['name'] for g in candidate.genres]) & favorite_genres) / len(favorite_genres or [1])
    score += recommendation_weights["genre_similarity"] * genre_match

    # Cast/Crew match
    actor_match = len(set([a.name for a in list(candidate.cast)[:3]]) & favorite_actors) / len(favorite_actors or [1])
    score += recommendation_weights["cast_crew"] * actor_match

    # Year of Release (rescaled to 0-1)
    year = int(candidate.release_date[:4]) if candidate.release_date else 0
    current_year = datetime.now().year
    if current_year - year <= 2:
        score += recommendation_weights["release_year"] * 0.53
    elif current_year - year <= 5:
        score += recommendation_weights["release_year"] * 0.33
    elif current_year - year <= 15:
        score += recommendation_weights["release_year"] * 0.13

    # Ratings (IMDB proxy only for now)
    score += recommendation_weights["ratings"] * (candidate.vote_average or 0) / 10

    # Streaming availability boost
    platform = getattr(candidate, 'platform', None)
    if platform in user_preferences["subscribed_platforms"]:
        boost = streaming_platform_priority.get(platform, 0.5)
        score += recommendation_weights["streaming_availability"] * boost

    # Mood tone match
    mood_score = get_mood_score(candidate.genres, user_preferences.get("preferred_moods", set()))
    score += recommendation_weights["mood_tone"] * mood_score

    # Maturity level adjustment
    score -= get_maturity_penalty(candidate.genres)

    return score

# Function to safely fetch and enrich similar movie data with basic logging
cache = {}
def fetch_similar_movie_details(m_id):
    if m_id in cache:
        return m_id, cache[m_id]
    try:
        m_details = movie.details(m_id)
        m_credits = movie.credits(m_id)
        m_details.genres = m_details.genres
        m_details.cast = list(m_credits['cast'])[:3]
        m_details.platform = None  # Placeholder
        m_details.plot = m_details.overview or ""
        cache[m_id] = m_details
        print(f"Fetched: {m_details.title}")
        return m_id, m_details
    except Exception as e:
        print(f"Failed to fetch movie ID {m_id}: {e}")
        return m_id, None

# Function: Recommend similar movies based on weighted scoring
def recommend_movies(favorite_titles):
    start_time = time.time()
    favorite_genres = set()
    favorite_actors = set()
    candidate_movie_ids = set()
    inferred_plot_moods = set()

    for title in favorite_titles:
        search_result = movie.search(title)
        if not search_result:
            print(f"Movie '{title}' not found. Skipping.")
            continue

        movie_details = movie.details(search_result[0].id)
        movie_credits = movie.credits(search_result[0].id)

        # Collect genres and top 3 cast members
        favorite_genres.update([g['name'] for g in movie_details.genres])
        favorite_actors.update([c['name'] for c in list(movie_credits['cast'])[:3]])

        # Infer mood from plot description
        inferred_plot_moods.add(infer_mood_from_plot(movie_details.overview or ""))

        # Collect similar movie IDs
        similar = movie.similar(movie_details.id)
        candidate_movie_ids.update([m.id for m in list(similar)])

    inferred_moods = infer_moods_from_genres(favorite_genres) | inferred_plot_moods
    user_preferences = {
        "subscribed_platforms": ["netflix", "hbo_max", "prime_video"],
        "preferred_moods": inferred_moods
    }

    # Use concurrent futures to fetch movie details in parallel
    candidate_movies = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_id = {executor.submit(fetch_similar_movie_details, m_id): m_id for m_id in candidate_movie_ids}
        for future in concurrent.futures.as_completed(future_to_id):
            m_id, m = future.result()
            if m:
                candidate_movies[m_id] = m

    # Score each candidate movie
    scored = [(m.title, compute_score(m, favorite_genres, favorite_actors, user_preferences)) for m in candidate_movies.values()]
    top_scored = sorted(scored, key=lambda x: x[1], reverse=True)[:10]

    print("\nTop 10 Recommendations Based on Your Favorites:")
    for title, score in top_scored:
        print(f"- {title} (score: {round(score, 2)})")

    print(f"\nTotal execution time: {round(time.time() - start_time, 2)} seconds")

import streamlit as st

st.title("🎬 Movie AI Recommender")
st.write("Enter up to 5 of your favorite movies, and we'll recommend similar ones.")

# User input
movie_1 = st.text_input("Favorite Movie 1")
movie_2 = st.text_input("Favorite Movie 2")
movie_3 = st.text_input("Favorite Movie 3")
movie_4 = st.text_input("Favorite Movie 4")
movie_5 = st.text_input("Favorite Movie 5")

if st.button("Get Recommendations"):
    favorites = [m for m in [movie_1, movie_2, movie_3, movie_4, movie_5] if m]
    
    if not favorites:
        st.warning("Please enter at least one movie.")
    else:
        st.write("🔍 Finding recommendations based on your favorites...")
        with st.spinner("Crunching data..."):
            recommendations = []

            def capture_output():
                # Run and capture print output
                from io import StringIO
                import sys
                buffer = StringIO()
                sys.stdout = buffer
                recommend_movies(favorites)
                sys.stdout = sys.__stdout__
                return buffer.getvalue()

            output = capture_output()
            st.text_area("📋 Recommendation Log", output, height=300)

favorite_movies = [
    "The Bourne Identity",
    "Knocked Up",
    "Manchester by the Sea",
    "Gone Girl",
    "Miami Vice"
]

recommend_movies(favorite_movies)
