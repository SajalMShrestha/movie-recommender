import streamlit as st
from tmdbv3api import TMDb, Movie
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from datetime import datetime
import concurrent.futures
import time

# Setup NLTK and Sentiment
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# TMDb Setup
tmdb = TMDb()
tmdb.api_key = st.secrets["TMDB_API_KEY"]
tmdb.language = 'en'
movie = Movie()

# Recommendation Weights
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

def get_maturity_penalty(genres):
    return 0.15 if any(g['name'] in immature_genres for g in genres) else 0

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
    score = 0.2  # base similarity boost
    genre_match = len(set([g['name'] for g in candidate.genres]) & favorite_genres) / len(favorite_genres or [1])
    score += recommendation_weights["genre_similarity"] * genre_match

    actor_match = len(set([a.name for a in list(candidate.cast)[:3]]) & favorite_actors) / len(favorite_actors or [1])
    score += recommendation_weights["cast_crew"] * actor_match

    year = int(candidate.release_date[:4]) if candidate.release_date else 0
    current_year = datetime.now().year
    if current_year - year <= 2:
        score += recommendation_weights["release_year"] * 0.53
    elif current_year - year <= 5:
        score += recommendation_weights["release_year"] * 0.33
    elif current_year - year <= 15:
        score += recommendation_weights["release_year"] * 0.13

    score += recommendation_weights["ratings"] * (candidate.vote_average or 0) / 10

    platform = getattr(candidate, 'platform', None)
    if platform in user_preferences["subscribed_platforms"]:
        boost = streaming_platform_priority.get(platform, 0.5)
        score += recommendation_weights["streaming_availability"] * boost

    mood_score = get_mood_score(candidate.genres, user_preferences.get("preferred_moods", set()))
    score += recommendation_weights["mood_tone"] * mood_score

    score -= get_maturity_penalty(candidate.genres)
    return score

cache = {}
def fetch_similar_movie_details(m_id):
    if m_id in cache:
        return m_id, cache[m_id]
    try:
        m_details = movie.details(m_id)
        m_credits = movie.credits(m_id)
        m_details.genres = m_details.genres
        m_details.cast = list(m_credits['cast'])[:3]
        m_details.platform = None
        m_details.plot = m_details.overview or ""
        cache[m_id] = m_details
        return m_id, m_details
    except:
        return m_id, None

def recommend_movies(favorite_titles):
    favorite_genres = set()
    favorite_actors = set()
    candidate_movie_ids = set()
    inferred_plot_moods = set()

    for title in favorite_titles:
        search_result = movie.search(title)
        if not search_result:
            continue
        movie_details = movie.details(search_result[0].id)
        movie_credits = movie.credits(search_result[0].id)
        favorite_genres.update([g['name'] for g in movie_details.genres])
        favorite_actors.update([c['name'] for c in list(movie_credits['cast'])[:3]])
        inferred_plot_moods.add(infer_mood_from_plot(movie_details.overview or ""))
        similar = movie.similar(movie_details.id)
        candidate_movie_ids.update([m.id for m in list(similar)])

    inferred_moods = infer_moods_from_genres(favorite_genres) | inferred_plot_moods
    user_preferences = {
        "subscribed_platforms": ["netflix", "hbo_max", "prime_video"],
        "preferred_moods": inferred_moods
    }

    candidate_movies = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_id = {executor.submit(fetch_similar_movie_details, m_id): m_id for m_id in candidate_movie_ids}
        for future in concurrent.futures.as_completed(future_to_id):
            m_id, m = future.result()
            if m:
                candidate_movies[m_id] = m

    scored = [(m.title, compute_score(m, favorite_genres, favorite_actors, user_preferences)) for m in candidate_movies.values()]
    top_scored = sorted(scored, key=lambda x: x[1], reverse=True)[:10]

    return top_scored

# Streamlit UI
st.title("🎬 Movie AI Recommender")
st.write("Enter up to 5 of your favorite movies, and we'll recommend similar ones.")

movie_inputs = []
for i in range(5):
    movie_inputs.append(st.text_input(f"Favorite Movie {i+1}"))

if st.button("Get Recommendations"):
    valid_movies = [m for m in movie_inputs if m.strip() != ""]
    if len(valid_movies) < 3:
        st.warning("Please enter at least 3 movies.")
    else:
        with st.spinner("Finding recommendations..."):
            recs = recommend_movies(valid_movies)
            st.subheader("🎯 Top 10 Recommendations")
            for title, score in recs:
                st.markdown(f"- **{title}** (score: {round(score, 2)})")
