# Streamlit Movie AI Recommender
import streamlit as st
from tmdbv3api import TMDb, Movie, Person
from collections import defaultdict
from datetime import datetime
import concurrent.futures
import time
import requests
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# TMDb Setup
tmdb = TMDb()
tmdb.api_key = st.secrets["TMDB_API_KEY"]
tmdb.language = 'en'
tmdb.debug = True
movie = Movie()
person = Person()
sia = SentimentIntensityAnalyzer()

# Recommendation weights and platform priorities
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

def estimate_user_age(favorite_movies_years):
    if not favorite_movies_years:
        return 30
    median_year = sorted(favorite_movies_years)[len(favorite_movies_years)//2]
    return datetime.now().year - median_year + 18

def compute_score(candidate, favorite_genres, favorite_actors, user_preferences):
    score = 0.0
    candidate_genres = set(g['name'] for g in candidate.genres)
    genre_overlap = len(candidate_genres & favorite_genres)
    genre_score = genre_overlap / len(favorite_genres or [1])
    score += recommendation_weights["genre_similarity"] * genre_score

    candidate_cast = set(a.name for a in candidate.cast)
    candidate_directors = set(getattr(candidate, 'directors', []))
    total_people = favorite_actors or set()
    overlap = (candidate_cast | candidate_directors) & total_people
    actor_score = len(overlap) / len(total_people or [1])
    score += recommendation_weights["cast_crew"] * actor_score

    current_year = datetime.now().year
    try:
        year = int(candidate.release_date[:4]) if candidate.release_date else 0
        year_diff = current_year - year
        if year_diff <= 2:
            year_score = 1.0
        elif year_diff <= 5:
            year_score = 0.66
        elif year_diff <= 15:
            year_score = 0.33
        else:
            year_score = 0.0
    except:
        year_score = 0.0
    score += recommendation_weights["release_year"] * year_score

    rating_score = (candidate.vote_average or 0) / 10.0
    score += recommendation_weights["ratings"] * rating_score

    platform = getattr(candidate, 'platform', None)
    if platform in user_preferences["subscribed_platforms"]:
        availability_score = streaming_platform_priority.get(platform, 0.5)
    else:
        availability_score = 0.0
    score += recommendation_weights["streaming_availability"] * availability_score

    mood_score = get_mood_score(candidate.genres, user_preferences.get("preferred_moods", set()))
    score += recommendation_weights["mood_tone"] * mood_score

    trending_score = 0.5
    score += recommendation_weights["trending_factor"] * trending_score

    maturity_penalty = get_maturity_penalty(candidate.genres)
    score -= maturity_penalty

    try:
        user_age = user_preferences.get("estimated_age", 30)
        release_year = int(candidate.release_date[:4]) if candidate.release_date else None
        if release_year:
            user_age_at_release = user_age - (datetime.now().year - release_year)
            if 15 <= user_age_at_release <= 25:
                age_score = 1.0
            elif 10 <= user_age_at_release < 15 or 25 < user_age_at_release <= 30:
                age_score = 0.5
            else:
                age_score = 0.0
        else:
            age_score = 0.0
    except:
        age_score = 0.0
    score += recommendation_weights["age_alignment"] * age_score
    return max(score, 0)

cache = {}
def fetch_similar_movie_details(m_id):
    if m_id in cache:
        return m_id, cache[m_id]
    try:
        m_details = movie.details(m_id)
        m_credits = movie.credits(m_id)
        m_details.genres = m_details.genres
        m_details.cast = list(m_credits['cast'])[:3]
        m_details.directors = [d['name'] for d in m_credits['crew'] if d['job'] == 'Director'] if 'crew' in m_credits else []
        try:
            tmdb_id = m_details.id
            headers = {
                'x-rapidapi-key': st.secrets["RAPIDAPI_KEY"],
                'x-rapidapi-host': 'streaming-availability.p.rapidapi.com'
            }
            response = requests.get(
                f"https://streaming-availability.p.rapidapi.com/get/basic?tmdb_id=movie%2F{tmdb_id}&output_language=en",
                headers=headers
            )
            if response.status_code == 200:
                data = response.json()
                known_platforms = list(streaming_platform_priority.keys())
                platforms = data.get("streamingInfo", {}).get("us", {})
                selected_platform = None
                for p in platforms:
                    if p.lower() in known_platforms:
                        selected_platform = p.lower()
                        break
                m_details.platform = selected_platform
            else:
                m_details.platform = None
        except:
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
    favorite_years = []

    for title in favorite_titles:
        search_result = movie.search(title)
        if not search_result:
            continue
        movie_details = movie.details(search_result[0].id)
        movie_credits = movie.credits(search_result[0].id)
        favorite_genres.update([g['name'] for g in movie_details.genres])
        favorite_actors.update([c['name'] for c in list(movie_credits['cast'])[:3]])
        if 'crew' in movie_credits:
            favorite_actors.update([d['name'] for d in movie_credits['crew'] if d['job'] == 'Director'])
        inferred_plot_moods.add(infer_mood_from_plot(movie_details.overview or ""))
        if movie_details.release_date:
            favorite_years.append(int(movie_details.release_date[:4]))
        similar = movie.similar(movie_details.id)
        candidate_movie_ids.update([m.id for m in list(similar)])

    inferred_moods = infer_moods_from_genres(favorite_genres) | inferred_plot_moods
    estimated_age = estimate_user_age(favorite_years)
    user_preferences = {
        "subscribed_platforms": ["netflix", "hbo_max", "prime_video"],
        "preferred_moods": inferred_moods,
        "estimated_age": estimated_age
    }

    candidate_movies = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_id = {executor.submit(fetch_similar_movie_details, m_id): m_id for m_id in candidate_movie_ids}
        for future in concurrent.futures.as_completed(future_to_id):
            m_id, m = future.result()
            if m and getattr(m, 'vote_count', 0) >= 20:
                candidate_movies[m_id] = m

    scored = [(m.title, compute_score(m, favorite_genres, favorite_actors, user_preferences) + min(getattr(m, 'vote_count', 0), 1000)/20000.0) for m in candidate_movies.values()]
    top_scored = []
    low_vote_count = 0
    for title, score in sorted(scored, key=lambda x: x[1], reverse=True):
        movie_obj = next((m for m in candidate_movies.values() if m.title == title), None)
        if movie_obj:
            votes = getattr(movie_obj, 'vote_count', 0)
            if votes < 100:
                if low_vote_count >= 2:
                    continue
                low_vote_count += 1
            top_scored.append((title, score))
            if len(top_scored) == 10:
                break
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
