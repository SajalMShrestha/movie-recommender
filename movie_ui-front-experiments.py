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
def recommend_movies(favorite_titles):
    favorite_genres, favorite_actors = set(), set()
    candidate_movie_ids, plot_moods, favorite_years = set(), set(), []

    for title in favorite_titles:
        search_result = movie.search(title)
        if not search_result:
            continue
        details = movie.details(search_result[0].id)
        credits = movie.credits(search_result[0].id)
        favorite_genres.update([g['name'] for g in details.genres])
        favorite_actors.update([c['name'] for c in list(credits['cast'])[:3]])
        favorite_actors.update([d['name'] for d in credits['crew'] if d['job'] == 'Director'])
        plot_moods.add(infer_mood_from_plot(details.overview or ""))
        if details.release_date:
            favorite_years.append(int(details.release_date[:4]))
        similar = movie.similar(details.id)
        candidate_movie_ids.update([m.id for m in similar])

    user_prefs = {
        "subscribed_platforms": ["netflix", "hbo_max", "prime_video"],
        "preferred_moods": plot_moods,
        "estimated_age": estimate_user_age(favorite_years)
    }

    candidate_movies = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_id = {executor.submit(fetch_similar_movie_details, m_id): m_id for m_id in candidate_movie_ids}
        for future in concurrent.futures.as_completed(future_to_id):
            m_id, m = future.result()
            if m and getattr(m, 'vote_count', 0) >= 20:
                candidate_movies[m_id] = m

    def compute_score(candidate):
        score = 0.0
        genre_overlap = len(set(g['name'] for g in candidate.genres) & favorite_genres)
        score += recommendation_weights["genre_similarity"] * (genre_overlap / max(len(favorite_genres), 1))
        cast = set(a.name for a in candidate.cast)
        directors = set(getattr(candidate, 'directors', []))
        overlap = (cast | directors) & favorite_actors
        score += recommendation_weights["cast_crew"] * (len(overlap) / max(len(favorite_actors), 1))
        try:
            year_diff = datetime.now().year - int(candidate.release_date[:4])
            if year_diff <= 2:
                score += recommendation_weights["release_year"] * 1.0
            elif year_diff <= 5:
                score += recommendation_weights["release_year"] * 0.66
            elif year_diff <= 15:
                score += recommendation_weights["release_year"] * 0.33
        except:
            pass
        score += recommendation_weights["ratings"] * ((candidate.vote_average or 0) / 10.0)
        score += recommendation_weights["mood_tone"] * get_mood_score(candidate.genres, user_prefs["preferred_moods"])
        score += recommendation_weights["trending_factor"] * 0.5
        score -= get_maturity_penalty(candidate.genres)
        try:
            release_year = int(candidate.release_date[:4])
            user_age_at_release = user_prefs["estimated_age"] - (datetime.now().year - release_year)
            if 15 <= user_age_at_release <= 25:
                score += recommendation_weights["age_alignment"] * 1.0
            elif 10 <= user_age_at_release < 15 or 25 < user_age_at_release <= 30:
                score += recommendation_weights["age_alignment"] * 0.5
        except:
            pass
        return max(score, 0)

    scored = [(m.title, compute_score(m) + min(m.vote_count, 1000)/20000.0) for m in candidate_movies.values()]
    top_scored = []
    low_votes = 0
    for title, score in sorted(scored, key=lambda x: x[1], reverse=True):
        m_obj = next((m for m in candidate_movies.values() if m.title == title), None)
        if m_obj:
            if m_obj.vote_count < 100:
                if low_votes >= 2:
                    continue
                low_votes += 1
            top_scored.append((title, score))
            if len(top_scored) == 10:
                break
    return top_scored, candidate_movies

# --- Display Movie Cards ---
def display_movie_card(movie_obj, index):
    title = getattr(movie_obj, 'title', 'Untitled')
    overview = getattr(movie_obj, 'overview', '') or "No description available."
    poster_path = getattr(movie_obj, 'poster_path', None)
    tmdb_link = f"https://www.themoviedb.org/movie/{getattr(movie_obj, 'id', '')}"

    words = overview.split()
    if len(words) > 50:
        short_description = " ".join(words[:50]) + "..."
        read_more_html = f"<a href='{tmdb_link}' target='_blank' style='color: #3399ff;'>Read more</a>"
    else:
        short_description = overview
        read_more_html = ""

    card_html = f"""
    <div style='display: flex; flex-direction: column; justify-content: flex-start; align-items: center; text-align: center; height: 500px; width: 100%; border: 1px solid #333; border-radius: 10px; padding: 10px; box-sizing: border-box; margin-bottom: 10%;'>
        <img src='https://image.tmdb.org/t/p/w300{poster_path}' alt='Poster' width='150' style='margin-bottom: 10px; border-radius: 4px;'>
        <h4 style='margin: 10px 0 4px;'>{index}. <a href='{tmdb_link}' target='_blank' style='text-decoration: none; color: #3399ff;'>{title}</a></h4>
        <div style='margin-top: 4px; font-size: 12px;'>
            {short_description} <br> {read_more_html}
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

# --- Streamlit App UI ---
st.title("🎬 Movie AI Recommender")
st.markdown("Enter up to 5 of your favorite movies, and we'll recommend similar ones.")

# Initialize session state
if "favorite_movies" not in st.session_state:
    st.session_state.favorite_movies = []

if "movie_input" not in st.session_state:
    st.session_state["movie_input"] = ""

new_movie = st.text_input("Search and add your favorite movie", key="movie_input")

# Automatically add movie on Enter
if new_movie:
    if new_movie not in st.session_state.favorite_movies:
        if len(st.session_state.favorite_movies) < 5:
            st.session_state.favorite_movies.append(new_movie)
        else:
            st.warning("You can only add up to 5 movies.")
    else:
        st.info("This movie is already in your favorites.")
    st.session_state["movie_input"] = ""  # Reset input field
    st.experimental_rerun()

if st.session_state.favorite_movies:
    st.subheader("🎥 Your Favorite Movies")
    for i, title in enumerate(st.session_state.favorite_movies, 1):
        st.markdown(f"{i}. {title}")

if st.button("❌ Clear All"):
    st.session_state.favorite_movies = []
    st.session_state["movie_input"] = ""
    st.experimental_rerun()

if st.button("🎬 Get Recommendations"):
    st.session_state.movie_input = ""
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
