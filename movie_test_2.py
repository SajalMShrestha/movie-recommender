import streamlit as st
from tmdbv3api import TMDb, Movie
from datetime import datetime
import concurrent.futures
import requests
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import time
import json
import os

nltk.download('vader_lexicon')

# TMDb setup
tmdb = TMDb()
tmdb.api_key = st.secrets["TMDB_API_KEY"]
st.write("TMDB API Key loaded:", tmdb.api_key)  # Debug print

tmdb.language = 'en'
tmdb.debug = True
movie = Movie()
sia = SentimentIntensityAnalyzer()

# Session persistence via file storage
SESSION_FILE = "session_state.json"
def load_session():
    if os.path.exists(SESSION_FILE):
        with open(SESSION_FILE, "r") as f:
            return json.load(f)
    return {}

def save_session(session_data):
    with open(SESSION_FILE, "w") as f:
        json.dump(session_data, f)

# Restore session state
saved_state = load_session()
if "favorite_movies" not in st.session_state:
    st.session_state.favorite_movies = saved_state.get("favorite_movies", [])
if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = None

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

    # Gather user preferences
    for title in favorite_titles:
        search_result = movie.search(title)
        if not search_result:
            continue
        details = movie.details(search_result[0].id)
        credits = movie.credits(search_result[0].id)
        favorite_genres.update([g['name'] for g in details.genres])
        favorite_actors.update([c['name'] for c in list(credits['cast'])[:3]])
        favorite_actors.update([d['name'] for d in credits['crew'] if d['job']=='Director'])
        plot_moods.add(infer_mood_from_plot(details.overview or ""))
        if details.release_date:
            favorite_years.append(int(details.release_date[:4]))
        # Similar movies
        similar_list = movie.similar(details.id)
        candidate_movie_ids.update([m.id for m in similar_list])

    user_prefs = {
        "subscribed_platforms": [k for k,v in streaming_platform_priority.items() if v>0],
        "preferred_moods": plot_moods,
        "estimated_age": estimate_user_age(favorite_years)
    }

    # Fetch details in parallel
    candidate_movies = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_similar_movie_details, mid): mid for mid in candidate_movie_ids}
        for fut in concurrent.futures.as_completed(futures):
            mid, m = fut.result()
            if m and getattr(m,'vote_count',0)>=20:
                candidate_movies[mid] = m

    # Scoring function
    def compute_score(m):
        score = 0.0
        genres = {g['name'] for g in m.genres}
        score += recommendation_weights['genre_similarity'] * (len(genres & favorite_genres)/max(len(favorite_genres),1))
        cast_dir = set(a.name for a in m.cast) | set(getattr(m,'directors',[]))
        score += recommendation_weights['cast_crew'] * (len(cast_dir & favorite_actors)/max(len(favorite_actors),1))
        # Year recency
        try:
            year_diff = datetime.now().year - int(m.release_date[:4])
            if year_diff<=2: score += recommendation_weights['release_year']
            elif year_diff<=5: score += recommendation_weights['release_year']*0.66
            elif year_diff<=15: score += recommendation_weights['release_year']*0.33
        except: pass
        score += recommendation_weights['ratings'] * ((m.vote_average or 0)/10)
        score += recommendation_weights['mood_tone'] * get_mood_score(m.genres, user_prefs['preferred_moods'])
        score += recommendation_weights['trending_factor'] * 0.5
        score -= get_maturity_penalty(m.genres)
        # Age alignment
        try:
            release_year=int(m.release_date[:4])
            user_age_at_release = user_prefs['estimated_age'] - (datetime.now().year - release_year)
            if 15<=user_age_at_release<=25: score += recommendation_weights['age_alignment']
            elif 10<=user_age_at_release<15 or 25<user_age_at_release<=30: score += recommendation_weights['age_alignment']*0.5
        except: pass
        return max(score,0)

    # Score and select top 10
    scored = [(m, compute_score(m) + min(m.vote_count,1000)/20000) for m in candidate_movies.values()]
    scored.sort(key=lambda x:x[1], reverse=True)
    top = []
    low_votes=0
    for m, s in scored:
        if m.vote_count<100:
            if low_votes>=2: continue
            low_votes+=1
        top.append((m.title, s))
        if len(top)==10: break
    return top, candidate_movies

st.title("🎬 Movie AI Recommender")

# --- Movie Search and Selection ---
def search_movies(prefix):
    if not prefix or len(prefix) < 2:
        return []
    time.sleep(0.3)
    try:
        url = "https://api.themoviedb.org/3/search/movie"
        params = {
            "api_key": st.secrets["TMDB_API_KEY"],
            "query": prefix
        }
        response = requests.get(url, params=params)
        data = response.json()
        results = data.get("results", [])
        # Return list of tuples with (display_text, movie_id)
        return [
            (f"{m['title']} ({m['release_date'][:4]})" if m.get('release_date') else m['title'], m['id'])
            for m in results[:5]
        ]
    except Exception as e:
        st.error(f"Error searching for movies: {e}")
        return []

# Create a search box and dropdown for movie selection
search_query = st.text_input("Search for a movie (type at least 2 characters)", key="movie_search")
search_results = []
if search_query and len(search_query) >= 2:
    try:
        search_results = search_movies(search_query)
        if not search_results:
            st.info("No results found. Try a different movie name.")
    except Exception as e:
        st.error(f"Error searching for movies: {e}")

if search_results:
    selected_option = st.selectbox(
        "Select a movie from the results",
        options=[result[0] for result in search_results],
        key="movie_select"
    )
    if selected_option and st.button("Add Movie"):
        if len(st.session_state.favorite_movies) >= 5:
            st.warning("You can only add up to 5 movies. Please remove some movies first.")
        else:
            clean_title = selected_option.split(" (", 1)[0]
            if clean_title not in [title.split(" (", 1)[0] for title in st.session_state.favorite_movies]:
                st.session_state.favorite_movies.append(clean_title)
                save_session({"favorite_movies": st.session_state.favorite_movies})
                st.experimental_rerun()

# Display selected movies with remove option
if st.session_state.favorite_movies:
    st.subheader("🎥 Your Selected Movies (5 max)")
    for i, title in enumerate(st.session_state.favorite_movies, 1):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"{i}. {title}")
        with col2:
            if st.button("Remove", key=f"remove_{i}"):
                st.session_state.favorite_movies.pop(i-1)
                save_session({"favorite_movies": st.session_state.favorite_movies})
                st.experimental_rerun()

if st.button("❌ Clear All"):
    st.session_state.favorite_movies = []
    save_session({"favorite_movies": []})
    st.experimental_rerun()

st.markdown("---")

# --- Recommendation Button and Feedback UI ---
if st.button("🎬 Get Recommendations"):
    if len(st.session_state.favorite_movies) != 5:
        st.warning("Please select exactly 5 movies to get recommendations.")
    else:
        with st.spinner("Finding personalized movie recommendations..."):
            recs, candidate_movies = recommend_movies(st.session_state.favorite_movies)
        st.subheader("🎯 Your Top 10 Movie Recommendations")

        def save_feedback_to_csv():
            feedback_rows = []
            for key, val in st.session_state.items():
                if key.startswith("feedback_") and isinstance(val, dict):
                    feedback_rows.append({
                        "timestamp": datetime.now().isoformat(),
                        "movie_title": val["title"],
                        "response": val["response"],
                        "rating": val.get("rating")
                    })
            if feedback_rows:
                df = pd.DataFrame(feedback_rows)
                df.to_csv("user_feedback_log.csv", mode="a", header=False, index=False)
                st.success("✅ Feedback saved!")

        for idx, (title, _) in enumerate(recs, 1):
            movie_obj = next((m for m in candidate_movies.values() if m.title == title), None)
            if not movie_obj:
                continue
            st.markdown(f"### {idx}. {movie_obj.title}")
            col1, col2, col3 = st.columns([1, 2, 1])

            # Poster
            with col1:
                poster_url = f"https://image.tmdb.org/t/p/w300{movie_obj.poster_path}" if movie_obj.poster_path else None
                if poster_url:
                    st.image(poster_url, width=150)
                else:
                    st.text("No image available")

            # Description
            with col2:
                overview = movie_obj.overview or "No description available."
                short_desc = " ".join(overview.split()[:50]) + ("..." if len(overview.split())>50 else "")
                st.write(short_desc)

            # Feedback
            with col3:
                fb_key = f"feedback_{movie_obj.id}"
                response = st.radio("Would you watch this?", ["Yes","No","Already watched"], key=fb_key)
                rating = None
                if response == "Already watched":
                    rate_key = f"rating_{movie_obj.id}"
                    rating = st.number_input("Rate this movie out of 10", min_value=1, max_value=10, key=rate_key)
                st.session_state[fb_key] = {"title": movie_obj.title, "response": response, "rating": rating}
            st.markdown("---")

        if st.button("📥 Submit Feedback"):
            save_feedback_to_csv()
            # Persist session state including favorites and feedback
            save_session({
                "favorite_movies": st.session_state.favorite_movies,
                # Optionally store feedback entries
                "feedback": {k: v for k, v in st.session_state.items() if k.startswith("feedback_")}
            })
