import streamlit as st
from tmdbv3api import TMDb, Movie
from datetime import datetime
import concurrent.futures
import requests
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
import textstat
import pandas as pd
import time
import json
import os

nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# TMDb setup
tmdb = TMDb()
tmdb.api_key = st.secrets["TMDB_API_KEY"]
# st.write("TMDB API Key loaded:", tmdb.api_key)  # Debug print

tmdb.language = 'en'
tmdb.debug = True
movie_api = Movie()
sia = SentimentIntensityAnalyzer()

# Fetch and normalize trending popularity scores
def get_trending_popularity(api_key):
    try:
        url = f"https://api.themoviedb.org/3/trending/movie/week?api_key={api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json().get("results", [])
            if not data:
                return {}
            max_pop = max([m.get("popularity", 1) for m in data])
            return {m["id"]: m.get("popularity", 0) / max_pop for m in data}
    except:
        return {}

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
if "recommendations" not in st.session_state:
    st.session_state.recommendations = None
if "candidates" not in st.session_state:
    st.session_state.candidates = None
if "recommend_triggered" not in st.session_state:
    st.session_state.recommend_triggered = False
if "favorite_movie_posters" not in st.session_state:
    st.session_state.favorite_movie_posters = {}

# --- Updated recommendation weights ---
recommendation_weights = {
    "mood_tone": 0.28,          # ↑ stronger emotional fit
    "genre_similarity": 0.16,   # ↑ still useful, but secondary
    "cast_crew": 0.15,          # ↓ modest influence
    "narrative_style": 0.10,    # ↔️ not yet implemented, placeholder for future
    "ratings": 0.08,            # ↓ modest quality filter
    "trending_factor": 0.10,    # NEW: popularity boost
    "release_year": 0.05,       # ↔️ subtle influence
    "discovery_boost": 0.08,    # ↔️ minor novelty/obscurity encouragement
    "age_alignment": 0.0        # temporarily removed to keep total = 1.0
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
        m_details = movie_api.details(m_id)
        m_credits = movie_api.credits(m_id)
        m_details.genres = m_details.genres
        m_details.cast = list(m_credits['cast'])[:3]
        m_details.directors = [d['name'] for d in m_credits['crew'] if d['job'] == 'Director']
        m_details.plot = m_details.overview or ""
        m_details.narrative_style = infer_narrative_style(m_details.plot)
        cache[m_id] = m_details
        return m_id, m_details
    except:
        return m_id, None

# Text analysis functions for narrative style detection
stop_words = set(stopwords.words('english'))

# Helper to tokenize and preprocess text once
def preprocess_text(plot):
    tokens = word_tokenize(plot.lower())
    words = [word for word in tokens if word.isalpha() and word not in stop_words]
    return words

# 1. Tone (Sentiment)
def infer_tone(plot):
    sentiment = sia.polarity_scores(plot)
    if sentiment['compound'] >= 0.3:
        return "positive"
    elif sentiment['compound'] <= -0.3:
        return "negative"
    else:
        return "neutral"

# 2. Narrative Complexity
def infer_narrative_complexity(plot):
    readability = textstat.flesch_kincaid_grade(plot)
    words = preprocess_text(plot)
    unique_words_ratio = len(set(words)) / max(len(words), 1)

    # Adjusted complexity thresholds (more suitable for shorter texts)
    if readability >= 9 or unique_words_ratio >= 0.5:
        return "complex"
    else:
        return "simple"

# Keywords refined as sets for accurate checking
action_keywords = {"chase", "fight", "escape", "war", "battle", "mission", "rescue"}
emotion_keywords = {"love", "friendship", "betrayal", "grief", "romance", "relationship", "emotional"}
concept_keywords = {"reality", "consciousness", "philosophy", "dream", "mystery", "existential"}

# 3. Genre Indicators
def infer_genre_indicator(plot):
    words = set(preprocess_text(plot))
    if words & action_keywords:
        return "action-oriented"
    elif words & emotion_keywords:
        return "emotion/character-oriented"
    elif words & concept_keywords:
        return "idea/concept-oriented"
    else:
        return "general"

realistic_keywords = {"city", "suburb", "historical", "town", "village", "real-life", "everyday", "ordinary"}
fantasy_keywords = {"galaxy", "kingdom", "future", "space", "magical", "supernatural", "alien", "fantasy", "alternate"}

# 4. Setting Context
def infer_setting_context(plot):
    words = set(preprocess_text(plot))
    if words & fantasy_keywords:
        return "fantastical/surreal"
    elif words & realistic_keywords:
        return "realistic"
    else:
        return "neutral"

def infer_narrative_style(plot):
    if not plot:
        return {
            "tone": "neutral",
            "complexity": "simple",
            "genre_indicator": "general",
            "setting_context": "neutral"
        }

    plot_lower = plot.lower()
    tokens = word_tokenize(plot_lower)
    words = [word for word in tokens if word.isalpha() and word not in stop_words]
    word_set = set(words)

    sentiment = sia.polarity_scores(plot)
    tone = ("positive" if sentiment['compound'] >= 0.3 else
            "negative" if sentiment['compound'] <= -0.3 else
            "neutral")

    readability = textstat.flesch_kincaid_grade(plot)
    unique_words_ratio = len(word_set) / max(len(words), 1)
    complexity = "complex" if readability >= 9 or unique_words_ratio >= 0.5 else "simple"

    genre_indicator = ("action-oriented" if word_set & action_keywords else
                       "emotion/character-oriented" if word_set & emotion_keywords else
                       "idea/concept-oriented" if word_set & concept_keywords else
                       "general")

    setting_context = ("fantastical/surreal" if word_set & fantasy_keywords else
                       "realistic" if word_set & realistic_keywords else
                       "neutral")

    return {
        "tone": tone,
        "complexity": complexity,
        "genre_indicator": genre_indicator,
        "setting_context": setting_context
    }

# --- Recommendation Logic ---
def recommend_movies(favorite_titles):
    favorite_genres, favorite_actors = set(), set()
    candidate_movie_ids, plot_moods, favorite_years = set(), set(), []

    for title in favorite_titles:
        search_result = movie_api.search(title)
        if not search_result:
            continue
        details = movie_api.details(search_result[0].id)
        credits = movie_api.credits(search_result[0].id)
        favorite_genres.update([g['name'] for g in details.genres])
        favorite_actors.update([c['name'] for c in list(credits['cast'])[:3]])
        favorite_actors.update([d['name'] for d in credits['crew'] if d['job']=='Director'])
        plot_moods.add(infer_mood_from_plot(details.overview or ""))
        if details.release_date:
            favorite_years.append(int(details.release_date[:4]))
        try:
            similar_list = movie_api.similar(details.id)
            if similar_list:
                candidate_movie_ids.update([m.id for m in similar_list if hasattr(m, 'id')])
        except:
            continue

    # Add trending movies to candidate set
    trending_scores = get_trending_popularity(tmdb.api_key)
    candidate_movie_ids.update(trending_scores.keys())

    user_prefs = {
        "subscribed_platforms": [k for k,v in streaming_platform_priority.items() if v>0],
        "preferred_moods": plot_moods,
        "estimated_age": estimate_user_age(favorite_years)
    }

    candidate_movies = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_similar_movie_details, mid): mid for mid in candidate_movie_ids}
        for fut in concurrent.futures.as_completed(futures):
            mid, m = fut.result()
            if m and getattr(m,'vote_count',0)>=20:
                candidate_movies[mid] = m

    # Fetch trending scores before computing movie scores
    trending_scores = get_trending_popularity(tmdb.api_key)

    def compute_score(m):
        narrative = m.narrative_style
        st.write(f"{m.title} narrative style: {narrative}")
        score = 0.0
        genres = {g['name'] for g in m.genres}
        score += recommendation_weights['genre_similarity'] * (len(genres & favorite_genres)/max(len(favorite_genres),1))
        cast_dir = set(a.name for a in m.cast) | set(getattr(m,'directors',[]))
        score += recommendation_weights['cast_crew'] * (len(cast_dir & favorite_actors)/max(len(favorite_actors),1))
        try:
            year_diff = datetime.now().year - int(m.release_date[:4])
            if year_diff<=2: score += recommendation_weights['release_year']
            elif year_diff<=5: score += recommendation_weights['release_year']*0.66
            elif year_diff<=15: score += recommendation_weights['release_year']*0.33
        except: pass
        score += recommendation_weights['ratings'] * ((m.vote_average or 0)/10)
        score += recommendation_weights['mood_tone'] * get_mood_score(m.genres, user_prefs['preferred_moods'])
        movie_trend_score = trending_scores.get(m.id, 0)
        mood_match_score = get_mood_score(m.genres, user_prefs['preferred_moods'])
        genre_overlap_score = len({g['name'] for g in m.genres} & favorite_genres) / max(len(favorite_genres), 1)

        # Only apply trending boost if both mood and genre are somewhat aligned
        if mood_match_score > 0.3 and genre_overlap_score > 0.2:
            score += recommendation_weights['trending_factor'] * movie_trend_score

        # Apply a small penalty for very old movies (e.g., released more than 20 years ago)
        try:
            release_year = int(m.release_date[:4])
            if datetime.now().year - release_year > 20:
                score -= 0.03  # small age penalty
        except:
            pass
        st.write(f"{m.title}: mood={mood_match_score:.2f}, genre_overlap={genre_overlap_score:.2f}, trending={movie_trend_score:.2f}")
        score -= get_maturity_penalty(m.genres)
        try:
            release_year=int(m.release_date[:4])
            user_age_at_release = user_prefs['estimated_age'] - (datetime.now().year - release_year)
            if 15<=user_age_at_release<=25: score += recommendation_weights['age_alignment']
            elif 10<=user_age_at_release<15 or 25<user_age_at_release<=30: score += recommendation_weights['age_alignment']*0.5
        except: pass
        st.write(f"{m.title} → total_score: {score:.4f}, trending_boost: {movie_trend_score:.2f}")
        return max(score,0)

    scored = [(m, compute_score(m) + min(m.vote_count, 500)/50000) for m in candidate_movies.values()]
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

# Movie Search & Selection UI
search_query = st.text_input("Search for a movie (type at least 2 characters)", key="movie_search")
search_results = []
if search_query and len(search_query) >= 2:
    try:
        url = "https://api.themoviedb.org/3/search/movie"
        params = {"api_key": st.secrets["TMDB_API_KEY"], "query": search_query}
        response = requests.get(url, params=params)
        data = response.json()
        results = data.get("results", [])
        search_results = [
            {
                "label": f"{m.get('title')} ({m.get('release_date')[:4]})" if m.get("release_date") else m.get('title'),
                "id": m.get("id"),
                "poster_path": m.get("poster_path")
            }
            for m in results[:5]
            if m.get("title") and m.get("id")
        ]
    except Exception as e:
        st.error(f"Error searching for movies: {e}")

if search_results:
    selected_label = st.selectbox(
        "Select a movie from the results",
        options=[item["label"] for item in search_results if "label" in item],
        key="movie_select"
    )
    selected_movie = next((m for m in search_results if m["label"] == selected_label), None)

    if selected_movie and st.button("Add Movie"):
        clean_title = selected_label.split(" (", 1)[0]
        existing_titles = [m["title"] for m in st.session_state.favorite_movies if isinstance(m, dict)]
        if len(st.session_state.favorite_movies) >= 5:
            st.warning("You can only add up to 5 movies. Please remove some movies first.")
        elif clean_title not in existing_titles:
            st.session_state.favorite_movies.append({
                "title": clean_title,
                "year": selected_label.split("(", 1)[1].replace(")", "") if "(" in selected_label else "",
                "poster_path": selected_movie.get("poster_path")
            })
            save_session({"favorite_movies": st.session_state.favorite_movies})
            # ✅ Show the poster *after* Add Movie is clicked
            if selected_movie['poster_path']:
                st.image(f"https://image.tmdb.org/t/p/w300{selected_movie['poster_path']}", width=150)
            else:
                st.text("No image available")
            st.experimental_rerun()

# --- Display Favorite Movies with Posters in a Grid ---
if st.session_state.favorite_movies:
    st.subheader("🎥 Your Selected Movies (5 max)")
    cols = st.columns(5)
    for i, movie in enumerate(st.session_state.favorite_movies):
        title = movie["title"]
        year = movie["year"]
        poster = movie.get("poster_path")
        with cols[i % 5]:
            if poster:
                st.image(f"https://image.tmdb.org/t/p/w200{poster}", use_column_width=True)
            st.markdown(f"**{title} ({year})**")
            if st.button("Remove", key=f"remove_{i}"):
                st.session_state.favorite_movies.pop(i)
                save_session({"favorite_movies": st.session_state.favorite_movies})
                st.experimental_rerun()

if st.button("❌ Clear All"):
    st.session_state.favorite_movies = []
    save_session({"favorite_movies": []})
    st.experimental_rerun()

# --- Get Recommendations ---
if st.button("🎬 Get Recommendations"):
    if len(st.session_state.favorite_movies) != 5:
        st.warning("Please select exactly 5 movies to get recommendations.")
    else:
        with st.spinner("Finding personalized movie recommendations..."):
            favorite_titles = [m["title"] for m in st.session_state.favorite_movies if isinstance(m, dict)]
            try:
                recs, candidate_movies = recommend_movies(favorite_titles)
                st.session_state.recommendations = recs
                st.session_state.candidates = candidate_movies
                st.session_state.recommend_triggered = True
            except Exception as e:
                st.error(f"❌ Failed to generate recommendations: {e}")

# Display recommendations and feedback
if st.session_state.recommend_triggered and st.session_state.recommendations:
    st.subheader("🌟 Your Top 10 Movie Recommendations")

    def save_feedback_to_csv():
        feedback_rows = []
        for key, val in st.session_state.items():
            if key.startswith("feedback_obj_") and isinstance(val, dict):
                feedback_rows.append({
                    "timestamp": datetime.now().isoformat(),
                    "movie_title": val["title"],
                    "response": val["response"],
                    "liked": val.get("liked")
                })
        if feedback_rows:
            df = pd.DataFrame(feedback_rows)
            df.to_csv("user_feedback_log.csv", mode="a", header=False, index=False)
            st.success("✅ Feedback saved!")

    for idx, (title, _) in enumerate(st.session_state.recommendations, 1):
        movie_obj = next((m for m in st.session_state.candidates.values() if m.title == title), None)
        if not movie_obj:
            continue
        st.markdown(f"### {idx}. {movie_obj.title}")
        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            poster_url = f"https://image.tmdb.org/t/p/w300{movie_obj.poster_path}" if movie_obj.poster_path else None
            if poster_url:
                st.image(poster_url, width=150)
            else:
                st.text("No image available")

        with col2:
            overview = movie_obj.overview or "No description available."
            short_desc = " ".join(overview.split()[:50]) + ("..." if len(overview.split())>50 else "")
            st.write(short_desc)

        with col3:
            fb_key = f"feedback_{movie_obj.id}"
            response = st.radio("Would you watch this?", ["Yes", "No", "Already watched"], key=fb_key, index=None)

            liked = None
            liked_key = f"liked_{movie_obj.id}"
            if response == "Already watched":
                liked = st.radio("Did you like it?", ["Yes", "No"], key=liked_key, index=None)

            feedback_entry = {
                "title": movie_obj.title,
                "response": response,
                "liked": liked
            }
            st.session_state[f"feedback_obj_{movie_obj.id}"] = feedback_entry
        st.markdown("---")

    if st.button("Submit Feedback"):
        save_feedback_to_csv()
        save_session({
            "favorite_movies": st.session_state.favorite_movies,
            "feedback": {k: v for k, v in st.session_state.items() if k.startswith("feedback_")}
        })