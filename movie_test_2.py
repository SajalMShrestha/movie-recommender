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
import csv
from collections import Counter
from sentence_transformers import SentenceTransformer
import numpy as np
from numpy.linalg import norm
from numpy import mean, dot
from sentence_transformers.util import cos_sim
import uuid
import gspread
from google.oauth2.service_account import Credentials

# Feedback system constants and functions
FEEDBACK_FILE = "user_feedback.csv"
SESSION_MAP_FILE = "session_map.csv"

def initialize_feedback_csv():
    if not os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([
                "numeric_session_id",
                "session_id",
                "movie_id",
                "movie_title",
                "watched_status",
                "liked_status",
                "timestamp"
            ])

def get_or_create_numeric_session_id():
    if not os.path.exists(SESSION_MAP_FILE):
        pd.DataFrame(columns=["numeric_session_id", "session_id"]).to_csv(SESSION_MAP_FILE, index=False)

    session_id = st.session_state.get("session_id", str(uuid.uuid4()))
    st.session_state["session_id"] = session_id

    df = pd.read_csv(SESSION_MAP_FILE)
    if session_id in df["session_id"].values:
        numeric_id = df[df["session_id"] == session_id]["numeric_session_id"].values[0]
    else:
        numeric_id = df["numeric_session_id"].max() + 1 if not df.empty else 1
        new_entry = pd.DataFrame([[numeric_id, session_id]], columns=["numeric_session_id", "session_id"])
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(SESSION_MAP_FILE, index=False)

    return numeric_id, session_id

def save_feedback(numeric_id, session_id, movie_id, movie_title, watched_status, liked_status):
    with open(FEEDBACK_FILE, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            numeric_id,
            session_id,
            movie_id,
            movie_title,
            watched_status,
            liked_status,
            datetime.utcnow().isoformat()
        ])

# Set up credentials using streamlit secrets
def get_gsheet_client():
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        
        # Check if the secret exists
        if "gcp_service_account" not in st.secrets:
            st.error("❌ Google Cloud service account credentials not found in Streamlit secrets")
            return None
            
        # Build credentials dictionary from individual TOML fields
        creds_dict = {
            "type": st.secrets["gcp_service_account"]["type"],
            "project_id": st.secrets["gcp_service_account"]["project_id"],
            "private_key_id": st.secrets["gcp_service_account"]["private_key_id"],
            "private_key": st.secrets["gcp_service_account"]["private_key"],
            "client_email": st.secrets["gcp_service_account"]["client_email"],
            "client_id": st.secrets["gcp_service_account"]["client_id"],
            "auth_uri": st.secrets["gcp_service_account"]["auth_uri"],
            "token_uri": st.secrets["gcp_service_account"]["token_uri"],
            "auth_provider_x509_cert_url": st.secrets["gcp_service_account"]["auth_provider_x509_cert_url"],
            "client_x509_cert_url": st.secrets["gcp_service_account"]["client_x509_cert_url"],
            "universe_domain": st.secrets["gcp_service_account"]["universe_domain"]
        }
        
        # Fix private key format - convert literal \n to actual newlines
        private_key = creds_dict.get("private_key", "")
        if "\\n" in private_key:
            private_key = private_key.replace("\\n", "\n")
            creds_dict["private_key"] = private_key
        
        # Additional cleanup - remove any extra quotes or formatting issues
        private_key = private_key.strip()
        if private_key.startswith('"') and private_key.endswith('"'):
            private_key = private_key[1:-1]
        creds_dict["private_key"] = private_key
        
        # Debug: Check private key format
        if not private_key.startswith("-----BEGIN PRIVATE KEY-----"):
            st.error("❌ Private key is not in correct PEM format")
            st.info("💡 Make sure your private key starts with '-----BEGIN PRIVATE KEY-----'")
            return None
            
        creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
        client = gspread.authorize(creds)
        return client
        
    except Exception as e:
        st.error(f"❌ Error setting up Google Sheets client: {str(e)}")
        st.info("💡 This usually means your service account JSON is corrupted or improperly formatted.")
        return None

# Append a row of user feedback
def record_feedback_to_sheet(numeric_session_id, uuid_session_id, movie_id, movie_title, would_watch, liked_if_seen):
    try:
        sheet_name = "user_feedback"  # your sheet name
        client = get_gsheet_client()
        if client is None:
            st.error("❌ Could not connect to Google Sheets. Please check your credentials.")
            return False

        sheet = client.open(sheet_name).sheet1  # first worksheet

        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        # ✅ Explicitly convert all values to safe Python built-ins
        row = [
            int(numeric_session_id),
            str(uuid_session_id),
            str(movie_id),
            str(movie_title),
            str(would_watch),
            str(liked_if_seen),
            str(timestamp)
        ]

        sheet.append_row(row)
        return True

    except Exception as e:
        st.error(f"❌ Error saving to Google Sheets: {str(e)}")
        return False

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

# Initialize embedding model for semantic analysis
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

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

# Add a persistent UUID for the session
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Initialize feedback system
initialize_feedback_csv()
numeric_id, session_uuid = get_or_create_numeric_session_id()
st.session_state.numeric_session_id = numeric_id

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
    "mood_tone": 0.224,
    "genre_similarity": 0.128,
    "cast_crew": 0.120,
    "narrative_style": 0.080,
    "ratings": 0.064,
    "trending_factor": 0.080,
    "release_year": 0.040,
    "discovery_boost": 0.064,
    "age_alignment": 0.0,
    "embedding_similarity": 0.20
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

from collections import Counter

def compute_narrative_similarity(candidate_style, reference_styles):
    similarity = 0
    for key in candidate_style:
        if not reference_styles[key]: continue
        dominant = Counter(reference_styles[key]).most_common(1)[0][0]
        if candidate_style[key] == dominant:
            similarity += 1
    return similarity / len(candidate_style)

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

        # 🚫 Skip if plot is missing or too short to embed meaningfully
        if not m_details.plot or len(m_details.plot.split()) < 5:
            st.write(f"⚠️ Skipping {m_details.title} due to missing/short plot")
            return m_id, None

        m_details.narrative_style = infer_narrative_style(m_details.plot)

        # ✅ Generate embedding
        embedding = embedding_model.encode(m_details.plot)

        cache[m_id] = (m_details, embedding)
        return m_id, (m_details, embedding)

    except Exception as e:
        st.warning(f"Embedding fetch failed for ID {m_id}: {e}")
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

# Generate embeddings for semantic similarity
def generate_embedding(text):
    if not text:
        return np.zeros(384)  # Embedding size for MiniLM
    return embedding_model.encode(text, convert_to_numpy=True, normalize_embeddings=True)

def cosine_similarity(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2) + 1e-8)

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

def construct_enriched_description(movie_details, credits, keywords=None):
    title = movie_details.title
    genres = [g['name'] for g in movie_details.genres]
    cast = [c['name'] for c in credits.get('cast', [])[:3]]
    directors = [c['name'] for c in credits.get('crew', []) if c['job'] == 'Director']
    tagline = getattr(movie_details, 'tagline', '')
    overview = getattr(movie_details, 'overview', '')
    keyword_list = [k['name'] for k in keywords] if keywords else []

    enriched_text = f"{title} is a {', '.join(genres)} movie"
    if directors:
        enriched_text += f" directed by {', '.join(directors)}"
    if cast:
        enriched_text += f", starring {', '.join(cast)}"
    enriched_text += ". "
    if tagline:
        enriched_text += f"Tagline: {tagline}. "
    if keyword_list:
        enriched_text += f"Keywords: {', '.join(keyword_list)}. "
    enriched_text += f"Plot: {overview}"

    return enriched_text

# --- Recommendation Logic ---
def recommend_movies(favorite_titles):
    favorite_genres, favorite_actors = set(), set()
    candidate_movie_ids, plot_moods, favorite_years = set(), set(), []
    favorite_narrative_styles = {"tone": [], "complexity": [], "genre_indicator": [], "setting_context": []}
    favorite_embeddings = []

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
        narr_style = infer_narrative_style(details.overview or "")
        for key in favorite_narrative_styles:
            favorite_narrative_styles[key].append(narr_style.get(key, ""))
        if details.release_date:
            favorite_years.append(int(details.release_date[:4]))
        
        enriched_plot = f"{details.overview} Genres: {', '.join([g['name'] for g in details.genres])}."
        embedding = generate_embedding(enriched_plot)
        favorite_embeddings.append(embedding)
        
        try:
            similar_list = movie_api.similar(details.id)
            if similar_list:
                candidate_movie_ids.update([m.id for m in similar_list if hasattr(m, 'id')])
        except:
            continue

    # Compute average embedding of favorite movies
    favorite_embeddings = []
    for title in favorite_titles:
        results = movie_api.search(title)
        if results:
            details = movie_api.details(results[0].id)
            overview = details.overview or ""
            emb = embedding_model.encode(overview, convert_to_tensor=True)
            favorite_embeddings.append(emb)

    if favorite_embeddings:
        from torch import stack
        avg_fav_embedding = stack(favorite_embeddings).mean(dim=0)
    else:
        avg_fav_embedding = None

    # Add trending movies to candidate set
    trending_scores = get_trending_popularity(tmdb.api_key)

    user_prefs = {
        "subscribed_platforms": [k for k,v in streaming_platform_priority.items() if v>0],
        "preferred_moods": plot_moods,
        "estimated_age": estimate_user_age(favorite_years)
    }

    candidate_movies = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_similar_movie_details, mid): mid for mid in candidate_movie_ids}
        for fut in concurrent.futures.as_completed(futures):
            result = fut.result()
            if result is None:
                continue
            mid, payload = result
            if payload is None:
                continue
            m, embedding = payload
            if m is None or embedding is None:
                continue
            if getattr(m, 'vote_count', 0) < 20:
                continue
            candidate_movies[mid] = (m, embedding)

    st.write(f"🎯 Favorite titles: {favorite_titles}")
    st.write(f"🔍 Candidate Movie IDs fetched: {len(candidate_movie_ids)}")
    st.write(f"🧠 Candidate Movies with embeddings: {len(candidate_movies)}")

    valid_titles = [m.title for m, emb in candidate_movies.values() if m and emb is not None]
    st.markdown(f"✅ Valid candidate titles: {valid_titles}")

    if not candidate_movies:
        st.warning("No candidate movies with valid plots or embeddings were found.")

    # Fetch trending scores before computing movie scores
    trending_scores = get_trending_popularity(tmdb.api_key)

    def compute_score(m, avg_fav_embedding):
        narrative = m.narrative_style
        # st.write(f"{m.title} narrative style: {narrative}")  # Removed from UI
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

        narrative = infer_narrative_style(m.plot)
        narrative_match_score = compute_narrative_similarity(narrative, favorite_narrative_styles)
        score += recommendation_weights['narrative_style'] * narrative_match_score
        # st.write(f"{m.title} narrative_match={narrative_match_score:.2f}")

        # --- New: Embedding Similarity ---
        candidate_embedding = candidate_movies[m.id][1]  # (movie_obj, embedding)
        embedding_sim_score = float(cos_sim(candidate_embedding, avg_fav_embedding))

        score += recommendation_weights['embedding_similarity'] * embedding_sim_score

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
        # st.write(f"{m.title} → total_score: {score:.4f}, trending_boost: {movie_trend_score:.2f}")
        try:
            release_year=int(m.release_date[:4])
            user_age_at_release = user_prefs['estimated_age'] - (datetime.now().year - release_year)
            if 15<=user_age_at_release<=25: score += recommendation_weights['age_alignment']
            elif 10<=user_age_at_release<15 or 25<user_age_at_release<=30: score += recommendation_weights['age_alignment']*0.5
        except: pass
        # st.write(f"{m.title} → total_score: {score:.4f}, trending_boost: {movie_trend_score:.2f}")
        return max(score, 0)

    scored = []
    for movie_obj, embedding in candidate_movies.values():
        if movie_obj is None or embedding is None:
            continue
        score = compute_score(movie_obj, avg_fav_embedding) + min(movie_obj.vote_count, 500) / 50000
        scored.append((movie_obj, score))

    st.write(f"✅ Candidate movies count: {len(candidate_movies)}")
    st.write(f"✅ Valid scored movies: {len(scored)}")

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

# Setup flags
if "search_done" not in st.session_state:
    st.session_state["search_done"] = False
if "previous_query" not in st.session_state:
    st.session_state["previous_query"] = ""

# Get input
search_query = st.text_input("Search for a movie (type at least 2 characters)", key="movie_search")

# ✅ Reset search_done when user types a different movie
if search_query != st.session_state["previous_query"]:
    st.session_state["search_done"] = False
    st.session_state["previous_query"] = search_query

search_results = []

# --- Keep track of whether to show Top 5 ---
if search_query and len(search_query) >= 2:
    st.session_state.show_search_results = True

# --- Only show if toggle is ON ---
if st.session_state.get("show_search_results", False) and search_results:
    st.markdown("### Top 5 Matches")
    cols = st.columns(5)

    for idx, movie in enumerate(search_results):
        with cols[idx]:
            poster_url = f"https://image.tmdb.org/t/p/w200{movie['poster_path']}" if movie.get("poster_path") else None
            if poster_url:
                st.image(poster_url, use_column_width=True)
            st.write(f"**{movie['label']}**")

            if st.button("Add Movie", key=f"add_{idx}"):
                clean_title = movie["label"].split(" (", 1)[0]
                movie_id = movie["id"]

                existing_titles = [m["title"] for m in st.session_state.favorite_movies if isinstance(m, dict)]
                if len(st.session_state.favorite_movies) >= 5:
                    st.warning("You can only add up to 5 movies.")
                elif clean_title not in existing_titles:
                    st.session_state.favorite_movies.append({
                        "title": clean_title,
                        "year": movie["label"].split("(", 1)[1].replace(")", "") if "(" in movie["label"] else "",
                        "poster_path": movie.get("poster_path", ""),
                        "id": movie_id
                    })
                    save_session({"favorite_movies": st.session_state.favorite_movies})
                    st.toast(f"✅ Added {clean_title}")
                    # ✅ Hide Top 5 matches after adding
                    st.session_state.show_search_results = False

# --- Display Favorite Movies with Posters in a Grid ---
st.subheader("🎥 Your Selected Movies (5 max)")

# Build a single HTML string for all cards
movie_cards_html = """
<style>
.movie-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    justify-content: flex-start;
    margin-bottom: 20px;
}
.movie-card {
    width: 120px;
    text-align: center;
}
.movie-card img {
    height: 180px;
    width: 100%;
    object-fit: cover;
    border-radius: 6px;
}
</style>
<div class="movie-grid">
"""

for i, movie in enumerate(st.session_state.favorite_movies):
    title = movie["title"]
    year = movie["year"]
    poster = movie.get("poster_path")

    movie_cards_html += f'<div class="movie-card">'
    if poster:
        poster_url = f"https://image.tmdb.org/t/p/w200{poster}"
        movie_cards_html += f'<img src="{poster_url}" alt="{title}">'
    else:
        movie_cards_html += '<div>No image available</div>'

    movie_cards_html += f'<div><strong>{title} ({year})</strong></div>'
    movie_cards_html += f'''
        <form action="" method="post">
            <button type="submit" name="remove_index" value="{i}">Remove</button>
        </form>
    '''
    movie_cards_html += '</div>'

movie_cards_html += "</div>"

# Render all cards at once
st.markdown(movie_cards_html, unsafe_allow_html=True)

# Buttons rendered separately
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
if st.session_state.recommend_triggered:
    if not st.session_state.recommendations:
        st.warning("⚠️ No recommendations could be generated. Please try different favorite movies.")
        st.info("Tip: Make sure your selected movies have plot summaries and at least some popularity.")
    else:
        st.subheader("🌟 Your Top 10 Movie Recommendations")

    # 1. Create placeholders and gather all responses in a dictionary
    user_feedback = {}

    for idx, (title, _) in enumerate(st.session_state.recommendations, 1):
        movie_obj = next((m[0] for m in st.session_state.candidates.values() if m and m[0].title == title), None)
        if movie_obj is None:
            continue
        release_year = "N/A"
        try:
            release_year = movie_obj.release_date[:4]
        except:
            pass
        st.markdown(f"### {idx}. {movie_obj.title} ({release_year})")
        st.image(f"https://image.tmdb.org/t/p/w300{movie_obj.poster_path}" if movie_obj.poster_path else None, width=150)
        st.write(movie_obj.overview or "No description available.")

        fb_key = f"watch_{idx}"
        liked_key = f"liked_{idx}"

        response = st.radio("Would you watch this?", ["Yes", "No", "Already watched"], key=fb_key, index=None)

        liked = None
        if response == "Already watched":
            liked = st.radio("Did you like it?", ["Yes", "No"], key=liked_key, index=None)

        user_feedback[idx] = {
            "movie": movie_obj.title,
            "movie_id": movie_obj.id,
            "response": response,
            "liked": liked,
        }
        
        st.markdown("---")

    # 2. Submit button at the end only
    if st.button("Submit All Responses"):
        # Store all responses in Google Sheet
        success_count = 0
        total_responses = 0
        
        for index, feedback in user_feedback.items():
            if feedback["response"]:  # Only save if user provided a response
                total_responses += 1
                if record_feedback_to_sheet(
                    numeric_session_id=st.session_state.numeric_session_id,
                    uuid_session_id=st.session_state.session_id,
                    movie_id=feedback["movie_id"],
                    movie_title=feedback["movie"],
                    would_watch=feedback["response"],
                    liked_if_seen=feedback["liked"] or ""
                ):
                    success_count += 1
        
        if success_count == total_responses and total_responses > 0:
            st.success(f"✅ All {success_count} responses saved successfully!")
        elif success_count > 0:
            st.warning(f"⚠️ {success_count}/{total_responses} responses saved. Some failed to save.")
        else:
            st.error("❌ Failed to save any responses. Please check your Google Sheets setup.")

# --- Display Feedback Log ---
if os.path.exists("user_feedback_log.csv"):
    st.success("✅ Feedback file found!")
    df = pd.read_csv("user_feedback_log.csv")
    st.dataframe(df.tail(10))  # Show the last 10 rows for quick inspection
else:
    st.info("📝 Feedback log not found yet. Submit feedback after getting recommendations.")