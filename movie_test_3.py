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
from torch import stack
import torch
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

# Global cache for movie details to avoid repeated API calls
MOVIE_DETAILS_CACHE = {}
MOVIE_CREDITS_CACHE = {}

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

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

# Initialize embedding model for semantic analysis
embedding_model = get_embedding_model()

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
    "mood_tone": 0.15,
    "genre_similarity": 0.10,
    "cast_crew": 0.10,
    "narrative_style": 0.08,
    "ratings": 0.05,
    "trending_factor": 0.07,
    "release_year": 0.05,
    "discovery_boost": 0.05,
    "age_alignment": 0.0,
    "embedding_similarity": 0.35  # Increased but will use cluster-based approach
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
    for g in genres:
        genre_name = g.get('name', '') if isinstance(g, dict) else getattr(g, 'name', '')
        if genre_name in immature_genres:
            return 0.15
    return 0

def get_mood_score(genres, preferred_moods):
    matched_moods = set()
    for g in genres:
        # Handle both dict and object formats
        genre_name = g.get('name', '') if isinstance(g, dict) else getattr(g, 'name', '')
        if genre_name:
            for mood, tags in mood_tone_map.items():
                if genre_name in tags:
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

        # Robust genre, cast, director extraction
        genres = []
        genres_list = getattr(m_details, 'genres', [])
        for g in genres_list:
            if isinstance(g, dict):
                name = g.get('name', '')
            else:
                name = getattr(g, 'name', '')
            if name:
                genres.append(name)
        cast_list_raw = m_credits.get('cast', []) if isinstance(m_credits, dict) else getattr(m_credits, 'cast', [])
        crew_list = m_credits.get('crew', []) if isinstance(m_credits, dict) else getattr(m_credits, 'crew', [])
        if hasattr(cast_list_raw, '__iter__'):
            m_details.cast = list(cast_list_raw)[:3] if cast_list_raw else []
        else:
            m_details.cast = []
        directors = []
        for c in crew_list:
            is_director = False
            name = ''
            if isinstance(c, dict):
                is_director = c.get('job', '') == 'Director'
                name = c.get('name', '')
            else:
                is_director = getattr(c, 'job', '') == 'Director'
                name = getattr(c, 'name', '')
            
            if is_director and name:
                directors.append(name)
        m_details.directors = directors
        m_details.plot = getattr(m_details, 'overview', '') or ''

        # 🚫 Skip if plot is missing or too short to embed meaningfully
        if not m_details.plot or len(m_details.plot.split()) < 5:
            st.write(f"⚠️ Skipping {getattr(m_details, 'title', 'Unknown')} due to missing/short plot")
            cache[m_id] = None
            return m_id, None

        m_details.narrative_style = infer_narrative_style(m_details.plot)

        # ✅ Generate embedding
        embedding = embedding_model.encode(m_details.plot, convert_to_tensor=True)

        cache[m_id] = (m_details, embedding)
        return m_id, (m_details, embedding)

    except Exception as e:
        st.warning(f"Embedding fetch failed for ID {m_id}: {e}")
        cache[m_id] = None
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
        import torch
        return torch.zeros(384)  # same size, but torch tensor
    return embedding_model.encode(text, convert_to_tensor=True, normalize_embeddings=True)

def compute_cosine_similarity(vec1, vec2):
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
    title = getattr(movie_details, 'title', 'Unknown')
    genres = []
    genres_list = getattr(movie_details, 'genres', [])
    for g in genres_list:
        if isinstance(g, dict):
            name = g.get('name', '')
        else:
            name = getattr(g, 'name', '')
        if name:
            genres.append(name)
    
    # Handle credits safely
    if isinstance(credits, dict):
        cast_list_raw = credits.get('cast', [])
        crew_list = credits.get('crew', [])
    else:
        cast_list_raw = getattr(credits, 'cast', [])
        crew_list = getattr(credits, 'crew', [])

    # Safely slice cast_list
    if hasattr(cast_list_raw, '__iter__'):
        cast_list = list(cast_list_raw)[:3] if cast_list_raw else []
    else:
        cast_list = []
    
    favorite_actors = set()
    favorite_directors = set()

    # ✅ Collect actor and director names - Fixed attribute access
    for c in cast_list:
        if isinstance(c, dict):
            name = c.get('name', '')
        else:
            name = getattr(c, 'name', '')
        if name:
            favorite_actors.add(name)

    # Process director names
    for c in crew_list:
        is_director = False
        name = ''
        if isinstance(c, dict):
            is_director = c.get('job', '') == 'Director'
            name = c.get('name', '')
        else:
            is_director = getattr(c, 'job', '') == 'Director'
            name = getattr(c, 'name', '')
        
        if is_director and name:
            favorite_directors.add(name)

    tagline = getattr(movie_details, 'tagline', '')
    overview = getattr(movie_details, 'overview', '')
    keyword_list = []
    if keywords:
        for k in keywords:
            if isinstance(k, dict):
                name = k.get('name', '')
            else:
                name = getattr(k, 'name', '')
            if name:
                keyword_list.append(name)

    enriched_text = f"{title} is a {', '.join([g for g in genres if g])} movie"
    if favorite_directors:
        enriched_text += f" directed by {', '.join([d for d in favorite_directors if d])}"
    if favorite_actors:
        enriched_text += f", starring {', '.join([c for c in favorite_actors if c])}"
    enriched_text += ". "
    if tagline:
        enriched_text += f"Tagline: {tagline}. "
    if keyword_list:
        enriched_text += f"Keywords: {', '.join([k for k in keyword_list if k])}. "
    enriched_text += f"Plot: {overview}"

    return enriched_text

def build_custom_candidate_pool(favorite_genre_ids, favorite_cast_ids, favorite_director_ids, favorite_years, tmdb_api_key):
    """
    Build a pool of 100-200 candidate movies using custom criteria
    """
    candidate_movie_ids = set()
    
    # Strategy 1: Discover by Genre (40-60 movies)
    # st.write("🎭 Discovering movies by genre...")
    for genre_id in list(favorite_genre_ids)[:3]:  # Top 3 genres to avoid too much overlap
        try:
            # Get popular movies in this genre
            url = f"https://api.themoviedb.org/3/discover/movie"
            params = {
                "api_key": tmdb_api_key,
                "with_genres": str(genre_id),
                "sort_by": "popularity.desc",
                "vote_count.gte": 50,  # Minimum votes for quality
                "page": 1
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                movies = response.json().get("results", [])
                candidate_movie_ids.update([m["id"] for m in movies[:20]])  # Top 20 per genre
        except Exception as e:
            st.warning(f"Error discovering by genre {genre_id}: {e}")
    
    # Strategy 2: Discover by Cast (30-40 movies)
    # st.write("🎬 Discovering movies by favorite actors...")
    for person_id in list(favorite_cast_ids)[:5]:  # Top 5 actors
        try:
            url = f"https://api.themoviedb.org/3/discover/movie"
            params = {
                "api_key": tmdb_api_key,
                "with_cast": str(person_id),
                "sort_by": "popularity.desc",
                "vote_count.gte": 30,
                "page": 1
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                movies = response.json().get("results", [])
                candidate_movie_ids.update([m["id"] for m in movies[:8]])  # Top 8 per actor
        except Exception as e:
            st.warning(f"Error discovering by cast {person_id}: {e}")
    
    # Strategy 3: Discover by Directors (20-30 movies)
    # st.write("🎥 Discovering movies by favorite directors...")
    for person_id in list(favorite_director_ids)[:3]:  # Top 3 directors
        try:
            url = f"https://api.themoviedb.org/3/discover/movie"
            params = {
                "api_key": tmdb_api_key,
                "with_crew": str(person_id),
                "sort_by": "popularity.desc", 
                "vote_count.gte": 30,
                "page": 1
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                movies = response.json().get("results", [])
                candidate_movie_ids.update([m["id"] for m in movies[:10]])  # Top 10 per director
        except Exception as e:
            st.warning(f"Error discovering by director {person_id}: {e}")
    
    # Strategy 4: Year-based Discovery (20-30 movies)
    # st.write("📅 Discovering movies from similar time periods...")
    if favorite_years:
        # Get movies from the same decades as user's favorites
        decades = set()
        for year in favorite_years:
            decade_start = (year // 10) * 10
            decades.add(decade_start)
        
        for decade_start in list(decades)[:2]:  # Top 2 decades
            try:
                url = f"https://api.themoviedb.org/3/discover/movie"
                params = {
                    "api_key": tmdb_api_key,
                    "primary_release_date.gte": f"{decade_start}-01-01",
                    "primary_release_date.lte": f"{decade_start + 9}-12-31",
                    "sort_by": "vote_average.desc",
                    "vote_count.gte": 100,
                    "page": 1
                }
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    movies = response.json().get("results", [])
                    candidate_movie_ids.update([m["id"] for m in movies[:15]])  # Top 15 per decade
            except Exception as e:
                st.warning(f"Error discovering by decade {decade_start}: {e}")
    
    # Strategy 5: Multi-criteria Discovery (20-30 movies)
    # st.write("🎯 Discovering with combined criteria...")
    try:
        # Combine top genres and cast for more targeted results
        top_genres = ",".join(str(id) for id in list(favorite_genre_ids)[:2])
        top_cast = ",".join(str(id) for id in list(favorite_cast_ids)[:3])
        
        if top_genres and top_cast:
            url = f"https://api.themoviedb.org/3/discover/movie"
            params = {
                "api_key": tmdb_api_key,
                "with_genres": top_genres,
                "with_cast": top_cast,
                "sort_by": "popularity.desc",
                "vote_count.gte": 20,
                "page": 1
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                movies = response.json().get("results", [])
                candidate_movie_ids.update([m["id"] for m in movies[:20]])
    except Exception as e:
        st.warning(f"Error with multi-criteria discovery: {e}")
    
    # Strategy 6: Trending/Popular Backup (10-20 movies)
    # st.write("📈 Adding trending movies as backup...")
    try:
        # Add some trending movies to ensure we have enough candidates
        url = f"https://api.themoviedb.org/3/trending/movie/week"
        params = {"api_key": tmdb_api_key}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            movies = response.json().get("results", [])
            candidate_movie_ids.update([m["id"] for m in movies[:15]])
    except Exception as e:
        st.warning(f"Error getting trending movies: {e}")
    
    # Strategy 7: High-rated movies in favorite genres (backup)
    # st.write("⭐ Adding highly-rated movies in favorite genres...")
    for genre_id in list(favorite_genre_ids)[:2]:
        try:
            url = f"https://api.themoviedb.org/3/discover/movie"
            params = {
                "api_key": tmdb_api_key,
                "with_genres": str(genre_id),
                "sort_by": "vote_average.desc",
                "vote_count.gte": 200,  # High vote count for quality
                "vote_average.gte": 7.0,  # High rating
                "page": 1
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                movies = response.json().get("results", [])
                candidate_movie_ids.update([m["id"] for m in movies[:10]])
        except Exception as e:
            st.warning(f"Error getting high-rated movies for genre {genre_id}: {e}")
    
    # st.write(f"🎬 Built candidate pool with {len(candidate_movie_ids)} movies")
    return candidate_movie_ids

# NEW: Multi-cluster approach for preserving diverse taste profiles
def identify_taste_clusters(favorite_embeddings, favorite_movies_info):
    """
    Identify distinct taste clusters from user's favorite movies
    Returns cluster centers and movie assignments
    """
    if len(favorite_embeddings) <= 2:
        # Too few movies to cluster meaningfully
        return None, None
    
    # Convert embeddings to numpy array
    embeddings_array = torch.stack(favorite_embeddings).cpu().numpy()
    
    # Determine optimal number of clusters (2-3 for 5 movies)
    n_clusters = min(3, max(2, len(favorite_embeddings) // 2))
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings_array)
    cluster_centers = kmeans.cluster_centers_
    
    # Convert back to torch tensors
    cluster_centers_torch = [torch.from_numpy(center) for center in cluster_centers]
    
    return cluster_centers_torch, cluster_labels

def compute_multi_cluster_similarity(candidate_embedding, cluster_centers):
    """
    Compute similarity to multiple cluster centers
    Returns the maximum similarity (best match to any cluster)
    """
    if cluster_centers is None:
        return 0.0
    
    max_similarity = 0.0
    for center in cluster_centers:
        similarity = float(cos_sim(candidate_embedding, center))
        max_similarity = max(max_similarity, similarity)
    
    return max_similarity

def analyze_taste_diversity(favorite_embeddings, favorite_genres, favorite_years):
    """
    Analyze how diverse the user's taste is
    Returns a diversity score and taste profile
    """
    diversity_metrics = {
        "genre_diversity": len(favorite_genres) / 5.0,  # Normalized by max possible
        "temporal_spread": 0.0,
        "embedding_variance": 0.0,
        "taste_profile": "focused"  # or "diverse" or "eclectic"
    }
    
    # Temporal spread
    if len(favorite_years) > 1:
        year_range = max(favorite_years) - min(favorite_years)
        diversity_metrics["temporal_spread"] = min(year_range / 50.0, 1.0)  # Normalize by 50 years
    
    # Embedding variance
    if len(favorite_embeddings) > 1:
        embeddings_array = torch.stack(favorite_embeddings).cpu().numpy()
        # Use sklearn's cosine_similarity for pairwise computation
        pairwise_similarities = sklearn_cosine_similarity(embeddings_array)
        # Get off-diagonal elements (excluding self-similarity)
        mask = ~np.eye(pairwise_similarities.shape[0], dtype=bool)
        avg_similarity = pairwise_similarities[mask].mean()
        diversity_metrics["embedding_variance"] = 1.0 - avg_similarity
    
    # Determine taste profile
    overall_diversity = (diversity_metrics["genre_diversity"] + 
                        diversity_metrics["temporal_spread"] + 
                        diversity_metrics["embedding_variance"]) / 3.0
    
    if overall_diversity < 0.3:
        diversity_metrics["taste_profile"] = "focused"
    elif overall_diversity < 0.6:
        diversity_metrics["taste_profile"] = "diverse"
    else:
        diversity_metrics["taste_profile"] = "eclectic"
    
    return diversity_metrics

# --- Enhanced Recommendation Logic ---
def recommend_movies(favorite_titles):
    # Check cache first
    cache_key = "|".join(sorted(favorite_titles))
    if "recommendation_cache" not in st.session_state:
        st.session_state.recommendation_cache = {}
    
    if cache_key in st.session_state.recommendation_cache:
        cached_result = st.session_state.recommendation_cache[cache_key]
        st.write(f"✅ Using cached results")
        return cached_result
    
    favorite_genres = set()
    favorite_actors = set()
    favorite_directors = set()
    # New sets for IDs
    favorite_genre_ids = set()
    favorite_cast_ids = set()
    favorite_director_ids = set()
    candidate_movie_ids, plot_moods, favorite_years = set(), set(), []
    favorite_narrative_styles = {"tone": [], "complexity": [], "genre_indicator": [], "setting_context": []}
    favorite_embeddings = []
    favorite_movies_info = []  # Store full movie info for clustering analysis

    for title in favorite_titles:
        try:
            search_result = movie_api.search(title)
            if not search_result:
                continue
            
            movie_id = search_result[0].id
            
            # Check cache first
            if movie_id in MOVIE_DETAILS_CACHE:
                details = MOVIE_DETAILS_CACHE[movie_id]
                credits = MOVIE_CREDITS_CACHE[movie_id]
            else:
                # Fetch and cache
                details = movie_api.details(movie_id)
                credits = movie_api.credits(movie_id)
                MOVIE_DETAILS_CACHE[movie_id] = details
                MOVIE_CREDITS_CACHE[movie_id] = credits
            
            # Store movie info for clustering
            movie_info = {
                "title": title,
                "genres": [],
                "year": None
            }
            
            # ✅ Collect genre names - Fixed attribute access
            genres_list = getattr(details, 'genres', [])
            for g in genres_list:
                if isinstance(g, dict):
                    name = g.get('name', '')
                else:
                    name = getattr(g, 'name', '')
                if name:
                    favorite_genres.add(name)
                    movie_info["genres"].append(name)

            # ✅ Collect actor and director names - Fixed attribute access
            cast_list_raw = credits.get('cast', []) if isinstance(credits, dict) else getattr(credits, 'cast', [])
            crew_list = credits.get('crew', []) if isinstance(credits, dict) else getattr(credits, 'crew', [])
            
            # Process cast names
            # Convert to list if needed and safely slice
            if hasattr(cast_list_raw, '__iter__'):
                cast_list = list(cast_list_raw)[:3] if cast_list_raw else []
            else:
                cast_list = []
            
            for c in cast_list:
                if isinstance(c, dict):
                    name = c.get('name', '')
                else:
                    name = getattr(c, 'name', '')
                if name:
                    favorite_actors.add(name)

            # Process director names
            for c in crew_list:
                is_director = False
                name = ''
                if isinstance(c, dict):
                    is_director = c.get('job', '') == 'Director'
                    name = c.get('name', '')
                else:
                    is_director = getattr(c, 'job', '') == 'Director'
                    name = getattr(c, 'name', '')
                
                if is_director and name:
                    favorite_directors.add(name)

            # ✅ Collect genre IDs - Fixed attribute access
            for g in genres_list:
                if hasattr(g, 'id'):
                    favorite_genre_ids.add(g.id)
                elif isinstance(g, dict) and 'id' in g:
                    favorite_genre_ids.add(g['id'])

            # Fixed overview access
            overview = getattr(details, 'overview', '') or ''
            plot_moods.add(infer_mood_from_plot(overview))
            narr_style = infer_narrative_style(overview)
            for key in favorite_narrative_styles:
                favorite_narrative_styles[key].append(narr_style.get(key, ""))
            
            # Fixed release_date access
            release_date = getattr(details, 'release_date', None)
            if release_date:
                try:
                    year = int(release_date[:4])
                    favorite_years.append(year)
                    movie_info["year"] = year
                except (ValueError, TypeError):
                    pass
            
            # ✅ Directly encode as torch tensor
            emb = embedding_model.encode(overview, convert_to_tensor=True)
            favorite_embeddings.append(emb)
            favorite_movies_info.append(movie_info)
            
            # ✅ Collect top 3 cast IDs
            for c in cast_list:
                if isinstance(c, dict):
                    cast_id = c.get('id', 0)
                else:
                    cast_id = getattr(c, 'id', 0)
                if cast_id:
                    favorite_cast_ids.add(cast_id)

            # ✅ Collect directors' IDs  
            for c in crew_list:
                is_director = False
                if isinstance(c, dict):
                    is_director = c.get('job', '') == 'Director'
                    person_id = c.get('id', 0)
                else:
                    is_director = getattr(c, 'job', '') == 'Director'
                    person_id = getattr(c, 'id', 0)
                
                if is_director and person_id:
                    favorite_director_ids.add(person_id)
            
            # We'll build the candidate pool after processing all favorites
            pass
                
        except Exception as e:
            st.warning(f"Error processing {title}: {e}")
            continue

    # Build custom candidate pool using multiple strategies
    candidate_movie_ids = build_custom_candidate_pool(
        favorite_genre_ids, 
        favorite_cast_ids, 
        favorite_director_ids, 
        favorite_years, 
        tmdb.api_key
    )

    st.write(f"✅ Custom candidate pool size: {len(candidate_movie_ids)} movies")

    # Limit to first 150 candidates
    candidate_movie_ids = list(candidate_movie_ids)[:150]

    # Analyze taste diversity
    diversity_metrics = analyze_taste_diversity(favorite_embeddings, favorite_genres, favorite_years)
    st.write(f"🎯 Taste profile: {diversity_metrics['taste_profile']}")
    
    # Identify taste clusters
    cluster_centers, cluster_labels = identify_taste_clusters(favorite_embeddings, favorite_movies_info)
    
    if cluster_centers:
        st.write(f"🎬 Identified {len(cluster_centers)} distinct taste clusters")

    # Add trending movies to candidate set
    trending_scores = get_trending_popularity(tmdb.api_key)

    user_prefs = {
        "subscribed_platforms": [k for k,v in streaming_platform_priority.items() if v>0],
        "preferred_moods": plot_moods,
        "estimated_age": estimate_user_age(favorite_years),
        "taste_diversity": diversity_metrics
    }

    candidate_movies = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_similar_movie_details, mid): mid for mid in candidate_movie_ids}
        for fut in concurrent.futures.as_completed(futures):
            try:
                result = fut.result()
                if result is None:
                    continue
                mid, payload = result
                if payload is None:
                    continue
                m, embedding = payload
                if m is None or embedding is None:
                    continue
                # Fixed vote_count access
                vote_count = getattr(m, 'vote_count', 0)
                if vote_count < 20:
                    continue
                candidate_movies[mid] = (m, embedding)
            except Exception as e:
                st.warning(f"Error processing candidate movie: {e}")
                continue

    # Get valid titles for debugging
    valid_titles = [getattr(m, 'title', 'Unknown') for m, emb in candidate_movies.values() if m and emb is not None]

    if not candidate_movies:
        st.warning("No candidate movies with valid plots or embeddings were found.")
        return [], {}

    # Fetch trending scores before computing movie scores
    trending_scores = get_trending_popularity(tmdb.api_key)

    def compute_score(m, cluster_centers, diversity_metrics):
        try:
            narrative = getattr(m, 'narrative_style', {})
            score = 0.0
            
            # Fixed genres access
            genres = set()
            genres_list = getattr(m, 'genres', [])
            for g in genres_list:
                if isinstance(g, dict):
                    name = g.get('name', '')
                else:
                    name = getattr(g, 'name', '')
                if name:
                    genres.add(name)
            
            score += recommendation_weights['genre_similarity'] * (len(genres & favorite_genres) / max(len(favorite_genres),1))
            
            # Fixed cast and directors access - handle both dict and object formats
            cast_names = set()
            cast_list = getattr(m, 'cast', [])
            for actor in cast_list:
                if isinstance(actor, dict):
                    name = actor.get('name', '')
                else:
                    name = getattr(actor, 'name', '')
                if name:
                    cast_names.add(name)
            
            directors = getattr(m, 'directors', [])
            director_names = set(directors) if isinstance(directors, list) else set()
            
            cast_dir = cast_names | director_names
            score += recommendation_weights['cast_crew'] * (len(cast_dir & favorite_actors) / max(len(favorite_actors),1))
            
            # Fixed release_date access
            try:
                release_date = getattr(m, 'release_date', None)
                if release_date:
                    year_diff = datetime.now().year - int(release_date[:4])
                    if year_diff<=2: score += recommendation_weights['release_year']
                    elif year_diff<=5: score += recommendation_weights['release_year']*0.66
                    elif year_diff<=15: score += recommendation_weights['release_year']*0.33
            except (ValueError, TypeError, AttributeError):
                pass
            
            # Fixed vote_average access
            vote_average = getattr(m, 'vote_average', 0) or 0
            score += recommendation_weights['ratings'] * (vote_average/10)
            
            # Fixed genres access for mood score
            movie_genres = getattr(m, 'genres', [])
            score += recommendation_weights['mood_tone'] * get_mood_score(movie_genres, user_prefs['preferred_moods'])

            # Fixed plot access
            plot = getattr(m, 'plot', '') or getattr(m, 'overview', '') or ''
            narrative = infer_narrative_style(plot)
            narrative_match_score = compute_narrative_similarity(narrative, favorite_narrative_styles)
            score += recommendation_weights['narrative_style'] * narrative_match_score

            # --- NEW: Multi-Cluster Embedding Similarity ---
            movie_id = getattr(m, 'id', None)
            if movie_id and movie_id in candidate_movies:
                candidate_data = candidate_movies[movie_id]
                if len(candidate_data) >= 2 and candidate_data[1] is not None:
                    candidate_embedding = candidate_data[1]
                    
                    # Use multi-cluster similarity for diverse tastes
                    if cluster_centers and diversity_metrics['taste_profile'] in ['diverse', 'eclectic']:
                        embedding_sim_score = compute_multi_cluster_similarity(candidate_embedding, cluster_centers)
                    else:
                        # For focused tastes, use average embedding as before
                        if favorite_embeddings:
                            avg_embedding = torch.mean(torch.stack(favorite_embeddings), dim=0)
                            embedding_sim_score = float(cos_sim(candidate_embedding, avg_embedding))
                        else:
                            embedding_sim_score = 0.0
                    
                    score += recommendation_weights['embedding_similarity'] * embedding_sim_score

            movie_trend_score = trending_scores.get(getattr(m, 'id', 0), 0)
            mood_match_score = get_mood_score(movie_genres, user_prefs['preferred_moods'])
            genre_overlap_score = len(genres & favorite_genres) / max(len(favorite_genres), 1)

            # Adjust trending boost based on taste diversity
            if diversity_metrics['taste_profile'] == 'eclectic':
                # Eclectic users might enjoy trending movies more
                trending_weight = recommendation_weights['trending_factor'] * 1.5
            elif diversity_metrics['taste_profile'] == 'focused':
                # Focused users need stronger alignment for trending boost
                if mood_match_score > 0.5 and genre_overlap_score > 0.4:
                    trending_weight = recommendation_weights['trending_factor']
                else:
                    trending_weight = 0
            else:
                # Standard approach for diverse users
                if mood_match_score > 0.3 and genre_overlap_score > 0.2:
                    trending_weight = recommendation_weights['trending_factor']
                else:
                    trending_weight = 0
            
            score += trending_weight * movie_trend_score

            # Discovery boost for eclectic users
            if diversity_metrics['taste_profile'] == 'eclectic':
                # Boost movies that are somewhat different but not completely unrelated
                if 0.1 < genre_overlap_score < 0.5:
                    score += recommendation_weights['discovery_boost'] * 1.5

            # Apply a small penalty for very old movies
            try:
                if release_date:
                    release_year = int(release_date[:4])
                    if datetime.now().year - release_year > 20:
                        score -= 0.03  # small age penalty
            except (ValueError, TypeError):
                pass

            # Age alignment scoring
            try:
                if release_date:
                    release_year = int(release_date[:4])
                    user_age_at_release = user_prefs['estimated_age'] - (datetime.now().year - release_year)
                    if 15 <= user_age_at_release <= 25:
                        score += recommendation_weights['age_alignment']
                    elif 10 <= user_age_at_release < 15 or 25 < user_age_at_release <= 30:
                        score += recommendation_weights['age_alignment'] * 0.5
            except (ValueError, TypeError):
                pass
                
            return max(score, 0)
        except Exception as e:
            st.warning(f"Error computing score for movie: {e}")
            return 0

    scored = []
    for movie_obj, embedding in candidate_movies.values():
        if movie_obj is None or embedding is None:
            continue
        try:
            score = compute_score(movie_obj, cluster_centers, diversity_metrics)
            vote_count = getattr(movie_obj, 'vote_count', 0)
            score += min(vote_count, 500) / 50000
            scored.append((movie_obj, score))
        except Exception as e:
            st.warning(f"Error scoring movie {getattr(movie_obj, 'title', 'Unknown')}: {e}")
            continue

    st.write(f"✅ Candidate movies count: {len(candidate_movies)}")
    st.write(f"✅ Valid scored movies: {len(scored)}")

    scored.sort(key=lambda x:x[1], reverse=True)
    
    # NEW: Diversify recommendations for eclectic users
    top = []
    low_votes = 0
    used_genres = set()
    
    # Create a set of favorite movie titles for easy checking
    favorite_titles_set = {title.lower() for title in favorite_titles}

    for m, s in scored:
        vote_count = getattr(m, 'vote_count', 0)
        movie_title = getattr(m, 'title', 'Unknown Title')
        
        # Skip if this movie is in the user's favorites
        if movie_title.lower() in favorite_titles_set:
            continue
        
        # Get movie genres
        movie_genres = set()
        genres_list = getattr(m, 'genres', [])
        for g in genres_list:
            if isinstance(g, dict):
                name = g.get('name', '')
            else:
                name = getattr(g, 'name', '')
            if name:
                movie_genres.add(name)
        
        # For eclectic users, ensure genre diversity in recommendations
        if diversity_metrics['taste_profile'] == 'eclectic' and len(top) >= 3:
            # Check if we already have too many movies from the same genre
            genre_overlap = len(movie_genres & used_genres) / max(len(movie_genres), 1)
            if genre_overlap > 0.7:  # Skip if too similar to already selected
                continue
        
        if vote_count < 100:
            if low_votes >= 2: 
                continue
            low_votes += 1
        
        top.append((movie_title, s))
        used_genres.update(movie_genres)
        
        if len(top) == 10: 
            break

    # Before returning, cache the result
    result = (top, candidate_movies)
    st.session_state.recommendation_cache[cache_key] = result
    return result
# Add this code at the END of your existing file (after the recommend_movies function)

def fetch_multiple_movie_details(movie_ids):
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_id = {
            executor.submit(fetch_similar_movie_details, mid): mid 
            for mid in movie_ids[:50]  # Limit to first 50
        }
        for future in concurrent.futures.as_completed(future_to_id):
            mid = future_to_id[future]
            try:
                result = future.result()
                if result and result[1]:
                    results[mid] = result[1]
            except:
                pass
    return results

# ============ STREAMLIT UI CODE ============

st.title("🎬 Screen or Skip")

# Setup flags
if "search_done" not in st.session_state:
    st.session_state["search_done"] = False
if "previous_query" not in st.session_state:
    st.session_state["previous_query"] = ""

# Get input
search_query = st.text_input("search for a movie", key="movie_search")

# ✅ Reset search_done when user types a different movie
if search_query != st.session_state["previous_query"]:
    st.session_state["search_done"] = False
    st.session_state["previous_query"] = search_query

search_results = []

# 3️⃣ Only search if user hasn't just added a movie
if search_query and len(search_query) >= 2 and not st.session_state["search_done"]:
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

# 4️⃣ Show Top 5 only if we have results AND no movie was just added
if search_results:
    st.markdown("### Top 5 Matches")
    cols = st.columns(5)
    for idx, movie in enumerate(search_results):
        with cols[idx]:
            poster_url = f"https://image.tmdb.org/t/p/w200{movie['poster_path']}" if movie.get("poster_path") else None
            if poster_url:
                st.image(poster_url, use_column_width=True)
            st.write(f"**{movie['label']}**")
            if st.button("Add Movie", key=f"add_{idx}"):  # ✅ Simpler button text
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
                    st.session_state["search_done"] = True  # ✅ Hide Top 5
                    st.success(f"✅ Added {clean_title}")
                    st.rerun()

# --- Display Favorite Movies with Posters in a Grid ---
st.subheader("🎥 Your Selected Movies (5 max)")

if st.session_state.favorite_movies:
    cols = st.columns(5)
    for i, movie in enumerate(st.session_state.favorite_movies):
        with cols[i % 5]:
            title = movie["title"]
            year = movie.get("year", "")
            poster = movie.get("poster_path")
            
            if poster:
                poster_url = f"https://image.tmdb.org/t/p/w200{poster}"
                st.image(poster_url, use_column_width=True)
            else:
                st.write("🎬 No poster")
            
            st.write(f"**{title}**")
            if year:
                st.write(f"({year})")
            
            if st.button(f"Remove", key=f"remove_{i}"):
                st.session_state.favorite_movies.pop(i)
                save_session({"favorite_movies": st.session_state.favorite_movies})
                st.rerun()
else:
    st.info("👆 Search and add your 5 favorite movies to get personalized recommendations!")

# Buttons below the grid
col1, col2 = st.columns(2)
with col1:
    if st.button("❌ Clear All"):
        st.session_state.favorite_movies = []
        save_session({"favorite_movies": []})
        st.rerun()

with col2:
    # --- Get Recommendations ---
    if st.button("🎬 Get Recommendations", type="primary"):
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
                    import traceback
                    st.error(traceback.format_exc())

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
            # Find the movie object from candidates
            movie_obj = None
            for m, _ in st.session_state.candidates.values():
                if m and getattr(m, 'title', '') == title:
                    movie_obj = m
                    break
            
            if movie_obj is None:
                continue
            
            st.markdown(f"### {idx}. {movie_obj.title}")
            
            # Create columns for poster and details
            col1, col2 = st.columns([1, 3])
            
            with col1:
                if movie_obj.poster_path:
                    st.image(f"https://image.tmdb.org/t/p/w300{movie_obj.poster_path}", width=150)
                else:
                    st.write("🎬 No poster")
            
            with col2:
                # Show year
                release_year = "N/A"
                try:
                    if hasattr(movie_obj, 'release_date') and movie_obj.release_date:
                        release_year = movie_obj.release_date[:4]
                except:
                    pass
                st.write(f"**Year:** {release_year}")
                
                # Show plot
                overview = getattr(movie_obj, 'overview', None) or getattr(movie_obj, 'plot', None) or "No description available."
                st.write(f"**Plot:** {overview}")

            # Feedback section
            fb_key = f"watch_{idx}"
            liked_key = f"liked_{idx}"

            response = st.radio(
                "Would you watch this?", 
                ["Yes", "No", "Already watched"], 
                key=fb_key, 
                index=None,
                horizontal=True
            )

            liked = None
            if response == "Already watched":
                liked = st.radio(
                    "Did you like it?", 
                    ["Yes", "No"], 
                    key=liked_key, 
                    index=None,
                    horizontal=True
                )

            user_feedback[idx] = {
                "movie": movie_obj.title,
                "movie_id": movie_obj.id,
                "response": response,
                "liked": liked,
            }
            
            st.markdown("---")

        # 2. Submit button at the end only
        if st.button("Submit All Responses", type="primary"):
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
                st.balloons()
            elif success_count > 0:
                st.warning(f"⚠️ {success_count}/{total_responses} responses saved. Some failed to save.")
            else:
                if total_responses == 0:
                    st.warning("⚠️ Please provide at least one response before submitting.")
                else:
                    st.error("❌ Failed to save any responses. Please check your Google Sheets setup.")