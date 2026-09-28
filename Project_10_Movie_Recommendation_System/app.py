# app/app.py (Enhanced Netflix-Style Movie Recommender - FIXED)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --------------------------
# SET WORKING DIRECTORY
# --------------------------
# This ensures file paths work regardless of where you run the script
BASE_DIR = Path("G:/Machine Learning Projects/Movie_Recommendation_System")  # Goes up one level from app/
MODELS_DIR = BASE_DIR / 'models'
DATA_DIR = BASE_DIR / 'data' / 'ml-latest-small'

# --------------------------
# PAGE CONFIG
# --------------------------
st.set_page_config(
    page_title="🎬 CineMatch - Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# --------------------------
# CUSTOM CSS (Netflix Style)
# --------------------------
st.markdown("""
<style>
    /* Global background */
    .stApp {
        background: linear-gradient(180deg, #141414 0%, #1a1a1a 100%);
        color: #ffffff;
    }
    
    /* Title styling */
    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #e50914, #f5c518);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        margin-bottom: 0;
        text-align: center;
    }
    .sub-title {
        color: #b3b3b3;
        font-size: 1.2rem;
        margin-top: 0;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    /* Card-style movie posters */
    .movie-card {
        background: #1f1f1f;
        border-radius: 8px;
        padding: 10px;
        margin: 5px;
        transition: transform 0.3s;
        border: 1px solid #2a2a2a;
        text-align: center;
    }
    .movie-card:hover {
        transform: scale(1.05);
        border-color: #e50914;
        box-shadow: 0 0 20px rgba(229, 9, 20, 0.3);
    }
    .movie-title {
        color: #ffffff;
        font-weight: 500;
        font-size: 0.85rem;
        margin-top: 5px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .match-score {
        color: #46d369;
        font-size: 0.75rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1a1a1a;
    }
    
    /* Search box */
    .stTextInput > div > div > input {
        background-color: #2a2a2a;
        border: 1px solid #333;
        color: white;
        border-radius: 20px;
        padding: 10px 20px;
    }
    .stTextInput > div > div > input:focus {
        border-color: #e50914;
        box-shadow: 0 0 0 2px rgba(229, 9, 20, 0.3);
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        background-color: #2a2a2a;
        border-radius: 8px;
        color: white;
    }
    
    /* Metrics */
    .stMetric {
        background: #1f1f1f;
        border-radius: 8px;
        padding: 10px;
        border-left: 3px solid #e50914;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #2a2a2a;
        border-radius: 20px;
        padding: 8px 20px;
        color: #b3b3b3;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e50914;
        color: white;
    }
    
    /* Custom banner */
    .banner-container {
        position: relative;
        width: 100%;
        overflow: hidden;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    .banner-overlay {
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 30px;
        background: linear-gradient(transparent, rgba(0,0,0,0.8));
    }
</style>
""", unsafe_allow_html=True)

# --------------------------
# HEADER WITH BANNER
# --------------------------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<p class="main-title">🎬 CineMatch</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Your Personal Movie Recommender</p>', unsafe_allow_html=True)

# Try to display banner image
try:
    # Look for banner in multiple locations
    banner_paths = [
        BASE_DIR / 'assets' / 'vjpg',
        BASE_DIR / 'assets' / 'bg.jpg',
        BASE_DIR / 'app' / 'assets' / 'vjpg',
    ]
    
    banner_found = False
    for path in banner_paths:
        if path.exists():
            from PIL import Image
            banner = Image.open(path)
            st.image(banner, use_container_width=True)
            banner_found = True
            break
    
    if not banner_found:
        # Display a Netflix-style banner using pure CSS
        st.markdown("""
        <div style="
            background: linear-gradient(90deg, #e50914 0%, #141414 70%);
            padding: 40px;
            border-radius: 12px;
            text-align: center;
            margin-bottom: 20px;
        ">
            <h2 style="color: white; font-size: 2rem; margin: 0;">
                 Find Your Next Favorite Movie
            </h2>
            <p style="color: #b3b3b3; margin: 10px 0 0 0;">
                Powered by Collaborative Filtering & SVD
            </p>
        </div>
        """, unsafe_allow_html=True)
except Exception as e:
    # Fallback banner
    st.markdown("""
    <div style="
        background: linear-gradient(90deg, #e50914 0%, #141414 70%);
        padding: 40px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 20px;
    ">
        <h2 style="color: white; font-size: 2rem; margin: 0;">
             Find Your Next Favorite Movie
        </h2>
        <p style="color: #b3b3b3; margin: 10px 0 0 0;">
            Powered by Collaborative Filtering & SVD
        </p>
    </div>
    """, unsafe_allow_html=True)

# --------------------------
# LOAD MODELS
# --------------------------
@st.cache_resource
def load_models():
    # Load models from the correct paths
    svd_path = MODELS_DIR / 'svd_model.pkl'
    sim_path = MODELS_DIR / 'item_similarity.pkl'
    mapping_path = MODELS_DIR / 'movie_mapping.pkl'
    metadata_path = MODELS_DIR / 'metadata.pkl'
    
    # Check if files exist
    if not all([p.exists() for p in [svd_path, sim_path, mapping_path, metadata_path]]):
        st.error("Model files not found! Please run train_model.py first.")
        st.stop()
    
    svd_model = joblib.load(svd_path)
    item_similarity = joblib.load(sim_path)
    movie_mapping = joblib.load(mapping_path)
    metadata = joblib.load(metadata_path)
    return svd_model, item_similarity, movie_mapping, metadata

svd_model, item_similarity, movie_mapping, metadata = load_models()

# Create movie DataFrame
movie_df = pd.DataFrame(list(movie_mapping.items()), columns=['movieId', 'title'])
movie_df['title_lower'] = movie_df['title'].str.lower()

# --------------------------
# TMDB API CONFIG
# --------------------------
TMDB_API_KEY = "31d16158972ce9ab21415da26da32946"  # Your actual key
TMDB_BASE_URL = "https://api.themoviedb.org/3"
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w200"

@st.cache_data(ttl=86400)
def get_movie_poster(title, year=None):
    """Fetch poster URL from TMDB."""
    try:
        search_url = f"{TMDB_BASE_URL}/search/movie"
        params = {
            'api_key': TMDB_API_KEY,
            'query': title,
            'language': 'en-US'
        }
        if year:
            params['year'] = year
        response = requests.get(search_url, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if data['results']:
                poster_path = data['results'][0].get('poster_path')
                if poster_path:
                    return f"{POSTER_BASE_URL}{poster_path}"
    except Exception as e:
        pass
    return None

@st.cache_data(ttl=86400)
def get_movie_details(title, year=None):
    """Get movie details including poster and year."""
    try:
        search_url = f"{TMDB_BASE_URL}/search/movie"
        params = {
            'api_key': TMDB_API_KEY,
            'query': title,
            'language': 'en-US'
        }
        if year:
            params['year'] = year
        response = requests.get(search_url, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if data['results']:
                result = data['results'][0]
                poster_path = result.get('poster_path')
                release_date = result.get('release_date', '')
                return {
                    'poster': f"{POSTER_BASE_URL}{poster_path}" if poster_path else None,
                    'year': release_date[:4] if release_date else None,
                    'title': result.get('title', title)
                }
    except:
        pass
    return None

# --------------------------
# SIDEBAR
# --------------------------
with st.sidebar:
    st.image("https://www.themoviedb.org/assets/2/v4/logos/v2/blue_square_2-d537fb228cf3ded904ef09b136fe3fec72548ebc1fea3fbbd1ad9e36364db38b.svg", width=150)
    st.header(" Dataset Info")
    col1, col2 = st.columns(2)
    col1.metric("Users", metadata['n_users'])
    col2.metric("Movies", metadata['n_movies'])
    st.metric("Ratings", f"{metadata['n_ratings']:,}")
    st.metric("Sparsity", f"{metadata['sparsity']:.1f}%")
    st.divider()
    st.caption(f"SVD RMSE: {metadata['svd_rmse']:.4f}")
    st.caption(f"SVD MAE: {metadata['svd_mae']:.4f}")
    st.divider()
    st.markdown("** How it works:**")
    st.caption("1. We use collaborative filtering to find movies similar to your selection.")
    st.caption("2. Posters are fetched from TMDB (if available).")
    st.caption("3. The SVD model predicts ratings with ~0.88 RMSE.")

# --------------------------
# MAIN TABS
# --------------------------
tab1, tab2, tab3 = st.tabs([" Get Recommendations", " Rate a Movie", " Dashboard"])

# ============================================================
# TAB 1: RECOMMENDATIONS
# ============================================================
with tab1:
    st.subheader("🔍 Find Movies You'll Love")
    
    # Search bar
    search_term = st.text_input("Search for a movie", "The Usual Suspects")
    
    # Filter movies based on search
    if search_term:
        filtered = movie_df[movie_df['title_lower'].str.contains(search_term.lower())]
    else:
        filtered = movie_df.sample(min(100, len(movie_df)))
    
    if not filtered.empty:
        selected_movie = st.selectbox(
            "Select a movie",
            filtered['title'].tolist()
        )
        
        if selected_movie:
            movie_id = movie_df[movie_df['title'] == selected_movie]['movieId'].values[0]
            
            # Get recommendations
            try:
                similar_movies = item_similarity[movie_id].sort_values(ascending=False).head(11)
                similar_ids = similar_movies.index[1:11]  # Skip itself
                
                # Build recommendations with posters
                rec_data = []
                with st.spinner("Fetching movie posters..."):
                    for mid in similar_ids:
                        if mid in movie_mapping:
                            title = movie_mapping[mid]
                            sim = similar_movies[mid] * 100
                            
                            # Try to extract year from title
                            year = None
                            if '(' in title and ')' in title:
                                try:
                                    year = int(title.split('(')[-1].split(')')[0])
                                except:
                                    pass
                            
                            # Get poster
                            movie_info = get_movie_details(title, year)
                            poster = movie_info['poster'] if movie_info else None
                            
                            rec_data.append({
                                'title': title,
                                'similarity': sim,
                                'poster': poster,
                                'year': movie_info['year'] if movie_info else None
                            })
                
                # Display in a grid
                st.subheader(f"🎥 Because you liked '{selected_movie}'...")
                
                # Show in rows of 5
                cols = st.columns(5)
                for idx, item in enumerate(rec_data[:10]):
                    with cols[idx % 5]:
                        if item['poster']:
                            try:
                                st.image(item['poster'], use_container_width=True)
                            except:
                                st.image("https://via.placeholder.com/200x300/2a2a2a/ffffff?text=No+Poster", use_container_width=True)
                        else:
                            st.image("https://via.placeholder.com/200x300/2a2a2a/ffffff?text=No+Poster", use_container_width=True)
                        
                        # Show title with match score
                        display_title = item['title'][:25] + "..." if len(item['title']) > 25 else item['title']
                        st.markdown(f"<div class='movie-title'>{display_title}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='match-score'>⭐ {item['similarity']:.1f}% match</div>", unsafe_allow_html=True)
                
            except Exception as e:
                st.warning(f"Could not get recommendations: {e}")
    else:
        st.info("No movies found. Try a different search term.")

# ============================================================
# TAB 2: RATE A MOVIE
# ============================================================
with tab2:
    st.subheader("⭐ Predict Your Rating")
    st.caption("We'll predict how much you'd like a movie based on your other ratings.")
    
    user_id = st.number_input("Enter your User ID (1-610)", min_value=1, max_value=610, value=1)
    
    # Random sample for selection
    sample_movies = movie_df.sample(min(100, len(movie_df)))['title'].tolist()
    movie_to_rate = st.selectbox("Select a movie to predict your rating for", sample_movies)
    
    if st.button("Predict My Rating"):
        if movie_to_rate:
            movie_id = movie_df[movie_df['title'] == movie_to_rate]['movieId'].values[0]
            try:
                pred = svd_model.predict(user_id, movie_id)
                
                # Get poster for the movie
                movie_info = get_movie_details(movie_to_rate)
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    if movie_info and movie_info['poster']:
                        st.image(movie_info['poster'], width=150)
                    else:
                        st.image("https://via.placeholder.com/150x225/2a2a2a/ffffff?text=No+Poster", width=150)
                
                with col2:
                    st.metric(
                        label=f"Predicted Rating for '{movie_to_rate}'",
                        value=f"⭐ {pred.est:.2f} / 5.0",
                        delta=f"Confidence: {pred.details['was_impossible']}"
                    )
            except Exception as e:
                st.error(f"Error making prediction: {e}")

# ============================================================
# TAB 3: DASHBOARD
# ============================================================
with tab3:
    st.subheader(" Dataset Dashboard")
    
    try:
        ratings_path = DATA_DIR / 'ratings.csv'
        movies_path = DATA_DIR / 'movies.csv'
        
        if ratings_path.exists():
            ratings_df = pd.read_csv(ratings_path)
        else:
            st.error("Ratings data not found!")
            st.stop()
        
        if movies_path.exists():
            movies_df = pd.read_csv(movies_path)
        else:
            movies_df = None
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Rating Distribution**")
            fig, ax = plt.subplots(figsize=(8, 4))
            ratings_df['rating'].value_counts().sort_index().plot(kind='bar', ax=ax, color='steelblue')
            ax.set_title('Rating Distribution')
            ax.set_xlabel('Rating')
            ax.set_ylabel('Count')
            ax.set_facecolor('#1a1a1a')
            fig.patch.set_facecolor('#1a1a1a')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            st.pyplot(fig)
        
        with col2:
            if movies_df is not None:
                st.write("**Top Genres**")
                all_genres = movies_df['genres'].str.split('|').explode()
                genre_counts = all_genres.value_counts().head(10)
                fig, ax = plt.subplots(figsize=(8, 4))
                genre_counts.plot(kind='barh', ax=ax, color='coral')
                ax.set_title('Top 10 Genres')
                ax.set_xlabel('Count')
                ax.set_facecolor('#1a1a1a')
                fig.patch.set_facecolor('#1a1a1a')
                ax.tick_params(colors='white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.title.set_color('white')
                st.pyplot(fig)

        st.divider()
        st.caption("Data source: MovieLens Latest Small (100k ratings)")
    except Exception as e:
        st.error(f"Error loading dashboard data: {e}")

# --------------------------
# FOOTER
# --------------------------
st.divider()
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.caption("Built with using Streamlit, Surprise, and TMDB API")
    st.caption("Movie posters provided by TMDB")

# --------------------------
# RUN: streamlit run app/app.py
# --------------------------