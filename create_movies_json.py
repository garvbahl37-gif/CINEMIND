import pandas as pd
import json
import os
import numpy as np
import requests
import time

# Script to merge MovieLens data into a single JSON for the frontend
# and clean up titles/genres.

# TMDB API Key for patching
TMDB_API_KEY = "3fd2be6f0c70a2a598f084ddfb75487c"

def fetch_tmdb_poster(tmdb_id, media_type='movie'):
    """Fetch the actual poster path from TMDB to permanently fix missing images."""
    try:
        url = f"https://api.themoviedb.org/3/{media_type}/{tmdb_id}?api_key={TMDB_API_KEY}"
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            data = response.json()
            return data.get('poster_path')
        elif response.status_code == 429:
            time.sleep(1) # Backoff
            return fetch_tmdb_poster(tmdb_id, media_type)
    except:
        pass
    return None

def create_metadata(movies_csv_path, links_csv_path, ratings_csv_path, tags_csv_path, output_path):
    print("Loading movies.csv...")
    movies_df = pd.read_csv(movies_csv_path)
    print(f"Loaded {len(movies_df)} movies")

    links_df = pd.read_csv(links_csv_path)
    print(f"Loaded {len(links_df)} links")
    
    # Merge movies with TMDb links
    movies_df = movies_df.merge(links_df[['movieId', 'tmdbId']], on='movieId', how='left')
    
    # Helper to clean titles
    def clean_title(title):
        return title.strip()
    
    movies_df['title'] = movies_df['title'].apply(clean_title)

    # Load ratings to get popularity
    print("Loading ratings to find popular movies...")
    # Read only necessary columns to save memory
    ratings = pd.read_csv(ratings_csv_path, usecols=['movieId', 'rating'])
    
    # Calculate average rating and vote count
    # Group by movieId and calculate count and mean
    stats = ratings.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
    stats.columns = ['movieId', 'vote_average', 'vote_count']
    
    # Filter for statistically significant movies
    popular_stats = stats[stats['vote_count'] >= 50].sort_values('vote_count', ascending=False)
    # Taking top 20000 to cover enough ground
    popular_movies = popular_stats['movieId'].head(20000).tolist()
    
    stats_map = popular_stats.set_index('movieId').to_dict('index')
    
    # Filter movies_df to only popular ones
    movies_df = movies_df[movies_df['movieId'].isin(popular_movies)]
    
    # Load tags
    print("Loading tags...")
    try:
        tags_df = pd.read_csv(tags_csv_path)
        tags_df = tags_df[tags_df['movieId'].isin(popular_movies)]
        tags_df['tag'] = tags_df['tag'].astype(str).str.lower()
        tag_counts = tags_df.groupby(['movieId', 'tag']).size().reset_index(name='count')
        top_tags = tag_counts.sort_values(['movieId', 'count'], ascending=[True, False]).groupby('movieId').head(5)
        movie_tags_map = top_tags.groupby('movieId')['tag'].apply(list).to_dict()
    except Exception as e:
        print(f"Warning: Could not load tags: {e}")
        movie_tags_map = {}

    print("Creating metadata dictionary...")
    metadata = {}
    
    # Convert movies dataframe to dictionary for faster iteration
    movies_data = movies_df.set_index('movieId').to_dict('index')
    
    count = 0
    
    # Known TV Shows to patch
    tv_patches = {
        "Band of Brothers": 4613,
        "Planet Earth": 1920, # Will require careful matching
        "Planet Earth II": 66902,
        "The Wire": 1438,
        "Breaking Bad": 1396,
        "Game of Thrones": 1399,
        "Rick and Morty": 60625,
        "Cosmos": 58474,
        "Avatar: The Last Airbender": 246,
        "Sherlock": 19885,
        "Firefly": 1437,
        "True Detective": 46648,
        "Black Mirror": 42009,
        "Over the Garden Wall": 61617,
        "Blue Planet II": 74313,
        "Stranger Things": 66732,
        "The Mandalorian": 82856,
        "Chernobyl": 87108,
        "The Crown": 65499,
        "Westworld": 63247,
        "Fargo": 60622,
        "Better Call Saul": 60059,
        "Dark": 70523,
        "The Office": 2316, 
        "Friends": 1668,
        "Seinfeld": 1400,
        "The Sopranos": 1398,
        "Arcane": 94605,
        "The Last of Us": 100088,
        "Succession": 76331,
        "Severance": 115036, 
        "Andor": 83867
    }

    poster_overrides = {
        66902: "/u88QjNj1xyGwnZMtSZjA9jDNNro.jpg", # Planet Earth II
        4613: "/8JMXquNmdMUy2n2RgW8gfOM0O3l.jpg", # Band of Brothers
        1920: "/lA9CNSdo50iQPZ8A2fyVpMvJZAf.jpg"   # Planet Earth
    }

    metadata_overrides = {
        4613: {"title": "Band of Brothers", "date": "2001-09-09"},
        1920: {"title": "Planet Earth", "date": "2006-03-05"},
        74313: {"title": "Blue Planet II", "date": "2017-10-29"}
    }

    for movieId, movie in movies_data.items():
        title = movie['title']
        encoded_id = movieId # Use original movieId as key
        
        genres = movie['genres'].split('|') if pd.notna(movie['genres']) else []
        tmdb_id = int(movie['tmdbId']) if pd.notna(movie['tmdbId']) else None
        poster_path_val = None 
        media_type = "movie"
        
        # Check for TV Show patches
        is_patched_tv = False
        
        # HARD EXCLUSION: Prevent "Escape from Planet Earth" from being patched as TV
        if "Escape from Planet Earth" in title:
             is_patched_tv = False
        # Special case for Planet Earth to avoid "II" collision
        elif "Planet Earth" in title and "II" not in title and ("Planet Earth (" in title or title == "Planet Earth"):
             tmdb_id = tv_patches["Planet Earth"]
             media_type = "tv"
             is_patched_tv = True
        else:
            for tv_name, tv_id in tv_patches.items():
                if "Escape from Planet Earth" in title and tv_name == "Planet Earth":
                     print(f"DEBUG: Checking {title} against {tv_name}")
                     print(f"  Strict: {title == tv_name or title.startswith(f'{tv_name} (')}")
                     print(f"  In Title: {tv_name in title}")
                     print(f"  Escape Not In: {'Escape' not in title}")

                # Strict matching details:
                # 1. Exact match
                # 2. "Title (" for years
                # 3. SPECIAL OVERRIDES for known tricky titles
                match = False
                if title == tv_name or title.startswith(f"{tv_name} ("):
                    match = True
                
                # Special looser overrides for specific requested failures:
                if not match and tv_name in ["Band of Brothers", "Planet Earth"]:
                     # Only match if it actually contains the name properly, preventing "Escape from Planet Earth"
                     # For Band of Brothers, it's unique enough usually.
                     # For Planet Earth, avoid "Escape from"
                     if tv_name in title and "Escape" not in title:
                         match = True

                if "Escape" in title:
                    match = False

                if match:
                    # Skip Planet Earth here as it's handled above (actually, wait, the loop handles it if I didn't verify the special case above?)
                    # The special case block above handles Planet Earth logic partly, let's keep it safe.
                    if tv_name == "Planet Earth": 
                        # Check "II" collision again just in case
                        if "II" in title: continue
                        
                    tmdb_id = tv_id
                    media_type = "tv"
                    is_patched_tv = True
                    print(f"MATCH: '{title}' matched TV '{tv_name}'")
                    break
        
        # Fetch Poster if it's a patched TV show (PERMANENT FIX)
        if is_patched_tv and tmdb_id:
             # Check override first
             if tmdb_id in poster_overrides:
                 poster_path_val = poster_overrides[tmdb_id]
             else:
                 print(f"Fetching poster for TV: {title} ({tmdb_id})...")
                 fetched_poster = fetch_tmdb_poster(tmdb_id, 'tv')
                 if fetched_poster:
                     poster_path_val = fetched_poster
                     print(f"  -> Found: {poster_path_val}")
                 else:
                     print(f"  -> Failed to fetch.")
        
        # Apply Metadata Overrides (Title, Date) if patched TV
        if is_patched_tv and tmdb_id and tmdb_id in metadata_overrides:
             meta = metadata_overrides[tmdb_id]
             if "title" in meta: title = meta["title"]
             # We don't have a 'date' field in the final json yet, but we can update 'release_date' or just rely on title fix.
             # Actually, movies.json structure (from app.py viewing) uses: tmdbId, title, poster.
             # It doesn't seem to store 'year' or 'date' explicitly in the root?
             # Let's check App.tsx usage. App.tsx uses 'releaseDate' from API details.
             # The search suggestion uses title from movies.json.
             # So updating title here fixes the displayed title in search/lists.
             
        # Get tags for this movie
        movie_tags = movie_tags_map.get(movieId, [])
        
        # Extract year
        year = None
        if '(' in title and ')' in title:
             try:
                 year_str = title[title.rfind('(')+1:title.rfind(')')]
                 if year_str.isdigit() and len(year_str) == 4:
                     year = int(year_str)
                     title = title[:title.rfind('(')].strip()
             except:
                 pass
        
        # Override Year from Metadata (Fix for "1956")
        if is_patched_tv and tmdb_id and tmdb_id in metadata_overrides:
             meta = metadata_overrides[tmdb_id]
             if "date" in meta:
                 try:
                     year = int(meta["date"].split('-')[0])
                 except: pass
        
        # Get stats
        stats = stats_map.get(movieId, {'vote_average': 0, 'vote_count': 0})

        # Metadata dictionary
        movie_data = {
            "title": title,
            "year": year,
            "releaseDate": f"{year}-01-01" if year else None, # Add releaseDate for frontend
            "genres": genres,
            "tags": movie_tags,
            "tmdbId": tmdb_id,
            "vote_average": round(float(stats.get('vote_average', 0)), 1),
            "vote_count": int(stats.get('vote_count', 0)),
            "poster_path": poster_path_val,
            "media_type": media_type
        }
        
        metadata[str(encoded_id)] = movie_data
        
        count += 1
        if count % 1000 == 0:
            print(f"Processed {count} movies...")

    print(f"Created metadata for {len(metadata)} items")
    print("Saving to deployment/movies.json...")
    
    # Save full version
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False)
        
    print("Saved!")

if __name__ == "__main__":
    create_metadata(
        'ml-32m/ml-32m/movies.csv',
        'ml-32m/ml-32m/links.csv',
        'ml-32m/ml-32m/ratings.csv',
        'ml-32m/ml-32m/tags.csv',
        'deployment/movies.json'
    )
