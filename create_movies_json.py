import pandas as pd
import json
import os
import numpy as np
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Script to merge MovieLens data into a single JSON for the frontend
# and clean up titles/genres.

# TMDB API Key for patching
TMDB_API_KEY = "87292fa08d459899972b4236bbe540aa"

def fetch_tmdb_details(tmdb_id, media_type='movie'):
    """Fetch details + credits from TMDB."""
    try:
        url = f"https://api.themoviedb.org/3/{media_type}/{tmdb_id}?api_key={TMDB_API_KEY}&append_to_response=credits"
        response = requests.get(url, timeout=3)
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract basic info
            poster_path = data.get('poster_path')
            overview = data.get('overview', '')
            
            # Date
            release_date = data.get('release_date') if media_type == 'movie' else data.get('first_air_date')
            year = None
            if release_date:
                try:
                    year = int(release_date.split('-')[0])
                except: pass
                
            # Cast (Top 5)
            credits = data.get('credits', {})
            cast = [c['name'] for c in credits.get('cast', [])[:5]]
            
            # Genres (if needed to patch)
            genres = [g['name'] for g in data.get('genres', [])]
            
            return {
                'poster_path': poster_path,
                'overview': overview,
                'releaseDate': release_date,
                'year': year,
                'cast': cast,
                'genres': genres,
                'original_language': data.get('original_language')
            }
        elif response.status_code == 429:
            time.sleep(1) # Backoff
            return fetch_tmdb_details(tmdb_id, media_type)
    except Exception as e:
        print(f"Error fetching {tmdb_id}: {e}")
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
    popular_stats = stats[stats['vote_count'] >= 20].sort_values('vote_count', ascending=False)
    # LIMIT to top 500 for testing speed, but ensure we have enough
    popular_movies = popular_stats['movieId'].head(500).tolist()
    
    print(f"Processing Top {len(popular_movies)} movies...")
    
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
    
    # Known TV Shows to patch (simplified for brevity, keeping major ones)
    tv_patches = {
        "Band of Brothers": 4613,
        "Planet Earth": 1920,
        "Planet Earth II": 66902,
        "The Wire": 1438,
        "Breaking Bad": 1396,
        "Game of Thrones": 1399,
        "Rick and Morty": 60625,
        "Avatar: The Last Airbender": 246,
        "Sherlock": 19885,
        "Stranger Things": 66732,
        "The Office": 2316, 
        "Friends": 1668,
        "Seinfeld": 1400,
        "The Sopranos": 1398,
        "Arcane": 94605,
        "Succession": 76331,
        "Severance": 115036
    }
    # Add some common Indian movies to ensure they are treated as movies if needed (not strictly necessary as default is movie)
    
    # Process Loop
    items_to_process = []
    
    for movieId, movie in movies_data.items():
        title = movie['title']
        encoded_id = movieId
        
        genres = movie['genres'].split('|') if pd.notna(movie['genres']) else []
        tmdb_id = int(movie['tmdbId']) if pd.notna(movie['tmdbId']) else None
        
        if not tmdb_id: continue

        media_type = "movie"
        is_patched_tv = False
        
        # Simple TV detection logic (simplified from previous script)
        for tv_name, tv_id in tv_patches.items():
             if tv_name in title and "Escape" not in title: # Basic guard
                 # Strict check
                 if title == tv_name or title.startswith(f"{tv_name} ("):
                     tmdb_id = tv_id
                     media_type = "tv"
                     is_patched_tv = True
                     break
        
        items_to_process.append({
            "movieId": movieId,
            "tmdbId": tmdb_id,
            "media_type": media_type,
            "title": title,
            "original_genres": genres,
            "stats": stats_map.get(movieId, {'vote_average': 0, 'vote_count': 0})
        })

    print(f"Fetching details for {len(items_to_process)} items using threads...")
    
    # Parallel Fetching
    final_metadata = {}
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_item = {
            executor.submit(fetch_tmdb_details, item['tmdbId'], item['media_type']): item 
            for item in items_to_process
        }
        
        count = 0
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            try:
                details = future.result()
                if details:
                    # Merge details
                    movie_data = {
                        "title": item['title'], # Keep original title unless overridden by details? Prefer Original with Year usually.
                        "year": details['year'],
                        "releaseDate": details['releaseDate'],
                        "genres": details['genres'] if details['genres'] else item['original_genres'],
                        "tags": movie_tags_map.get(item['movieId'], []),
                        "tmdbId": item['tmdbId'],
                        "vote_average": round(float(item['stats'].get('vote_average', 0)), 1),
                        "vote_count": int(item['stats'].get('vote_count', 0)),
                        "poster_path": details['poster_path'],
                        "media_type": item['media_type'],
                        "overview": details['overview'],
                        "cast": details['cast'],
                        "original_language": details['original_language']
                    }
                    
                    # Update title if year missing in original but found in details
                    # Actually, let's trust the TMDB title? No, keep MovieLens title for consistency with dataset, 
                    # but maybe use TMDB for cleaner display? Let's use MovieLens title but strip year if needed.
                    
                    final_metadata[str(item['movieId'])] = movie_data
                else:
                    # Fallback if fetch failed
                    final_metadata[str(item['movieId'])] = {
                        "title": item['title'],
                        "genres": item['original_genres'],
                        "tmdbId": item['tmdbId'],
                        "media_type": item['media_type'],
                        "poster_path": None
                    }
            except Exception as exc:
                print(f"Generated an exception for {item['title']}: {exc}")
            
            count += 1
            if count % 50 == 0:
                print(f"Completed {count}/{len(items_to_process)}")

    print(f"Created metadata for {len(final_metadata)} items")
    print("Saving to deployment/movies.json...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_metadata, f, ensure_ascii=False)
        
    print("Saved!")

if __name__ == "__main__":
    create_metadata(
        'ml-32m/ml-32m/movies.csv',
        'ml-32m/ml-32m/links.csv',
        'ml-32m/ml-32m/ratings.csv',
        'ml-32m/ml-32m/tags.csv',
        'deployment/movies.json'
    )

