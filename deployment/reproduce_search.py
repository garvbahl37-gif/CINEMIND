
import json
import os
import logging
from typing import List, Dict, Any

# Mock setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MOVIES_PATH = "movies.json"
movies_data = {}

def load_movies():
    global movies_data
    if os.path.exists(MOVIES_PATH):
        with open(MOVIES_PATH, 'r', encoding='utf-8') as f:
            movies_data = json.load(f)
        print(f"Loaded {len(movies_data)} movies")
    else:
        print("movies.json not found!")

def search_movies(q: str):
    print(f"\nSearching for: '{q}'")
    q_lower = q.lower()
    scored_results = []
    
    KEYWORD_MAP = {
        "hindi": ["bollywood", "india", "indian"],
        "sci-fi": ["science fiction", "scifi", "futuristic"],
        "romance": ["romantic", "love"],
        "animated": ["animation", "cartoon", "anime"]
    }
    
    LANGUAGE_MAP = {
        "english": "en",
        "hindi": "hi",
        "french": "fr",
        "spanish": "es",
        "korean": "ko",
        "japanese": "ja"
    }

    target_lang = None
    for lang, code in LANGUAGE_MAP.items():
        if lang in q_lower:
            target_lang = code
            break
            
    print(f"Target Lang: {target_lang}")

    expanded_tokens = set(q_lower.split())
    for token in q_lower.split():
        if token in KEYWORD_MAP:
            expanded_tokens.update(KEYWORD_MAP[token])
    
    print(f"Expanded Tokens: {expanded_tokens}")

    # DEBUG: Check map keys
    # print(f"DEBUG MAP KEYS: {[repr(k) for k in KEYWORD_MAP.keys()]}")

    # Mock intent filters (empty for 1 word)
    intent_filters = {} 
            
    for item_id, movie in movies_data.items():
        score = 0
        title = movie.get("title", "").lower()
        tags = [t.lower() for t in movie.get("tags", [])]
        genres = [g.lower() for g in movie.get("genres", [])]
        
        # DEBUG: Trace movie 11
        is_debug = (item_id == "11")
        if is_debug:
            print(f"DEBUG MOVIE 11: q={repr(q_lower)} genres={[repr(g) for g in genres]}")

        # 0. Language Match
        if target_lang and movie.get("original_language") == target_lang:
            score += 100
            if is_debug: print("  +100 Lang match")
        
        # 1. Title Match
        if q_lower == title:
            score += 100
            if is_debug: print("  +100 Exact Title")
        elif q_lower in title:
            score += 50
            if is_debug: print("  +50 Partial Title")
            
        # 2. Tag Match
        for tag in tags:
            if q_lower == tag:
                score += 30
            elif q_lower in tag:
                score += 20
            elif tag in q_lower and len(tag) > 2:
                score += 25
            
            if tag in expanded_tokens:
                 score += 15

        # 3. Genre Match
        for genre in genres:
            if q_lower == genre:
                score += 20
                if is_debug: print(f"  +20 Exact Genre '{genre}'")
            elif genre in q_lower:
                score += 20
                if is_debug: print(f"  +20 Partial Genre '{genre}'")
        
        if is_debug: print(f"  FINAL SCORE: {score}")

        if score > 0:
            scored_results.append({
                "item_id": item_id,
                "title": movie.get("title"),
                "score": score,
                "genres": movie.get("genres"),
                "lang": movie.get("original_language")
            })
    
    scored_results.sort(key=lambda x: x["score"], reverse=True)
    
    print(f"Found {len(scored_results)} results")
    for r in scored_results[:5]:
        print(f" - {r['title']} (Score: {r['score']}) [{r['genres']}] Lang: {r['lang']}")

if __name__ == "__main__":
    load_movies()
    search_movies("comedy")
    search_movies("hindi")
