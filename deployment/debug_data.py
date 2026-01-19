
import json
import os

MOVIES_PATH = "movies.json"

def debug_search():
    if not os.path.exists(MOVIES_PATH):
        print("movies.json not found")
        return

    with open(MOVIES_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} movies")

    # Check for Movie ID 11 (American President - matches Comedy)
    movie11 = data.get("11")
    if movie11:
        print(f"Movie 11 Genres: {movie11.get('genres')}")
        g = [x.lower() for x in movie11.get('genres', [])]
        print(f"Movie 11 Genres (lower): {g}")
        if "comedy" in g:
            print("Movie 11 has comedy!")
        else:
            print("Movie 11 missing comedy??")
    else:
        print("Movie 11 not found")

    # Check for Hindi existence
    hindi_count = 0
    bollywood_count = 0
    comedy_count = 0
    
    for mid, m in data.items():
        # Check Lang
        if m.get("original_language") == "hi":
            hindi_count += 1
            print(f"Found Hindi movie: {m.get('title')}")
        
        # Check Genres for Comedy
        gs = [x.lower() for x in m.get("genres", [])]
        if "comedy" in gs:
            comedy_count += 1
            
        # Check Tags/Keywords
        tags = [str(t).lower() for t in m.get("tags", [])]
        title = m.get("title", "").lower()
        if "bollywood" in tags or "india" in tags or "hindi" in tags or "bollywood" in title:
            bollywood_count += 1

    print(f"Total Hindi Language Movies: {hindi_count}")
    print(f"Total Bollywood/India Tagged Movies: {bollywood_count}")
    print(f"Total Comedy Movies: {comedy_count}")

if __name__ == "__main__":
    debug_search()
