
import requests
import json
import os

BASE_URL = "http://localhost:8003"
MOVIES_PATH = "deployment/movies.json"

def check_tv_content():
    if os.path.exists(MOVIES_PATH):
        with open(MOVIES_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tv_count = 0
        media_types = {}
        for mid, m in data.items():
            mt = m.get("media_type", "unknown")
            media_types[mt] = media_types.get(mt, 0) + 1
            if mt == "tv":
                tv_count += 1
                
        print(f"Media Type Counts: {media_types}")
    else:
        print("movies.json not found")

def check_similar_endpoint():
    # Try item ID 11 (American President)
    try:
        r = requests.get(f"{BASE_URL}/similar/11?k=5")
        if r.status_code == 200:
            res = r.json()
            items = res.get("similar_items", [])
            print(f"Similar items for ID 11 (k=5): Found {len(items)}")
            for i in items:
                print(f" - {i}")
        else:
            print(f"Similar endpoint failed: {r.status_code} {r.text}")
    except Exception as e:
        print(f"Request failed (backend might be down?): {e}")

if __name__ == "__main__":
    print("--- Checking Content ---")
    check_tv_content()
    print("\n--- Checking Endpoint ---")
    check_similar_endpoint()
