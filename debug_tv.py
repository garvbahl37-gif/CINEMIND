import json

data = json.load(open('deployment/movies.json', encoding='utf-8', errors='ignore'))
targets = ["Planet Earth", "Band of Brothers", "Escape from Planet Earth", "Blue Planet II"]

with open('debug_output.txt', 'w', encoding='utf-8') as f:
    for k, v in data.items():
        title = v['title']
        if any(t in title for t in targets):
             f.write(f"Title: '{title}' | ID: {v['tmdbId']} | Type: {v['media_type']} | Poster: {v.get('poster_path')} | Date: {v.get('releaseDate')}\n")
