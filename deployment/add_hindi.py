
import json
import os

MOVIES_PATH = "movies.json"

def add_hindi_movies():
    if not os.path.exists(MOVIES_PATH):
        print("movies.json not found")
        return

    with open(MOVIES_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Find a safe ID
    existing_ids = [int(k) for k in data.keys() if k.isdigit()]
    next_id = max(existing_ids) + 1 if existing_ids else 10000

    new_movies = [
        {
            "title": "3 Idiots (2009)",
            "genres": ["Comedy", "Drama"],
            "year": 2009,
            "original_language": "hi",
            "tags": ["bollywood", "india", "engineering", "friendship", "romance"],
            "overview": "Two friends start a quest for a lost buddy, who was once an optimistic and unconventional student...",
            "vote_average": 8.4
        },
        {
            "title": "Lagaan: Once Upon a Time in India (2001)",
            "genres": ["Drama", "Romance", "Music"],
            "year": 2001,
            "original_language": "hi",
            "tags": ["bollywood", "cricket", "india", "colonialism", "oscar nominee"],
            "overview": "The people of a small village in Victorian India stake their future on a game of cricket against their ruthless British rulers...",
             "vote_average": 8.1
        },
        {
             "title": "Dilwale Dulhania Le Jayenge (1995)",
             "genres": ["Comedy", "Drama", "Romance"],
             "year": 1995,
             "original_language": "hi",
             "tags": ["bollywood", "romance", "classic", "shah rukh khan", "kajol"],
             "overview": "A young man and woman fall in love on a trip to Europe...",
             "vote_average": 8.2
        }
    ]

    for m in new_movies:
        mid = str(next_id)
        data[mid] = m
        print(f"Added {m['title']} as ID {mid}")
        next_id += 1

    with open(MOVIES_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    
    print("Successfully saved movies.json")

if __name__ == "__main__":
    add_hindi_movies()
