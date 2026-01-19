
import json
import os

MOVIES_PATH = "deployment/movies.json"

tv_shows_to_add = [
    {
        "title": "Breaking Bad (2008)",
        "genres": ["Crime", "Drama", "Thriller"],
        "year": 2008,
        "original_language": "en",
        "tags": ["drugs", "meth", "cancer", "chemistry", "teacher"],
        "overview": "A high school chemistry teacher diagnosed with inoperable lung cancer turns to manufacturing and selling methamphetamine in order to secure his family's future.",
        "vote_average": 9.3,
        "vote_count": 14000,
        "media_type": "tv",
        "tmdbId": 1396
    },
    {
        "title": "Game of Thrones (2011)",
        "genres": ["Sci-Fi", "Fantasy", "Drama", "Action", "Adventure"],
        "year": 2011,
        "original_language": "en",
        "tags": ["dragons", "politics", "war", "kings", "queens"],
        "overview": "Seven noble families fight for control of the mythical land of Westeros. Friction between the houses leads to full-scale war. All while a very ancient evil awakens in the farthest north.",
        "vote_average": 8.4,
        "vote_count": 22000,
        "media_type": "tv",
        "tmdbId": 1399
    },
    {
        "title": "Stranger Things (2016)",
        "genres": ["Mystery", "Sci-Fi", "Fantasy", "Horror"],
        "year": 2016,
        "original_language": "en",
        "tags": ["80s", "supernatural", "monster", "government conspiracy", "kids"],
        "overview": "When a young boy vanishes, a small town uncovers a mystery involving secret experiments, terrifying supernatural forces, and one strange little girl.",
        "vote_average": 8.6,
        "vote_count": 16000,
        "media_type": "tv",
        "tmdbId": 66732
    },
    {
        "title": "Rick and Morty (2013)",
        "genres": ["Animation", "Comedy", "Sci-Fi", "Adventure"],
        "year": 2013,
        "original_language": "en",
        "tags": ["adult animation", "space", "mad scientist", "black comedy", "interdimensional travel"],
        "overview": "Rick is a mentally-unbalanced but scientifically gifted old man who has recently reconnected with his family. He spends most of his time involving his young grandson Morty in dangerous, outlandish adventures throughout space and alternate universes.",
        "vote_average": 8.7,
        "vote_count": 9500,
        "media_type": "tv",
        "tmdbId": 60625
    },
    {
        "title": "The Office (2005)",
        "genres": ["Comedy"],
        "year": 2005,
        "original_language": "en",
        "tags": ["mockumentary", "workplace", "sitcom", "boss", "cringe"],
        "overview": "A mockumentary on a group of typical office workers, where the workday consists of ego clashes, inappropriate behavior, and tedium.",
        "vote_average": 8.6,
        "vote_count": 14000,
        "media_type": "tv",
        "tmdbId": 2316
    }
]

def add_tv():
    if not os.path.exists(MOVIES_PATH):
        print("movies.json not found")
        return

    with open(MOVIES_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Find safe Start ID
    ids = [int(k) for k in data.keys() if k.isdigit()]
    next_id = max(ids) + 1 if ids else 20000

    for show in tv_shows_to_add:
        key = str(next_id)
        data[key] = show
        print(f"Added TV Show: {show['title']} as ID {key}")
        next_id += 1

    with open(MOVIES_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
        
    print("Successfully updated movies.json with TV shows.")

if __name__ == "__main__":
    add_tv()
