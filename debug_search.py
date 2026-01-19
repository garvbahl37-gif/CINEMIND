
import json

def test_search():
    with open('deployment/movies.json', 'r', encoding='utf-8') as f:
        movies_data = json.load(f)

    q = "Hindi movies"
    q_lower = q.lower()
    scored_results = []

    print(f"Searching for: {q}")

    for item_id, movie in movies_data.items():
        score = 0
        title = movie.get("title", "").lower()
        tags = [t.lower() for t in movie.get("tags", [])]
        genres = [g.lower() for g in movie.get("genres", [])]
        
        # 1. Title Match
        if q_lower == title:
            score += 100
        elif q_lower in title:
            score += 50
            
        # 2. Tag Match
        query_tokens = set(q_lower.split())
        for tag in tags:
            if q_lower == tag:
                score += 30
            elif q_lower in tag:
                score += 20
            elif tag in q_lower and len(tag) > 2:
                score += 25
            
            if tag in query_tokens:
                 score += 15

        # 3. Genre Match
        for genre in genres:
            if q_lower == genre:
                score += 20
            elif genre in q_lower:
                score += 20
        
        if score > 0:
            scored_results.append({
                "item_id": item_id,
                "score": score,
                "title": movie.get("title"),
                "tags": tags
            })
    
    scored_results.sort(key=lambda x: x["score"], reverse=True)
    
    print(f"Found {len(scored_results)} matches.")
    for res in scored_results[:5]:
        print(f"Score: {res['score']} - {res['title']} (Tags: {res['tags']})")

if __name__ == "__main__":
    test_search()
