
import json

def run():
    with open("movies.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    hindi = 0
    comedy = 0
    for m in data.values():
        val = m.get("original_language")
        if val == 'hi': hindi += 1
        
        genres = [g.lower() for g in m.get("genres", [])]
        if 'comedy' in genres: comedy += 1
        
    with open("counts.txt", "w") as f:
        f.write(f"Hindi: {hindi}\nComedy: {comedy}\n")

if __name__ == "__main__":
    run()
