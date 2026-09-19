# Cinemind

A film recommendation engine over **17,719 films** and **32 million ratings**.
Item-item collaborative filtering and a two-tower neural network, reranked against
what the films are actually about.

Live: https://cinemind.vercel.app

---

## Why the recommendations are different

Most recommenders drift toward "what else was popular that year". Cinemind blends several
signals so the answer stays about the *film*:

| Signal | Weight | What it contributes |
|---|---|---|
| Item-item CF over 32M ratings | 1.35 | People who rated this highly also rated that highly |
| Two-tower embedding cosine | 0.75 | Learned 64-d taste geometry; finds non-obvious neighbours |
| Genre overlap, IDF-weighted | 0.60 | Rare genres count more than "Drama" |
| Tag overlap | 0.50 | Theme and subject matter |
| Thin-data penalty | −0.70 | Holds back films with too few ratings to trust |
| Franchise bonus | 0.40 | Sequels and series entries surface together |

Power users and blockbusters are damped when building the CF matrix, so neither a
single prolific account nor sheer popularity can masquerade as similarity.

Sample of what that produces:

```
Sense and Sensibility (1995)  ->  Emma · Persuasion · The Remains of the Day · Il Postino
The Matrix (1999)             ->  Matrix Reloaded · Blade Runner · The Terminator · Total Recall
Inception (2010)              ->  The Dark Knight · Shutter Island · Interstellar
Toy Story (1995)              ->  Toy Story 2 · Aladdin · Monsters Inc. · The Lion King
```

## Architecture

```
MovieLens 20M ──> recover encoder ──┐
                  (item_classes.npy)│
MovieLens 32M ──> ratings / tags ───┼──> pipelines/build_artifacts.py
TMDB          ──> posters, overview ┘              │
                                                   v
                              data/artifacts/  catalog.json
                                               neighbors.npz
                                               browse.json
                                                   │
                     Vercel Python Function  <─────┘   (NumPy only)
                              api/index.py
                                   │
                     React + Vite static frontend
```

Every ranking is computed offline. A request is an array lookup, not a search,
which is why `/api/similar` returns in single-digit milliseconds.

## The bug this project had

The training pipeline encoded `movieId` into contiguous indices with a
`LabelEncoder`, but **never persisted the encoder's classes**. The serving layer
then treated FAISS row numbers as MovieLens IDs, so every recommendation was
mis-decoded:

```
/similar/17   "Sense and Sensibility"  ->  Die Hard, Die Hard 2, The Cable Guy
```

The mapping was recovered by reproducing the exact training sample
(`ratings.sample(n=2_000_000, random_state=42)` over ml-20m) and verifying it
yields precisely 135,697 users and 17,719 items. `build_artifacts.py` now writes
`item_classes.npy` and refuses to emit artifacts if that check fails.

## Running it

```bash
# 1. raw data (~1.1 GB)
./pipelines/fetch_data.sh data/raw

# 2. build artifacts
python -m venv .venv && source .venv/bin/activate
pip install -r pipelines/requirements-build.txt
python pipelines/build_artifacts.py \
  --ml20m data/raw/ml-20m --ml32m data/raw/ml-32m \
  --index models/production.index

# 3. API
pip install -r requirements.txt
uvicorn api.index:app --reload --port 8000

# 4. frontend
cd frontend-app && npm install && npm run dev
```

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/browse` | Hero picks, top 50, series and every genre shelf in one response |
| `GET /api/similar/{item_id}?k=12` | Ranked neighbours with match strength and shared genres/tags |
| `GET /api/search?q=` | Title, genre, tag, language and decade search |
| `GET /api/suggest?q=` | Typeahead |
| `POST /api/blend` | Recommendations from several films at once |
| `GET /api/movie/{item_id}` | Single film |
| `GET /api/health` | Liveness and catalogue size |

## Licence

MIT. Ratings and tags from [MovieLens](https://grouplens.org/datasets/movielens/)
(GroupLens Research). Artwork and synopses from [TMDB](https://www.themoviedb.org/);
this product uses the TMDB API but is not endorsed or certified by TMDB.
