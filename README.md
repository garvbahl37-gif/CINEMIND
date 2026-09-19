# CINEMIND

A film recommendation engine over **17,719 films** and **32 million ratings**.
Item-item collaborative filtering blended with a two-tower neural network, then
reranked against what the films are actually about.

**Live:** https://cinemind-theta.vercel.app

```
Sense and Sensibility (1995)  →  Emma · Il Postino · Leaving Las Vegas · The Remains of the Day
The Matrix (1999)             →  Matrix Reloaded · Fight Club · The Empire Strikes Back · Blade Runner
Spirited Away (2001)          →  Howl's Moving Castle · Princess Mononoke · My Neighbor Totoro · Nausicaä
Alien (1979)                  →  Aliens · Alien³ · Alien: Resurrection · The Terminator
```

---

## Contents

- [What it does](#what-it-does)
- [How recommendations are built](#how-recommendations-are-built)
- [Search and filtering](#search-and-filtering)
- [Architecture](#architecture)
- [The ID-mapping bug](#the-id-mapping-bug)
- [Performance](#performance)
- [API reference](#api-reference)
- [Repository layout](#repository-layout)
- [Running it locally](#running-it-locally)
- [Deployment](#deployment)
- [Data and licence](#data-and-licence)

---

## What it does

Pick any film in the catalogue and get twenty-four ranked neighbours, each with a
match strength and the genres and themes it shares with your pick. Or search by
category — `sci-fi`, `korean thrillers`, `90s horror`, `christopher nolan` — and
narrow the result with genre, decade, language and sort filters that report how
many films each option would leave.

Three things it deliberately avoids:

- **Popularity as a proxy for similarity.** Raw co-occurrence drifts toward
  whatever else was big that year. Blockbusters are damped with an IDF term, and
  a content rerank pulls the ranking back toward the film itself.
- **A single prolific account skewing the matrix.** Each user's contribution is
  damped by `1/√(films rated)`, so someone who rated 4,000 films does not
  outvote a thousand people who rated four.
- **Thin data presented as confidence.** A film with 30 ratings can look like a
  perfect match by accident, so it carries a penalty until the evidence is there.

## How recommendations are built

Neighbours for all 17,719 films are computed offline and stored. At request time
the API does an array lookup, not a search.

### 1. Co-occurrence over 32M ratings

For every pair of films, how often the same viewers rated both highly — as a
sparse item×item matrix, with two corrections:

| Correction | Formula | Why |
|---|---|---|
| Power-user damping | weight `1/√len(user_ratings)` | One prolific account should not dominate the matrix |
| Blockbuster damping | IDF over rating counts | Stops "everyone saw it" reading as "it is similar to everything" |

### 2. Two-tower embeddings

A dual-encoder network (PyTorch) maps users and films into a shared 64-dimensional
space, trained with **InfoNCE** and in-batch negatives on a 2M-rating sample.
Cosine distance there is a learned measure of taste, and it surfaces neighbours
co-occurrence alone would miss — films that rarely share an audience but sit in
the same region of the space.

### 3. Candidate union

Both models nominate their closest films and the union is taken: the precision of
co-occurrence with the reach of the embedding space, rather than trusting either
on its own.

### 4. Content rerank

Candidates are rescored on a weighted sum:

| Signal | Weight | What it contributes |
|---|---|---|
| Item-item CF | **1.35** | People who rated this highly also rated that highly |
| Two-tower cosine | **0.75** | Learned 64-d taste geometry |
| Genre overlap, IDF-weighted | **0.60** | Rare genres count for more than "Drama" |
| Tag overlap | **0.50** | Theme and subject matter |
| Thin-data penalty | **−0.70** | Holds back films with too few ratings to trust |
| Franchise bonus | **0.40** | Keeps series entries together |

The top 24 per film are written to `neighbors.npz` — IDs and scores, 17,719 × 24.

### Integrity checks

`build_artifacts.py` verifies the output across the whole catalogue before it
writes anything:

| Check | Result |
|---|---|
| Self-recommendations | 0 |
| Out-of-range or duplicate IDs | 0 |
| Score rows sorted descending | 17,719 / 17,719 |
| Neighbours missing a title | 0 |
| Recommendations sharing ≥1 genre with the query | 94.8% |
| Mean year gap to the query film | 8.5 years |

## Search and filtering

Search resolves to **categories as well as titles**. Typing `rom` offers *Romance*
(2,843 films) as a filter, not only films with those letters in the name.

| Input | Resolves to |
|---|---|
| `korean thrillers` | `lang=ko` + `genres=Thriller` → 26 films |
| `90s sci-fi` | `year 1990–1999` + `genres=Sci-Fi` → 221 films |
| `christopher nolan` | Director match across cast and crew |
| `heist` | Tag match → 109 films |
| `scary` | Aliased to the Horror genre |

Structural terms are stripped from the query before the free-text match runs,
which is what stops `80s horror` returning *The Rocky Horror Picture Show*.
Queries are singularised (`thrillers` → `thriller`), and hyphenated genres are
handled as units — `sci-fi` splits into two tokens, and a stray `fi` used to match
*Final Fantasy VII*.

**Filters.** Genre, decade, language and sort, as four menus. Genres combine with
**AND** — Action plus Comedy means action-comedies (527 films), not everything
that is either. Every response carries facet counts for the current result set,
so each option states how many films it would leave and a dead end reads as `0`
before it is clicked. Counts for one dimension ignore that dimension's own
selection, which is what lets you move from Korean to French without first
clearing Korean.

Typed structure and the filter menus are the same state: `90s sci-fi` arrives on
the results page with *Sci-Fi* and *1990s* already selected, and deselecting one
really removes it — an explicit parameter overrides whatever the query implied.

## Architecture

```
MovieLens 20M ──> recover encoder ──┐
                  (item_classes.npy)│
MovieLens 32M ──> ratings / tags ───┼──> pipelines/build_artifacts.py
TMDB          ──> posters, credits ─┘         (offline, ~20 min)
                                                   │
                              data/artifacts/  catalog.json    11 MB
                                               neighbors.npz  1.5 MB
                                               item_vectors.npy
                                               browse.json
                                                   │
                     Vercel Python Function  <─────┘   NumPy only — no torch,
                              api/index.py             no FAISS at request time
                                   │
                     React 19 + Vite static frontend
```

**Serving.** FastAPI on Vercel Python Functions (Fluid Compute). Artifacts are
parsed once per warm instance and reused across requests; because it is
serverless there is no instance to spin down, so nothing sleeps between visits.

**Frontend.** React 19, TypeScript, Vite 7, TailwindCSS 3, framer-motion 12,
lucide-react. No UI framework beyond a handful of local primitives.

**Deliberately not in the request path:** PyTorch, FAISS, pandas, Redis, Kafka and
MLflow are all build-time only. The deployed function imports NumPy and nothing
else heavy, which is what keeps cold starts short and the bundle small.

## The ID-mapping bug

This is the defect the project was rebuilt around, and it is worth writing down.

The training pipeline encoded raw MovieLens `movieId` values into contiguous
indices with `sklearn.preprocessing.LabelEncoder`, but **never persisted the
encoder's classes**. The serving layer then read FAISS row numbers as if they
were MovieLens IDs. Every recommendation was decoded against the wrong key:

```
/similar/17   "Sense and Sensibility"  →  Die Hard, Die Hard 2, The Cable Guy
```

The model was fine. The lookup table was gone.

**Recovery.** The mapping was reconstructed by reproducing the exact training
sample — `ratings.sample(n=2_000_000, random_state=42)` over `ml-20m` — and
verifying it yields *precisely* 135,697 users and 17,719 items. Both counts have
to match, not just one, before the classes are accepted:

```python
if (nu, ni) != (EXPECT_USERS, EXPECT_ITEMS):
    sys.exit(f"[encoder] FAILED to reproduce sample: got {nu} users / {ni} items")
```

`build_artifacts.py` now writes `item_classes.npy` and **refuses to emit any
artifact if that check fails**, so the mapping cannot be silently lost a second
time. `*.npy` was also removed from `.gitignore`, which is what discarded it
originally.

## Performance

Measured against production:

| Endpoint | Latency |
|---|---|
| `GET /api/similar/{id}` | **~1 ms** |
| `GET /api/search` | **2–16 ms** |
| `GET /api/suggest` | **~9 ms** |
| `GET /api/browse` | **~22 ms** |

Every ranking is precomputed, so a similar-items request is an array slice. The
previous implementation ran an O(n) scan plus a blocking call to a hosted LLM on
every search.

Catalogue coverage: 17,719 films, 17,542 with artwork (98.6%), 46,249 director
and cast names (72,982 index keys, counting surnames) and 26,802 indexed tags.

## API reference

Interactive schema at [`/docs`](https://cinemind-theta.vercel.app/docs).

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Liveness and catalogue size |
| `GET /api/browse` | Hero picks, top 50, series and every genre shelf in one response |
| `GET /api/similar/{item_id}?k=12` | Ranked neighbours with match strength and shared genres/tags |
| `GET /api/movie/{item_id}` | A single film |
| `GET /api/search` | Category and free-text search with filters and facets |
| `GET /api/suggest?q=` | Typeahead — categories and titles |
| `GET /api/categories` | The filter vocabulary: genres, decades, languages with counts |
| `GET /api/top50` | Bayesian-ranked top 50 |
| `POST /api/blend` | Recommendations from several films at once — a taste profile |

### Search parameters

`GET /api/search` and `GET /api/movies/search` both accept:

| Parameter | Type | Notes |
|---|---|---|
| `q` | string | Free text; may also carry structure (`korean thrillers`) |
| `genres` | CSV | Combined with AND. Overrides any the query implied |
| `year_min`, `year_max` | int | Inclusive bounds |
| `lang` | ISO code | `ko`, `fr`, `ja`, … |
| `media` | `movie`\|`tv` | |
| `sort` | enum | `relevance`, `rating`, `newest`, `oldest`, `title` |
| `limit`, `offset` | int | Paging, max 120 per page |
| `facets` | bool | Include counts per genre, decade and language |

```bash
curl "https://cinemind-theta.vercel.app/api/search?genres=Action,Comedy&sort=newest&facets=true"
curl "https://cinemind-theta.vercel.app/api/similar/16?k=5"
curl "https://cinemind-theta.vercel.app/api/suggest?q=kor"
```

Top 50 ranking uses a Bayesian average (`C` = catalogue mean, `m` = 2,500) so a
film with nine 5-star ratings does not outrank *The Godfather*.

## Repository layout

```
api/
  index.py            FastAPI app — every endpoint
  _engine.py          NumPy serving engine: search, filters, facets, neighbours
pipelines/
  build_artifacts.py  The whole offline build, including encoder recovery
  fetch_data.sh       Downloads MovieLens 20M + 32M
  requirements-build.txt
data/artifacts/       Generated — committed so the deploy needs no build step
models/
  best_model.pt       Two-tower checkpoint (75 MB, excluded from deploys)
  production.index    FAISS index (build-time only)
notebooks/
  kaggle_training.ipynb   Two-tower training
frontend-app/
  src/components/     SearchCommand, FilterBar, MovieRow, DetailsOverlay, …
  src/lib/search.ts   Single client for the search API
requirements.txt      Serving dependencies only — fastapi, pydantic, numpy
vercel.json           Build, function config and /api rewrites
```

## Running it locally

**Prerequisites:** Python 3.12, Node 20+, ~1.5 GB free for the raw datasets.

```bash
git clone https://github.com/garvbahl37-gif/CINEMIND.git
cd CINEMIND
```

The artifacts in `data/artifacts/` are committed, so you can skip straight to
step 3 unless you want to rebuild them.

```bash
# 1. raw data (~1.1 GB) — only needed to rebuild artifacts
./pipelines/fetch_data.sh data/raw

# 2. rebuild artifacts (~20 min)
python -m venv .venv && source .venv/bin/activate
pip install -r pipelines/requirements-build.txt
python pipelines/build_artifacts.py \
  --ml20m data/raw/ml-20m --ml32m data/raw/ml-32m \
  --index models/production.index

# 3. API on :8000
pip install -r requirements.txt
uvicorn index:app --app-dir api --reload --port 8000

# 4. frontend on :5173
cd frontend-app && npm install
VITE_API_BASE=http://localhost:8000 npm run dev
```

> **Pin your serving dependencies.** `requirements.txt` pins
> `fastapi==0.118.0` / `pydantic==2.11.7`, which is what Vercel installs. A local
> environment on newer versions can pass tests that fail in production — FastAPI
> resolves dependency annotations differently across those releases. Smoke-test
> against the pinned versions before deploying.

## Deployment

One Vercel project serves both halves: the static frontend from
`frontend-app/dist`, and the API as a Python Function with `/api/(.*)` rewritten
to `api/index`. Artifacts ship with the function via `includeFiles`.

```bash
vercel deploy --prod
```

The project root directory must be the repository root, **not** `frontend-app`.
If it points at the frontend, Vercel never builds the Python function and every
`/api/*` path quietly serves `index.html` with a 200 — so verify a response body,
not just a status code:

```bash
curl https://cinemind-theta.vercel.app/api/health
# {"status":"healthy","items":17719,"neighbors_per_item":24}
```

## Data and licence

MIT.

Ratings and tags from [MovieLens](https://grouplens.org/datasets/movielens/)
(GroupLens Research, University of Minnesota). Artwork, synopses and credits from
[TMDB](https://www.themoviedb.org/) — this product uses the TMDB API but is not
endorsed or certified by TMDB.
