# CINEMIND

A film recommendation engine over **17,719 films** and **32 million ratings**.
Item-item collaborative filtering blended with a two-tower neural network, then
reranked against what the films are actually about. Describe a film in plain
words and it finds the ones that fit; like a few and an adaptive engine learns
your taste — measured against the baselines on held-out ratings, not assumed.

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
- [The film assistant](#the-film-assistant)
- [Describing a film](#describing-a-film)
- [For you: the adaptive engine](#for-you-the-adaptive-engine)
- [Evaluation](#evaluation)
- [Profiles and privacy](#profiles-and-privacy)
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

The reverse also holds: a query of two or more words that names a title, or
begins one, is searched as a title and none of its words become filters. Without
that, `star wars` applied the *War* genre and returned *Starship Troopers*,
`children of men` became the *Children* genre, and `the french connection`
became a French-language filter with no results. Titles are matched in every
form MovieLens files them under — article-last (`Good, the Bad and the Ugly,
The`), with alternates in brackets (`Seven (a.k.a. Se7en)`, `Postman, The
(Postino, Il)`), and with or without a leading article.
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

## The film assistant

`POST /api/chat/message` answers questions about films from the catalogue and
the recommendation graph, not from a hosted language model — so every fact it
states is one the API itself would return, and a reply takes milliseconds.

| Ask | It answers with |
|---|---|
| `tell me about heat` | Director, cast, genres, runtime, synopsis, rating and the three closest films |
| `who directed it?` · `who's in it` · `how long is it` · `is it any good` | The fact, about the film already under discussion |
| `is alien a horror movie` | Yes or no, from its genres and tags |
| `films like inception but funnier` · `older ones` · `darker ones` | The film's neighbours, then their neighbours, filtered by the mood |
| `something uplifting under two hours` · `a heist that isn't too violent` | Discovery — see [Describing a film](#describing-a-film) |
| `films between heat and drive` | A blend of both films' neighbourhoods |
| `tom hanks movies` · `films by miyazaki` | A person's films, best rated first |
| `90s horror` · `recommend me something funny` | A category, best rated first |
| `tell me about la la land` | That it isn't here — the catalogue ends in 2015 — and the nearest titles it has |

The server keeps no sessions: each reply names the film it was about as
`focus`, and the client sends it back as `context`, which is how "it" resolves.
Titles resolve exactly first, then by prefix — but only to a well-known film, so
`get out` does not quietly become *Get Out Your Handkerchiefs* (1978) — then by
spelling (`incepshun` → *Inception*). When a title the query begins is ten times
better known than an exact match, it wins: `lord of the rings` means Peter
Jackson's, not the 1978 animated film.

## Describing a film

`GET /api/discover?q=` takes a sentence instead of a title:

```
Movies that feel like Interstellar but are more emotional
  → Like Interstellar · more emotional
  → Her · The Theory of Everything · Up · Intouchables · Gravity

Psychological thrillers with unreliable narrators and a shocking ending
  → Thriller · psychological unreliable narrators shocking ending
  → The Game · Shutter Island · Memento · The Prestige

Something uplifting under two hours for tonight
  → Under 2 hours · uplifting
  → Intouchables · August Rush · Life Is Beautiful · About a Boy

A heist movie that isn't too violent
  → Less violent · heist
  → Rififi · Heist · Bandits · Out of Sight · Thief · The Town
```

1. **Read the sentence.** Runtime (`under two hours`, `not too long`), era
   (`from the 2000s`, `classics`), language (`korean`, `in french`) and genre
   become filters. Exclusions are read before inclusions, so `nothing scary`
   removes Horror instead of adding it, and `isn't too violent` steers away
   from violence instead of towards it. A film after `like`, `feels like` or
   `than` becomes an anchor — but `longer than two hours` stays a runtime.
   Curly apostrophes are straightened first, because phones type them.
2. **Embed the mood.** What is left is encoded by
   [Model2Vec](https://github.com/MinishLab/model2vec) `potion-base-8M`, a
   256-dimension static sentence model distilled from a transformer. It is
   reimplemented in NumPy in `api/_semantic.py` — BERT normalisation,
   WordPiece, mean pooling — and verified against the library at build time
   (identical token ids, cosine ≥ 0.9995). Words it spells badly, like *heist*
   (`he` + `##ist`), are also read through the films viewers tagged with them.
3. **Rank by meaning.** Every film's genres, tags and synopsis sit in the same
   space. Anchors add what their audience also loved (two-tower and the
   co-rating graph); the mood leans the ranking towards it; a quality prior
   breaks ties. When genres leave fewer than twelve films, genre turns into a
   preference and the response says so.

Each result carries `why` — the anchor it shares an audience with, the tags that
match, its runtime — and the response returns `understood`, the reading as
chips, so the interface can show *I read that as …*. The film assistant hands
descriptive messages to the same code.

## For you: the adaptive engine

`POST /api/foryou` turns what one browser has done into a ranked shelf:

```
signals ─▶ taste ─▶ candidates ─▶ rerank ─▶ picks + evidence ─▶ feedback log
```

| Stage | What happens |
|---|---|
| **Signals** | Like 1.0 · watchlist 0.6 · opened 0.25 · rating (r−3)/2 · not for me −1, plus searches from Discover. The latest stance on a film wins. |
| **Taste** | The two-tower user tower reads your likes twice: decayed over a year (long-run) and over three days (lately). Recent activity gets up to 40% of the say, reached at about five fresh likes — that is preference drift. |
| **Candidates** | Nearest films in the two-tower space, co-rating neighbours of what you liked, films that fit your Discover searches, and a pool of well-rated films. |
| **Rerank** | z-scored blend of those sources, a quality prior, a penalty for resembling films you passed on, and maximal marginal relevance so one niche can't fill the row. |
| **Exploration** | `explore` (0–1, the *Adventurousness* slider, default 0.3) spends round(k · 0.35 · explore) slots on relevant, well-rated films outside your usual genres or far from your likes. They are *sampled*, not ranked, and each carries the probability it was shown with. |
| **Evidence** | Every pick says why (`Because you liked …` cites only films you liked), with a confidence built from where it ranks among the candidates, how many of your likes support it and how many ratings it has — scaled down for exploration. |
| **Feedback** | The browser logs what was shown, where, whether it was exploration, its propensity, and whether it was opened or liked — the record counterfactual evaluation needs. |

The two-tower model is retrained for this (`pipelines/train_two_tower.py`): the
user tower is a mean over liked films plus a residual MLP, so it needs no
per-user parameters; the item tower adds a projection of each film's sentence
embedding to its ID embedding. In-batch softmax at τ = 0.07 with a logQ
correction, trained on pre-2016 likes only. Half the examples use a 1–10 film
history, because that is what a browser has: trained on long histories alone, a
single like of *The Dark Knight* landed among *2 Fast 2 Furious* and
*Police Academy 2*; now it lands on *Batman Begins*, *The Dark Knight Rises*
and *Inception*. Validation recall@20 (inside the training period): 0.170 from a
full history, 0.208 from the last three likes.

Same inputs on the same day give the same list, so a reload doesn't reshuffle.
Serving is NumPy: about 6 ms per request.

## Evaluation

`pipelines/evaluate.py` compares the recommenders on the same held-out data and
writes `data/artifacts/eval.json`, which the *How it works* page draws from.

**Split.** Temporal, at 2016-01-01: 20.8M MovieLens ratings before it to learn
from, 8.9M after it to test on. 3,000 people drawn from the 5,983 with ten or
more likes before the cutoff and three or more new ones after. A later rating of
4★ or more on a film they hadn't rated is a hit. The co-rating graph is rebuilt
from training ratings with the production formula, because the shipped graph
saw all 32M.

| Policy | Recall@20 (95% CI) | NDCG@20 | Hit rate@10 | Coverage | Beyond top 1,000 |
|---|---|---|---|---|---|
| Most popular | 0.139 (0.134–0.145) | 0.144 | 54.1% | 1.8% | 0.0% |
| Item-item CF | 0.201 (0.194–0.207) | 0.204 | 67.0% | 4.6% | 1.8% |
| Two-tower alone | 0.133 (0.127–0.138) | 0.129 | 53.9% | 26.3% | 67.9% |
| **Adaptive engine** | **0.224 (0.216–0.230)** | **0.227** | **72.8%** | 14.3% | 19.8% |

The adaptive engine beats item-item CF by **+2.3 recall points** (paired 95%
interval +1.9 to +2.7) while recommending three times as much of the catalogue.
The two-tower model alone ranks worse than CF but reaches far more of the long
tail; combined, each covers the other's weakness.

**Ablations** — each part switched off for the same 3,000 people, compared
person by person:

| Switched off | Recall change (paired 95% CI) | Reading |
|---|---|---|
| Diversity pass | −0.30 pts (−0.48 to −0.10) | helps |
| Passes and low ratings | −0.28 pts (−0.43 to −0.15) | helps |
| Drift | +0.15 pts (−0.03 to +0.34) | no measurable effect |
| Exploration | +1.08 pts (+0.93 to +1.27) | costs recall offline, by design |

Drift shows nothing here, and this split can't really test it: the test period
runs to 2023 and recency is measured from each person's last rating before
2016. Exploration costs a point because offline recall only rewards films people
already went on to rate — it counts the slots exploration spends and none of
what it learns. That is the case for logging propensities.

**Counterfactual evaluation.** 30,000 simulated impressions from a
popularity-leaning logging policy with known probabilities, ten per person over
a candidate set of each policy's top 100 plus random films. Each estimator
predicts how often a policy's single pick (from its own top 10, rank-weighted,
with 5% spread for overlap) would be a later favourite; the truth is computed
exactly.

| Policy | Truth | IPS | SNIPS | DM | DR (95% CI) | Effective logs |
|---|---|---|---|---|---|---|
| Most popular | 0.144 | 0.157 | 0.153 | 0.127 | 0.153 (0.139–0.168) | 3,032 |
| Item-item CF | 0.201 | 0.229 | 0.222 | 0.186 | 0.223 (0.206–0.242) | 2,176 |
| Two-tower alone | 0.122 | 0.126 | 0.115 | 0.076 | 0.130 (0.102–0.160) | 27 |
| Adaptive engine | 0.221 | 0.238 | 0.232 | 0.149 | 0.233 (0.210–0.259) | 1,252 |

IPS, SNIPS and doubly robust put the four policies in their true order (Kendall
τ = 1), though they mostly read high, by up to 14%; the direct method, which trusts a reward
model alone, reads every policy low and swaps two of them (τ = 0.67). Overlap
decides everything: the two-tower policy picks films the logging policy rarely
showed, so its estimate rests on 27 effective logs.

**What these numbers can't tell you.** A film not rated after the cutoff counts
as a miss, though the person may never have seen it — MovieLens ratings are
missing not at random. The logging policy is simulated; slates and position
effects are not modelled. Viewer tags were collected up to 2023, a small
content-side leak into the test period. These are comparisons between methods on
the same data, not predictions of live click-through.

```bash
python pipelines/evaluate.py --ml32m data/raw/ml-32m --users 3000   # ~2 min
```

## Profiles and privacy

There are no accounts, and the server stores nothing about anyone.

- **Per browser.** Taste lives in `localStorage`, so another browser, device or
  private window starts empty. Only the active profile's signals are sent, with
  no name, id or cookie attached.
- **Per person.** People sharing a browser each get a profile (the + in the
  taste tray), stored under its own key, `cinemind.taste.v1.<id>`. Switching remounts
  the picks, Discover and the assistant, so nothing from the last person stays
  on screen; deleting a profile deletes its data.
- **Per tab.** Two tabs of the same profile merge through `storage` events
  instead of overwriting each other.
- **No shared caches.** `/api/foryou` and `/api/chat/message` answer with
  `Cache-Control: private, no-store`. Discovery stays publicly cacheable,
  because its answer depends on the words alone.
- **Tested.** `tests/test_discovery_and_adaptive.py` checks that one person's
  request cannot change what the next person gets, and that non-finite numbers
  are refused with a 422 rather than quietly erasing a signal.

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
potion-base-8M ─> build_semantic.py ──────────>  semantic_*.npy  24 MB
ML-32M < 2016 ──> train_two_tower.py ─────────>  tower_*.npy/npz 4.4 MB
ML-32M split ───> evaluate.py ────────────────>  eval.json
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
| `GET /api/discover` | **2–6 ms** (measured locally) |
| `POST /api/foryou` | **~6 ms** (measured locally) |

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
| `POST /api/blend` | Recommendations from several films at once |
| `GET /api/discover?q=&limit=` | Films that fit a description, each with `why`, plus the reading as `understood` |
| `POST /api/foryou` | The adaptive engine: `{signals, searches, explore, k}` → picks with `reason`, `evidence`, `confidence`, `exploring`, `propensity` |
| `GET /api/evaluation` | The offline comparison in `eval.json` |
| `POST /api/chat/message` | The film assistant: `{message, context?}` → `{response, recommendations, focus, follow_ups}` |

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
  _chat.py            The film assistant
  _semantic.py        Model2Vec sentence encoder, in NumPy
  _discover.py        Natural-language discovery
  _tower.py           The two-tower user tower, in NumPy
  _adaptive.py        The adaptive engine: profile, drift, candidates, rerank, exploration
pipelines/
  build_artifacts.py  The whole offline build, including encoder recovery
  repair_tv_metadata.py  Fixes items whose TMDB data belonged to another title
  build_semantic.py   Converts and verifies the sentence model; embeds every film
  train_two_tower.py  Retrains the two-tower model on pre-2016 likes
  evaluate.py         Baselines vs two-tower vs adaptive on held-out ratings
  fetch_data.sh       Downloads MovieLens 20M + 32M
  requirements-build.txt
tests/                Search, assistant, discovery, adaptive and API behaviour
data/artifacts/       Generated — committed so the deploy needs no build step
models/
  best_model.pt       Two-tower checkpoint (75 MB, excluded from deploys)
  production.index    FAISS index (build-time only)
notebooks/
  kaggle_training.ipynb   Two-tower training
frontend-app/
  src/components/     MatchStage, HeroRail, Collections, Shelf, MatchSearch, …
  src/lib/api.ts      One normaliser for every endpoint's films
  src/lib/search.ts   Single client for the search API
  src/lib/taste.ts    Likes, passes, watchlist, searches and the feedback log
  src/lib/profiles.ts Who is watching: one storage slot per profile
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

# 2b. the semantic layer, two-tower model and evaluation (optional)
huggingface-cli download minishlab/potion-base-8M --local-dir data/raw/potion-base-8M
python pipelines/build_semantic.py --model-dir data/raw/potion-base-8M
python pipelines/train_two_tower.py --ml32m data/raw/ml-32m          # ~3 min on a laptop CPU
python pipelines/evaluate.py --ml32m data/raw/ml-32m --users 3000    # ~2 min

# 3. API on :8000
pip install -r requirements.txt
uvicorn index:app --app-dir api --reload --port 8000

# 4. frontend on :5173
cd frontend-app && npm install
API_PROXY=http://localhost:8000 npm run dev    # proxies /api; defaults to the deployed API

# 5. tests, from the repo root, against the pinned serving dependencies
cd .. && pip install pytest httpx && pytest tests/
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
