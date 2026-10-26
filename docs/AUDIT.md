# CINEMIND audit — before the catalogue expansion

Date: 2026-10-25. Commit audited: `8291d79` (deployed to production the same day).
Everything below was checked against the code and data, not the existing docs.

## What runs

| Part | Entry point | Notes |
|---|---|---|
| API | `api/index.py` (FastAPI on a Vercel Python Function) | NumPy only at request time. Loads `data/artifacts/*` once per instance. |
| Serving engine | `api/_engine.py` | Columnar `catalog.json`, precomputed `neighbors.npz`, `browse.json`; search indexes built in Python at startup. |
| Discovery | `api/_discover.py`, `api/_semantic.py` | Model2Vec `potion-base-8M` reimplemented in NumPy; `semantic_items.npy`. |
| Adaptive picks | `api/_adaptive.py`, `api/_tower.py` | Two-tower user tower in NumPy; `tower_items.npy`, `tower_user.npz`. |
| Assistant | `api/_chat.py` | Rule-based over the catalogue and graph; no LLM. |
| Offline build | `pipelines/build_artifacts.py` | Catalogue, neighbour graph, browse rows. Needs ML-20M (to recover the original encoder), ML-32M and the FAISS index of the original two-tower. |
| Offline models | `pipelines/build_semantic.py`, `pipelines/train_two_tower.py` | |
| Evaluation | `pipelines/evaluate.py` → `data/artifacts/eval.json` | |
| Frontend | `frontend-app/` (React 19, Vite) | Talks only to `/api/*`. |

**Not part of the running system:** `serving/`, `candidate_generation/`, `ranking/`
(LightGBM LambdaRank), `reranking/`, `evaluation/offline_metrics.py`,
`preprocessing.py`, `feature_engineering.py`, `run.py` — an earlier prototype that
nothing deployed imports. `mlops/train.py` is a **simulation**: it logs random
"loss" and "accuracy" values to MLflow and writes a placeholder
`production.index`. Nothing in the repository uses Kafka or Redis.

`docs/SYSTEM_DOCUMENTATION.md` describes LightGBM ranking, FAISS IVF-PQ/HNSW
retrieval ("Recall@100 95%+", "~3 ms") and a Redis feature store. None of that is
in the deployed system, and those numbers are not reproduced by anything in the
repository. The README's line that Redis, Kafka and MLflow are "build-time only"
is also wrong: none of them is used at all.

## The catalogue today

| | |
|---|---|
| Films served | **17,719** (`item_id` 0–17,718) |
| How the set was chosen | Every film in a 2M-rating random sample of **ML-20M**, the original two-tower's training set (`item_classes.npy`, the recovered encoder) |
| Rating statistics | From **ML-32M**; 959 served films have **no ML-32M ratings** (they are in ML-20M only) |
| Metadata | MovieLens titles/genres/tags; TMDB posters, synopses, runtime, language, credits (17,564 with a poster) |
| Media | 17,639 movie, 80 TV (verified in the 2026-10-22 repair) |
| Newest films | Effectively 2015 — the assistant tells people so |

`item_id` is the row number, and it is what browsers store for likes, passes and
watchlists (`localStorage`, see `frontend-app/src/lib/taste.ts`). It must never be
reassigned.

## Data available locally

`data/raw/ml-32m` (MovieLens 32M, released 2023-10):

| | |
|---|---|
| Films (`movies.csv`) | **87,585** unique `movieId` |
| With ≥ 1 rating | 84,432 (3,153 have tags only) |
| With ≥ 5 / 20 / 100 ratings | 43,884 / 23,350 / 12,191 |
| IMDb id | 87,585 (all) |
| TMDB id | 87,461 (124 missing) |
| TMDB ids shared by two `movieId`s | 36 groups — duplicate candidates |
| Exact duplicate titles | 198 — **not** evidence of duplication on their own |
| No year in title | 617 |
| `(no genres listed)` | 7,080 |
| Not yet served | 70,825 |
| Served films missing from ML-32M | 959; 22 of them share a TMDB id with an ML-32M film |

## Licensing — what each source allows

- **MovieLens 32M** (GroupLens): research use; redistribution of the data *and
  transformations* allowed under the same terms; **no commercial or
  revenue-bearing use** without permission; acknowledgement required.
- **TMDB API terms** (last updated 2023-10-20):
  - §1.C: you must not "use the TMDB APIs or TMDB Content in connection with, including for training, a machine learning (ML) or artificial intelligence (AI) based Application" — no exception.
  - §2.A: training or validating an ML system with TMDB content is commercial use, which needs a written agreement.
  - §1.C: nothing obtained from TMDB may be cached for more than six months.
  - §3: the TMDB logo and the notice "… uses TMDB and the TMDB APIs but is not endorsed, certified, or otherwise approved by TMDB" are required.
- **IMDb non-commercial datasets**: personal, non-commercial use; not for
  republishing. Only IMDb *identifiers* (via MovieLens `links.csv`) are used.
- **Wikidata**: CC0 (public domain). No restriction on use, including ML.

## Existing metrics

From `data/artifacts/eval.json` (temporal split at 2016-01-01, 3,000 users, the
17,719-film universe only):

| Policy | Recall@20 | NDCG@20 | Coverage |
|---|---|---|---|
| Popularity | 0.139 | 0.144 | 1.8% |
| Item-item CF | 0.201 | 0.204 | 4.6% |
| Two-tower | 0.133 | 0.129 | 26.3% |
| Adaptive (hybrid) | 0.224 | 0.227 | 14.3% |

Plus ablations, paired intervals and simulated counterfactual estimators. The
two-tower model's epoch was chosen on a validation set inside the training period.

## Gaps and risks

1. **Licensing (highest).** TMDB content (synopses) feeds `semantic_items.npy`,
   and through it the discovery ranker and the two-tower item tower — use "in
   connection with" an ML application that §1.C forbids without an agreement.
   The cached TMDB data has no fetch timestamps, so the six-month limit cannot be
   checked, and the site lacks the TMDB logo and the exact notice. **This predates
   the expansion; whether to keep TMDB content is the owner's decision.**
   *Decision, 2026-10-25:* the owner chose to fetch TMDB details for the added
   films with their own API key, in rate-limited batches. Every TMDB value is
   recorded as such in `provenance.json.gz` with its fetch date (for the
   six-month limit). The site showed the TMDB logo and notice briefly; the owner
   removed them on 2026-10-26.
   An agreement with TMDB for ML use is still outstanding — see the README's
   *Status* section.
2. **Catalogue size.** 17,719 films, none after 2015. ML-32M has 87,585.
3. **Cold films.** 959 served films have no ML-32M ratings, yet are treated like
   any other; nothing tells a user that a match is content-only.
4. **Evaluation gaps.** No @10, precision or MAP; no latency percentiles; no
   cold-start metrics; one candidate universe; no separate validation period for
   tuning; no machine-readable experiment record (dataset/config versions).
5. **No artifact checks at startup.** A catalogue and a vector file of different
   lengths would load and fail later, per request.
6. **Non-atomic writes** in the build; an interrupted build can leave a mix of
   old and new artifacts.
7. **Inaccurate documentation** (above) and a simulated MLflow script that looks
   like real training.
8. **Scale unknowns.** At 17,719 films the API peaks at **284 MB** RSS and loads
   all engines in **0.74 s**. Five times the catalogue has not been measured.

## Proposed changes

- Expand to every valid ML-32M film (target ≥ 50,000), appended **after** the
  existing 17,719 so no `item_id` changes; duplicates resolved by external ids
  with year/title agreement checks; provenance per record.
- Enrich new films from **Wikidata** (CC0): director, cast, runtime, release
  date, country, language, genre, description, sitelink count. No plots are
  invented; missing stays missing. No new TMDB content.
- Content embeddings rebuilt for every film from a documented, license-clean
  text recipe, with a cache and checkpoints; two-tower retrained on the same
  architecture so films with ratings get learned vectors, with a content branch
  for films without them; neighbour graph rebuilt with an explicit
  collaborative / hybrid / content label per film.
- Startup validation, atomic writes, a data-quality report, an evaluation with
  @10/@20, precision, MAP, cold-start and latency percentiles under fixed
  before/after protocols, and tests for all of it.
