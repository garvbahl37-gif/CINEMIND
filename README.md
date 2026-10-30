# CINEMIND

A film recommendation engine over **88,514 films** and **32 million ratings**.
Item-item collaborative filtering blended with a two-tower neural network, then
reranked against what the films are actually about. Describe a film in plain
words and it finds the ones that fit; like a few and an adaptive engine learns
your taste — measured against the baselines on held-out ratings, not assumed.

**Live:** https://cinemind-theta.vercel.app

```
Sense and Sensibility (1995)  →  Emma · Il Postino · Leaving Las Vegas · The Remains of the Day
The Matrix (1999)             →  Fight Club · The Empire Strikes Back · Matrix Reloaded · Blade Runner
Spirited Away (2001)          →  Howl's Moving Castle · Princess Mononoke · My Neighbor Totoro · Laputa
Alien (1979)                  →  Aliens · The Terminator · Blade Runner · Alien³
```

---

## Contents

- [What it does](#what-it-does)
- [The catalogue](#the-catalogue)
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
- [Status](#status)
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

## The catalogue

88,514 films: the 17,719 served before the expansion, unchanged and in the same
order, and 70,795 added from MovieLens 32M. 84,404 have MovieLens ratings;
4,104 are metadata-only. The full report, regenerated with every build,
is [docs/DATA_QUALITY.md](docs/DATA_QUALITY.md).

| Source | What it contributes | Terms |
|---|---|---|
| [MovieLens 32M](https://grouplens.org/datasets/movielens/32m/) | Films, titles, years, genres, ratings, tags, IMDb and TMDB ids | Research use; no commercial use without permission |
| [Wikidata](https://www.wikidata.org/) (SPARQL, by IMDb id) | Runtime, genres, countries, directors, cast, release dates | CC0 |
| [TMDB API](https://developer.themoviedb.org/) | Synopses, posters, language, credits, runtime | [TMDB API terms](https://www.themoviedb.org/api-terms-of-use) — see [Status](#status) |
| IMDb | Identifiers only, from MovieLens `links.csv` | — |

**Ids never move.** Browsers store item_ids for likes and watchlists, so the
expansion appends: films 0–17,718 keep their ids, and every new film comes after
them. `id_map.json` maps every MovieLens id to its item_id, including ids merged
as duplicates, so ratings for either land on the same film.

**Duplicates merge only on evidence.** Two MovieLens entries are one film when
they share a TMDB id *and* their years are within one *and* their titles match.
A shared TMDB id alone is not enough — MovieLens files some series under a
film's id, which is how *Top Gun* and the series *Jekyll* came to share one. Titles
alone never merge anything. 30 pairs were merged; 7 that share an id are
different films, kept apart, and both lists are in the report. (A series served
before the expansion has an id in TMDB's TV namespace, which only coincides
with a film's; those are never compared.) When both duplicates were already
served, both ids may sit in browsers, so the second stays as an alias: it
resolves, is never listed, and its ratings count for the first. A wrong merge
hides a film and pools its ratings with another's, so the rule stays strict
where titles differ; pairs like *Pokémon 4Ever* and *Pokémon 4Ever: Celebi* are
aliased only once a person has confirmed them from evidence beyond the titles —
IMDb, TMDB and Wikidata records, directors, runtimes — in
[`pipelines/confirmed_duplicates.json`](pipelines/confirmed_duplicates.json):
four so far, six aliases in all.

**Nothing is invented.** Each field comes from the first source that has it and
agrees with MovieLens. A TMDB record is this film when it carries MovieLens' own
IMDb id, whatever its year says (*Andrei Rublev* was made in 1966 and released
in 1969); with a different IMDb id, only when its title — in any of MovieLens'
alternate forms — and year agree; with no IMDb id to compare, only when the
year is within two. Anything else is rejected and flagged, and so is a
disagreeing Wikidata record. A film with no title in any source is left out
rather than given one; a runtime of zero is cleared rather than kept. Every
field's source is recorded per film in `provenance.json.gz`, with the date TMDB
data was fetched — for the films served before the expansion too, whose TMDB
data was fetched again. That refetch found seven whose own TMDB id belongs to
another film (*The Wife*, 1995, had been showing the poster of *Wife*, 1953):
their old artwork and synopsis are cleared, and Wikidata supplies what it can.

**Metadata-only films** have no ratings anywhere in MovieLens 32M. The API
returns them with `rating_avg: null` and `rating_backed: false`, never with a
rating borrowed from elsewhere, and their matches are found by what they are
about — see [Cold start](#cold-start-what-a-match-is-based-on).

**Checked before it is served.** `pipelines/validate_artifacts.py` runs 59
checks across the catalogue and every file built from it — ids, ranges,
vocabularies, checksums, and that each film's vectors are the encoding of its
own text — and the API refuses to start on artifacts that disagree in length.
Poster paths are checked for form at build time; whether they load was checked
on a fixed random sample rather than all 86,188, to keep requests to TMDB's
image servers modest: 400 of 400 (300 added films, 100 served before) returned
an image, on 2026-10-26.

## How recommendations are built

Neighbours for all 88,514 films are computed offline and stored. At request time
the API does an array lookup, not a search.

### 1. Co-occurrence over 32M ratings

For every pair of films, how often the same viewers rated both highly — as a
sparse item×item matrix, with two corrections:

| Correction | Formula | Why |
|---|---|---|
| Power-user damping | weight `1/√len(user_ratings)` | One prolific account should not dominate the matrix |
| Blockbuster damping | IDF over rating counts | Stops "everyone saw it" reading as "it is similar to everything" |

### 2. Two-tower embeddings

A dual-encoder network (PyTorch) maps people and films into a shared
64-dimensional space, trained on likes before 2016 — see
[For you](#for-you-the-adaptive-engine) for the model. Cosine distance there is
a learned measure of taste, and it surfaces neighbours co-occurrence alone would
miss — films that rarely share an audience but sit in the same region of the
space. Each film's vector adds a projection of what it is about to what its
audience taught the model, so a film nobody has rated still has a place in it.

### 3. Candidate union

Co-occurrence, the two-tower space and — for films with little rating evidence —
the sentence embeddings each nominate their 200 closest films, and the union is
ranked: the precision of co-occurrence with the reach of the embeddings, rather
than trusting any one on its own.

### 4. Content rerank

Candidates are rescored on a weighted sum:

| Signal | Weight | What it contributes |
|---|---|---|
| Item-item CF | **1.35 · e** | People who rated this highly also rated that highly |
| Two-tower cosine | **0.75** | Learned 64-d taste geometry |
| Genre overlap, IDF-weighted | **0.60** | Rare genres count for more than "Drama" |
| Tag overlap | **0.50** | Theme and subject matter |
| Sentence-embedding similarity | **0.5 · (1 − e)** | What the films are about, where co-rating evidence is thin |
| Era gap, \|Δyear\| / 40 | **−1.0 · (1 − e)** | Keeps content matches near their film's time, as co-rating does |
| Thin-data penalty | **−0.70** | Holds back films with too few ratings to trust |
| Franchise bonus | **0.40** | Keeps series entries together |

`e = min(1, positive raters / 20)` is how much co-rating evidence the film
has. The top 24 per film are written to `neighbors.npz` — ids, scores and the
basis of each film's matches, 88,514 × 24 (`pipelines/build_neighbors.py`).

### Cold start: what a match is based on

A film with twenty or more positive raters is matched exactly as before. With
fewer, co-rating counts for less and content for more; with none, co-rating
counts for nothing — the film is matched on what it is about, never on
co-ratings it does not have. `/api/similar` returns which, as `match_basis`, and
the match page says so:

| `match_basis` | Positive raters | Films | Matched on |
|---|---|---|---|
| `collaborative` | 20 or more | 17,280 | Co-rating, two-tower, genres, tags |
| `hybrid` | 1–19 | 47,730 | Co-rating weighted by evidence, the rest by content |
| `content` | none | 23,504 | Sentence embedding, two-tower content projection, genres, tags |

Both content weights were chosen on validation, not on the test set: for 400
films with plenty of co-ratings, the co-ratings were hidden and each pair of
weights' content-only matches compared with the film's real collaborative ones.
Content alone recovers 16.2% of them; with the era term, 19.2%.
Without it, a content-matched film's matches were 22.9 years away from it on
average, against 8.3 for a co-rated one.

### Integrity checks

`pipelines/build_neighbors.py` refuses to write a graph with a self-match, an
out-of-range or repeated id or a non-finite score, and
`pipelines/validate_artifacts.py` checks the result against the catalogue again
before it is promoted. Over the whole graph:

| Check | Result |
|---|---|
| Self-recommendations | 0 |
| Out-of-range or duplicate IDs | 0 |
| Score rows sorted descending | 88,514 / 88,514 |
| Metadata-only films matched by co-rating | 0 |
| Matches sharing ≥1 genre with their film | 96.9% (collaborative 95.6%, hybrid 97.9%, content 95.9%) |
| Mean year gap to their film | 6.5 years |

## Search and filtering

Search resolves to **categories as well as titles**. Typing `rom` offers *Romance*
(11,132 films) as a filter, not only films with those letters in the name.

| Input | Resolves to |
|---|---|
| `korean thrillers` | `lang=ko` + `genres=Thriller` → 237 films |
| `90s sci-fi` | `year 1990–1999` + `genres=Sci-Fi` → 623 films |
| `christopher nolan` | Director match across cast and crew |
| `heist` | Tag match → 307 films |
| `scary` | Aliased to the Horror genre |

Structural terms are stripped from the query before the free-text match runs,
which is what stops `80s horror` returning *The Rocky Horror Picture Show*.

The reverse also holds: a query of two or more words that names a title, or
begins one, is searched as a title and none of its words become filters. Without
that, `star wars` applied the *War* genre and returned *Starship Troopers*,
`children of men` became the *Children* genre, and `the french connection`
became a French-language filter with no results. A year that is also a well-known film's
whole title — `1917`, `2012` — finds the film rather than the year's films;
`1990` still browses 1990. Titles are matched in every
form MovieLens files them under — article-last (`Good, the Bad and the Ugly,
The`), with alternates in brackets (`Seven (a.k.a. Se7en)`, `Postman, The
(Postino, Il)`), and with or without a leading article.
Queries are singularised (`thrillers` → `thriller`), and hyphenated genres are
handled as units — `sci-fi` splits into two tokens, and a stray `fi` used to match
*Final Fantasy VII*.

**Filters.** Genre, decade, language and sort, as four menus. Genres combine with
**AND** — Action plus Comedy means action-comedies (1,852 films), not everything
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
| `tell me about anora` | That it isn't here — the ratings behind the catalogue end in 2023 — and the nearest titles it has |

The server keeps no sessions: each reply names the film it was about as
`focus`, and the client sends it back as `context`, which is how "it" resolves.
Titles resolve exactly first, then by prefix — but only to a well-known film, so
`get out` does not quietly become *Get Out Your Handkerchiefs* (1978) — then by
spelling (`interstelar` → *Interstellar*, `forest gump` → *Forrest Gump*). When a title the query begins is ten times
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
| **Not yet rated** | Films nobody has rated never earn a place in the list, which is built on ratings. They get a shelf of their own — up to six, matched on what they are about (the two-tower model's content vector and the sentence embedding) and labelled *No ratings yet*. Offline, for people who went on to love a film nobody had rated, it held one for 1.4% of them — 15 times six random unrated films. Weighted on the two-tower content vector alone it did twice as well on 2015, but not by enough to clear the rule fixed in advance, so the even mix stays. |

The two-tower model is retrained for this (`pipelines/train_two_tower.py`): the
user tower is a mean over liked films plus a residual MLP, so it needs no
per-user parameters; both towers add a projection of each film's sentence
embedding to its ID embedding, so a film nobody has rated still has a vector.
Softmax at τ = 0.07 over the batch's other films and 4,096 films drawn uniformly
from the catalogue, each with a logQ correction, trained on pre-2016 likes only.
Half the examples use a 1–10 film history, because that is what a browser has:
trained on long histories alone, a single like of *The Dark Knight* landed among
*2 Fast 2 Furious* and *Police Academy 2*; now it lands on *Inception*,
*Iron Man*, *The Departed* and *Batman Begins*.

Two things the expansion forced. With most of the catalogue unrated before 2016,
in-batch negatives alone never pushed those films down, and they crowded every
list; uniform negatives fix that. And a film with no training data is served by
its content alone, which a model trained only with ID embeddings has no
calibrated score for — exported that way, films with no training data took
26–66% of every list on validation and recall fell each epoch. So each film's
ID embedding is dropped at random in training (DropoutNet), and the model is
selected the way it is served. Selection is temporal and inside the training
period: a model trained on likes before 2015 ranks people's 2015 likes, the
dropout rate and epoch are chosen there, and the model is refit on all
pre-2016 likes. Dropout 0.3 at epoch 4 won — 2015 Recall@20 of 0.108 from a
full history and 0.095 from the last three likes, against 0.083 at best with
no dropout.

Same inputs on the same day give the same list, so a reload doesn't reshuffle.
Serving is NumPy: about 30 ms per request over the full catalogue.

## Evaluation

`pipelines/evaluate.py` compares the recommenders on the same held-out data and
writes an experiment record — code commit, artifact checksums, dataset
checksums, split, seed, catalogue size and population — as JSON.
`pipelines/benchmark_report.py` turns the records into
[docs/BENCHMARK.md](docs/BENCHMARK.md), which has every metric, the ablations,
the commands to reproduce it and the two-tower selection grid. The *How it
works* page draws from the full-catalogue run (`/api/evaluation`).

**Split.** Temporal, at 2016-01-01: MovieLens ratings before it to learn from,
after it to test on, and nothing after it used to choose anything — the
two-tower model, the match graph's content weights and the adaptive engine's
blend are all selected on 2015, inside the training period. Of the blends,
similarity thresholds and similarity spaces tried for the adaptive engine, two
settings have cleared the rule fixed in advance: a quality-prior weight of 0.35
(by 0.03 recall points) and a harder-pushing diversity pass, `mmr_scale` 5.0
(by 0.12, 95% CI +0.01 to +0.23). Both are served; neither makes a measurable
difference on the test period. 3,000 people, drawn once and fixed
(`docs/benchmarks/users.json`), each with ten or more likes before the cutoff
and three or more new ones after.
A later rating of 4★ or more on a film they hadn't rated is a hit. Only films
that existed at the cutoff — released or rated before it — are candidates or
targets: a list made on 1 January 2016 cannot hold a film from 2019, and
counting one as a miss would be as wrong as recommending it. The co-rating graph
is rebuilt from training ratings with the production formula, because the
shipped graph saw all 32M.

**Over the whole expanded catalogue** (67,599 candidate films — those that
existed at the cutoff; 1,231 of the 3,000 people later loved one that nobody
had rated before it):

| Policy | Recall@20 (95% CI) | NDCG@20 | MAP@20 | Hit rate@10 | Coverage | Beyond top 1,000 |
|---|---|---|---|---|---|---|
| Most popular | 0.133 (0.127–0.138) | 0.140 | 0.063 | 54.1% | 0.5% | 0.0% |
| Item-item CF | 0.190 (0.183–0.197) | 0.198 | 0.097 | 67.3% | 1.3% | 2.0% |
| Two-tower alone | 0.159 (0.153–0.165) | 0.161 | 0.070 | 62.2% | 4.5% | 50.5% |
| **Adaptive engine (hybrid)** | **0.207 (0.200–0.213)** | **0.216** | **0.102** | **72.4%** | 4.3% | 24.2% |

**Before and after, person by person.** The same 3,000 people, the same
held-out ratings, and only the old catalogue's films as candidates and
targets — every policy retrieving within that set — so the two systems can be
compared on each person's own list. "Before" is the deployed system as it was,
quality-prior weight and all:

| Policy | Recall@20 before | after | Change, points (paired 95% CI) |
|---|---|---|---|
| Most popular | 0.139 | 0.139 | 0.00 |
| Item-item CF | 0.201 | 0.201 | +0.04 (−0.03 to +0.11) |
| Two-tower alone | 0.133 | 0.166 | **+3.29 (+2.94 to +3.64)** |
| Adaptive engine (hybrid) | 0.223 | 0.219 | −0.46 (−0.68 to −0.21) |

The retrained two-tower model is the clear gain: 3.3 recall points on the same
people and films. Popularity and item-item CF don't move, as they shouldn't —
they see the same training ratings. The adaptive engine is 0.46 points (2.1%)
worse, and the [benchmark](docs/BENCHMARK.md#why-the-hybrid-dipped) finds why
by putting parts back one at a time: learning only from the old films' ratings
changes nothing (−0.61 points), but putting the old two-tower model back in
their rows closes almost all of the gap (−0.09).

Which of the model's jobs costs the points was then found on 2015, without
touching the test period, by handing each job in turn to an old-style model
trained before 2015
([details](docs/BENCHMARK.md#which-of-the-towers-roles-on-2015)). Not
retrieving candidates or scoring them — the new model does both at least as
well — but the diversity pass, which keeps a row from being twelve versions of
one film. It judges which films are alike by the model's similarities, and
among a person's best candidates the new model packs those about half as
widely as the old one did (a spread of 0.07 against 0.12), so the pass barely
reorders the list. Run on the old model's similarities, the same pass gains
0.27 points. Pushing harder (`mmr_scale` 5.0) cleared the 2015 rule and is
served, but changes nothing measurable on the test period (+0.03 points, −0.09
to +0.15); judging alikeness by what films are about costs 0.34 points on
2015; and the model's history table, whose similarities track the old model's
more closely still, doesn't help either. The dip stands (see [Status](#status)).

Over the whole catalogue the adaptive engine still leads every baseline. No
ranked list holds a film nobody had rated before the cutoff — cold recall is
0.000 for every policy — so those films get a shelf of their own: for the 1,231
people who went on to love one, the six-film shelf held it for 1.4% of them
(95% CI 0.8–2.1%), where six unrated films drawn at random would for 0.1%.

**What these numbers can't tell you.** A film not rated after the cutoff counts
as a miss, though the person may never have seen it — MovieLens ratings are
missing not at random. Offline recall only rewards films people went on to
rate, so it counts what exploration costs and none of what it learns. The
counterfactual study in the benchmark uses a simulated logging policy. Viewer
tags were collected up to 2023, a small content-side leak into the test period.
The before/after difference is the whole system's — retrained two-tower, new
content vectors, longer histories for people who rated films the old catalogue
lacked — and isolates none of them. These are comparisons between methods on the
same data, not predictions of live click-through.

```bash
# commands for every run: docs/BENCHMARK.md
python pipelines/benchmark_report.py
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
MovieLens 32M ─┬─────────────────────────────┐
               │  ingest_wikidata.py (CC0)    │
               ├─> data/raw/wikidata/ ────────┤
               │  ingest_tmdb.py (your key)   │
               └─> data/raw/tmdb/ ────────────┤
                                              ▼
                   build_catalog.py ─> catalog.json.gz, id_map.json,
                                       provenance.json.gz, DATA_QUALITY.md
potion-base-8M ──> build_semantic.py ─> semantic_*.npy   (cached by text)
ML-32M < 2016 ───> train_two_tower.py ─> tower_*.npy/npz
ML-32M ──────────> build_neighbors.py ─> neighbors.npz (ids, scores, basis),
                                         browse.json
                   validate_artifacts.py ─> promote: data/artifacts/
ML-32M split ────> evaluate.py ─────────> eval.json, docs/benchmarks/
                                              │
                     Vercel Python Function <─┘   NumPy only — no torch,
                              api/index.py         no FAISS at request time
                                   │
                     React 19 + Vite static frontend
```

`pipelines/rebuild.py` runs the build steps in order into a staging copy,
validates it, and with `--promote` swaps the changed files into
`data/artifacts/` one atomic rename at a time. A failed step leaves the served
artifacts as they were. Every artifact is listed in `data/artifacts/manifest.json`
with its checksum, row count and build settings.

**Serving.** FastAPI on Vercel Python Functions (Fluid Compute). Artifacts are
parsed once per warm instance and reused across requests; because it is
serverless there is no instance to spin down, so nothing sleeps between visits.

**Frontend.** React 19, TypeScript, Vite 7, TailwindCSS 3, framer-motion 12,
lucide-react. No UI framework beyond a handful of local primitives.

**Deliberately not in the request path:** PyTorch, FAISS and pandas are
build-time only. The deployed function imports NumPy and nothing else heavy,
which is what keeps cold starts short and the bundle small. (There is no Redis,
Kafka or MLflow anywhere; `mlops/` holds an early simulation, kept for history.)

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

Measured locally over HTTP on the expanded artifacts — uvicorn with the pinned
serving dependencies, an Apple-silicon laptop, 30 warm requests each:

| Endpoint | p50 (ms) | p95 (ms) |
|---|---|---|
| `GET /api/similar/{id}` | 0.8 | 0.9 |
| `GET /api/suggest` | 0.6 | 0.7 |
| `POST /api/chat/message` | 7.9 | 8.0 |
| `GET /api/browse` | 10.6 | 10.9 |
| `GET /api/discover` | 20.6 | 23.1 |
| `GET /api/search` (filters and facets) | 24.1 | 24.2 |
| `POST /api/foryou` | 29.2 | 29.6 |

Five times the films made the full-catalogue passes four to five times slower
(discovery was 4 ms, *For you* 6 ms); typeahead, which runs on every
keystroke, now finds titles, people and themes by prefix in sorted lists
rather than scanning, and stayed under a millisecond.

Every similar-films ranking is precomputed, so that request is an array slice;
search, discovery and *For you* rank at request time over all 88,514 films in
NumPy. Production latency adds the network and Vercel's routing on top.

**Start-up.** A cold start used to build the search index — title keys, title
words, tags, and every director and cast name with its surname — from the
catalogue: about 1.8 of the engine's 2.2 s to start on a laptop. The index now
ships prebuilt (`search_index.npz`, 8.9 MB, from
`pipelines/build_search_index.py`), stamped with the checksum of the catalogue
it was built from; the engine loads it in about a quarter of a second, and
builds its own only when the stamp doesn't match, so a stale index is never
served. The engine now starts in 0.8 s locally, and the whole API warms up in
1.5 s rather than 2.7 s, for about 14 MB more at peak. On Vercel, the first
request to an idle deployment took 10.4, 12.7 and 12.8 s without it and 4.7,
7.8 and 7.4 s with it (`/api/health`, three cold starts of each build, timed
from a client far from the iad1 region, so network time is included).

**Opening.** The site no longer waits for the API at all. The home page — hero,
top 50, series and every genre shelf — depends on the served artifacts alone,
so `pipelines/build_home.py` writes it to `frontend-app/public/home.json` with
the engine's own code, and it arrives from Vercel's CDN with the rest of the
site; a test fails if it differs from what `/api/browse` answers, which remains
the fallback. Once it has loaded, the page wakes the API in the background, so a
first search doesn't wait out the cold start. The opening animation is shorter
(0.7 s at least on screen, down from 1.5 s, and a quicker gate) and plays once
per tab: a reload goes straight to the site. From a fresh browser, the site
opened in 3.6 s before and 2.2 s after, data in hand at 0.4 s in both.

Catalogue coverage: 88,514 films, 86,188 with artwork, 245,683 director and
cast names (351,458 index keys, counting surnames) and 46,370 indexed tags.

## API reference

Interactive schema at [`/docs`](https://cinemind-theta.vercel.app/docs).

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Liveness, catalogue size, how many films have ratings, and whether the search index was loaded prebuilt |
| `GET /api/browse` | Hero picks, top 50, series and every genre shelf in one response, with the catalogue `total` — also shipped as the static `home.json` the site opens from |
| `GET /api/similar/{item_id}?k=12` | Ranked neighbours with match strength, shared genres/tags, and `match_basis` — `collaborative`, `hybrid` or `content` |
| `GET /api/movie/{item_id}` | A single film |
| `GET /api/search` | Category and free-text search with filters and facets |
| `GET /api/suggest?q=` | Typeahead — categories and titles |
| `GET /api/categories` | The filter vocabulary: genres, decades, languages with counts |
| `GET /api/top50` | Bayesian-ranked top 50 |
| `POST /api/blend` | Recommendations from several films at once |
| `GET /api/discover?q=&limit=` | Films that fit a description, each with `why`, plus the reading as `understood` |
| `POST /api/foryou` | The adaptive engine: `{signals, searches, explore, k, exclude}` → picks with `reason`, `evidence`, `confidence`, `exploring`, `propensity`, plus `unrated`: films nobody has rated, matched on content |
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

Every film object carries `imdb_id` and `rating_backed`; a metadata-only film
has `rating_backed: false` and `rating_avg: null`. These fields were added; none
was removed or renamed.

Top 50 ranking uses a Bayesian average (`C` = catalogue mean, `m` = 2,500) so a
film with nine 5-star ratings does not outrank *The Godfather*.

## Repository layout

```
api/
  index.py            FastAPI app — every endpoint
  _engine.py          NumPy serving engine: search, filters, facets, neighbours;
                      checks the artifacts agree before serving
  _chat.py            The film assistant
  _semantic.py        Model2Vec sentence encoder, in NumPy
  _discover.py        Natural-language discovery
  _tower.py           The two-tower user tower, in NumPy
  _adaptive.py        The adaptive engine: profile, drift, candidates, rerank, exploration
pipelines/
  ingest_wikidata.py  Runtime, genres, countries, credits from Wikidata (cached, resumable)
  ingest_tmdb.py      TMDB details in rate-limited batches (cached, resumable; needs a key)
  build_catalog.py    The expanded catalogue: ids, duplicates, provenance, quality report
  build_semantic.py   Converts and verifies the sentence model; embeds every film (cached)
  train_two_tower.py  The two-tower model, selected on 2015 and refit on all pre-2016 likes
  build_neighbors.py  24 matches per film with their basis; home page rows
  validate_artifacts.py  Consistency checks run before promotion and by the tests
  rebuild.py          All of the above into staging, validated, then promoted
  evaluate.py         Baselines vs two-tower vs hybrid on held-out ratings
  benchmark_report.py docs/BENCHMARK.md from the experiment records
  build_artifacts.py  The original build, including encoder recovery
  repair_tv_metadata.py  Fixes items whose TMDB data belonged to another title
  fetch_data.sh       Downloads MovieLens 20M + 32M
  requirements-build.txt
tests/                Catalogue, pipelines, search, assistant, discovery, adaptive and API
data/artifacts/       Generated — committed so the deploy needs no build step
docs/
  DATA_QUALITY.md     Generated with the catalogue
  BENCHMARK.md        Generated from docs/benchmarks/*.json
  AUDIT.md            The audit that preceded the expansion
models/
  best_model.pt       Original two-tower checkpoint (75 MB, excluded from deploys)
  production.index    FAISS index of the original item vectors (build-time only)
frontend-app/
  src/components/     MatchStage, HeroRail, Collections, Shelf, MatchSearch, …
  src/lib/api.ts      One normaliser for every endpoint's films
  src/lib/search.ts   Single client for the search API
  src/lib/taste.ts    Likes, passes, watchlist, searches and the feedback log
  src/lib/profiles.ts Who is watching: one storage slot per profile
requirements.txt      Serving dependencies only — fastapi, pydantic, numpy
vercel.json           Build, function config and /api rewrites
.env.example          The one secret the pipelines read (TMDB_API_KEY); the API needs none
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
python -m venv .venv && source .venv/bin/activate
pip install -r pipelines/requirements-build.txt
huggingface-cli download minishlab/potion-base-8M --local-dir data/raw/potion-base-8M

# 2. fetch metadata for the films MovieLens lists (both cached and resumable)
python pipelines/ingest_wikidata.py --ml32m data/raw/ml-32m          # ~35 min, no key
export TMDB_API_KEY=...                                              # optional; see Status
python pipelines/ingest_tmdb.py --ml32m data/raw/ml-32m               # batches of 500, rate-limited
python pipelines/ingest_tmdb.py --ml32m data/raw/ml-32m --refresh-days 180   # monthly: TMDB's cache limit

# 2b. rebuild everything into data/raw/staging, validate, then promote
python pipelines/rebuild.py --ml32m data/raw/ml-32m --model-dir data/raw/potion-base-8M
python pipelines/rebuild.py --ml32m data/raw/ml-32m --model-dir data/raw/potion-base-8M \
  --skip catalog,semantic,tower,neighbors --promote                  # validate again and swap in

# 2c. evaluate and regenerate the benchmark — commands in docs/BENCHMARK.md

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
# {"status":"healthy","items":88514,"neighbors_per_item":24,"rating_backed":84404}
```

**Resources.** The function is configured with 1,024 MB (`vercel.json`). With
the expanded artifacts loaded — catalogue, search indices, the match graph, the
two-tower and sentence-embedding tables — one instance peaks at about 650 MB
while loading and is ready 4 s after start (measured on macOS with the pinned
serving dependencies; on Linux the API also hands freed memory back with
`malloc_trim`). To fit, the per-film tables are float16 (multiplied
in float32 chunks), repeated strings are interned, and the search indices are
packed integer arrays. The artifacts ship with the function:

| File | Size | Holds |
|---|---|---|
| `catalog.json.gz` | 22.7 MB | Every film's metadata, columnar — 57 MB of JSON, ASCII-escaped because a wide character anywhere doubles the parse's memory |
| `semantic_items.npy` | 45.3 MB | Sentence embedding per film, 88,514 × 256, float16 |
| `semantic_tokens.npy` | 15.1 MB | The sentence model's token table, for encoding queries |
| `tower_items.npy` | 11.3 MB | Two-tower item vector per film, 88,514 × 64, float16 |
| `tower_user.npz` | 10.6 MB | History embeddings and the user tower's weights |
| `neighbors.npz` | 8.6 MB | 24 matches per film, scores, basis, positive raters |
| `item_vectors.npy` | 2.3 MB | The original model's vectors (17,719 films), kept for the FAISS index check |
| `id_map.json`, `provenance.json.gz`, `browse.json`, `manifest.json`, … | 3 MB | Ids, sources, home rows, checksums |

## Status

What is built and checked, what is not, and what is waiting on something
outside this repository.

**Implemented and tested** (`pytest tests/`, run against the committed artifacts)

- The expanded catalogue: stable ids, evidence-only duplicate merging,
  per-field provenance, a regenerated data-quality report
  (`tests/test_catalog_and_vectors.py`, `tests/test_pipelines.py`).
- Artifact validation before promotion and at API start-up: every per-film
  file has one row per film, checksums match the manifest, and sampled films'
  vectors re-encode from their own text (`pipelines/validate_artifacts.py`).
- Content-based matching for films with no ratings, labelled by basis
  (`match_basis`), and no invented ratings for them; and their own shelf in
  *For you*, which never takes a place in the ranked list, honours `exclude`
  and is measured on its own (`tests/test_discovery_and_adaptive.py`).
- Duplicates a person confirmed are listed once and resolve by both ids
  (`tests/test_catalog_and_vectors.py`).
- The search index ships prebuilt, stamped with the checksum of the catalogue
  it was built from: the engine refuses a stale one and builds its own, and
  what it loads is checked identical to a fresh build, key order included
  (`tests/test_search_and_chat.py`, `pipelines/validate_artifacts.py`).
- The evaluation harness: metric definitions checked on hand-worked cases, a
  reproducible run on a fixture, fixed populations and a common candidate
  universe for before/after comparisons, and a refusal to score a model trained
  past the held-out period.
- Every tuned setting chosen on validation inside the training period, never on
  the test set: the two-tower model's ID dropout and epoch, the match graph's
  content and era weights, and the adaptive engine's blend, similarity
  thresholds and how its diversity pass measures alikeness — each choice and its
  alternatives recorded in `manifest.json` and [docs/BENCHMARK.md](docs/BENCHMARK.md).
- The static home page the site opens from is checked identical to
  `/api/browse` (`tests/test_discovery_and_adaptive.py`).
- Search, the assistant, discovery, the adaptive engine and per-profile
  isolation, as before (`tests/test_search_and_chat.py`,
  `tests/test_discovery_and_adaptive.py`).
- Whether a TMDB record is the film it is filed under, on the cases that taught
  the rule (*Andrei Rublev*, *The Wife*, *The Mikado*, *Pokémon 4Ever*).
- The TMDB cache: a refresh refetches exactly the records older than the limit,
  a found record outranks a failed one and a newer one an older
  (`tests/test_pipelines.py`, with the network faked); the catalogue reads
  compressed or plain; promotion leaves `data/artifacts` equal to the staged
  build.

**Implemented, not covered by automated tests**

- The ingestion scripts' network paths (`pipelines/ingest_wikidata.py`,
  `pipelines/ingest_tmdb.py`): retries, pacing and resumption were exercised
  by the real runs that built this catalogue, not by tests.
- Each deploy's API reads carry its build id, so a browser never shows the
  previous deploy's cached answers; checked by hand after deploying.
- `pipelines/diagnose_tower_roles.py`, the diagnostic that hands the two-tower
  model's roles to another model one at a time: its records are in
  `docs/benchmarks/diagnosis/`, and it refuses to run if the engine's source
  has moved on from the lines it patches.

**Planned**

- Recovering the adaptive engine's last 0.46 points. The cause is narrowed to
  the similarities the diversity pass judges films by
  ([Evaluation](#evaluation)), and nothing the shipped model offers — its item
  vectors pushed harder, its history table, content, or a mix — recovers it.
  The next thing to try is a similarity learned for the job: an embedding
  trained, as the old model was, with in-batch negatives only (whose crowding
  of unrated films matters for retrieval, not for telling apart candidates
  already chosen), over the whole catalogue, validated on 2015 the same way.

**Considered and dropped**

- A release-recency signal for new films. Offline, the films nobody had rated
  at the cutoff looked like new releases — but a list made on 1 January 2016
  cannot hold a film from 2019, and the evaluation now leaves such films out
  ([Evaluation](#evaluation)). In production the catalogue ends where
  MovieLens does, so there are no new releases to surface.

**Blocked on the owner**

- **TMDB terms.** TMDB's API terms (§1.C) forbid using TMDB content "in
  connection with … a machine learning (ML) or artificial intelligence (AI)
  based Application" without a written agreement. TMDB synopses feed the
  sentence embeddings, and through them discovery and the two-tower model. The
  owner chose to use their own key; an agreement with TMDB, or rebuilding the
  embeddings without TMDB text, is theirs to decide. Details in
  [docs/AUDIT.md](docs/AUDIT.md).
- **TMDB attribution.** The site does not show TMDB's logo or its notice
  ("This product uses TMDB and the TMDB APIs but is not endorsed, certified,
  or otherwise approved by TMDB"), which the terms (§3) require wherever TMDB
  content appears — the owner removed them on 2026-10-26.
- **Rotate the TMDB key.** An earlier commit contained it, and the repository is
  public. The pipelines now read it only from the `TMDB_API_KEY` environment
  variable (`.env.example`).

## Data and licence

Code: MIT.

- **Ratings, tags, genres and ids** from
  [MovieLens 32M](https://grouplens.org/datasets/movielens/) (F. Maxwell Harper
  and Joseph A. Konstan, *The MovieLens Datasets: History and Context*, ACM TiiS
  2015; GroupLens Research, University of Minnesota). Research use only; the
  artifacts derived from it carry the same terms, and they are not for
  commercial use.
- **Artwork, synopses and credits** from [TMDB](https://www.themoviedb.org/).
  This product uses TMDB and the TMDB APIs but is not endorsed, certified, or
  otherwise approved by TMDB.
- **Runtimes, genres, countries and credits** for films TMDB doesn't cover,
  from [Wikidata](https://www.wikidata.org/) (CC0).
- **IMDb** identifiers only, as published in MovieLens' `links.csv`.
