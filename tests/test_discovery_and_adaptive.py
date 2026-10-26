"""
Behaviour natural-language discovery, the adaptive engine and their endpoints
must keep. Runs against the committed artifacts, with no server:

    pip install -r requirements.txt pytest httpx
    pytest tests/

Each parsing case here was a real misreading before it was fixed.
"""
import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "api"))

from _adaptive import FORMATS, Adaptive, Signal  # noqa: E402
from _discover import Discover  # noqa: E402
from _engine import get_engine  # noqa: E402

NOW = 1_790_000_000.0
DAY = 86400.0


@pytest.fixture(scope="module")
def e():
    return get_engine()


@pytest.fixture(scope="module")
def d(e):
    return Discover(e)


@pytest.fixture(scope="module")
def a(e):
    return Adaptive(e)


def idx(e, title):
    return e.c["title"].index(title)


def titles(e, out, n=8):
    return [e.c["title"][r["item_id"]] for r in out["results"][:n]]


def labels(r):
    return [c["label"] for c in r["understood"]["chips"]]


# -- discovery: the three requests it was built for --------------------------

def test_like_a_film_but_more_emotional(e, d):
    r = d.search("Movies that feel like Interstellar but are more emotional", 8)
    assert [x["title"] for x in r["understood"]["anchors"]] == ["Interstellar"]
    assert "Interstellar" not in titles(e, r)
    assert {"Her", "The Theory of Everything"} & set(titles(e, r))


def test_psychological_thrillers_with_twists(e, d):
    r = d.search("Psychological thrillers with unreliable narrators and a shocking ending", 8)
    assert "Thriller" in labels(r)
    assert {"Shutter Island", "Memento", "The Game", "The Prestige"} & set(titles(e, r))


def test_uplifting_under_two_hours_is_a_hard_filter(e, d):
    r = d.search("Something uplifting under two hours for tonight", 24)
    assert "Under 2 hours" in labels(r)
    assert all(0 < (e.c["runtime"][x["item_id"]] or 0) <= 120 for x in r["results"])
    assert "Intouchables" in titles(e, r)


def test_language_era_and_genre_together(e, d):
    r = d.search("korean thrillers from the 2000s", 12)
    for x in r["results"]:
        i = x["item_id"]
        assert e.c["lang"][i] == "ko" and 2000 <= e.c["year"][i] <= 2009 and "thriller" in e.genre_of[i]


# -- discovery: negation, comparatives, and what isn't a film ----------------

@pytest.mark.parametrize("apostrophe", ["'", "\u2019"])   # phones type the curly one
def test_a_negated_mood_steers_away(e, d, apostrophe):
    r = d.search(f"a heist movie that isn{apostrophe}t too violent", 8)
    assert r["understood"]["avoid"] == "violent"
    assert "violent" not in (r["understood"]["mood"] or "")
    tagged = [x for x in r["results"] if "heist" in [t.lower() for t in e.c["tags"][x["item_id"]] or []]]
    assert len(tagged) >= 4, titles(e, r)


def test_nothing_scary_rules_out_horror(e, d):
    r = d.search("nothing scary, something cozy", 24)
    assert "No Horror" in labels(r)
    assert not any("horror" in e.genre_of[x["item_id"]] for x in r["results"])


@pytest.mark.parametrize("q, avoid", [
    ("I'm not in the mood for anything sad", "sad"),
    ("nothing too dark, something cozy", "dark"),
])
def test_other_ways_of_saying_not(d, q, avoid):
    assert d.read(q).avoid == avoid


def test_than_names_a_film_but_not_a_runtime(d):
    assert d.read("darker than toy story").anchors
    r = d.read("something longer than two hours")
    assert not r.anchors and r.runtime_min == 120


@pytest.mark.parametrize("q", ["no idea, just something fun", "not sure, something like amelie"])
def test_filler_is_not_a_constraint(d, q):
    r = d.read(q)
    assert not r.avoid and not r.avoid_genres
    assert "not" not in r.mood.split() and "idea" not in r.mood.split()


def test_impossible_constraints_explain_themselves(d):
    r = d.search("horror under 10 minutes from the 1920s", 8)
    assert r["notes"]


# -- the adaptive engine -----------------------------------------------------

HORROR = ["The Conjuring", "Insidious", "Sinister", "It Follows"]
ROMANCE = ["The Notebook", "Pride & Prejudice", "Before Sunrise", "Notting Hill", "About Time"]


def likes(e, names, age=0.0):
    return [Signal(idx(e, t), "like", None, NOW - age - k * 3600) for k, t in enumerate(names)]


def test_cold_start_still_answers(a):
    out = a.recommend([], [], k=12, explore=0.3, now=NOW)
    assert len(out["results"]) == 12 and out["profile"]["cold_start"]


def test_likes_steer_and_are_never_recommended_back(e, a):
    sig = likes(e, HORROR)
    out = a.recommend(sig, [], k=12, explore=0.0, now=NOW)
    ids = [r["item_id"] for r in out["results"]]
    assert not {s.item for s in sig} & set(ids)
    assert sum("horror" in e.genre_of[i] for i in ids) >= 9


def test_recent_likes_move_the_list(e, a):
    old = likes(e, HORROR, age=300 * DAY)
    before = a.recommend(old, [], k=12, explore=0.0, now=NOW)
    after = a.recommend(old + likes(e, ROMANCE), [], k=12, explore=0.0, now=NOW)
    romance = lambda out: sum("romance" in e.genre_of[r["item_id"]] for r in out["results"])  # noqa: E731
    assert romance(after) >= romance(before) + 6
    assert after["profile"]["lately"] and after["profile"]["drift"] > 0.2


def test_a_pass_removes_the_film(e, a):
    sig = likes(e, HORROR)
    top = a.recommend(sig, [], k=12, explore=0.0, now=NOW)["results"][0]["item_id"]
    out = a.recommend(sig + [Signal(top, "skip", None, NOW)], [], k=12, explore=0.0, now=NOW)
    assert top not in [r["item_id"] for r in out["results"]]


def test_exploration_is_flagged_logged_and_switchable(e, a):
    sig = likes(e, HORROR)
    none = a.recommend(sig, [], k=24, explore=0.0, now=NOW)
    assert not any(r["exploring"] for r in none["results"])
    out = a.recommend(sig, [], k=24, explore=1.0, now=NOW)
    ex = [r for r in out["results"] if r["exploring"]]
    assert ex and all(0 < r["propensity"] < 1 for r in ex)
    assert all(r["propensity"] == 1.0 for r in out["results"] if not r["exploring"])
    # Same inputs, same day: the same list, so a reload doesn't reshuffle.
    again = a.recommend(sig, [], k=24, explore=1.0, now=NOW)
    assert [r["item_id"] for r in again["results"]] == [r["item_id"] for r in out["results"]]


def test_excluded_films_never_come_back_as_exploration(e, a):
    """`exclude` is part of /api/foryou's contract, and binds the exploration slots too."""
    sig = likes(e, HORROR)
    out = a.recommend(sig, [], k=24, explore=1.0, now=NOW, seed=3)
    explored = {r["item_id"] for r in out["results"] if r["exploring"]}
    assert explored
    again = a.recommend(sig, [], k=24, explore=1.0, now=NOW, seed=3, exclude=explored)
    assert not explored & {r["item_id"] for r in again["results"]}


def test_an_allowed_set_bounds_every_candidate_source(e, a):
    """The offline evaluation's common universe: candidates come from these films only."""
    import numpy as np
    allowed = np.zeros(e.n, bool)
    allowed[::2] = True
    for sig in (likes(e, HORROR), []):
        out = a.recommend(sig, [], k=24, explore=1.0, now=NOW, seed=3, allowed=allowed, explain=False)
        assert len(out["ids"]) == 24 and all(allowed[i] for i in out["ids"])


def test_explanations_are_true(e, a):
    sig = likes(e, ["The Dark Knight"] + HORROR[:2])
    liked = {s.item for s in sig}
    out = a.recommend(sig, [], k=24, explore=1.0, now=NOW)
    for r in out["results"]:
        for ev in r["evidence"]:
            if ev["kind"] == "liked":
                assert ev["item_id"] in liked
        if r["reason"].startswith("Exploring: outside your usual "):
            usual = r["reason"].split("outside your usual ")[1].split(",")[0].split(" and ")
            names = {e.genre_label.get(g, g) for g in e.genre_of[r["item_id"]]}
            assert not names & set(usual), (r["title"], usual)
    assert not {e.genre_label.get(g, g) for g in FORMATS} & set(out["profile"]["long_term"])


# -- the API, and keeping one person's picks from another ---------------------

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from index import app
    return TestClient(app)


def foryou(client, body):
    return client.post("/api/foryou", json=body)


def test_personal_answers_are_never_cached(client):
    for r in (foryou(client, {"k": 2}), client.post("/api/chat/message", json={"message": "films like heat"})):
        assert r.status_code == 200
        cc = r.headers["cache-control"]
        assert "no-store" in cc and "private" in cc and "public" not in cc


def test_one_persons_request_never_reaches_another(e, client):
    """The server keeps nothing: a request is answered from its own body alone."""
    fresh = [r["item_id"] for r in foryou(client, {"k": 12}).json()["results"]]
    horror = {"signals": [{"item_id": idx(e, t), "kind": "like"} for t in HORROR], "k": 12}
    first = [r["item_id"] for r in foryou(client, horror).json()["results"]]
    after = [r["item_id"] for r in foryou(client, {"k": 12}).json()["results"]]
    assert after == fresh, "a stranger's request changed what a new visitor sees"
    again = [r["item_id"] for r in foryou(client, horror).json()["results"]]
    assert again == first


def test_bad_numbers_are_refused_cleanly(client):
    for body in ('{"explore": NaN}', '{"signals": [{"item_id": 1, "kind": "like", "ts": Infinity}]}',
                 '{"signals": [{"item_id": 1, "kind": "rating", "value": NaN}]}'):
        r = client.post("/api/foryou", content=body, headers={"content-type": "application/json"})
        assert r.status_code == 422, body
        assert r.json()["detail"][0]["type"] == "finite_number"


def test_unknown_signals_are_counted_not_trusted(e, client):
    body = {"signals": [{"item_id": idx(e, "The Dark Knight"), "kind": "like", "ts": time.time() * 1000},
                        {"item_id": 10 ** 9, "kind": "like"}, {"item_id": -1, "kind": "like"},
                        {"item_id": 5, "kind": "admin"}],
            "explore": 7, "k": 500}
    out = foryou(client, body).json()
    assert out["ignored_signals"] == 3
    assert len(out["results"]) == 48 and out["profile"]["adventurousness"] == 1.0
    huge = foryou(client, {"signals": [{"item_id": 1, "kind": "rating", "value": 1e308}], "k": 3})
    assert huge.status_code == 200 and huge.json()["profile"]["liked"] == 1


def test_discover_endpoint(client):
    assert client.get("/api/discover", params={"q": ""}).status_code == 422
    r = client.get("/api/discover", params={"q": "something uplifting under two hours", "limit": 5})
    assert r.status_code == 200 and len(r.json()["results"]) == 5
    assert "public" in r.headers["cache-control"]   # the same for everyone: depends on the words alone


def test_evaluation_compares_every_policy(client):
    r = client.get("/api/evaluation")
    assert r.status_code == 200
    m = r.json()["metrics"]
    assert {"popularity", "item-cf", "two-tower", "adaptive"} <= set(m)
    assert all("vs_adaptive" in v for k, v in m.items() if k != "adaptive")
    json.dumps(r.json(), allow_nan=False)


# -- response schemas: everything that was there still is -----------------------

ITEM_FIELDS = {"item_id", "title", "year", "genres", "tags", "tmdbId", "movieId", "poster_path", "backdrop_path",
               "overview", "runtime", "media_type", "original_language", "rating_avg", "rating_count", "tmdb_vote",
               "directors", "cast"}


def test_film_fields_are_a_superset_of_what_clients_read(client):
    d = client.get("/api/movie/0").json()
    assert ITEM_FIELDS <= set(d), ITEM_FIELDS - set(d)
    assert isinstance(d["genres"], list) and isinstance(d["cast"], list)
    assert "rating_backed" in d and "imdb_id" in d


def test_similar_says_how_matches_were_found(e, client):
    d = client.get("/api/similar/0", params={"k": 5}).json()
    assert set(d) >= {"item_id", "source", "similar_items", "match_basis"}
    assert d["match_basis"] in ("collaborative", "hybrid", "content")
    assert all(ITEM_FIELDS <= set(x) and x["match_basis"] == d["match_basis"] for x in d["similar_items"])
    unrated = next(i for i, r in enumerate(e.c["rc"]) if r == 0)
    u = client.get(f"/api/similar/{unrated}").json()
    assert u["match_basis"] == "content" and u["source"]["rating_backed"] is False
    assert u["source"]["rating_avg"] is None


def test_browse_and_health_report_the_catalogue(e, client):
    b = client.get("/api/browse").json()
    assert {"hero", "top50", "tv", "rows", "total"} <= set(b) and b["total"] == e.n
    h = client.get("/api/health").json()
    assert h["items"] == e.n and 0 < h["rating_backed"] <= e.n


def test_films_nobody_has_rated_get_their_own_labelled_shelf(e, a):
    sig = likes(e, HORROR)
    out = a.recommend(sig, [], k=24, explore=0.3, now=NOW)
    shelf, main = out["unrated"], {r["item_id"] for r in out["results"]}
    assert 0 < len(shelf) <= 6
    for r in shelf:
        assert r["rating_backed"] is False and r["rating_avg"] is None      # never a borrowed rating
        assert r["item_id"] not in main and not e.hidden[r["item_id"]]
        assert r["reason"].startswith("No ratings yet")
    assert sum("horror" in e.genre_of[r["item_id"]] for r in shelf) >= len(shelf) // 2
    gone = {r["item_id"] for r in shelf}
    again = a.recommend(sig, [], k=24, explore=0.3, now=NOW, exclude=gone)["unrated"]
    assert not gone & {r["item_id"] for r in again}
    assert a.recommend([], [], k=12, now=NOW)["unrated"] == []              # nothing to match a taste to
