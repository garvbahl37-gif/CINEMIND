"""
The build and evaluation pipelines: matching rules, source parsing, metric
definitions, and a reproducible evaluation on a small fixture. Needs the build
dependencies (pandas, scipy — pipelines/requirements-build.txt); skipped in a
serving-only environment.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("scipy")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "api"))
sys.path.insert(0, str(ROOT / "pipelines"))

from build_catalog import imdb_of, same_film, wd_genres, wd_runtime, year_of  # noqa: E402
from evaluate import average_precision, boot_ci, build_cf, ndcg, topk  # noqa: E402


# -- duplicates: a shared TMDB id is not enough on its own ----------------------

@pytest.mark.parametrize("a, b, same", [
    (("Clockstoppers", 2002), ("Clockstoppers", 2002), True),
    (("November", 2004), ("November", 2003), True),                  # festival vs release year
    (("From the Earth to the Moon", 1998), ("Mad Dog Coll", 1961), False),
    (("Top Gun", 1986), ("Jekyll", 2007), False),                    # a series filed under a film's id
    (("Emma", 1996), ("Emma", 2009), False),                         # same title, different film
])
def test_same_film_needs_matching_year_and_title(a, b, same):
    assert same_film(a[0], a[1], b[0], b[1])[0] is same


# -- is a TMDB record this film? --------------------------------------------------------

@pytest.mark.parametrize("ml, imdb, tmdb, verdict", [
    # MovieLens' own IMDb id settles it, whatever the years: made 1966, released 1969
    (("Andrei Rublev (Andrey Rublyov)", 1969), "tt0060107", ("Andrei Rublev", "1966-12-16", "tt0060107"), None),
    # a different IMDb id, but the same title and year: an outdated duplicate id
    (("Clockstoppers", 2002), "tt0250305", ("Clockstoppers", "2002-03-29", "tt0157472"), None),
    # ... in any of MovieLens' alternate titles
    (("Pokemon 4 Ever (a.k.a. Pokémon 4: The Movie)", 2002), "tt0313487",
     ("Pokémon 4Ever", "2001-07-07", "tt0287635"), None),
    # ... or without a subtitle, "&" and all
    (("Nightmares in Red White & Blue: The Evolution of the American Horror Film", 2009), "tt1337118",
     ("Nightmares in Red, White and Blue", "2009-08-06", "tt1337117"), None),
    # a different special in the same series
    (("Louis Theroux: Law & Disorder", 2008), "tt1347439",
     ("Louis Theroux: Law and Disorder in Johannesburg", "2008-12-07", "tt2585192"), "imdb_mismatch"),
    # another film: the title alone is not enough
    (("The Wife", 1995), "tt0114936", ("Wife", "1953-01-01", "tt0046468"), "imdb_mismatch"),
    (("The Mikado", 1939), "tt0031650", ("The Mikado", "1987-01-01", "tt0187283"), "imdb_mismatch"),
    # no IMDb id to compare: the year has to agree
    (("Eureka", 1986), None, ("Eureka", "1983-01-01", None), "year_conflict"),
    (("Eureka", 1984), None, ("Eureka", "1983-01-01", None), None),
])
def test_a_tmdb_record_is_used_only_for_its_own_film(ml, imdb, tmdb, verdict):
    from build_catalog import tmdb_mismatch
    t = {"title": tmdb[0], "release_date": tmdb[1], "imdb_id": tmdb[2]}
    assert tmdb_mismatch(ml[0], ml[1], imdb, t) == verdict


# -- parsing the sources ----------------------------------------------------------

def test_imdb_ids_are_normalised():
    assert imdb_of("114709") == "tt0114709"
    assert imdb_of("15398776") == "tt15398776"
    assert imdb_of(None) is None and imdb_of("abc") is None


def test_years_outside_film_history_are_rejected():
    assert year_of("1995-11-22") == 1995
    assert year_of("1850-01-01") is None and year_of("") is None and year_of(None) is None


def test_wikidata_durations_are_converted_by_unit():
    assert wd_runtime(["80@Q7727"]) == 80                    # minutes
    assert wd_runtime(["5400@Q11574"]) == 90                 # seconds
    assert wd_runtime(["2@Q25235"]) == 120                   # hours
    assert wd_runtime(["3@Q11573"]) is None                  # metres: not a duration
    assert wd_runtime([]) is None


def test_wikidata_genres_map_onto_the_movielens_vocabulary():
    assert wd_genres({"comedy-drama film"}) == ["Drama", "Comedy"]
    assert wd_genres({"science fiction film", "action film"}) == ["Action", "Sci-Fi"]
    assert wd_genres({"biographical film"}) == []            # no counterpart: left out, not guessed


# -- metric definitions, on cases worked by hand ------------------------------------

def test_ranking_metrics():
    rec, rel = [1, 2, 3, 4], {2, 4, 9}
    assert ndcg(rec, rel, 4) == pytest.approx((1 / math.log2(3) + 1 / math.log2(5)) /
                                              (1 + 1 / math.log2(3) + 1 / math.log2(4)))
    assert average_precision(rec, rel, 4) == pytest.approx((1 / 2 + 2 / 4) / 3)
    assert ndcg([5, 6], rel, 2) == 0 and average_precision([5, 6], rel, 2) == 0


def test_topk_excludes_seen_and_breaks_ties_the_same_way_every_time():
    s = np.array([0.5, 0.9, 0.9, 0.1, 0.9])
    first = topk(s, np.array([1]), 2).tolist()
    assert 1 not in first and all(topk(s, np.array([1]), 2).tolist() == first for _ in range(5))


# -- a small fixture, evaluated twice -----------------------------------------------

def fixture_ratings(seed=0):
    """Three taste groups of five users over twelve films, with a little noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(15):
        group = u // 5
        for item in range(12):
            liked = item // 4 == group
            if liked or rng.random() < 0.15:
                rows.append((u, item, 4.5 if liked else 2.0, 1_400_000_000 + item))
    return pd.DataFrame(rows, columns=["userId", "item", "rating", "timestamp"])


def test_item_cf_finds_each_films_own_group():
    nbr, sco = build_cf(fixture_ratings(), 12)
    for i in range(12):
        top = [int(j) for j, s in zip(nbr[i], sco[i]) if s > 0][:3]
        assert top and all(j // 4 == i // 4 for j in top), (i, top)


def test_evaluation_is_reproducible():
    def run():
        r = fixture_ratings()
        nbr, sco = build_cf(r, 12)
        scores = []
        for u in range(15):
            liked = r[(r.userId == u) & (r.rating >= 4)].item.values
            s = np.zeros(12, np.float32)
            np.add.at(s, nbr[liked].ravel(), sco[liked].ravel())
            scores.append(topk(s, liked, 3).tolist())
        return scores, boot_ci(np.arange(15, dtype=float), np.random.default_rng(1))
    assert run() == run()


# -- what the report and the About page quote -----------------------------------------

def test_graph_checks_count_shared_genres_and_year_gaps_by_basis():
    from build_neighbors import graph_checks
    cols = {"genres": [["Drama"], ["Drama", "War"], ["Comedy"], []],
            "year": [2000, 2004, None, 1990]}
    nbr = np.array([[1, 2], [0, 2], [0, 1], [0, 1]])
    basis = np.array([0, 0, 2, 2], np.int8)
    c = graph_checks(cols, nbr, basis)
    assert c["all"]["share_a_genre"] == pytest.approx(2 / 6, abs=1e-4)  # films with genres: 0, 1, 2
    assert c["all"]["mean_year_gap"] == pytest.approx((4 + 4 + 10 + 14) / 4)
    assert c["collaborative"]["films"] == 2 and c["content"]["share_a_genre"] == 0
    assert c["hybrid"] == {"films": 0, "share_a_genre": None, "mean_year_gap": None}


def test_before_after_differences_are_paired_by_person():
    from benchmark_report import paired
    a = {"users": [1, 2, 3], "recall_20": {"x": [0.1, 0.2, 0.3]}}
    b = {"users": [3, 1, 4], "recall_20": {"x": [0.5, 0.1, 0.9]}}     # user 4 only in b; order differs
    mean, ci, n = paired(a, b, "recall_20", "x")
    assert n == 2 and mean == pytest.approx((0.2 + 0.0) / 2)
    assert ci[0] <= mean <= ci[1] and paired(a, b, "recall_20", "x") == (mean, ci, n)


# -- the catalogue file, the TMDB cache and promotion ----------------------------------

def test_the_catalogue_loads_compressed_or_plain(tmp_path):
    import gzip
    import json as js
    from _engine import catalog_path, load_catalog
    cols = {"title": ["Amélie"], "year": [2001]}
    (tmp_path / "catalog.json").write_text(js.dumps(cols, ensure_ascii=False), encoding="utf-8")
    assert load_catalog(tmp_path) == cols and catalog_path(tmp_path).name == "catalog.json"
    (tmp_path / "catalog.json.gz").write_bytes(gzip.compress(js.dumps({**cols, "year": [2002]}).encode()))
    assert load_catalog(tmp_path)["year"] == [2002]          # the compressed build wins
    assert load_catalog(tmp_path / "catalog.json")["year"] == [2001]


def test_a_tmdb_batch_refetches_only_missing_unreachable_and_stale_films(tmp_path, monkeypatch):
    import json as js
    import ingest_tmdb
    asked = []

    def fake_get(path, key, pace, **params):
        tid = int(path.rsplit("/", 1)[1])
        asked.append(tid)
        return {"id": tid, "title": f"film {tid}"}
    monkeypatch.setattr(ingest_tmdb, "get", fake_get)
    ids = [1, 2, 3, 4]
    path = ingest_tmdb.batch_path(tmp_path, 0, ids)
    path.write_text(js.dumps({"ids": ids, "unreachable": [3], "results": {
        "1": {"status": "ok", "fetched": "2026-10-25T00:00:00Z"},
        "2": {"status": "ok", "fetched": "2025-01-01T00:00:00Z"},
        "3": {"status": "ok", "fetched": "2026-10-25T00:00:00Z"}}}))
    pace = ingest_tmdb.Pace(1000)
    ingest_tmdb.run_batch(0, ids, tmp_path, "key", pace, 2)
    assert sorted(asked) == [3, 4]                             # unreachable, and never fetched
    asked.clear()
    ingest_tmdb.run_batch(0, ids, tmp_path, "key", pace, 2, stale_before="2026-04-25T00:00:00Z")
    assert asked == [2]                                        # only the record older than the limit
    assert js.loads(path.read_text())["results"]["2"]["fetched"] > "2026-04-25"


def test_the_tmdb_cache_keeps_a_found_record_then_the_newest(tmp_path, monkeypatch):
    import json as js
    import build_catalog
    d = tmp_path / "tmdb" / "movie"
    d.mkdir(parents=True)
    (d / "batch-00000-a.json").write_text(js.dumps({"results": {
        "7": {"status": "ok", "fetched": "2026-01-01T00:00:00Z", "title": "old"},
        "8": {"status": "ok", "fetched": "2026-01-01T00:00:00Z", "title": "found"}}}))
    (d / "batch-00000-b.json").write_text(js.dumps({"results": {
        "7": {"status": "ok", "fetched": "2026-10-01T00:00:00Z", "title": "new"},
        "8": {"status": "not_found", "fetched": "2026-10-01T00:00:00Z"}}}))
    monkeypatch.setattr(build_catalog, "RAW", tmp_path)
    tm = build_catalog.tmdb_cache()
    assert tm[7]["title"] == "new" and tm[8]["title"] == "found"


def test_promotion_makes_the_artifacts_the_staged_build(tmp_path, monkeypatch):
    import rebuild
    art, stage = tmp_path / "artifacts", tmp_path / "staging"
    art.mkdir(); stage.mkdir()
    (art / "catalog.json").write_text("old")
    (art / "same.txt").write_text("x")
    (stage / "catalog.json.gz").write_text("new")
    (stage / "same.txt").write_text("x")
    monkeypatch.setattr(rebuild, "ART", art)
    rebuild.promote(stage)
    assert sorted(f.name for f in art.iterdir()) == ["catalog.json.gz", "same.txt"]
