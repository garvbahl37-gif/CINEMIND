"""
The catalogue and every vector built from it agree — and the expanded
catalogue keeps its promises. Runs against the committed artifacts:

    pytest tests/

The heavy lifting is pipelines/validate_artifacts.py; these tests run it and
add the guarantees that matter to people using the site: ids never move,
duplicates are merged only on evidence, and films nobody has rated are never
presented as if they had been.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "data" / "artifacts"
sys.path.insert(0, str(ROOT / "api"))
sys.path.insert(0, str(ROOT / "pipelines"))

from validate_artifacts import LEGACY_N, validate  # noqa: E402


@pytest.fixture(scope="module")
def c():
    from _engine import load_catalog
    return load_catalog(ART)


def test_every_artifact_agrees_with_the_catalogue():
    r = validate(ART)
    assert not r.failed, "\n".join(r.failed)


def test_at_least_50000_valid_films(c):
    valid = [i for i in range(len(c["title"]))
             if c["title"][i].strip() and c["movieId"][i] is not None and c["media"][i] == "movie"]
    assert len(valid) >= 50_000, f"{len(valid):,} valid films"


def test_ids_already_in_browsers_never_move(c):
    """item_ids 0-17,718 are stored in browsers for likes and watchlists."""
    classes = np.load(ART / "item_classes.npy")
    assert [int(m) for m in c["movieId"][:LEGACY_N]] == [int(x) for x in classes]
    id_map = json.loads((ART / "id_map.json").read_text())
    assert all(id_map[str(int(m))] == i for i, m in enumerate(classes))


def test_no_film_is_listed_twice(c):
    from _engine import _key
    seen = {}
    alias = c.get("alias_of") or [None] * len(c["title"])
    for i, (t, y, tm) in enumerate(zip(c["title"], c["year"], c["tmdbId"])):
        if tm is None or alias[i] is not None:
            continue
        k = (tm, y, _key(t))
        assert k not in seen, f"films {seen.get(k)} and {i} are the same film"
        seen[k] = i


def test_metadata_only_films_carry_no_invented_ratings(c):
    z = np.load(ART / "neighbors.npz")
    unrated = [i for i, r in enumerate(c["rc"]) if r == 0]
    assert unrated, "expected some metadata-only films"
    for i in unrated:
        assert c["ra"][i] is None
        assert z["basis"][i] == 2          # matched on content, never on co-rating


def test_provenance_covers_every_film(c):
    import gzip
    p = json.loads(gzip.decompress((ART / "provenance.json.gz").read_bytes()))
    n = len(c["title"])
    assert all(len(v) == n for v in p.values())
    assert all(p["sources"][i] for i in range(n)), "a film with no recorded source"
    added_tmdb = [i for i in range(LEGACY_N, n) if "tmdb" in p["sources"][i]]
    assert all(p["tmdb_fetched"][i] for i in added_tmdb), "TMDB data without a fetch time"


def test_the_quality_report_matches_the_catalogue(c):
    q = json.loads((ART / "data_quality.json").read_text())
    assert q["films"] == len(c["title"])
    assert q["rating_backed"] == sum(1 for r in c["rc"] if r)


def test_a_film_served_twice_resolves_by_both_ids_but_is_listed_once(c):
    """MovieLens 20M listed Offside (2006) and Pokémon the Movie 2000 under two ids
    each, and both ids were served. Browsers may hold either, so neither row goes."""
    from _adaptive import Adaptive, Signal
    from _engine import get_engine
    e = get_engine()
    pairs = [(i, a) for i, a in enumerate(c["alias_of"]) if a is not None]
    assert {(c["movieId"][a], c["movieId"][i]) for i, a in pairs} >= {(48682, 80330), (3799, 91752)}
    for i, a in pairs:
        assert e.item(i)["title"] == c["title"][i]                      # the stored id still resolves
        ids = [r["item_id"] for r in e.search(c["title"][a], limit=40)["results"]]
        assert a in ids and i not in ids
    offside, alias = 10522, c["alias_of"].index(10522)
    out = Adaptive(e).recommend([Signal(alias, "like", None, 1.79e9)], [], k=24, explore=0.3, now=1.79e9)
    ids = {r["item_id"] for r in out["results"]}
    assert offside not in ids and alias not in ids                     # liked once, under either id
