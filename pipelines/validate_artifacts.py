"""
Check that a set of artifacts is internally consistent before it is served.

    python pipelines/validate_artifacts.py [--art DIR] [--faiss models/production.index]

Exits non-zero, naming every failed check, if anything is wrong. The tests run
the same checks (tests/test_catalog_and_vectors.py), and the promote step runs
them before replacing data/artifacts.

WHAT IT CHECKS
  catalogue   every column has one row per film; the first 17,719 rows are the
              films served before the expansion, in the same order (their
              item_ids are stored in browsers); MovieLens and IMDb ids are
              unique and well-formed; titles present; years, ratings and
              runtimes in range; genres from the known vocabulary; tags
              lower-case; metadata-only films carry no rating; an alias duplicates
              a listed film with its TMDB id and holds no ratings of its own
  id map      every MovieLens id maps into the catalogue; each film's own id
              maps to its own row
  vectors     one finite row per film in every matrix; unit length (or zero
              for a film with no text); a sample of films re-encoded from
              their text matches their own row — the vector belongs to the
              film it sits next to; films the two-tower model never saw have
              exactly their content projection as their item vector
  neighbours  ids in range, no film its own match, no repeats, no alias, scores
              finite and sorted; the basis label agrees with the co-rating evidence
  browse      one Bayesian score per film; every listed film exists and is
              not an alias
  search      the prebuilt search index is present, stamped with this
              catalogue's checksum, and identical to a fresh build from it
  manifest    checksums and row counts match the files on disk
  legacy index (optional) the FAISS index of the original two-tower model
              reconstructs to item_vectors.npy
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "data" / "artifacts"
sys.path.insert(0, str(ROOT / "api"))
sys.path.insert(0, str(ROOT / "pipelines"))
from _engine import load_catalog  # noqa: E402

LEGACY_N = 17_719
GENRES = {"Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
          "Film-Noir", "Horror", "IMAX", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"}
PER_FILM = ("semantic_items.npy", "tower_items.npy", "tower_user.npz", "neighbors.npz", "browse.json")


class Report:
    def __init__(self):
        self.failed: list[str] = []
        self.passed = 0

    def check(self, ok: bool, what: str):
        if ok:
            self.passed += 1
        else:
            self.failed.append(what)


def check_catalog(art: Path, r: Report) -> dict:
    c = load_catalog(art)
    n = len(c["title"])
    r.check(all(len(v) == n for v in c.values() if isinstance(v, list)), "catalog.json: columns differ in length")
    classes = np.load(art / "item_classes.npy") if (art / "item_classes.npy").exists() else None
    if classes is not None:
        r.check([int(m) for m in c["movieId"][:LEGACY_N]] == [int(x) for x in classes],
                "catalog.json: the first 17,719 films are not the ones served before the expansion")
    r.check(len(set(c["movieId"])) == n, "catalog.json: duplicate MovieLens ids")
    if "imdbId" in c:
        imdb = [x for x in c["imdbId"] if x]
        r.check(len(imdb) == len(set(imdb)), "catalog.json: two films share an IMDb id")
        r.check(all(re.fullmatch(r"tt\d{7,8}", x) for x in imdb), "catalog.json: malformed IMDb id")
    r.check(all(isinstance(t, str) and t.strip() for t in c["title"]), "catalog.json: a film without a title")
    r.check(all(y is None or 1874 <= y <= 2026 for y in c["year"]), "catalog.json: year out of range")
    r.check(all(isinstance(x, int) and x >= 0 for x in c["rc"]), "catalog.json: rating count not a count")
    r.check(all((rc == 0) == (ra is None) for rc, ra in zip(c["rc"], c["ra"])),
            "catalog.json: a film's average rating disagrees with its rating count (metadata-only films must have none)")
    r.check(all(ra is None or 0.5 <= ra <= 5.0 for ra in c["ra"]), "catalog.json: average rating outside 0.5-5")
    r.check(all(rt is None or rt > 0 for rt in c["runtime"]), "catalog.json: non-positive runtime")
    r.check(all(set(gs) <= GENRES for gs in c["genres"]), "catalog.json: a genre outside the MovieLens vocabulary")
    r.check(all(t == t.lower() for ts in c["tags"] for t in ts), "catalog.json: a tag that is not lower-case")
    r.check(all(m in ("movie", "tv") for m in c["media"]), "catalog.json: unknown media type")
    alias = c.get("alias_of") or [None] * n
    r.check(all(a is None or (0 <= a < n and alias[a] is None and c["tmdbId"][a] == c["tmdbId"][i] and c["rc"][i] == 0)
                for i, a in enumerate(alias)),
            "catalog.json: an alias that is not a duplicate of a listed film with the same TMDB id, "
            "or that kept ratings of its own")
    return c


def check_id_map(art: Path, c: dict, r: Report):
    f = art / "id_map.json"
    if not f.exists():
        return
    m = {int(k): int(v) for k, v in json.loads(f.read_text()).items()}
    n = len(c["title"])
    r.check(all(0 <= v < n for v in m.values()), "id_map.json: an id outside the catalogue")
    alias = c.get("alias_of") or [None] * n
    r.check(all(m.get(int(mid)) == (i if alias[i] is None else alias[i]) for i, mid in enumerate(c["movieId"])),
            "id_map.json: a film's own id maps elsewhere than its row (or, for an alias, its listed film)")


def unit_rows(M: np.ndarray, name: str, r: Report, allow_zero: bool = True):
    f = M.astype(np.float32)
    r.check(bool(np.isfinite(f).all()), f"{name}: non-finite values")
    norms = np.linalg.norm(f, axis=1)
    ok = np.isclose(norms, 1.0, atol=1e-2) | ((norms == 0) if allow_zero else False)
    r.check(bool(ok.all()), f"{name}: rows that are neither unit length nor empty")


def check_vectors(art: Path, c: dict, r: Report, sample: int = 300):
    n = len(c["title"])
    S = np.load(art / "semantic_items.npy")
    r.check(S.shape == (n, 256), f"semantic_items.npy is {S.shape}, expected ({n}, 256)")
    unit_rows(S, "semantic_items.npy", r)
    # Alignment: re-encode a sample of films from their text with the served model.
    from _semantic import SemanticModel
    from build_semantic import doc_text
    vocab = (art / "semantic_vocab.txt").read_text(encoding="utf-8").rstrip("\n").split("\n")
    model = SemanticModel(vocab, np.load(art / "semantic_tokens.npy"))
    rng = random.Random(3)
    ids = rng.sample(range(n), min(sample, n))
    fresh = model.encode_many([doc_text(c, i) for i in ids])
    cos = (fresh * S[ids].astype(np.float32)).sum(1)
    nz = np.linalg.norm(fresh, axis=1) > 0
    r.check(bool((cos[nz] > 0.999).all()), "semantic_items.npy: a film's row is not the encoding of its own text")

    V = np.load(art / "tower_items.npy")
    r.check(V.shape[0] == n and V.ndim == 2, f"tower_items.npy is {V.shape}, expected {n} rows")
    unit_rows(V, "tower_items.npy", r, allow_zero=False)
    z = np.load(art / "tower_user.npz")
    r.check(z["hist"].shape == V.shape, "tower_user.npz: history table does not match the item table")
    r.check(bool(np.isfinite(z["hist"].astype(np.float32)).all()), "tower_user.npz: non-finite history embeddings")
    if "seen" in z.files and "item_proj" in z.files:
        cold = np.flatnonzero(~z["seen"])
        if cold.size:
            pick = cold[np.random.default_rng(5).choice(cold.size, size=min(200, cold.size), replace=False)]
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):   # spurious BLAS flags
                q = S[pick].astype(np.float32) @ z["item_proj"]
            q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
            cos = (q * V[pick].astype(np.float32)).sum(1)
            r.check(bool((cos > 0.999).all()),
                    "tower_items.npy: a film with no training data is not its content projection (rows misaligned?)")


def check_neighbors(art: Path, c: dict, r: Report):
    n = len(c["title"])
    z = np.load(art / "neighbors.npz")
    ids, sc = z["ids"], z["scores"].astype(np.float32)
    r.check(ids.shape[0] == n and sc.shape == ids.shape, f"neighbors.npz is {ids.shape} for {n} films")
    r.check(bool(((ids >= 0) & (ids < n)).all()), "neighbors.npz: an id outside the catalogue")
    r.check(bool((ids != np.arange(n)[:, None]).all()), "neighbors.npz: a film is its own match")
    r.check(all(len(set(row.tolist())) == ids.shape[1] for row in ids), "neighbors.npz: a film matched twice")
    r.check(bool(np.isfinite(sc).all()), "neighbors.npz: non-finite scores")
    hidden = np.array([a is not None for a in c.get("alias_of") or [None] * n], bool)
    r.check(not hidden[ids].any(), "neighbors.npz: an alias offered as a match")
    r.check(bool((np.diff(sc, axis=1) <= 1e-3).all()), "neighbors.npz: scores not sorted best first")
    if "basis" in z.files:
        b, raters = z["basis"], z["positive_raters"]
        r.check(b.shape == (n,) and set(np.unique(b).tolist()) <= {0, 1, 2}, "neighbors.npz: bad basis labels")
        r.check(bool(((b == 2) == (raters == 0)).all()), "neighbors.npz: 'content' must mean no positive raters")
        rc = np.array(c["rc"])
        r.check(bool((b[rc == 0] == 2).all()), "neighbors.npz: a metadata-only film is matched by co-rating")


def check_browse(art: Path, c: dict, r: Report):
    n = len(c["title"])
    b = json.loads((art / "browse.json").read_text())
    r.check(len(b["bayes"]) == n, "browse.json: Bayesian scores do not cover every film")
    listed = b["top50"] + b["tv"] + b["hero"] + [i for ids in b["genres"].values() for i in ids]
    r.check(all(0 <= i < n for i in listed), "browse.json: lists a film outside the catalogue")
    r.check(len(set(b["top50"])) == len(b["top50"]), "browse.json: a film twice in the top 50")
    alias = c.get("alias_of") or [None] * n
    r.check(all(alias[i] is None for i in listed), "browse.json: lists an alias")


def check_search(art: Path, c: dict, r: Report):
    from _engine import SEARCH_INDEX, aliases, build_search, catalog_path, load_search
    if not (art / SEARCH_INDEX).exists():
        r.check(False, f"{SEARCH_INDEX} is missing — run pipelines/build_search_index.py")
        return
    loaded = load_search(art, catalog_path(art), len(c["title"]))
    r.check(loaded is not None, f"{SEARCH_INDEX}: built from another catalogue — rerun pipelines/build_search_index.py")
    if loaded is not None:
        built = build_search(c, *aliases(c))
        r.check(all(loaded[k] == v and (not isinstance(v, dict) or list(loaded[k]) == list(v)) for k, v in built.items()),
                f"{SEARCH_INDEX}: differs from a fresh build of the catalogue")


def check_manifest(art: Path, c: dict, r: Report):
    f = art / "manifest.json"
    if not f.exists():
        return
    from _artifacts import sha256
    n = len(c["title"])
    for name, meta in json.loads(f.read_text()).get("artifacts", {}).items():
        if not (art / name).exists():
            r.check(False, f"manifest.json lists {name}, which is missing")
            continue
        r.check(sha256(art / name) == meta["sha256"], f"{name}: checksum differs from manifest.json")
        if name in PER_FILM:
            r.check(meta.get("rows") == n, f"{name}: built for {meta.get('rows')} films, catalogue has {n}")


def check_faiss(art: Path, index: Path, r: Report):
    try:
        import faiss
    except ImportError:
        print("[validate] faiss not installed — skipping the legacy index check")
        return
    ix = faiss.read_index(str(index))
    V = np.load(art / "item_vectors.npy").astype(np.float32)
    r.check(ix.ntotal == V.shape[0], f"{index}: {ix.ntotal} vectors, item_vectors.npy has {V.shape[0]}")
    rec = ix.reconstruct_n(0, ix.ntotal)
    r.check(bool(np.allclose(rec, V, atol=1e-2)), f"{index}: does not reconstruct to item_vectors.npy")
    _, nn = ix.search(V[:200], 1)
    r.check(bool((nn[:, 0] == np.arange(200)).all()), f"{index}: a vector's nearest neighbour is not itself")


def validate(art: Path, faiss_index: Path | None = None) -> Report:
    r = Report()
    c = check_catalog(art, r)
    check_id_map(art, c, r)
    check_vectors(art, c, r)
    check_neighbors(art, c, r)
    check_browse(art, c, r)
    check_search(art, c, r)
    check_manifest(art, c, r)
    if faiss_index and faiss_index.exists() and (art / "item_vectors.npy").exists():
        check_faiss(art, faiss_index, r)
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--art", type=Path, default=ART)
    ap.add_argument("--faiss", type=Path, default=ROOT / "models" / "production.index")
    args = ap.parse_args()
    r = validate(args.art, args.faiss)
    for f in r.failed:
        print(f"[validate] FAILED {f}")
    print(f"[validate] {args.art}: {r.passed} checks passed, {len(r.failed)} failed")
    return 1 if r.failed else 0


if __name__ == "__main__":
    sys.exit(main())
