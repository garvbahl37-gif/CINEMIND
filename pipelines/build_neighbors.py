"""
Build each film's 24 matches, and the home page rows, for the whole catalogue.

    python pipelines/build_neighbors.py --ml32m data/raw/ml-32m [--art DIR]

THE SCORE
For film i and a candidate j, the same weighted sum pipelines/build_artifacts.py
has always used —

    1.35·e·CF + 0.75·two-tower + 0.60·genre + 0.50·tags − 0.70·thin + 0.40·franchise
      + (1 − e)·(w_sem·sentence-embedding similarity − w_era·era gap)

— where e = min(1, positive raters of i / 20) says how much co-rating evidence
film i has. A film with twenty or more positive raters gets exactly the old
formula (labelled `collaborative`). A film with fewer leans on what it is about
(`hybrid`), and a film nobody rated positively is matched on content alone
(`content`) — never on co-ratings it does not have. The label is stored per film
so the API can say which kind of match it is showing.

Candidates are the union of each film's top 200 by co-rating, by two-tower
similarity and (for films with e < 1) by sentence-embedding similarity, so a
film is only ever ranked against films that one of the signals put forward.

The era gap is |Δyear| / 40, capped at 1 (0 when either year is unknown): what
a film is about says nothing about when it was made, and without it a
content-matched film's matches sat about 20 years away on average, against 8
for a co-rated one — people who watch 1930s melodramas watch 1930s melodramas.

w_sem and w_era are chosen on validation, not on the test set: for a sample of
films with plenty of co-rating evidence, the co-ratings are hidden and each
pair of weights' content-only matches are compared with the film's real
collaborative matches.

INPUTS  catalog.json, id_map.json, tower_items.npy, semantic_items.npy (in --art)
WRITES  neighbors.npz (ids, scores, basis, positive_raters) and browse.json,
        atomically, and records them in manifest.json.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "data" / "artifacts"
sys.path.insert(0, str(ROOT / "pipelines"))
sys.path.insert(0, str(ROOT / "api"))
from _artifacts import record, save_npz  # noqa: E402
from _engine import load_catalog  # noqa: E402
from build_artifacts import build_browse  # noqa: E402

TOPK, CAND = 24, 200
ERA_SPAN = 40.0
W = dict(icf=1.35, tt=0.75, gen=0.60, tag=0.50, thin=0.70, fran=0.40)
FULL_EVIDENCE = 20
BASIS = {0: "collaborative", 1: "hybrid", 2: "content"}


def log(msg):
    print(f"[neighbors {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def norm_title(s: str) -> str:
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    s = re.sub(r"\b(the|a|an|part|episode|[ivx]+|\d+)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def ratings_matrix(ml32m: Path, pos: dict[int, int], n: int):
    """Users × films of positive ratings (≥ 3.5), damped as build_artifacts.py damps them."""
    r = pd.read_csv(ml32m / "ratings.csv", usecols=["userId", "movieId", "rating"],
                    dtype={"userId": np.int32, "movieId": np.int32, "rating": np.float32})
    r = r[r.rating >= 3.5]
    it = r.movieId.map(pos)
    keep = it.notna().values
    it = it[keep].astype(np.int32).values
    _, ur = np.unique(r.userId.values[keep], return_inverse=True)
    del r
    nU = int(ur.max()) + 1
    X = sp.csr_matrix((np.ones(len(it), np.float32), (ur, it)), shape=(nU, n))
    X.sum_duplicates()
    X.data[:] = 1.0
    raters = np.asarray(X.sum(0)).ravel().astype(np.int32)           # positive raters per film
    ul = np.asarray(X.sum(1)).ravel(); ul[ul == 0] = 1
    X = sp.diags((1 / np.sqrt(ul)).astype(np.float32)) @ X           # damp power users
    df = raters.astype(np.float32).copy(); df[df == 0] = 1
    X = X @ sp.diags(np.log(1 + nU / df).astype(np.float32))         # damp blockbusters
    cn = np.sqrt(np.asarray(X.multiply(X).sum(0)).ravel()); cn[cn == 0] = 1
    X = (X @ sp.diags((1 / cn).astype(np.float32))).tocsr().astype(np.float32)
    return X, raters


def content(cols, n):
    GEN = sorted({g for gs in cols["genres"] for g in gs})
    gi = {g: i for i, g in enumerate(GEN)}
    G = np.zeros((n, len(GEN)), np.float32)
    for i, gs in enumerate(cols["genres"]):
        for g in gs:
            G[i, gi[g]] = 1
    Gw = G * np.log(n / np.maximum(G.sum(0), 1)).astype(np.float32)     # rare genres count more
    Gn = Gw / np.maximum(np.linalg.norm(Gw, axis=1, keepdims=True), 1e-6)
    TAG = sorted({t for ts in cols["tags"] for t in ts})
    ti = {t: i for i, t in enumerate(TAG)}
    rr, cc = [], []
    for i, ts in enumerate(cols["tags"]):
        for t in ts:
            rr.append(i); cc.append(ti[t])
    T = sp.csr_matrix((np.ones(len(rr), np.float32), (rr, cc)), shape=(n, len(TAG)))
    tn = np.sqrt(np.asarray(T.multiply(T).sum(1)).ravel()); tn[tn == 0] = 1
    Tn = (sp.diags(1 / tn).tocsr() @ T).tocsr().astype(np.float32)
    return Gn, Tn


class Graph:
    def __init__(self, cols, X, raters, V, S, P=None):
        self.n = len(cols["title"])
        self.Xc, self.X = X.tocsc(), X
        self.raters, self.V, self.S = raters, V, S
        self.P = P          # item tower's content projection, for simulated cold starts
        self.Gn, self.Tn = content(cols, self.n)
        self.TnT = self.Tn.T.tocsr()
        rc = np.array(cols["rc"], np.float32)
        self.thin = np.maximum(0, 1 - np.log10(rc + 1) / np.log10(300)).astype(np.float32)
        self.titles = [norm_title(t) for t in cols["title"]]
        self.year = np.array([y or 0 for y in cols["year"]], np.float32)
        # Aliases (a served film's second id) are never anyone's match; an alias's own
        # matches leave out the film it duplicates.
        self.alias = cols.get("alias_of") or [None] * self.n
        self.hidden = np.array([a is not None for a in self.alias], bool)

    def block(self, s0: int, e: int, w_sem: float, w_era: float = 0.0, hide_cf: bool = False):
        """Matches for films s0..e-1; hide_cf scores them as if nobody had rated them."""
        # macOS Accelerate raises spurious divide/overflow flags on finite inputs;
        # the output is checked for finiteness before anything is written.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            return self._block(s0, e, w_sem, w_era, hide_cf)

    def _block(self, s0: int, e: int, w_sem: float, w_era: float, hide_cf: bool):
        ids = np.arange(s0, e)
        icf = np.zeros((e - s0, self.n), np.float32) if hide_cf else (self.Xc[:, s0:e].T @ self.X).toarray()
        if hide_cf and self.P is not None:
            # A cold film's two-tower vector comes from its text alone.
            q = self.S[s0:e] @ self.P
            q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
            tt = q @ self.V.T
        else:
            tt = self.V[s0:e] @ self.V.T
        sem = self.S[s0:e] @ self.S.T
        gen = self.Gn[s0:e] @ self.Gn.T
        tag = (self.Tn[s0:e] @ self.TnT).toarray()
        ev = np.zeros(e - s0, np.float32) if hide_cf else np.minimum(1.0, self.raters[s0:e] / FULL_EVIDENCE)
        for M in (icf, tt, sem):
            M[np.arange(e - s0), ids] = -np.inf if M is not icf else 0
            if self.hidden.any():
                M[:, self.hidden] = -np.inf if M is not icf else 0
                for j, i in enumerate(ids):
                    if self.alias[i] is not None:
                        M[j, self.alias[i]] = -np.inf if M is not icf else 0
        out_ids = np.zeros((e - s0, TOPK), np.int32)
        out_sc = np.zeros((e - s0, TOPK), np.float32)
        for j in range(e - s0):
            i = s0 + j
            pools = [np.argpartition(-tt[j], CAND)[:CAND]]
            if ev[j] > 0:
                pools.append(np.argpartition(-icf[j], CAND)[:CAND])
            if ev[j] < 1:
                pools.append(np.argpartition(-sem[j], CAND)[:CAND])
            cand = np.unique(np.concatenate(pools))
            cand = cand[(cand != i) & ~self.hidden[cand] & (cand != (self.alias[i] if self.alias[i] is not None else -1))]
            a = icf[j, cand]
            a = a / a.max() if a.max() > 0 else a
            b = tt[j, cand]
            b = (b - b.min()) / max(float(np.ptp(b)), 1e-6)
            fr = np.zeros(len(cand), np.float32)
            na = self.titles[i]
            if len(na) >= 4:
                for k, c in enumerate(cand):
                    nb = self.titles[c]
                    if nb and (na == nb or na in nb or nb in na):
                        fr[k] = 1.0
            yc = self.year[cand]
            era = (np.minimum(np.abs(yc - self.year[i]), ERA_SPAN) / ERA_SPAN) * ((yc > 0) & (self.year[i] > 0))
            sc = (W["icf"] * ev[j] * a + W["tt"] * b + W["gen"] * gen[j, cand] + W["tag"] * tag[j, cand]
                  + (1 - ev[j]) * (w_sem * sem[j, cand] - w_era * era)
                  - W["thin"] * self.thin[cand] + W["fran"] * fr)
            o = np.argsort(-sc, kind="stable")[:TOPK]
            out_ids[j], out_sc[j] = cand[o], sc[o]
        return out_ids, out_sc, ev


def tune(g: Graph, sample: np.ndarray, sem_weights=(0.0, 0.5, 0.9, 1.3),
         era_weights=(0.0, 0.25, 0.5, 1.0, 1.5)) -> tuple[tuple[float, float], list]:
    """Pick (w_sem, w_era) by how well content-only matches recover real collaborative ones."""
    truth = {}
    for i in sample:
        ids, _, _ = g.block(int(i), int(i) + 1, 0.0)
        truth[int(i)] = set(ids[0].tolist())
    scores = {}
    for ws in sem_weights:
        for we in era_weights:
            hits = 0
            for i in sample:
                ids, _, _ = g.block(int(i), int(i) + 1, ws, we, hide_cf=True)
                hits += len(truth[int(i)] & set(ids[0].tolist()))
            scores[(ws, we)] = round(hits / (len(sample) * TOPK), 4)
            log(f"  w_sem {ws}, w_era {we}: content-only matches recover {scores[(ws, we)]:.1%} of collaborative ones")
    best = max(scores, key=lambda k: (scores[k], -k[0], -k[1]))
    return best, [{"w_sem": ws, "w_era": we, "recovered": v} for (ws, we), v in scores.items()]


def graph_checks(cols: dict, nbr: np.ndarray, basis: np.ndarray) -> dict:
    """How alike films and their matches are — over the whole graph and by basis.
    The About page and the README quote these."""
    gs = [set(g) for g in cols["genres"]]
    yr = cols["year"]
    out = {}
    for name, rows in (("all", range(len(gs))), *((BASIS[b], np.flatnonzero(basis == b)) for b in BASIS)):
        share, gaps = [], []
        for i in rows:
            if gs[i]:
                share.extend(bool(gs[i] & gs[j]) for j in nbr[i])
            if yr[i] is not None:
                gaps.extend(abs(yr[i] - yr[j]) for j in nbr[i] if yr[j] is not None)
        out[name] = {"films": len(rows),
                     "share_a_genre": round(float(np.mean(share)), 4) if share else None,
                     "mean_year_gap": round(float(np.mean(gaps)), 2) if gaps else None}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ml32m", type=Path, required=True)
    ap.add_argument("--art", type=Path, default=ART)
    ap.add_argument("--w-sem", type=float, default=None, help="skip tuning: the sentence-embedding weight")
    ap.add_argument("--w-era", type=float, default=0.0, help="with --w-sem: the era-gap weight")
    ap.add_argument("--tune-sample", type=int, default=400)
    ap.add_argument("--block", type=int, default=256)
    args = ap.parse_args()
    rng = np.random.default_rng(11)

    cols = load_catalog(args.art)
    n = len(cols["title"])
    id_map = {int(m): int(i) for m, i in json.loads((args.art / "id_map.json").read_text()).items()}
    V = np.load(args.art / "tower_items.npy").astype(np.float32)
    S = np.load(args.art / "semantic_items.npy").astype(np.float32)
    for name, M in (("tower_items", V), ("semantic_items", S)):
        if M.shape[0] != n:
            sys.exit(f"{name}.npy has {M.shape[0]} rows for {n} films — rebuild it first.")
    V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)

    log(f"{n:,} films; positive ratings from ml-32m ...")
    X, raters = ratings_matrix(args.ml32m, id_map, n)
    tu = np.load(args.art / "tower_user.npz")
    P = tu["item_proj"].astype(np.float32) if "item_proj" in tu.files else None
    g = Graph(cols, X, raters, V, S, P)
    basis = np.where(raters >= FULL_EVIDENCE, 0, np.where(raters > 0, 1, 2)).astype(np.int8)
    log("basis: " + ", ".join(f"{BASIS[b]} {int((basis == b).sum()):,}" for b in BASIS))

    tuning = []
    if args.w_sem is None:
        warm = np.flatnonzero(raters >= 200)
        sample = rng.choice(warm, size=min(args.tune_sample, len(warm)), replace=False)
        log(f"tuning w_sem and w_era on {len(sample)} well-rated films (validation by hidden co-ratings) ...")
        (w_sem, w_era), tuning = tune(g, sample)
    else:
        w_sem, w_era = args.w_sem, args.w_era
    log(f"w_sem = {w_sem}, w_era = {w_era}")

    nbr = np.zeros((n, TOPK), np.int32)
    sco = np.zeros((n, TOPK), np.float32)
    t0 = time.time()
    for s0 in range(0, n, args.block):
        e = min(s0 + args.block, n)
        nbr[s0:e], sco[s0:e], _ = g.block(s0, e, w_sem, w_era)
        if (s0 // args.block) % 40 == 0:
            log(f"  {e:,}/{n:,} ({time.time() - t0:.0f}s)")

    # Checks before anything is written.
    assert ((nbr >= 0) & (nbr < n)).all(), "neighbour id out of range"
    assert (nbr != np.arange(n)[:, None]).all(), "a film recommended to itself"
    assert np.isfinite(sco).all(), "non-finite neighbour score"
    assert all(len(set(r.tolist())) == TOPK for r in nbr), "duplicate neighbours"
    assert not g.hidden[nbr].any(), "an alias offered as a match"

    checks = graph_checks(cols, nbr, basis)
    for name, c in checks.items():
        log(f"  {name:13} {c['films']:>6,} films: {c['share_a_genre'] or 0:.1%} of matches share a genre, "
            f"{c['mean_year_gap'] or 0:.1f} years apart on average")

    save_npz(args.art / "neighbors.npz", ids=nbr, scores=sco.astype(np.float16), basis=basis,
             positive_raters=raters)
    record(args.art, "neighbors.npz", rows=n, k=TOPK, w_sem=w_sem, w_era=w_era, era_span_years=ERA_SPAN, tuning=tuning,
           weights=W, full_evidence_raters=FULL_EVIDENCE,
           basis={BASIS[b]: int((basis == b).sum()) for b in BASIS}, checks=checks)
    build_browse(cols, args.art)
    record(args.art, "browse.json", rows=n)
    log(f"wrote neighbors.npz and browse.json in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
