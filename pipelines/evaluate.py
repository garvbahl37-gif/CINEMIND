"""
Offline evaluation: baseline vs two-tower retrieval vs adaptive reranking, on
the same held-out data — plus a counterfactual-evaluation study.

    python pipelines/evaluate.py --ml32m data/raw/ml-32m [--users 3000] [--art DIR]
        [--universe CATALOG] [--users-file FILE] [--save-users FILE] [--label NAME] [--per-user FILE]
        [--cutoff DATE --until DATE] [--blend JSON] [--blend-grid FILE] [--no-ope]

Run it on any artifact directory, so a catalogue before and after a change can
be compared on the same people and the same held-out ratings:
  --users-file    evaluate exactly these userIds (the before/after population)
  --universe      only the films of this catalogue are candidates and test
                  targets — a common candidate universe for systems whose
                  catalogues differ. Every policy retrieves within it: the
                  adaptive engine's candidate sources are limited to it before
                  any top-k cut, not filtered afterwards. Targets are matched
                  by their own MovieLens id, so ratings filed under a duplicate
                  id this catalogue merged into an old film don't join them

And on a validation period instead of the test period, to choose settings:
  --cutoff 2015-01-01 --until 2016-01-01 --art data/raw/validation
                  everything trained before 2015 and scored on 2015 — with the
                  two-tower model trained before 2015 (rebuild.py assembles it).
                  A model trained past the cutoff is refused: it has seen the
                  held-out period
  --blend-grid F  each blend of the adaptive engine's weights in F as an extra
                  policy, compared person by person with the current weights

THE SPLIT
MovieLens 32M is split in time at 2016-01-01. Everything before is training,
everything after is the test period. That keeps every model honest:
  - the two-tower model was trained on MovieLens 32M ratings before the
    cutoff only (pipelines/train_two_tower.py), with its epoch chosen on a
    validation set inside the training period, so it has never seen a test
    rating;
  - item-item CF, popularity and the quality prior are rebuilt here from
    training ratings only, with the production formulas;
  - the adaptive engine is the production code in api/_adaptive.py, handed
    those training-only inputs. Its co-rating graph is the rebuilt item-item
    CF rather than the production hybrid graph, which was built from all 32M
    ratings and would leak the test period.
Evaluated users have ≥10 positive ratings (≥4★) before the cutoff and ≥3
positives after it on films they hadn't rated before; their training ratings
become the signals, with their real timestamps, and "now" is the cutoff.

THE POLICIES
  popularity   most-liked films in training, unseen ones first
  item-cf      Σ similarity to everything the user liked (damped co-occurrence)
  two-tower    nearest films to the mean of the user's liked films' vectors
  adaptive     the production engine: drift, negatives, quality, MMR
               diversity, exploration at the default setting
  + ablations  adaptive without each of drift / diversity / exploration /
               negative signals, to show what each piece trades

METRICS
Recall@10/20 (hits / min(|test|, K)), Precision@10/20, NDCG@10/20, MAP@20,
HitRate@10, catalogue coverage (of the candidate universe), intra-list
diversity (mean pairwise 1 − cosine of the films' content vectors), novelty
(mean −log2 of training popularity), long-tail share (outside the 1,000
most-liked), cold-start share (recommended films with no training ratings),
cold and warm Recall@20 (on test positives without / with training ratings),
and each policy's ranking latency per user (p50/p95/p99). Recall@20 and
NDCG@20 carry 95% bootstrap intervals; every policy also carries paired
differences against the adaptive engine.

COUNTERFACTUAL EVALUATION
Could logged data from one recommender tell us how another would do, without
deploying it? A semi-synthetic study: for each user, a candidate set of ~350
films; a popularity-leaning logging policy "shows" films and records whether
each was a held-out positive, with its propensity. Each policy's value — the
chance that a film it shows (one drawn from its own top 10, weighted by rank,
with 5% spread uniformly so the logging policy always covers it) is a hit —
is then estimated from those logs with IPS, SNIPS, the direct method and
doubly robust, and compared with the true value, which this setup can compute
exactly. Effective sample sizes and overlap are reported alongside, because
an estimate is only as good as the logs that support it. The assumptions are
written into the output.

Writes data/artifacts/eval.json, served at /api/evaluation.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "data" / "artifacts"
sys.path.insert(0, str(ROOT / "api"))

from _adaptive import Adaptive, Options, Signal  # noqa: E402
from _engine import Engine, load_catalog  # noqa: E402
from _semantic import item_matrix, mm  # noqa: E402
from _tower import Tower  # noqa: E402

K = 20
TOPN = 50          # neighbours kept per film in the rebuilt CF graph


def log(msg: str):
    print(f"[eval {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# -- data ----------------------------------------------------------------------

def item_map(art: Path, movie_ids: list[int]) -> dict[int, int]:
    """MovieLens id → item_id, through id_map.json (which folds in merged duplicates) when present."""
    f = art / "id_map.json"
    if f.exists():
        return {int(m): int(i) for m, i in json.loads(f.read_text()).items()}
    return {int(m): i for i, m in enumerate(movie_ids)}


def load(ml32m: Path, pos: dict[int, int]):
    r = pd.read_csv(ml32m / "ratings.csv", dtype={"userId": np.int32, "movieId": np.int32,
                                                  "rating": np.float32, "timestamp": np.int64})
    r["item"] = r.movieId.map(pos)
    r = r.dropna(subset=["item"])
    r["item"] = r["item"].astype(np.int32)
    return r[["userId", "movieId", "item", "rating", "timestamp"]]


def build_cf(train: pd.DataFrame, n_items: int):
    """Item-item CF exactly as pipelines/build_artifacts.py builds it, on training ratings only."""
    t = train[train.rating >= 3.5]
    _, ur = np.unique(t.userId.values, return_inverse=True)
    nU = int(ur.max()) + 1
    X = sp.csr_matrix((np.ones(len(t), np.float32), (ur, t.item.values)), shape=(nU, n_items))
    X.sum_duplicates()
    X.data[:] = 1.0
    ul = np.asarray(X.sum(1)).ravel()
    ul[ul == 0] = 1
    X = sp.diags((1 / np.sqrt(ul)).astype(np.float32)) @ X                  # damp power users
    df = np.asarray((X > 0).sum(0)).ravel().astype(np.float32)
    df[df == 0] = 1
    X = X @ sp.diags(np.log(1 + nU / df).astype(np.float32))                # damp blockbusters
    cn = np.sqrt(np.asarray(X.multiply(X).sum(0)).ravel())
    cn[cn == 0] = 1
    X = (X @ sp.diags((1 / cn).astype(np.float32))).tocsr().astype(np.float32)
    Xc = X.tocsc()
    nbr = np.zeros((n_items, TOPN), np.int32)
    sco = np.zeros((n_items, TOPN), np.float32)
    k = min(TOPN, n_items - 1)         # a catalogue smaller than TOPN keeps zero-evidence padding
    for s0 in range(0, n_items, 1024):
        e = min(s0 + 1024, n_items)
        blk = (Xc[:, s0:e].T @ X).toarray()
        blk[np.arange(e - s0), np.arange(s0, e)] = 0
        idx = np.argpartition(-blk, k, axis=1)[:, :k]
        vals = np.take_along_axis(blk, idx, 1)
        o = np.argsort(-vals, axis=1, kind="stable")
        nbr[s0:e, :k] = np.take_along_axis(idx, o, 1)
        sco[s0:e, :k] = np.take_along_axis(vals, o, 1)
    # A film with no co-ratings keeps an all-zero row: no evidence, which the
    # engine treats as contributing nothing (it never rescales a zero row).
    return nbr, sco


# -- policies ------------------------------------------------------------------

def topk(score: np.ndarray, seen: np.ndarray, k: int) -> np.ndarray:
    s = score.astype(np.float64).copy()
    s[seen] = -np.inf
    idx = np.argpartition(-s, k)[:k]
    return idx[np.argsort(-s[idx])]


def scores_cf(nbr, sco, liked: np.ndarray, n: int) -> np.ndarray:
    s = np.zeros(n, np.float32)
    np.add.at(s, nbr[liked].ravel(), sco[liked].ravel())
    return s


def scores_tt(tower, liked: np.ndarray) -> np.ndarray:
    """The user tower over the liked films, scored against every film's item vector."""
    u = tower.user(liked)
    return mm(tower.V, u) if u is not None else np.zeros(tower.V.shape[0], np.float32)


# -- metrics -------------------------------------------------------------------

def average_precision(rec: list[int], rel: set[int], k: int) -> float:
    hits, total = 0, 0.0
    for i, x in enumerate(rec[:k]):
        if x in rel:
            hits += 1
            total += hits / (i + 1)
    return total / min(len(rel), k) if rel else 0.0


def pct(xs: list[float]) -> dict:
    a = np.array(xs)
    return {f"p{q}": round(float(np.percentile(a, q)), 2) for q in (50, 95, 99)} if len(a) else {}


def ndcg(rec: list[int], rel: set[int], k: int) -> float:
    dcg = sum(1 / math.log2(i + 2) for i, x in enumerate(rec[:k]) if x in rel)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / idcg if idcg else 0.0


def ild(rec: list[int], content) -> float:
    """Intra-list diversity in the content space — neutral between the models compared."""
    if len(rec) < 2:
        return 0.0
    V = content[rec]
    S = mm(V, V.T)
    n = len(rec)
    return float((1 - (S.sum() - np.trace(S)) / (n * (n - 1))))


def boot_ci(x: np.ndarray, rng, n: int = 1000) -> list[float]:
    """95% percentile bootstrap interval of the mean. Given per-user differences
    between two policies, it is a paired interval: both were scored on the same
    people, so it is far tighter than comparing their separate intervals."""
    idx = rng.integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ml32m", type=Path, required=True)
    ap.add_argument("--users", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=ART / "eval.json")
    ap.add_argument("--art", type=Path, default=ART, help="artifact directory of the system to evaluate")
    ap.add_argument("--universe", type=Path, default=None,
                    help="a catalogue (file or artifact directory) whose films alone are candidates and targets")
    ap.add_argument("--users-file", type=Path, default=None, help="JSON list of userIds to evaluate")
    ap.add_argument("--save-users", type=Path, default=None, help="write the evaluated userIds here")
    ap.add_argument("--label", default="", help="a name for this experiment")
    ap.add_argument("--no-ope", action="store_true", help="skip the counterfactual study")
    ap.add_argument("--cutoff", default="2016-01-01",
                    help="the split: ratings before it train, after it test (2015-01-01 with --until 2016-01-01 "
                         "validates on 2015, inside the training period, with a model trained before 2015)")
    ap.add_argument("--until", default=None, help="ignore ratings from this date on")
    ap.add_argument("--blend", default=None,
                    help="JSON object or file of adaptive-engine blend weights to evaluate instead of the defaults")
    ap.add_argument("--blend-grid", type=Path, default=None,
                    help="JSON file: a list of blends, each evaluated as an extra policy (for validation only)")
    ap.add_argument("--train-in-universe", action="store_true",
                    help="diagnostic: learn only from ratings of films in --universe, as a system that "
                         "knew only those films would")
    ap.add_argument("--per-user", type=Path, default=None,
                    help="write each user's Recall@20 and NDCG@20 per policy here (.json.gz), for paired "
                         "comparisons between runs")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    t_start = time.time()
    day = lambda d: int(datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())  # noqa: E731
    cutoff = day(args.cutoff)
    until = day(args.until) if args.until else None

    e = Engine(args.art)
    n = e.n
    tower = Tower(args.art)
    content = item_matrix(args.art)
    meta = {}
    if tower.trained:
        meta = json.loads(str(np.load(args.art / "tower_user.npz")["meta"]))
        if meta.get("cutoff", "2016-01-01") > args.cutoff:
            sys.exit(f"the two-tower model in {args.art} was trained on ratings up to {meta.get('cutoff', '2016-01-01')}, "
                     f"after this split's cutoff {args.cutoff}: it has seen the test period")
    log(f"{args.label or args.art}: {n:,} films; two-tower "
        f"{'trained model ' + json.dumps(meta) if tower.trained else 'original item vectors (fallback)'}")

    pos = item_map(args.art, e.c["movieId"])
    allowed = np.ones(n, bool)
    base = None
    if args.universe:
        base = {int(m) for m in load_catalog(args.universe)["movieId"]}
        allowed[:] = False
        for m, i in pos.items():
            if m in base:
                allowed[i] = True
        log(f"common universe: {int(allowed.sum()):,} of {n:,} films ({args.universe})")
    allowed &= ~e.hidden                    # aliases of listed films are never offered, as in the API

    log("loading ratings ...")
    r = load(args.ml32m, pos)
    if until:
        r = r[r.timestamp < until]
    train, test = r[r.timestamp < cutoff], r[r.timestamp >= cutoff]
    if args.train_in_universe:
        train = train[allowed[train.item.values]]
    # Only a film that existed at the cutoff can be recommended then, or missed: released
    # before it by any source's date, or rated before it (MovieLens lists a few films before
    # release). A 2019 film is neither a candidate nor a target for a list made in 2016.
    released = e.c.get("released") or [None] * n
    existed = np.array([bool(d and d < args.cutoff) or (y is not None and y < int(args.cutoff[:4]))
                        for d, y in zip(released, e.c["year"])])
    existed |= np.bincount(train.item.values, minlength=n) > 0
    log(f"{int((allowed & ~existed).sum()):,} films released after the cutoff are left out")
    allowed &= existed
    blocked = np.flatnonzero(~allowed)
    test = test[allowed[test.item.values]]
    if base is not None:
        # Targets by their own MovieLens id, so a rating filed under a duplicate id that
        # this catalogue merged into an old film doesn't join the common test set.
        test = test[test.movieId.isin(base)]
    log(f"{len(r):,} catalogue ratings: {len(train):,} train, {len(test):,} test (in the universe)")

    log("rebuilding item-item CF from training ratings ...")
    nbr, sco = build_cf(train, n)

    # Popularity, quality prior and rating counts from training only.
    pos_cnt = np.bincount(train.item[train.rating >= 4].values, minlength=n).astype(np.float32)
    rc = np.bincount(train.item.values, minlength=n).astype(np.float32)
    rsum = np.bincount(train.item.values, weights=train.rating.values, minlength=n)
    ra = np.where(rc > 0, rsum / np.maximum(rc, 1), 0)
    C, m = float(ra[rc > 0].mean()), 2500.0
    bayes = (rc * ra + m * C) / (rc + m)
    quality = np.empty(n, np.float32)
    quality[np.argsort(bayes, kind="stable")] = np.arange(n) / (n - 1)
    n_train_users = train.userId.nunique()
    popular_1000 = set(np.argsort(-pos_cnt)[:1000].tolist())

    # Eligible users.
    tr_pos = train[train.rating >= 4].groupby("userId").size()
    seen_by = train.groupby("userId")["item"].apply(np.array)
    te = test[test.rating >= 4]
    eligible = []
    for u in tr_pos[tr_pos >= 10].index.intersection(te.userId.unique()):
        eligible.append(u)
    te_pos = te[te.userId.isin(eligible)].groupby("userId")["item"].apply(set)
    users = [u for u in eligible if len(te_pos.get(u, set()) - set(seen_by[u].tolist())) >= 3]
    n_eligible = len(users)
    if args.users_file:
        fixed = [int(u) for u in json.loads(args.users_file.read_text())]
        te_pos = te[te.userId.isin(fixed)].groupby("userId")["item"].apply(set)
        users = sorted(u for u in fixed if u in seen_by.index)
        log(f"fixed population: {len(users):,} of {len(fixed):,} listed users have training ratings")
    else:
        log(f"{n_eligible:,} eligible users; evaluating {min(args.users, n_eligible):,}")
        users = sorted(rng.choice(users, size=min(args.users, len(users)), replace=False).tolist())
    if args.save_users:
        args.save_users.write_text(json.dumps([int(u) for u in users]))
    tr_by_user = {u: g for u, g in train[train.userId.isin(users)].groupby("userId")}

    adaptive = Adaptive(e, args.art, nbr=nbr, nsc=sco, quality=quality, rating_count=rc,
                        tower=tower, semantic=content)
    blend = {}
    if args.blend:
        blend = json.loads(Path(args.blend).read_text() if Path(args.blend).exists() else args.blend)
    variants = {
        "adaptive": dict(explore=0.3, opt=Options(**blend)),
        "adaptive − drift": dict(explore=0.3, opt=Options(drift=False, **blend)),
        "adaptive − diversity": dict(explore=0.3, opt=Options(diversity=False, **blend)),
        "adaptive − exploration": dict(explore=0.0, opt=Options(exploration=False, **blend)),
        "adaptive − negatives": dict(explore=0.3, opt=Options(negatives=False, **blend)),
    }
    for b in json.loads(args.blend_grid.read_text()) if args.blend_grid else []:
        variants["blend " + json.dumps(b, separators=(",", ":"))] = dict(explore=0.3, opt=Options(**{**blend, **b}))
    names = ["popularity", "item-cf", "two-tower", *variants]
    recs: dict[str, list[list[int]]] = {k: [] for k in names}
    shelves: dict[str, list[list[int]]] = {k: [] for k in variants}     # films nobody had rated
    lat: dict[str, list[float]] = {k: [] for k in names}
    truth: list[set[int]] = []
    ope_rows = []

    def timed(name, f):
        t = time.perf_counter()
        out = f()
        lat[name].append((time.perf_counter() - t) * 1000)
        return out

    log("ranking ...")
    for n_u, u in enumerate(users):
        g = tr_by_user[u]
        seen = g.item.values
        seen_set = set(seen.tolist())
        rel = te_pos.get(u, set()) - seen_set
        truth.append(rel)
        liked = g.item.values[g.rating.values >= 4]
        hide = np.concatenate([seen, blocked]) if blocked.size else seen
        s_pop = pos_cnt
        recs["popularity"].append(timed("popularity", lambda: topk(s_pop, hide, K)).tolist())
        s_cf = scores_cf(nbr, sco, liked, n)
        recs["item-cf"].append(timed("item-cf", lambda: topk(scores_cf(nbr, sco, liked, n), hide, K)).tolist())
        s_tt = scores_tt(tower, liked)
        recs["two-tower"].append(timed("two-tower", lambda: topk(scores_tt(tower, liked), hide, K)).tolist())
        sig = [Signal(int(i), "rating", float(v), float(t)) for i, v, t in
               zip(g.item.values, g.rating.values, g.timestamp.values)]
        for name, kw in variants.items():
            out = timed(name, lambda: adaptive.recommend(sig, [], k=K, now=float(cutoff), seed=args.seed + n_u,
                                                          allowed=allowed if blocked.size else None,
                                                          explain=False, **kw))
            recs[name].append(out["ids"])
            shelves[name].append(out.get("unrated", []))
        if args.no_ope:
            if (n_u + 1) % 500 == 0:
                log(f"  {n_u + 1}/{len(users)}")
            continue

        # Counterfactual study: a candidate set, a logging policy, and the truth.
        A = set(topk(s_pop, hide, 100).tolist()) | set(topk(s_cf, hide, 100).tolist()) \
            | set(topk(s_tt, hide, 100).tolist()) | set(recs["adaptive"][-1])
        unseen = np.setdiff1d(np.flatnonzero(allowed), seen)
        A |= set(rng.choice(unseen, size=50, replace=False).tolist())
        A = np.array(sorted(A))
        reward = np.array([1.0 if a in rel else 0.0 for a in A])
        lp = np.log1p(pos_cnt[A])
        logits = 1.5 * (lp - lp.mean()) / (lp.std() or 1)
        p0 = np.exp(logits - logits.max())
        p0 /= p0.sum()
        first = {name: (recs[name][-1][0] if recs[name][-1] else None) for name in names}
        lists = {name: np.searchsorted(A, [i for i in recs[name][-1][:10] if i in set(A.tolist())])
                 for name in ("popularity", "item-cf", "two-tower", "adaptive")}
        ope_rows.append({"A": A, "r": reward, "p0": p0, "first": first, "lists": lists,
                         "pop_q": np.digitize(lp, np.quantile(lp, [0.2, 0.4, 0.6, 0.8])),
                         "cf_q": np.digitize(s_cf[A], np.quantile(s_cf[A], [0.5, 0.8, 0.95]))})
        if (n_u + 1) % 500 == 0:
            log(f"  {n_u + 1}/{len(users)}")

    # -- offline metrics ----------------------------------------------------------
    log("scoring ...")
    brng = np.random.default_rng(args.seed + 1)
    # Users with no held-out positive in the universe cannot score; they are counted, not averaged.
    keep = [j for j, t in enumerate(truth) if t]
    cold = rc == 0                                   # films with no training ratings at all
    metrics = {}
    per_user: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in names:
        R = [recs[name][j] for j in keep]
        T = [truth[j] for j in keep]
        def at(fn, k):
            return np.array([fn(x, t, k) for x, t in zip(R, T)])
        rec20 = at(lambda x, t, k: len(set(x[:k]) & t) / min(len(t), k), 20)
        rec10 = at(lambda x, t, k: len(set(x[:k]) & t) / min(len(t), k), 10)
        pre20 = at(lambda x, t, k: len(set(x[:k]) & t) / k, 20)
        pre10 = at(lambda x, t, k: len(set(x[:k]) & t) / k, 10)
        nd20, nd10 = at(ndcg, 20), at(ndcg, 10)
        map20 = at(average_precision, 20)
        hit = at(lambda x, t, k: 1.0 if set(x[:k]) & t else 0.0, 10)
        cold_r = [len(set(x[:K]) & {i for i in t if cold[i]}) / min(sum(1 for i in t if cold[i]), K)
                  for x, t in zip(R, T) if any(cold[i] for i in t)]
        warm_r = [len(set(x[:K]) & {i for i in t if not cold[i]}) / min(sum(1 for i in t if not cold[i]), K)
                  for x, t in zip(R, T) if any(not cold[i] for i in t)]
        per_user[name] = (rec20, nd20)
        items = [i for x in R for i in x[:K]]
        nov = np.mean([-math.log2((pos_cnt[i] + 1) / n_train_users) for i in items])
        metrics[name] = {
            "recall": round(float(rec20.mean()), 4), "recall_ci": boot_ci(rec20, brng),
            "ndcg": round(float(nd20.mean()), 4), "ndcg_ci": boot_ci(nd20, brng),
            "recall_10": round(float(rec10.mean()), 4), "ndcg_10": round(float(nd10.mean()), 4),
            "precision_10": round(float(pre10.mean()), 4), "precision_20": round(float(pre20.mean()), 4),
            "map_20": round(float(map20.mean()), 4),
            "hit_rate_10": round(float(hit.mean()), 4),
            "coverage": round(len(set(items)) / max(int(allowed.sum()), 1), 4),
            "diversity": round(float(np.mean([ild(x[:K], content) for x in R])), 4),
            "novelty": round(float(nov), 3),
            "long_tail": round(float(np.mean([i not in popular_1000 for i in items])), 4),
            "cold_start_share": round(float(np.mean([cold[i] for i in items])), 4),
            "cold_recall": round(float(np.mean(cold_r)), 4) if cold_r else None,
            "cold_recall_users": len(cold_r),
            "warm_recall": round(float(np.mean(warm_r)), 4) if warm_r else None,
            "latency_ms": pct(lat[name]),
        }
        log(f"  {name:24} recall@20 {metrics[name]['recall']:.4f}  ndcg@20 {metrics[name]['ndcg']:.4f}  "
            f"map@20 {metrics[name]['map_20']:.4f}  cov {metrics[name]['coverage']:.3f}  "
            f"cold {metrics[name]['cold_start_share']:.3f}  p95 {metrics[name]['latency_ms'].get('p95')} ms")

    # The shelf of films nobody had rated at the cutoff: for the people who went on to love
    # one, how often the shelf held it — against the same number of films drawn at random
    # from those nobody had rated, whose chance is computed exactly.
    pool_n = int(((rc == 0) & allowed).sum())
    cold_t = [{i for i in truth[j] if cold[i]} for j in keep]
    shelf_people = [x for x, ct in enumerate(cold_t) if ct]
    shelf_hits: dict[str, np.ndarray] = {}
    for name in variants:
        S = [shelves[name][j] for j in keep]
        k_ = max((len(x) for x in S), default=0)
        if not shelf_people or not k_:
            continue
        hits = np.array([1.0 if set(S[x]) & cold_t[x] else 0.0 for x in shelf_people])
        chance = float(np.mean([1 - math.prod((pool_n - len(cold_t[x]) - t) / (pool_n - t) for t in range(k_))
                                for x in shelf_people]))
        shelf_hits[name] = hits
        metrics[name]["unrated_shelf"] = {
            "k": k_, "pool": pool_n, "people": len(shelf_people),
            "hit_rate": round(float(hits.mean()), 4), "hit_rate_ci": boot_ci(hits, brng),
            "chance": round(chance, 4),
            "hits_per_shelf": round(float(np.mean([len(set(S[x]) & cold_t[x]) for x in shelf_people])), 4)}
        log(f"  {name:24} unrated shelf: {hits.mean():.4f} of {len(shelf_people):,} people "
            f"(chance {chance:.4f}, pool {pool_n:,})")
    for name, hits in shelf_hits.items():
        if name != "adaptive" and "adaptive" in shelf_hits:
            d = hits - shelf_hits["adaptive"]
            metrics[name]["unrated_shelf"]["vs_adaptive"] = {"hit_rate": round(float(d.mean()), 4),
                                                             "hit_rate_ci": boot_ci(d, brng)}

    # Every policy against the adaptive engine, user by user.
    base_rec, base_nd = per_user["adaptive"]
    for name in names:
        if name == "adaptive":
            continue
        d_rec, d_nd = per_user[name][0] - base_rec, per_user[name][1] - base_nd
        ci = boot_ci(d_rec, brng)
        metrics[name]["vs_adaptive"] = {
            "recall": round(float(d_rec.mean()), 4), "recall_ci": ci,
            "ndcg": round(float(d_nd.mean()), 4), "ndcg_ci": boot_ci(d_nd, brng),
            "significant": bool(ci[0] > 0 or ci[1] < 0),
        }
        log(f"  {name:24} vs adaptive: recall {d_rec.mean():+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    if args.per_user:
        rows = {"label": args.label, "users": [int(users[j]) for j in keep],
                "recall_20": {k: [round(float(x), 5) for x in v[0]] for k, v in per_user.items()},
                "ndcg_20": {k: [round(float(x), 5) for x in v[1]] for k, v in per_user.items()}}
        args.per_user.write_bytes(gzip.compress(json.dumps(rows).encode(), mtime=0))

    ope_out = None
    if not args.no_ope:
        # -- counterfactual evaluation -------------------------------------------------
        log("counterfactual evaluation ...")
        LOGS_PER_USER = 10
        lrng = np.random.default_rng(args.seed + 2)
        targets = ["popularity", "item-cf", "two-tower", "adaptive"]
        logged = []                       # (row, action index, reward, p0)
        for j, row in enumerate(ope_rows):
            for a in lrng.choice(len(row["A"]), size=LOGS_PER_USER, p=row["p0"]):
                logged.append((j, int(a), float(row["r"][a]), float(row["p0"][a])))

        # Direct-method reward model: logged reward by (popularity quintile, CF-score band).
        cells: dict[tuple, list[float]] = {}
        for j, a, rw, _ in logged:
            cells.setdefault((int(ope_rows[j]["pop_q"][a]), int(ope_rows[j]["cf_q"][a])), []).append(rw)
        prior = np.mean([rw for _, _, rw, _ in logged])
        rhat_cell = {c: (sum(v) + 5 * prior) / (len(v) + 5) for c, v in cells.items()}

        def pi(row, name):
            """A policy shows one film from its own top 10, weighted by rank like DCG,
            with 5% spread uniformly so the logging policy is never the only support."""
            A, L = row["A"], row["lists"][name]
            p = np.full(len(A), 0.05 / len(A))
            if len(L):
                w = 1 / np.log2(np.arange(len(L)) + 2)
                p[L] += 0.95 * w / w.sum()
            else:
                p += 0.95 / len(A)
            return p

        def rhat(row):
            return np.array([rhat_cell.get((int(a), int(b)), prior) for a, b in zip(row["pop_q"], row["cf_q"])])

        per_user = {name: {"truth": [], "ips": [], "dm": [], "dr": [], "w": [], "wr": []} for name in targets + ["logging"]}
        rh = [rhat(row) for row in ope_rows]
        for name in targets + ["logging"]:
            for j, row in enumerate(ope_rows):
                p = row["p0"] if name == "logging" else pi(row, name)
                per_user[name]["truth"].append(float(p @ row["r"]))
                per_user[name]["dm"].append(float(p @ rh[j]))
            ips, dr, ws, wr = [], [], [], []
            for j, a, rw, p0 in logged:
                row = ope_rows[j]
                p = row["p0"] if name == "logging" else pi(row, name)
                w = float(p[a] / p0)
                ips.append(w * rw)
                dr.append(float(p @ rh[j]) + w * (rw - rh[j][a]))
                ws.append(w)
                wr.append(w * rw)
            per_user[name].update(ips=ips, dr=dr, w=ws, wr=wr)

        # Overlap diagnostics: how likely the logging policy was to show each target's top pick.
        overlap = {}
        for name in targets:
            p0_top, hit_top = [], []
            for row in ope_rows:
                f = row["first"][name]
                if f is None:
                    continue
                k_ = int(np.searchsorted(row["A"], f))
                p0_top.append(float(row["p0"][k_]))
                hit_top.append(float(row["r"][k_]))
            overlap[name] = {"logging_prob_of_top_pick": round(float(np.mean(p0_top)), 4),
                             "top_pick_hit_rate": round(float(np.mean(hit_top)), 4)}
            log(f"  overlap {name:12} p0(top) {overlap[name]['logging_prob_of_top_pick']:.4f}  "
                f"hit(top) {overlap[name]['top_pick_hit_rate']:.4f}")

        ope = {}
        for name in targets + ["logging"]:
            d = per_user[name]
            ips, dr, w, wr = (np.array(d[k]) for k in ("ips", "dr", "w", "wr"))
            tru, dm = np.array(d["truth"]), np.array(d["dm"])
            bi = brng.integers(0, len(ips), (500, len(ips)))
            snips_b = wr[bi].sum(1) / np.maximum(w[bi].sum(1), 1e-12)
            ope[name] = {
                "truth": round(float(tru.mean()), 4),
                "ips": round(float(ips.mean()), 4), "ips_ci": boot_ci(ips, brng, 500),
                "snips": round(float(wr.sum() / max(w.sum(), 1e-12)), 4),
                "snips_ci": [round(float(np.percentile(snips_b, 2.5)), 4), round(float(np.percentile(snips_b, 97.5)), 4)],
                "dm": round(float(dm.mean()), 4),
                "dr": round(float(dr.mean()), 4), "dr_ci": boot_ci(dr, brng, 500),
                "ess": round(float(w.sum() ** 2 / max((w ** 2).sum(), 1e-12)), 1),
            }
            log(f"  {name:12} truth {ope[name]['truth']:.4f}  ips {ope[name]['ips']:.4f}  snips {ope[name]['snips']:.4f}  "
                f"dm {ope[name]['dm']:.4f}  dr {ope[name]['dr']:.4f}  ess {ope[name]['ess']}")

        def kendall(key):
            """Kendall's tau between an estimator's ranking of the policies and the true one."""
            conc = disc = 0
            for i_, a_ in enumerate(targets):
                for b_ in targets[i_ + 1:]:
                    s_ = (ope[a_]["truth"] - ope[b_]["truth"]) * (ope[a_][key] - ope[b_][key])
                    conc += s_ > 0
                    disc += s_ < 0
            return round((conc - disc) / max(conc + disc, 1), 3)
        agreement = {est: kendall(est) for est in ("ips", "snips", "dm", "dr")}

        ope_out = {
            "estimates": ope, "kendall_tau_vs_truth": agreement, "logs": len(logged), "overlap": overlap,
            "logging_policy": "Softmax over z-scored log-popularity (temperature 1/1.5) within each user's candidate set",
            "target_policies": "Each policy shows one film from its own top 10, weighted by rank (1/log2(rank+1)), "
                               "with 5% spread uniformly over the candidate set",
            "logs_per_user": LOGS_PER_USER,
        }

    manifest = json.loads((args.art / "manifest.json").read_text()) if (args.art / "manifest.json").exists() else {}
    arts = manifest.get("artifacts", {})
    sums = args.ml32m / "checksums.txt"
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, timeout=10).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                                    text=True, timeout=10).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    out = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "runtime_s": round(time.time() - t_start),
        "dataset": f"MovieLens 32M, {int(allowed.sum()):,} of the catalogue's {n:,} films as candidates",
        "experiment": {
            "label": args.label,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "code": {"commit": commit, "uncommitted_changes": dirty},
            "artifacts": str(args.art.resolve().relative_to(ROOT)) if args.art.resolve().is_relative_to(ROOT) else str(args.art),
            "dataset": {"name": "MovieLens 32M", "checksums": sums.read_text().split() if sums.exists() else None},
            "split": {"type": "temporal", "cutoff": args.cutoff, "until": args.until, "positive": "rating >= 4",
                      "eligible": ">= 10 positives before the cutoff and >= 3 new positives after it",
                      "candidate_universe": str(args.universe) if args.universe else "all films in the catalogue",
                      "availability": "only films released or rated before the cutoff are candidates or targets"},
            "seed": args.seed,
            "train_in_universe": bool(args.train_in_universe),
            "adaptive_options": {k: v for k, v in vars(Options(**blend)).items() if not isinstance(v, bool)},
            "catalog_films": n, "rating_backed_films": int(sum(1 for x in e.c["rc"] if x)),
            "candidate_films": int(allowed.sum()),
            "films_with_training_ratings": int((rc > 0).sum()),
            "evaluated_users": len(keep), "users_without_targets": len(users) - len(keep),
            "test_interactions": int(sum(len(truth[j]) for j in keep)),
            "models": {"two_tower": meta, "semantic": arts.get("semantic_items.npy"),
                       "neighbors": {k: v for k, v in (arts.get("neighbors.npz") or {}).items() if k != "tuning"} or None},
        },
        "split": {"cutoff": args.cutoff, "train_ratings": int(len(train)), "test_ratings": int(len(test)),
                  "eligible_users": n_eligible, "evaluated_users": len(keep), "items": n,
                  "candidate_items": int(allowed.sum())},
        "k": K,
        "two_tower": {"trained": tower.trained, **meta},
        "policies": {
            "popularity": "Most-liked films in training",
            "item-cf": "Item-item collaborative filtering (damped co-occurrence, cosine), rebuilt from training",
            "two-tower": "The user tower over the liked films, nearest films by the item tower",
            "adaptive": "The production adaptive engine: drift, negatives, quality, MMR diversity, 30% adventurousness",
        },
        "metrics": metrics,
        "ope": ope_out,
        "assumptions": [
            "A held-out positive (≥4★ after the cutoff) is a success; a film not rated afterwards counts as a miss, "
            "though the person may never have seen it. MovieLens ratings are missing not at random.",
            "The logging policy is simulated and its propensities are known exactly. A live system must log them — "
            "CINEMIND records the probability of every exploration slot it shows.",
            "Each logged event is one film shown to one user; slates and position effects are not modelled.",
            "Taste is assumed stable across the cutoff, apart from what the drift model captures.",
            "These are relative comparisons between methods on the same data, not predictions of live click-through.",
            "Offline recall only rewards films a person went on to rate, so it can measure what exploration costs but "
            "not what it learns. That is why exploration slots log their propensities: the counterfactual estimators "
            "need them once real feedback exists.",
            "The single temporal split tests drift only loosely: the test period runs to 2023, and recency is measured "
            "from each person's last rating before 2016, not from the moment they would see the list.",
        ],
        "notes": [
            "The adaptive engine runs here exactly as deployed, except that its co-rating graph is item-item CF rebuilt "
            "from training data: the production graph was built from all 32M ratings and would leak the test period.",
            "Film descriptions use viewer tags collected up to 2023, a small content-side leak into the test period.",
        ],
    }
    args.out.write_text(json.dumps(out, indent=1))
    log(f"wrote {args.out} in {out['runtime_s']}s")


if __name__ == "__main__":
    main()
