"""
Offline evaluation: baseline vs two-tower retrieval vs adaptive reranking, on
the same held-out data — plus a counterfactual-evaluation study.

    python pipelines/evaluate.py --ml32m data/raw/ml-32m [--users 3000]

THE SPLIT
MovieLens 32M is split in time at 2016-01-01. Everything before is training,
everything after is the test period. That keeps every model honest:
  - the two-tower model was trained on MovieLens 20M, which ends in March
    2015, so it has never seen a test rating;
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

METRICS (K = 20)
Recall@K (hits / min(|test|, K)), NDCG@K, HitRate@10, catalogue coverage,
intra-list diversity (mean pairwise 1 − cosine of the films' content vectors),
novelty (mean −log2 of training popularity) and long-tail share (outside
the 1,000 most-liked). Recall and NDCG carry 95% bootstrap intervals.

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
import json
import math
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
from _engine import get_engine  # noqa: E402
from _semantic import item_matrix, mm  # noqa: E402
from _tower import get_tower  # noqa: E402

CUTOFF = int(datetime(2016, 1, 1, tzinfo=timezone.utc).timestamp())
K = 20
TOPN = 50          # neighbours kept per film in the rebuilt CF graph


def log(msg: str):
    print(f"[eval {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# -- data ----------------------------------------------------------------------

def load(ml32m: Path, movie_ids: list[int]):
    pos = {int(m): i for i, m in enumerate(movie_ids)}
    r = pd.read_csv(ml32m / "ratings.csv", dtype={"userId": np.int32, "movieId": np.int32,
                                                  "rating": np.float32, "timestamp": np.int64})
    r["item"] = r.movieId.map(pos)
    r = r.dropna(subset=["item"])
    r["item"] = r["item"].astype(np.int32)
    return r[["userId", "item", "rating", "timestamp"]]


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
    for s0 in range(0, n_items, 1024):
        e = min(s0 + 1024, n_items)
        blk = (Xc[:, s0:e].T @ X).toarray()
        blk[np.arange(e - s0), np.arange(s0, e)] = 0
        idx = np.argpartition(-blk, TOPN, axis=1)[:, :TOPN]
        vals = np.take_along_axis(blk, idx, 1)
        o = np.argsort(-vals, axis=1)
        nbr[s0:e] = np.take_along_axis(idx, o, 1)
        sco[s0:e] = np.take_along_axis(vals, o, 1)
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
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    t_start = time.time()

    e = get_engine()
    n = e.n
    tower = get_tower()
    content = item_matrix()
    meta = {}
    if tower.trained:
        meta = json.loads(str(np.load(ART / "tower_user.npz")["meta"]))
    log(f"two-tower: {'trained model ' + json.dumps(meta) if tower.trained else 'original item vectors (fallback)'}")

    log("loading ratings ...")
    r = load(args.ml32m, e.c["movieId"])
    train, test = r[r.timestamp < CUTOFF], r[r.timestamp >= CUTOFF]
    log(f"{len(r):,} catalogue ratings: {len(train):,} train, {len(test):,} test")

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
    log(f"{n_eligible:,} eligible users; evaluating {min(args.users, n_eligible):,}")
    users = sorted(rng.choice(users, size=min(args.users, len(users)), replace=False).tolist())
    tr_by_user = {u: g for u, g in train[train.userId.isin(users)].groupby("userId")}

    adaptive = Adaptive(e, nbr=nbr, nsc=sco, quality=quality, rating_count=rc)
    variants = {
        "adaptive": dict(explore=0.3, opt=Options()),
        "adaptive − drift": dict(explore=0.3, opt=Options(drift=False)),
        "adaptive − diversity": dict(explore=0.3, opt=Options(diversity=False)),
        "adaptive − exploration": dict(explore=0.0, opt=Options(exploration=False)),
        "adaptive − negatives": dict(explore=0.3, opt=Options(negatives=False)),
    }
    names = ["popularity", "item-cf", "two-tower", *variants]
    recs: dict[str, list[list[int]]] = {k: [] for k in names}
    truth: list[set[int]] = []
    ope_rows = []

    log("ranking ...")
    for n_u, u in enumerate(users):
        g = tr_by_user[u]
        seen = g.item.values
        seen_set = set(seen.tolist())
        rel = te_pos[u] - seen_set
        truth.append(rel)
        liked = g.item.values[g.rating.values >= 4]
        s_pop, s_cf, s_tt = pos_cnt, scores_cf(nbr, sco, liked, n), scores_tt(tower, liked)
        recs["popularity"].append(topk(s_pop, seen, K).tolist())
        recs["item-cf"].append(topk(s_cf, seen, K).tolist())
        recs["two-tower"].append(topk(s_tt, seen, K).tolist())
        sig = [Signal(int(i), "rating", float(v), float(t)) for i, v, t in
               zip(g.item.values, g.rating.values, g.timestamp.values)]
        for name, kw in variants.items():
            out = adaptive.recommend(sig, [], k=K, now=float(CUTOFF), seed=args.seed + n_u, explain=False, **kw)
            recs[name].append(out["ids"])

        # Counterfactual study: a candidate set, a logging policy, and the truth.
        A = set(topk(s_pop, seen, 100).tolist()) | set(topk(s_cf, seen, 100).tolist()) \
            | set(topk(s_tt, seen, 100).tolist()) | set(recs["adaptive"][-1])
        unseen = np.setdiff1d(np.arange(n), seen)
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
    metrics = {}
    per_user: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in names:
        R = recs[name]
        rec_ = np.array([len(set(x[:K]) & t) / min(len(t), K) for x, t in zip(R, truth)])
        nd = np.array([ndcg(x, t, K) for x, t in zip(R, truth)])
        hit = np.array([1.0 if set(x[:10]) & t else 0.0 for x, t in zip(R, truth)])
        per_user[name] = (rec_, nd)
        items = [i for x in R for i in x[:K]]
        nov = np.mean([-math.log2((pos_cnt[i] + 1) / n_train_users) for i in items])
        metrics[name] = {
            "recall": round(float(rec_.mean()), 4), "recall_ci": boot_ci(rec_, brng),
            "ndcg": round(float(nd.mean()), 4), "ndcg_ci": boot_ci(nd, brng),
            "hit_rate_10": round(float(hit.mean()), 4),
            "coverage": round(len(set(items)) / n, 4),
            "diversity": round(float(np.mean([ild(x[:K], content) for x in R])), 4),
            "novelty": round(float(nov), 3),
            "long_tail": round(float(np.mean([i not in popular_1000 for i in items])), 4),
        }
        log(f"  {name:24} recall {metrics[name]['recall']:.4f}  ndcg {metrics[name]['ndcg']:.4f}  "
            f"div {metrics[name]['diversity']:.3f}  cov {metrics[name]['coverage']:.3f}")

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

    out = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "runtime_s": round(time.time() - t_start),
        "dataset": "MovieLens 32M, restricted to the 17,719 films in the catalogue",
        "split": {"cutoff": "2016-01-01", "train_ratings": int(len(train)), "test_ratings": int(len(test)),
                  "eligible_users": n_eligible, "evaluated_users": len(users), "items": n},
        "k": K,
        "two_tower": {"trained": tower.trained, **meta},
        "policies": {
            "popularity": "Most-liked films in training",
            "item-cf": "Item-item collaborative filtering (damped co-occurrence, cosine), rebuilt from training",
            "two-tower": "The user tower over the liked films, nearest films by the item tower",
            "adaptive": "The production adaptive engine: drift, negatives, quality, MMR diversity, 30% adventurousness",
        },
        "metrics": metrics,
        "ope": {
            "estimates": ope, "kendall_tau_vs_truth": agreement, "logs": len(logged), "overlap": overlap,
            "logging_policy": "Softmax over z-scored log-popularity (temperature 1/1.5) within each user's candidate set",
            "target_policies": "Each policy shows one film from its own top 10, weighted by rank (1/log2(rank+1)), "
                               "with 5% spread uniformly over the candidate set",
            "logs_per_user": LOGS_PER_USER,
        },
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
