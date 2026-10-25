"""
The adaptive recommendation engine.

    signals ──> taste profile ──> candidates ──> rerank ──> recommendations
    (likes,      (two-tower       (two-tower,     (relevance,   (with evidence
     skips,       space; long-     co-rating      quality,      and confidence)
     watchlist,   term and         graph,         diversity,          │
     ratings,     recent, blended  semantic       exploration)        │
     views,       for drift)       intent,                            ▼
     searches)                     quality pool)               logged outcomes
                                                                (propensities)

Taste. A person is represented by the two-tower model's user tower, run over
the films they reacted to: a like counts 1, a watchlist 0.6, a view 0.25, a
rating (r − 3) / 2; skips are kept apart as negatives. Two profiles are kept: long-term,
with a one-year half-life, and recent, with a three-day half-life measured from
the latest signal. The recent one is blended in with a weight that grows with
how much recent activity there is, so new interests move the recommendations
within a few signals without erasing years of taste — preference drift, without
forgetting.

Candidates. The nearest films to that profile in the two-tower space (an exact
inner-product search: what a FAISS flat index does, in NumPy, because 17,719
items don't need an approximate one), the co-rating neighbours of everything
liked, films close to recent searches in the sentence-embedding space, and a
pool of highly rated films to explore from.

Reranking. Relevance (two-tower, co-rating, content, search intent), a quality
prior, a penalty for resembling something skipped, then maximal marginal
relevance so the list isn't twelve variations on one film.

Exploration. A share of slots — set by "adventurousness", from none to 35% — goes to
well-rated films that are still relevant but sit outside the person's usual
territory. Those are sampled, not picked, and each carries the probability it
was shown with, so logged outcomes can be used for counterfactual evaluation.

Every recommendation says why it is there — which liked film, which search, or
that it is a deliberate stretch — and how confident the engine is.

All the inputs are injectable, so pipelines/evaluate.py runs this exact code
against a co-rating graph and quality prior rebuilt from training data only.
"""
from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from _engine import ART, Engine
from _semantic import dot, get_model, item_matrix, mm
from _tower import Tower, get_tower

WEIGHT = {"like": 1.0, "watchlist": 0.6, "view": 0.25, "skip": -1.0}
LONG_HALF_LIFE = 365 * 86400.0
SHORT_HALF_LIFE = 3 * 86400.0
DAY = 86400.0


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _z(x: np.ndarray) -> np.ndarray:
    if x.size < 2:
        return np.zeros_like(x)
    sd = float(x.std())
    return (x - float(x.mean())) / sd if sd > 1e-9 else np.zeros_like(x)


# MovieLens files IMAX as a genre; it is how a film was shown, not a taste.
FORMATS = frozenset({"imax"})


@dataclass
class Signal:
    item: int
    kind: str                       # like | skip | watchlist | view | rating
    value: float | None = None      # the star rating, for kind == "rating"
    ts: float = 0.0

    @property
    def weight(self) -> float:
        if self.kind == "rating":
            return float(np.clip(((self.value or 3.0) - 3.0) / 2.0, -1.0, 1.0))
        return WEIGHT.get(self.kind, 0.0)


@dataclass
class Options:
    """Switches for the offline ablations; production uses the defaults."""
    drift: bool = True
    diversity: bool = True
    exploration: bool = True
    negatives: bool = True


@dataclass
class Profile:
    u: np.ndarray | None
    u_long: np.ndarray | None
    u_recent: np.ndarray | None
    drift: float
    taste_sem: np.ndarray | None
    intent: np.ndarray | None
    positive: dict[int, float] = field(default_factory=dict)   # item -> effective weight
    recent: dict[int, float] = field(default_factory=dict)     # item -> recency weight, for "lately"
    negative: dict[int, float] = field(default_factory=dict)
    interacted: set[int] = field(default_factory=set)
    viewed: set[int] = field(default_factory=set)
    searches: list[tuple[str, np.ndarray, float]] = field(default_factory=list)


class Adaptive:
    def __init__(self, e: Engine, root: Path = ART, *, nbr=None, nsc=None, quality=None,
                 rating_count=None, tower: Tower | None = None, semantic=None, encoder=None):
        self.e = e
        self.tower = get_tower(root) if tower is None else tower
        self.TT = self.tower.V                  # item side of the two-tower model
        self.X = item_matrix(root) if semantic is None else semantic
        self.sem = get_model(root) if encoder is None else encoder
        self.nbr = e.nbr if nbr is None else nbr
        self.nsc = e.nsc if nsc is None else nsc
        if quality is None:
            order = np.argsort(e.bayes, kind="stable")
            quality = np.empty(e.n, np.float32)
            quality[order] = np.arange(e.n) / max(e.n - 1, 1)
        self.quality = np.asarray(quality, np.float32)
        self.rc = np.asarray([x or 0 for x in e.c["rc"]] if rating_count is None else rating_count, np.float32)
        self.n = e.n
        self.genre_of = e.genre_of

    # -- the profile ---------------------------------------------------------
    def profile(self, signals: list[Signal], searches: list[tuple[str, float]], now: float,
                opt: Options = Options()) -> Profile:
        sig = [s for s in signals if 0 <= s.item < self.n]
        last = max((s.ts for s in sig), default=now)
        pos_items, long_w, recent_w = [], [], []
        sem_v = np.zeros(self.X.shape[1], np.float32)
        pos: dict[int, float] = {}
        rec: dict[int, float] = {}
        neg: dict[int, float] = {}
        # The latest signal about a film wins: liking it after skipping it means a like.
        latest: dict[int, Signal] = {}
        for s in sorted(sig, key=lambda s: s.ts):
            if s.kind == "view" and s.item in latest and latest[s.item].kind != "view":
                continue
            latest[s.item] = s
        recent_mass = 0.0
        for s in latest.values():
            w = s.weight
            if w == 0:
                continue
            lw = w * 0.5 ** (max(now - s.ts, 0.0) / LONG_HALF_LIFE)
            rw = w * 0.5 ** (max(last - s.ts, 0.0) / SHORT_HALF_LIFE)
            if w > 0:
                pos_items.append(s.item)
                long_w.append(lw)
                recent_w.append(rw)
                sem_v += lw * self.X[s.item]
                pos[s.item] = max(pos.get(s.item, 0.0), lw if not opt.drift else 0.6 * lw + 0.4 * rw)
                rec[s.item] = rw
                recent_mass += rw
            elif opt.negatives:
                neg[s.item] = -w
        # The user tower reads the same history twice: weighted for the long run, and for lately.
        u_long = self.tower.user(pos_items, long_w)
        u_recent = self.tower.user(pos_items, recent_w)
        if u_long is None:
            u, drift = None, 0.0
        elif opt.drift and u_recent is not None:
            # More recent activity, more say — up to 40%, reached at about five fresh likes.
            drift = 0.4 * min(1.0, recent_mass / 5.0)
            u = _unit((1 - drift) * u_long + drift * u_recent)
        else:
            u, drift = u_long, 0.0
        intents = []
        intent = np.zeros(self.X.shape[1], np.float32)
        for q, ts in searches[-12:]:
            v = self.sem.encode(q)
            if not v.any():
                continue
            w = 0.5 ** (max(now - ts, 0.0) / (2 * DAY))
            intent += w * v
            intents.append((q, v, w))
        return Profile(
            u=u, u_long=u_long, u_recent=u_recent, drift=drift,
            taste_sem=_unit(sem_v) if sem_v.any() else None,
            intent=_unit(intent) if intent.any() else None,
            positive=pos, recent=rec, negative=neg,
            interacted={s.item for s in latest.values() if s.kind != "view"},
            viewed={s.item for s in latest.values() if s.kind == "view"},
            searches=intents,
        )

    # -- recommending ----------------------------------------------------------
    def recommend(self, signals: list[Signal], searches: list[tuple[str, float]] | None = None, *,
                  k: int = 24, explore: float = 0.3, exclude: set[int] | None = None,
                  now: float | None = None, opt: Options = Options(), seed: int | None = None,
                  explain: bool = True) -> dict:
        t0 = time.perf_counter()
        now = time.time() if now is None else now
        explore = float(np.clip(explore, 0.0, 1.0))
        P = self.profile(signals, searches or [], now, opt)
        exclude = set(exclude or ()) | P.interacted
        cold = P.u is None

        # Candidates.
        cand: set[int] = set()
        s_tt = dot(self.TT, P.u) if P.u is not None else np.zeros(self.n, np.float32)
        if P.u is not None:
            cand |= set(np.argpartition(-s_tt, 500)[:500].tolist())
        cf = np.zeros(self.n, np.float32)
        for p, w in P.positive.items():
            row, sc = self.nbr[p], self.nsc[p]
            cf[row] += w * sc / (float(sc.max()) or 1.0)
        cand |= set(np.flatnonzero(cf > 0).tolist())
        s_int = dot(self.X, P.intent) if P.intent is not None else np.zeros(self.n, np.float32)
        if P.intent is not None:
            cand |= set(np.argpartition(-s_int, 150)[:150].tolist())
        s_sem = dot(self.X, P.taste_sem) if P.taste_sem is not None else np.zeros(self.n, np.float32)
        q_pool = np.argsort(-self.quality)[:400]
        if cold and P.intent is None:
            cand |= set(q_pool[:300].tolist())
        cand -= exclude
        C = np.array(sorted(cand), np.int64)
        if C.size == 0:
            if not explain:
                return {"ids": [], "propensities": {}}
            return {"results": [], "profile": self._summary(P, explore, opt), "policy": {}}

        # Relevance over the candidates, each signal z-scored before blending.
        if cold:
            rel = 0.9 * _z(s_int[C]) if P.intent is not None else np.zeros(C.size, np.float32)
        else:
            rel = 0.45 * _z(s_tt[C]) + 0.30 * _z(cf[C]) + 0.15 * _z(s_sem[C])
            if P.intent is not None:
                rel = rel + 0.25 * _z(s_int[C])
        base = rel + (0.20 if not cold else 0.6) * _z(self.quality[C])
        base -= np.where(self.rc[C] < 30, 0.5, np.where(self.rc[C] < 100, 0.2, 0.0)).astype(np.float32)
        skip_sim = np.zeros(C.size, np.float32)
        if P.negative:
            N = self.TT[list(P.negative)]
            skip_sim = np.clip(mm(self.TT[C], N.T).max(axis=1), 0, 1)
            base -= 2.5 * np.clip(skip_sim - 0.6, 0, None)
        if P.viewed:
            base -= 0.3 * np.isin(C, list(P.viewed))

        # Exploitation, diversified with maximal marginal relevance.
        n_explore = int(round(k * 0.35 * explore)) if opt.exploration else 0
        n_exploit = max(k - n_explore, 0)
        order = np.argsort(-base)
        top = C[order[:250]]
        topb = base[order[:250]]
        delta = (0.25 + 0.50 * explore) if opt.diversity else 0.0
        chosen = self._mmr(top, topb, n_exploit, delta)

        # Exploration: relevant, well-rated films outside the usual territory, sampled.
        explored, props = [], {}
        if n_explore:
            explored, props = self._explore(P, C, rel, base, chosen, n_explore, now, explore, seed)
        ranked = self._interleave(chosen, explored)

        if not explain:                         # the offline evaluation needs the ranking only
            return {"ids": [int(i) for i in ranked], "propensities": {int(i): p for i, p in props.items()}}
        rel_pct = {int(i): float(p) for i, p in zip(C, (np.argsort(np.argsort(rel)) + 1) / C.size)}
        results = []
        for pos_, i in enumerate(ranked):
            d = self.e.item(int(i), None)
            ex = i in props
            ev, conf = self._evidence(int(i), P, s_tt, cf, s_int, rel_pct.get(int(i), 0.0), ex)
            d.update({"exploring": ex, "propensity": round(props[i], 4) if ex else 1.0,
                      "reason": ev[0]["text"] if ev else "", "evidence": ev,
                      "confidence": round(conf, 2),
                      "confidence_label": "High" if conf >= 0.66 else "Medium" if conf >= 0.4 else "Low",
                      "position": pos_})
            results.append(d)
        return {
            "results": results,
            "profile": self._summary(P, explore, opt),
            "policy": {"version": "adaptive-1", "epsilon": round(n_explore / max(k, 1), 3),
                       "diversity": round(delta, 3), "drift": round(P.drift, 3), "candidates": int(C.size),
                       "cold_start": cold},
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    def _mmr(self, items: np.ndarray, scores: np.ndarray, k: int, delta: float) -> list[int]:
        if k <= 0 or items.size == 0:
            return []
        V = self.TT[items]
        picked: list[int] = []
        best_sim = np.zeros(items.size, np.float32)
        avail = np.ones(items.size, bool)
        for _ in range(min(k, items.size)):
            val = scores - delta * 3.0 * best_sim
            val = np.where(avail, val, -np.inf)
            j = int(np.argmax(val))
            picked.append(int(items[j]))
            avail[j] = False
            best_sim = np.maximum(best_sim, np.clip(mm(V, V[j]), 0, 1))
        return picked

    def _explore(self, P: Profile, C, rel, base, chosen, n, now, explore, seed):
        pos_items = list(P.positive)
        usual = self._top_genres(P.positive, 3)
        pool = set(C.tolist()) | set(np.argsort(-self.quality)[:400].tolist())
        pool -= set(chosen) | P.interacted
        pool = np.array(sorted(pool), np.int64)
        if pool.size == 0:
            return [], {}
        if P.u is not None:
            r = dot(self.TT[pool], P.u)
            floor = float(np.percentile(dot(self.TT[C], P.u), 40))
        else:
            r = np.zeros(pool.size, np.float32)
            floor = -1.0
        near = np.clip(mm(self.TT[pool], self.TT[pos_items].T), 0, 1).max(axis=1) if pos_items else np.zeros(pool.size)
        outside = np.array([0.0 if (self.genre_of[i] & usual) else 1.0 for i in pool]) if usual else np.ones(pool.size)
        ok = (self.quality[pool] >= 0.6) & (self.rc[pool] >= 300) & (r >= floor) & ((near < 0.6) | (outside > 0))
        if not ok.any():
            return [], {}
        E, r, near, outside = pool[ok], r[ok], near[ok], outside[ok]
        novelty = (1 - near) + 0.5 * outside
        logits = _z(self.quality[E]) * 0.6 + _z(r) * 0.6 + _z(novelty) * 0.6
        rng = np.random.default_rng(seed if seed is not None else self._seed(P, now, explore))
        picked, props = [], {}
        avail = np.ones(E.size, bool)
        for _ in range(min(n, E.size)):
            w = np.exp(logits - logits[avail].max()) * avail
            pr = w / w.sum()
            j = int(rng.choice(E.size, p=pr))
            picked.append(int(E[j]))
            props[int(E[j])] = float(pr[j])            # the probability it was drawn when it was
            avail[j] = False
        return picked, props

    @staticmethod
    def _interleave(exploit: list[int], explore: list[int]) -> list[int]:
        """Spread exploration evenly through the list rather than bunching it."""
        if not explore:
            return exploit
        out, ex = list(exploit), list(explore)
        step = max(2, (len(out) + len(ex)) // (len(ex) + 1))
        for n, i in enumerate(ex):
            out.insert(min(len(out), step * (n + 1) - 1), i)
        return out

    def _seed(self, P: Profile, now: float, explore: float) -> int:
        # Stable within a day for the same taste, so a refresh doesn't reshuffle the list.
        key = f"{sorted(P.positive)}|{sorted(P.negative)}|{int(now // DAY)}|{explore:.2f}"
        return int(hashlib.sha256(key.encode()).hexdigest()[:12], 16)

    # -- explaining ------------------------------------------------------------
    def _evidence(self, i: int, P: Profile, s_tt, cf, s_int, rel_pct: float, exploring: bool):
        e, ev = self.e, []
        support = []
        for p, w in P.positive.items():
            sim = float(mm(self.TT[i], self.TT[p]))
            linked = bool((self.nbr[p] == i).any())
            strength = w * max(sim, 0.75 if linked else 0.0)
            if sim >= 0.55 or linked:
                support.append((strength, p))
        support.sort(reverse=True)
        for strength, p in support[:2]:
            ev.append({"kind": "liked", "item_id": p, "title": self._title(p), "strength": round(strength, 3),
                       "text": f"Because you liked {self._title(p)}"})
        if P.searches:
            q, v, _ = max(P.searches, key=lambda s: float(mm(self.X[i], s[1])))
            if float(mm(self.X[i], v)) >= 0.45 and s_int[i] > 0:
                ev.append({"kind": "search", "query": q, "strength": round(float(mm(self.X[i], v)), 3),
                           "text": f"Matches your search “{q}”"})
        if P.u_recent is not None and P.u_long is not None and P.drift > 0.05:
            gain = float(mm(self.TT[i], P.u_recent)) - float(mm(self.TT[i], P.u_long))
            if gain > 0.12:
                ev.append({"kind": "drift", "strength": round(gain, 3), "text": "Fits what you've been into lately"})
        if exploring:
            # A film explores by leaving your genres or by sitting far from your likes;
            # only claim the first when it is true.
            top = self._top_genres(P.positive, 2)
            ra, rc = e.c["ra"][i], e.c["rc"][i] or 0
            stretch = (f"outside your usual {' and '.join(sorted(e.genre_label.get(g, g) for g in top))}"
                       if top and not (self.genre_of[i] & top) else "a step off your usual path")
            ev.insert(0, {"kind": "explore", "strength": 0.0,
                          "text": f"Exploring: {stretch}, rated {ra:.1f}★ by {rc:,} viewers" if ra else f"Exploring: {stretch}"})
        if not ev:
            ra, rc = e.c["ra"][i], e.c["rc"][i] or 0
            ev.append({"kind": "quality", "strength": round(float(self.quality[i]), 3),
                       "text": f"Highly rated: {ra:.1f}★ from {rc:,} viewers" if ra else "Highly rated"})
        n_support = len(support)
        conf = 0.45 * rel_pct + 0.35 * min(1.0, n_support / 3) + 0.20 * min(1.0, math.log10(max(self.rc[i], 1)) / 4)
        if exploring:
            conf *= 0.6
        return ev, float(np.clip(conf, 0, 1))

    def _top_genres(self, weights: dict[int, float], k: int) -> set[str]:
        """The genres someone leans on most. Drama is set aside when there are
        others to choose from: nearly everything is Drama, so it says little."""
        tally: dict[str, float] = {}
        for i, w in weights.items():
            for g in self.genre_of[i] - FORMATS:
                tally[g] = tally.get(g, 0.0) + w
        ranked = [g for g, _ in sorted(tally.items(), key=lambda x: (-x[1], x[0]))]
        # Drop Drama before taking the top k, not after, or a Crime-Drama-Action
        # fan's "usual" shrinks to Crime and exploration offers them Action.
        rest = [g for g in ranked if g != "drama"]
        return set((rest if len(rest) >= k else ranked)[:k])

    def _summary(self, P: Profile, explore: float, opt: Options) -> dict:
        lab = self.e.genre_label
        long_w = dict(P.positive)
        recent = {i: w for i, w in P.recent.items() if w >= 0.25}
        return {
            "signals": len(P.positive) + len(P.negative),
            "liked": len(P.positive), "skipped": len(P.negative), "searches": len(P.searches),
            "long_term": [lab.get(g, g) for g in self._ranked_genres(long_w, 3)],
            "lately": [lab.get(g, g) for g in self._ranked_genres(recent, 3)] if P.drift > 0.05 else [],
            "drift": round(P.drift, 3), "adventurousness": round(explore, 2),
            "cold_start": P.u is None,
        }

    def _ranked_genres(self, weights: dict[int, float], k: int) -> list[str]:
        tally: dict[str, float] = {}
        for i, w in weights.items():
            for g in self.genre_of[i] - FORMATS:
                tally[g] = tally.get(g, 0.0) + w
        # Ties break by name: set order changes with each process's hash seed.
        return [g for g, _ in sorted(tally.items(), key=lambda x: (-x[1], x[0]))[:k]]

    def _title(self, i: int) -> str:
        t = re.sub(r"\s*\(.*\)\s*$", "", self.e.c["title"][i]).strip() or self.e.c["title"][i]
        m = re.match(r"^(.*), (The|A|An)$", t)
        return f"{m.group(2)} {m.group(1)}" if m else t
