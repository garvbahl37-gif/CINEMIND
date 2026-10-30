"""
Diagnostic: which of the two-tower model's roles in the adaptive engine moves
its recall, measured by handing each role to a second tower.

    python pipelines/diagnose_tower_roles.py --alt-tower DIR -- <pipelines/evaluate.py arguments>

The adaptive engine uses its tower for five things: retrieving candidates,
scoring their relevance, and, through how alike films are in its space, the
diversity pass, the penalty for resembling a film you passed on, and
exploration. This runs pipelines/evaluate.py unchanged except that the engine
carries a second tower (--alt-tower, e.g. the model from before the
expansion), and each entry of --blend-grid can name roles to take from it:

    {"roles": "oo"}      candidates, then relevance: o = the second tower, n = the artifacts' own
    {"geo": "mne"}       the diversity pass (m), the skip penalty (n), exploration (e) on the second tower
    {"roles": "all-old"} the whole engine on the second tower

Each entry is scored person by person against the unchanged engine, as any
blend is. The patched recommend() is generated from the production source by
exact text replacement, and refuses to run if the source has moved on.

--spread FILE also records how the similarities the diversity pass works from
are spread: over the first 300 lists of 250 films the unchanged engine hands
it, the pairwise cosine within a list, and — what decides how much the pass
reorders — the spread, among the 20 best, of each film's highest similarity to
a better one; and how closely each tracks the second tower's — for the
artifacts' tower (item and history tables), the second tower, the
sentence embedding and co-rating.

A second tower with fewer rows than the catalogue (one trained on the old
17,719 films) is padded with zero rows; run it with --universe and
--train-in-universe so only its films are in play. For the 2015 diagnosis in
docs/BENCHMARK.md, the second tower was the pre-expansion trainer
(`git show fe1b9b7:pipelines/train_two_tower.py`) run on likes before
2015-01-01 over data/raw/baseline-artifacts — its CUTOFF, the cutoff it
records, and its ART changed accordingly, nothing else.
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "api"))
sys.path.insert(0, str(ROOT / "pipelines"))

import numpy as np  # noqa: E402

import _adaptive  # noqa: E402
import evaluate  # noqa: E402
from _tower import Tower  # noqa: E402


@dataclass
class RoleOptions(_adaptive.Options):
    roles: str = "nn"
    geo: str = ""


SWAPS = [
    ("    s_tt = dot(self.TT, P.u) if P.u is not None else np.zeros(self.n, np.float32)\n",
     "    s_tt = dot(self.TT, P.u) if P.u is not None else np.zeros(self.n, np.float32)\n"
     "    s_cand = s_tt\n"
     "    if opt.roles != 'nn' and P.u is not None:\n"
     "        s_alt = dot(self.alt.TT, self.alt.profile(signals, searches or [], now, opt).u)\n"
     "        s_cand = s_alt if opt.roles[0] == 'o' else s_tt\n"
     "        s_tt = s_alt if opt.roles[1] == 'o' else s_tt\n"),
    ("        top = np.where(out_of_bounds, -np.inf, s_tt) if out_of_bounds is not None else s_tt\n"
     "        cand |= set(np.argpartition(-top, 500)[:500].tolist())\n",
     "        top = np.where(out_of_bounds, -np.inf, s_cand) if out_of_bounds is not None else s_cand\n"
     "        cand |= set(np.argpartition(-top, 500)[:500].tolist())\n"),
    ("    if P.negative:\n"
     "        N = self.TT[list(P.negative)]\n"
     "        skip_sim = np.clip(mm(self.TT[C], N.T).max(axis=1), 0, 1)\n",
     "    if P.negative:\n"
     "        G = self.alt.TT if 'n' in opt.geo else self.TT\n"
     "        N = G[list(P.negative)]\n"
     "        skip_sim = np.clip(mm(G[C], N.T).max(axis=1), 0, 1)\n"),
    ("    chosen = self._mmr(top, topb, n_exploit, delta * opt.mmr_scale, opt.diversity_space)\n",
     "    chosen = (self.alt if 'm' in opt.geo else self)._mmr(top, topb, n_exploit, delta * opt.mmr_scale,\n"
     "                                                       opt.diversity_space)\n"),
    ("        explored, props = self._explore(P, C, q_pool, chosen, exclude, n_explore, now, explore, seed,\n",
     "        _alt = 'e' in opt.geo\n"
     "        explored, props = (self.alt if _alt else self)._explore(\n"
     "            self.alt.profile(signals, searches or [], now, opt) if _alt else P,\n"
     "            C, q_pool, chosen, exclude, n_explore, now, explore, seed,\n"),
]


def patched_recommend():
    src = textwrap.dedent(inspect.getsource(_adaptive.Adaptive.recommend))
    for a, b in SWAPS:
        if src.count(a) != 1:
            sys.exit(f"api/_adaptive.py recommend() no longer contains, exactly once:\n{a}")
        src = src.replace(a, b)
    ns = dict(vars(_adaptive))
    exec(compile(src, "<diagnose_tower_roles>", "exec"), ns)
    return ns["recommend"]


def unit(M: np.ndarray) -> np.ndarray:
    M = M.astype(np.float32)
    return M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)


def corating(x, items: np.ndarray) -> np.ndarray:
    """Co-rating similarity within a list: a film's score in another's neighbour row, over that row's best."""
    at = {int(i): a for a, i in enumerate(items)}
    S = np.zeros((len(items), len(items)), np.float32)
    for a, i in enumerate(items):
        top = float(x.nsc[i].max()) or 1.0
        for j, sc in zip(x.nbr[i], x.nsc[i]):
            b = at.get(int(j))
            if b is not None:
                S[a, b] = max(S[a, b], sc / top)
    return np.maximum(S, S.T)


def spread(x, lists: list[np.ndarray]) -> dict:
    """For each similarity, over the captured lists: pairwise cosine, the spread that decides how much the
    diversity pass reorders, and how closely it tracks the second tower's (whose pass gained recall)."""
    tables = {"artifacts' tower": x.TT, "artifacts' tower, history table": unit(x.tower.H),
              "second tower": x.alt.TT, "sentence embedding": x.X}
    out = {name: {"within": [], "top": [], "corr": []} for name in [*tables, "co-rating"]}
    off = None
    for items in lists:
        off = ~np.eye(len(items), dtype=bool) if off is None or off.shape[0] != len(items) else off
        ref = unit(x.alt.TT[items]) @ unit(x.alt.TT[items]).T
        sims = {name: (lambda V: V @ V.T)(unit(M[items])) for name, M in tables.items()}
        sims["co-rating"] = corating(x, items)
        for name, S in sims.items():
            out[name]["within"].append((S[off].mean(), S[off].std()))
            out[name]["top"].append(np.std([S[j, :j].max() for j in range(1, min(20, len(items)))]))
            out[name]["corr"].append(np.corrcoef(ref[off], S[off])[0, 1])
    return {name: {"pairwise_mean": round(float(np.mean([w[0] for w in v["within"]])), 4),
                   "pairwise_sd": round(float(np.mean([w[1] for w in v["within"]])), 4),
                   "top20_max_similarity_to_a_better_film_sd": round(float(np.mean(v["top"])), 4),
                   "correlation_with_second_tower": round(float(np.nanmean(v["corr"])), 3)}
            for name, v in out.items()}


def padded_tower(path: Path, n: int) -> Tower:
    t = Tower(path)
    if t.V.shape[0] < n:
        pad = np.zeros((n - t.V.shape[0], t.V.shape[1]), t.V.dtype)
        t.V, t.H = np.vstack([t.V, pad]), np.vstack([t.H, pad.astype(t.H.dtype)])
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--alt-tower", type=Path, required=True, help="directory with tower_items.npy and tower_user.npz")
    ap.add_argument("--spread", type=Path, default=None, help="also write the similarity spread here (JSON)")
    ap.add_argument("rest", nargs=argparse.REMAINDER, help="-- then pipelines/evaluate.py's arguments")
    args = ap.parse_args()
    rest = args.rest[1:] if args.rest[:1] == ["--"] else args.rest
    recommend = patched_recommend()

    lists: list[np.ndarray] = []
    engines, base = [], []

    class Diagnosed(_adaptive.Adaptive):
        def recommend(self, *a, **kw):
            if kw["opt"].roles == "all-old":
                return self.alt.recommend(*a, **kw)
            # Only the unchanged engine's lists are measured: evaluate.py ranks it first for everyone.
            base.append(kw["opt"]) if not base else None
            self.capture = args.spread is not None and kw["opt"] == base[0] and len(lists) < 300
            return recommend(self, *a, **kw)

        def _mmr(self, items, scores, k, delta, space="tower"):
            if getattr(self, "capture", False):
                lists.append(np.asarray(items).copy())
            return super()._mmr(items, scores, k, delta, space)

    def make(e, *a, **kw):
        x = Diagnosed(e, *a, **kw)
        x.alt = _adaptive.Adaptive(e, *a, **{**kw, "tower": padded_tower(args.alt_tower, e.n)})
        engines.append(x)
        return x

    evaluate.Options, evaluate.Adaptive = RoleOptions, make
    sys.argv = ["evaluate.py", *rest]
    evaluate.main()
    if args.spread and lists:
        args.spread.write_text(json.dumps({"lists": len(lists), "films_per_list": int(lists[0].size),
                                           "spaces": spread(engines[0], lists)}, indent=1))
        print(f"[diagnose] wrote {args.spread}")


if __name__ == "__main__":
    main()
