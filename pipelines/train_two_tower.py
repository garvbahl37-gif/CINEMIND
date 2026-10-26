"""
Train the two-tower model the adaptive engine and discovery use.

    python pipelines/train_two_tower.py --ml32m data/raw/ml-32m [--art DIR] [--selection-log FILE]

WHY RETRAIN
The original item vectors (item_vectors.npy) came from a model trained on a
2M-rating random sample — about a hundred ratings a film. On their own they
make a poor taste space: The Notebook's nearest films were Saints and Soldiers
and Crimson Rivers 2, and films with no training data sat beside everything.
They still feed the precomputed similar-films graph, where co-rating evidence
dominates; this model replaces them only where a taste space is used directly.

THE MODEL
  user tower   the mean of the history's embeddings, plus a residual MLP,
               L2-normalised. No per-user parameters: it is exactly what the
               server can compute for a browser with three likes and no account.
               A history embedding is an ID embedding plus a learned projection
               of the film's sentence embedding, so a like for a film with no
               ratings still says something about taste.
  item tower   an ID embedding plus a learned projection of the film's
               sentence embedding, L2-normalised — so films with little rating
               data still land somewhere sensible, by what they are about.
  objective    softmax at temperature 0.07 over two kinds of negatives: the
               batch's other targets, and --uniform films drawn uniformly from
               the whole catalogue (mixed negative sampling, Yang et al. 2020).
               Each kind gets a logQ correction (scaled by --logq) for how
               likely it was to be drawn. In-batch negatives alone only ever
               push down films someone liked; with a catalogue where most films
               have no training data, those films were never pushed down at all
               and crowded the top of every list (validation recall fell from
               0.13 to 0.06 over six epochs). Uniform negatives fix that.
  id dropout   see FILMS WITHOUT TRAINING DATA.

THE DATA
Positive ratings (≥4★) before 2016-01-01 only, so pipelines/evaluate.py's test
period stays unseen. Each example is one liked film as the target and up to 50
of the same person's other likes as the history — but half the time the
history is cut to a random 1–10 of them, because that is what the site sees:
a browser with a handful of likes. Trained on long histories alone, a single
popular film (The Dark Knight, say) was out of distribution and its user
vector landed among unrelated films.

MODEL SELECTION — TEMPORAL, INSIDE THE TRAINING PERIOD
A selection model is trained on likes before 2015-01-01 and scored on 2015:
for up to 5,000 people with ≥3 earlier likes, their 2015 likes are the
targets, ranked from their earlier likes (all of them, and the last three).
Recall@K = hits / min(|targets|, K), as in pipelines/evaluate.py. Only films
that existed before 2015 (released or liked before it) are candidates and
targets. Those nobody had liked yet have no data in that model, so they are
scored exactly as a film without ratings is served — by content alone. The epoch with the best mean of the
two recalls@20 is kept; the model is then refit on every like before 2016 for
that many epochs. The test period (2016 on) is never seen.

FILMS WITHOUT TRAINING DATA
Films never seen in training keep their random initial ID embeddings, which are
noise. They are exported with those embeddings zeroed, so their item and
history vectors come from content alone. Zeroing alone is not enough: a model
that only ever saw films with their ID embeddings has no calibrated score for a
film without one, and content-only vectors crowded the top of every list (81%
of the two-tower policy's picks, recall@20 0.03). So during training each
film's ID embedding is dropped with probability p (DropoutNet, Volkovs et al.
2017) — for targets, negatives and history alike — and the model learns to
score a film from what it is about on the same scale as one it knows. Each p in
--id-dropout is tried on the 2015 validation and the best is refit; every
candidate's scores are kept in the model's metadata. Selection is scored on the
exported form, ID embeddings zeroed, so what is selected is what is served.

WRITES (into --art, default data/artifacts)
  tower_items.npy   (films × 64, float16) item tower outputs, row = item_id
  tower_user.npz    history embeddings (ID part + content part, precomputed)
                    and MLP weights, for the user tower in NumPy (api/_tower.py)
and records both in manifest.json.
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

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "data" / "artifacts"
sys.path.insert(0, str(ROOT / "pipelines"))
sys.path.insert(0, str(ROOT / "api"))
from _artifacts import record, save_npy, save_npz  # noqa: E402
from _engine import load_catalog  # noqa: E402
CUTOFF = int(datetime(2016, 1, 1, tzinfo=timezone.utc).timestamp())
VAL_CUT = int(datetime(2015, 1, 1, tzinfo=timezone.utc).timestamp())
HIST = 50


def log(msg):
    print(f"[tower {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ml32m", type=Path, required=True)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--logq", type=float, default=0.5, help="strength of the sampling correction, 0..1")
    ap.add_argument("--uniform", type=int, default=4096, help="uniformly sampled negatives per batch")
    ap.add_argument("--per-user", type=int, default=20, help="targets sampled per user per epoch")
    ap.add_argument("--short", type=float, default=0.5, help="share of examples with a 1-10 film history")
    ap.add_argument("--id-dropout", default="0,0.3,0.5",
                    help="chances a film's ID embedding is dropped in training (comma-separated: each is tried "
                         "on the 2015 validation and the best is refit)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--art", type=Path, default=ART, help="artifact directory: reads the catalogue, writes here")
    ap.add_argument("--selection-log", type=Path, default=None,
                    help="append this run's settings and validation scores (JSON lines)")
    ap.add_argument("--export-selection", type=Path, default=None,
                    help="also write the selected model as trained on likes before 2015 into this artifact "
                         "directory, to validate the adaptive engine on 2015")
    args = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))

    cat = load_catalog(args.art)
    n = len(cat["title"])
    # Every MovieLens id, including ones merged as duplicates, to its item_id.
    id_map = args.art / "id_map.json"
    pos = ({int(m): int(i) for m, i in json.loads(id_map.read_text()).items()} if id_map.exists()
           else {int(m): i for i, m in enumerate(cat["movieId"])})
    sem = np.load(args.art / "semantic_items.npy").astype(np.float32)
    assert sem.shape[0] == n, f"semantic_items.npy has {sem.shape[0]} rows for {n} films"
    sem_t = torch.tensor(sem)
    sem_pad = torch.cat([sem_t, torch.zeros(1, sem.shape[1])])        # row n: padding

    log("loading ratings ...")
    r = pd.read_csv(args.ml32m / "ratings.csv", dtype={"userId": np.int32, "movieId": np.int32,
                                                       "rating": np.float32, "timestamp": np.int64})
    r = r[(r.timestamp < CUTOFF) & (r.rating >= 4.0)]
    r["item"] = r.movieId.map(pos)
    r = r.dropna(subset=["item"]).sort_values(["userId", "timestamp"])
    r["item"] = r["item"].astype(np.int64)

    def groups_of(df):
        return [g.item.values for _, g in df.groupby("userId") if len(g) >= 3]

    # Validation: 2015 likes, ranked from each person's likes before 2015.
    early, late = r[r.timestamp < VAL_CUT], r[r.timestamp >= VAL_CUT]
    hist_by = early.groupby("userId")["item"].apply(np.array)
    late_by = late.groupby("userId")["item"].apply(set)
    cand = [u for u in hist_by.index.intersection(late_by.index) if len(hist_by[u]) >= 3]
    vrng = np.random.default_rng(args.seed)
    picked = sorted(vrng.choice(cand, size=min(5000, len(cand)), replace=False).tolist())
    # Only films that existed before 2015 — released before it, or liked before it — are
    # candidates or targets, as in pipelines/evaluate.py: a film released in 2015 could not
    # have been recommended on 2015-01-01.
    released = cat.get("released") or [None] * n
    existed = np.array([bool(d and d < "2015-01-01") or (y is not None and y < 2015)
                        for d, y in zip(released, cat["year"])])
    existed[early.item.unique()] = True
    val = [(hist_by[u], {i for i in late_by[u] - set(hist_by[u].tolist()) if existed[i]}) for u in picked]
    val = [(h, t) for h, t in val if t]
    log(f"selection: likes before 2015 train, {len(val):,} people's 2015 likes validate "
        f"({sum(len(t) for _, t in val):,} targets)")

    class Tower(nn.Module):
        def __init__(self):
            super().__init__()
            d = args.dim
            self.hist = nn.Embedding(n + 1, d, padding_idx=n)
            self.item = nn.Embedding(n, d)
            self.proj = nn.Linear(sem.shape[1], d, bias=False)
            self.hproj = nn.Linear(sem.shape[1], d, bias=False)
            self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))
            nn.init.normal_(self.hist.weight, std=0.05)
            nn.init.normal_(self.item.weight, std=0.05)
            with torch.no_grad():
                self.hist.weight[n].zero_()
            self.p = 0.0

        def keep(self, idx):
            """ID dropout: 1 keeps a film's ID embedding, 0 leaves it its content alone."""
            if not self.training or self.p <= 0:
                return 1.0
            return (torch.rand(idx.shape) >= self.p).float().unsqueeze(-1)

        def history(self, H):
            return self.hist(H) * self.keep(H) + self.hproj(sem_pad[H])

        def user(self, H, mask):
            x = (self.history(H) * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            return F.normalize(x + self.mlp(x), dim=-1)

        def items(self, idx):
            return F.normalize(self.item(idx) * self.keep(idx) + self.proj(sem_t[idx]), dim=-1)

    def exported(model, seen):
        """The model as served: films never seen in training lose their ID embeddings."""
        with torch.no_grad():
            cold = torch.from_numpy(~seen)
            model.item.weight[cold] = 0
            model.hist.weight[:n][cold] = 0
        return model

    def batches(groups, rng):
        """(history, target) batches: per user, up to --per-user liked films as targets."""
        ex_u, ex_t = [], []
        for ui, g in enumerate(groups):
            if len(g) < 2:
                continue
            k = min(len(g), args.per_user)
            for t in rng.choice(len(g), size=k, replace=False):
                ex_u.append(ui)
                ex_t.append(int(t))
        order = rng.permutation(len(ex_u))
        for s in range(0, len(order), args.batch):
            idx = order[s:s + args.batch]
            H = np.full((len(idx), HIST), n, np.int64)
            T = np.empty(len(idx), np.int64)
            for row, j in enumerate(idx):
                g, t = groups[ex_u[j]], ex_t[j]
                others = np.delete(g, t)
                if rng.random() < args.short:
                    keep = int(rng.integers(1, min(len(others), 10) + 1))
                    others = others[rng.choice(len(others), size=keep, replace=False)]
                elif len(others) > HIST:
                    others = others[rng.choice(len(others), size=HIST, replace=False)]
                H[row, :len(others)] = others
                T[row] = g[t]
            yield torch.from_numpy(H), torch.from_numpy(T)

    def validate(model, seen, last: int = HIST) -> dict:
        """Recall@20 of each person's 2015 likes from their earlier likes (their last `last`),
        scored on the exported model — new films by content alone."""
        model.eval()
        saved = {k: v.detach().clone() for k, v in model.state_dict().items()}
        exported(model, seen)
        new = ~seen
        with torch.no_grad():
            V = model.items(torch.arange(n)).numpy()
            rec, new_rec, new_slots = [], [], 0
            for s in range(0, len(val), 1000):
                chunk = val[s:s + 1000]
                H = np.full((len(chunk), HIST), n, np.int64)
                for row, (h, _) in enumerate(chunk):
                    h = h[-last:]
                    H[row, :len(h)] = h
                Ht = torch.from_numpy(H)
                U = model.user(Ht, (Ht != n).float()).numpy()
                with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                    S = U @ V.T  # Accelerate raises spurious flags on finite inputs
                S[:, ~existed] = -np.inf
                for row, (h, t) in enumerate(chunk):
                    S[row, h] = -np.inf  # everything they already liked, not just the last few
                    top = set(np.argpartition(-S[row], 20)[:20].tolist())
                    rec.append(len(top & t) / min(len(t), 20))
                    tn = {i for i in t if new[i]}
                    if tn:
                        new_rec.append(len(top & tn) / min(len(tn), 20))
                    new_slots += sum(1 for i in top if new[i])
        model.load_state_dict(saved)
        model.train()
        return {"recall": float(np.mean(rec)), "new_film_recall": float(np.mean(new_rec)) if new_rec else None,
                "new_film_people": len(new_rec), "new_film_share": new_slots / (20 * len(val))}

    def train(groups, epochs: int, select: bool, p: float):
        """Train for `epochs` with ID dropout `p`; with `select`, score every epoch on 2015 and keep the best."""
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        seen = np.zeros(n, bool)
        seen[np.concatenate(groups)] = True
        freq = np.bincount(np.concatenate(groups), minlength=n).astype(np.float64) + 1.0
        logq = torch.tensor(np.log(freq / freq.sum()), dtype=torch.float32)
        model = Tower()
        model.p = p
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        best, history = None, []
        for ep in range(epochs):
            t0, total, steps = time.time(), 0.0, 0
            for H, T in batches(groups, rng):
                mask = (H != n).float()
                U = model.user(H, mask)
                V = model.items(T)
                # logQ: subtract the log-probability that each negative was drawn —
                # B·p(j) for a batch target, M/n for a uniform draw.
                B = len(T)
                logits = U @ V.T / 0.07 - args.logq * (logq[T].unsqueeze(0) + math.log(B))
                dup = (T.unsqueeze(0) == T.unsqueeze(1)) & ~torch.eye(B, dtype=torch.bool)
                logits = logits.masked_fill(dup, -1e9)
                if args.uniform:
                    neg = torch.randint(0, n, (args.uniform,))
                    lu = U @ model.items(neg).T / 0.07 - args.logq * math.log(args.uniform / n)
                    lu = lu.masked_fill(neg.unsqueeze(0) == T.unsqueeze(1), -1e9)
                    logits = torch.cat([logits, lu], dim=1)
                loss = F.cross_entropy(logits, torch.arange(B))
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
                steps += 1
            msg = f"epoch {ep + 1}: loss {total / steps:.3f}"
            if select:
                full, short = validate(model, seen), validate(model, seen, 3)
                score = (full["recall"] + short["recall"]) / 2
                history.append({"epoch": ep + 1, "full": full, "last_three": short})
                msg += (f", 2015 recall@20 full {full['recall']:.4f}, last three {short['recall']:.4f}; "
                        f"new films: recall {full['new_film_recall'] or 0:.4f}, "
                        f"{full['new_film_share']:.1%} of slots")
                if best is None or score > best[0]:
                    best = (score, ep + 1, full, short, {k: v.detach().clone() for k, v in model.state_dict().items()})
            log(msg + f" ({time.time() - t0:.0f}s)")
        model.eval()
        return exported(model, seen), seen, best, history

    def write(model, seen, art: Path, meta: dict):
        """tower_items.npy and tower_user.npz, as api/_tower.py reads them, recorded in art's manifest."""
        with torch.no_grad():
            V = model.items(torch.arange(n)).numpy()
            Hx = model.history(torch.arange(n)).numpy()
            sd = model.state_dict()
        assert np.isfinite(V).all() and np.isfinite(Hx).all(), "non-finite tower outputs"
        art.mkdir(parents=True, exist_ok=True)
        save_npy(art / "tower_items.npy", V.astype(np.float16))
        save_npz(
            art / "tower_user.npz",
            hist=Hx.astype(np.float16),
            w1=sd["mlp.0.weight"].numpy().T.astype(np.float32), b1=sd["mlp.0.bias"].numpy().astype(np.float32),
            w2=sd["mlp.2.weight"].numpy().T.astype(np.float32), b2=sd["mlp.2.bias"].numpy().astype(np.float32),
            seen=seen,
            # The item tower's content projection: a film's vector from its text alone,
            # as a film with no training data gets it (used to simulate cold starts).
            item_proj=sd["proj.weight"].numpy().T.astype(np.float32),
            meta=np.array(json.dumps(meta)))
        record(art, "tower_items.npy", rows=n, dim=args.dim, dtype="float16", seed=args.seed,
               selection=meta.get("selection"), cutoff=meta["cutoff"], films_with_training_data=int(seen.sum()))
        record(art, "tower_user.npz", rows=n, dim=args.dim)

    grid, chosen = [], None
    early_groups = groups_of(early)
    for p in [float(x) for x in str(args.id_dropout).split(",")]:
        log(f"selection model: {args.epochs} epochs on likes before 2015, id dropout {p}")
        _, _, best, history = train(early_groups, args.epochs, select=True, p=p)
        grid.append({"id_dropout": p, "epoch": best[1], "score": round(best[0], 4),
                     "recall@20": round(best[2]["recall"], 4), "recall@20_last3": round(best[3]["recall"], 4),
                     "new_film_recall@20": best[2]["new_film_recall"],
                     "new_film_share_of_top20": round(best[2]["new_film_share"], 4), "epochs": history})
        if chosen is None or best[0] > chosen[1][0]:
            chosen = (p, best)
    p, (_, epochs, full, short, best_state) = chosen
    log(f"selected id dropout {p}, epoch {epochs}: 2015 recall@20 full {full['recall']:.4f}, "
        f"last three {short['recall']:.4f}")
    common = {"films": n, "content_history_branch": True, "seed": args.seed, "uniform_negatives": args.uniform,
              "id_dropout": p, "short_history_share": args.short, "dim": args.dim, "logq": args.logq,
              "epochs": epochs, "trained": datetime.now(timezone.utc).strftime("%Y-%m-%d")}
    if args.export_selection:
        # The selected model as trained on likes before 2015: what the adaptive engine's own
        # settings are validated with (pipelines/evaluate.py --cutoff 2015-01-01).
        sel = Tower()
        sel.load_state_dict(best_state)
        sel.eval()
        seen_early = np.zeros(n, bool)
        seen_early[np.concatenate(early_groups)] = True
        write(exported(sel, seen_early), seen_early, args.export_selection,
              {**common, "cutoff": "2015-01-01", "films_with_training_data": int(seen_early.sum()),
               "validation_recall@20": round(full["recall"], 4),
               "validation_recall@20_last3": round(short["recall"], 4)})
        log(f"wrote the selection model (likes before 2015) to {args.export_selection}")
    groups = groups_of(r)
    log(f"refit: {len(groups):,} users with ≥3 likes, {sum(len(g) for g in groups):,} likes before 2016, "
        f"{epochs} epochs")
    model, seen, _, _ = train(groups, epochs, select=False, p=p)
    selection = {"validation": "2015 likes from earlier likes; model trained on likes before 2015",
                 "id_dropout": p, "epoch": epochs, "recall@20": round(full["recall"], 4),
                 "recall@20_last3": round(short["recall"], 4),
                 "new_film_recall@20": None if full["new_film_recall"] is None else round(full["new_film_recall"], 4),
                 "new_film_people": full["new_film_people"],
                 "new_film_share_of_top20": round(full["new_film_share"], 4),
                 "grid": [{k: v for k, v in g.items() if k != "epochs"} for g in grid]}
    if args.selection_log:
        with args.selection_log.open("a") as f:
            f.write(json.dumps({"films": n, "uniform_negatives": args.uniform, "logq": args.logq, "dim": args.dim,
                                "seed": args.seed, "selected": selection, "grid": grid}) + "\n")
    log(f"{int(seen.sum()):,} films learned from training data, {int((~seen).sum()):,} from content alone")
    write(model, seen, args.art, {**common, "cutoff": "2016-01-01", "films_with_training_data": int(seen.sum()),
                                  "validation_recall@20": selection["recall@20"],
                                  "validation_recall@20_last3": selection["recall@20_last3"],
                                  "selection": selection})
    log("wrote tower_items.npy and tower_user.npz")


if __name__ == "__main__":
    sys.exit(main())
