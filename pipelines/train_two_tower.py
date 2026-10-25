"""
Train the two-tower model the adaptive engine and discovery use.

    python pipelines/train_two_tower.py --ml32m data/raw/ml-32m

WHY RETRAIN
The original item vectors (item_vectors.npy) came from a model trained on a
2M-rating random sample — about a hundred ratings a film. On their own they
make a poor taste space: The Notebook's nearest films were Saints and Soldiers
and Crimson Rivers 2, and films with no training data sat beside everything.
They still feed the precomputed similar-films graph, where co-rating evidence
dominates; this model replaces them only where a taste space is used directly.

THE MODEL
  user tower   the mean of the history's item embeddings, plus a residual MLP,
               L2-normalised. No per-user parameters: it is exactly what the
               server can compute for a browser with three likes and no account.
  item tower   an ID embedding plus a learned projection of the film's
               sentence embedding, L2-normalised — so films with little rating
               data still land somewhere sensible, by what they are about.
  objective    in-batch softmax at temperature 0.07, duplicates masked, with a
               logQ correction (scaled by --logq) so popular films are not
               learned as negatives just because they appear in many batches.

THE DATA
Positive ratings (≥4★) before 2016-01-01 only, so pipelines/evaluate.py's test
period stays unseen. Each example is one liked film as the target and up to 50
of the same person's other likes as the history — but half the time the
history is cut to a random 1–10 of them, because that is what the site sees:
a browser with a handful of likes. Trained on long histories alone, a single
popular film (The Dark Knight, say) was out of distribution and its user
vector landed among unrelated films.

For model selection, 5,000 users' last pre-2016 like is held out as a
validation target — inside the training period, never the test period — and
scored twice: from their full history and from their last three likes. The
epoch with the best mean of the two recalls@20 is kept.

WRITES
  data/artifacts/tower_items.npy   (films × 64, float16) item tower outputs
  data/artifacts/tower_user.npz    history embeddings and MLP weights, for the
                                   user tower in NumPy (api/_tower.py)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "data" / "artifacts"
CUTOFF = int(datetime(2016, 1, 1, tzinfo=timezone.utc).timestamp())
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
    ap.add_argument("--logq", type=float, default=0.5, help="strength of the popularity correction, 0..1")
    ap.add_argument("--per-user", type=int, default=20, help="targets sampled per user per epoch")
    ap.add_argument("--short", type=float, default=0.5, help="share of examples with a 1-10 film history")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=ART)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    cat = json.loads((ART / "catalog.json").read_text())
    n = len(cat["title"])
    pos = {int(m): i for i, m in enumerate(cat["movieId"])}
    sem = np.load(ART / "semantic_items.npy").astype(np.float32)

    log("loading ratings ...")
    r = pd.read_csv(args.ml32m / "ratings.csv", dtype={"userId": np.int32, "movieId": np.int32,
                                                       "rating": np.float32, "timestamp": np.int64})
    r = r[(r.timestamp < CUTOFF) & (r.rating >= 4.0)]
    r["item"] = r.movieId.map(pos)
    r = r.dropna(subset=["item"]).sort_values(["userId", "timestamp"])
    r["item"] = r["item"].astype(np.int64)
    groups = [g.item.values for _, g in r.groupby("userId") if len(g) >= 3]
    log(f"{len(groups):,} users with ≥3 likes, {sum(len(g) for g in groups):,} likes before the cutoff")

    # Validation: the last like of 5,000 users, held out of training for them.
    val_idx = set(rng.choice(len(groups), size=min(5000, len(groups)), replace=False).tolist())
    val = [(groups[i][:-1], int(groups[i][-1])) for i in sorted(val_idx)]
    train_groups = [g[:-1] if i in val_idx else g for i, g in enumerate(groups)]

    freq = np.bincount(np.concatenate(train_groups), minlength=n).astype(np.float64) + 1.0
    logq = torch.tensor(np.log(freq / freq.sum()), dtype=torch.float32)

    class Tower(nn.Module):
        def __init__(self):
            super().__init__()
            d = args.dim
            self.hist = nn.Embedding(n + 1, d, padding_idx=n)
            self.item = nn.Embedding(n, d)
            self.proj = nn.Linear(sem.shape[1], d, bias=False)
            self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))
            self.sem = torch.tensor(sem)
            nn.init.normal_(self.hist.weight, std=0.05)
            nn.init.normal_(self.item.weight, std=0.05)
            with torch.no_grad():
                self.hist.weight[n].zero_()

        def user(self, H, mask):
            x = (self.hist(H) * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            return F.normalize(x + self.mlp(x), dim=-1)

        def items(self, idx):
            return F.normalize(self.item(idx) + self.proj(self.sem[idx]), dim=-1)

    model = Tower()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def batches():
        """(history, mask, target) batches: per user, up to --per-user liked films as targets."""
        ex_u, ex_t = [], []
        for ui, g in enumerate(train_groups):
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
                g, t = train_groups[ex_u[j]], ex_t[j]
                others = np.delete(g, t)
                if rng.random() < args.short:
                    keep = int(rng.integers(1, min(len(others), 10) + 1))
                    others = others[rng.choice(len(others), size=keep, replace=False)]
                elif len(others) > HIST:
                    others = others[rng.choice(len(others), size=HIST, replace=False)]
                H[row, :len(others)] = others
                T[row] = g[t]
            yield torch.from_numpy(H), torch.from_numpy(T)

    def validate(model, last: int = HIST) -> float:
        """Recall@20 of the held-out like, from each person's last `last` likes."""
        model.eval()
        with torch.no_grad():
            V = model.items(torch.arange(n)).numpy()
            hits = 0
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
                for row, (h, t) in enumerate(chunk):
                    S[row, h] = -np.inf  # everything they already liked, not just the last few
                    top = np.argpartition(-S[row], 20)[:20]
                    hits += t in top
        model.train()
        return hits / len(val)

    def score(model):
        full, short = validate(model), validate(model, 3)
        return (full + short) / 2, full, short

    best, best_state, best_parts = -1.0, None, (0.0, 0.0)
    _, full, short = score(model)
    log(f"validation recall@20 before training: full history {full:.4f}, last three likes {short:.4f}")
    for ep in range(args.epochs):
        t0, total, steps = time.time(), 0.0, 0
        for H, T in batches():
            mask = (H != n).float()
            U = model.user(H, mask)
            V = model.items(T)
            logits = U @ V.T / 0.07 - args.logq * logq[T].unsqueeze(0)
            dup = (T.unsqueeze(0) == T.unsqueeze(1)) & ~torch.eye(len(T), dtype=torch.bool)
            logits = logits.masked_fill(dup, -1e9)
            loss = F.cross_entropy(logits, torch.arange(len(T)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            steps += 1
        rec, full, short = score(model)
        log(f"epoch {ep + 1}: loss {total / steps:.3f}, validation recall@20 full {full:.4f}, "
            f"last three {short:.4f} ({time.time() - t0:.0f}s)")
        if rec > best:
            best, best_parts = rec, (full, short)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        V = model.items(torch.arange(n)).numpy()
        sd = model.state_dict()
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "tower_items.npy", V.astype(np.float16))
    np.savez_compressed(
        args.out / "tower_user.npz",
        hist=sd["hist.weight"][:n].numpy().astype(np.float16),
        w1=sd["mlp.0.weight"].numpy().T.astype(np.float32), b1=sd["mlp.0.bias"].numpy().astype(np.float32),
        w2=sd["mlp.2.weight"].numpy().T.astype(np.float32), b2=sd["mlp.2.bias"].numpy().astype(np.float32),
        meta=np.array(json.dumps({"validation_recall@20": round(best_parts[0], 4),
                                  "validation_recall@20_last3": round(best_parts[1], 4),
                                  "short_history_share": args.short, "dim": args.dim, "logq": args.logq,
                                  "cutoff": "2016-01-01", "epochs": args.epochs, "trained": datetime.now(
                                      timezone.utc).strftime("%Y-%m-%d")})))
    log(f"best validation recall@20: full {best_parts[0]:.4f}, last three {best_parts[1]:.4f}; wrote tower_items.npy and tower_user.npz")


if __name__ == "__main__":
    sys.exit(main())
