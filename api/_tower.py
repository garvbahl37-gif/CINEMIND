"""
The two-tower model at serving time, in NumPy.

Trained by pipelines/train_two_tower.py on ratings before 2016:

    user  = normalise(x + MLP(x)),  x = weighted mean of history embeddings
    item  = precomputed: normalise(ID embedding + projected sentence embedding)

A person is their history — the films they liked, weighted by how strongly
and how recently — so a browser with three likes and no account gets a user
vector the same way the model was trained to produce one.

If the trained files are absent, it falls back to the original item vectors
(item_vectors.npy) with a plain weighted mean, so the API keeps working.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from _engine import ART
from _semantic import mm


class Tower:
    def __init__(self, root: Path = ART):
        items, user = root / "tower_items.npy", root / "tower_user.npz"
        if items.exists() and user.exists():
            self.V = np.load(items).astype(np.float32)
            z = np.load(user)
            self.H = z["hist"].astype(np.float32)
            self.w1, self.b1, self.w2, self.b2 = (z[k].astype(np.float32) for k in ("w1", "b1", "w2", "b2"))
            self.trained = True
        else:
            self.V = np.load(root / "item_vectors.npy").astype(np.float32)
            self.H = self.V
            self.w1 = None
            self.trained = False
        self.dim = self.V.shape[1]

    def user(self, items, weights=None) -> np.ndarray | None:
        """A unit user vector for a weighted history, or None for an empty one."""
        items = np.asarray(list(items), np.int64)
        if items.size == 0:
            return None
        w = np.ones(items.size, np.float32) if weights is None else np.asarray(list(weights), np.float32)
        if float(w.sum()) <= 0:
            return None
        x = mm(w, self.H[items]) / float(w.sum())
        if self.w1 is not None:
            h = np.maximum(mm(x, self.w1) + self.b1, 0)
            x = x + mm(h, self.w2) + self.b2
        n = float(np.linalg.norm(x))
        return x / n if n > 1e-12 else None


@lru_cache(maxsize=1)
def get_tower(root: Path = ART) -> Tower:
    return Tower(root)
