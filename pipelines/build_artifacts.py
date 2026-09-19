"""
CINEMIND — offline artifact builder.

Rebuilds every file the serving API needs, from the raw MovieLens data plus the
trained Two-Tower index.

Why this exists
---------------
The original pipeline trained a LabelEncoder over movieIds but never persisted
its classes. The serving layer therefore treated FAISS row numbers (0..17718)
as MovieLens movieIds, so every "similar movies" result was mis-decoded.
This script recovers that mapping, WRITES IT TO DISK (item_classes.npy), and
derives all serving artifacts from it. Never drop that file again.

Outputs (data/artifacts/):
    item_classes.npy   item_idx -> movieId          (the recovered encoder)
    catalog.json       columnar metadata, 17,719 items
    neighbors.npz      precomputed top-K similar items + scores
    browse.json        prebuilt home rows / top 50 / TV
    item_vectors.npy   float16 two-tower item embeddings

Usage:  python pipelines/build_artifacts.py --ml20m DIR --ml32m DIR --index PATH
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

SAMPLE_SIZE = 2_000_000   # must match kaggle_training.ipynb
SEED = 42                 # must match kaggle_training.ipynb
EXPECT_USERS, EXPECT_ITEMS = 135_697, 17_719
TOPK = 24


def recover_encoder(ml20m: Path, out: Path) -> np.ndarray:
    """Reproduce the exact training sample to recover item_idx -> movieId."""
    cached = out / "item_classes.npy"
    if cached.exists():
        c = np.load(cached)
        if len(c) == EXPECT_ITEMS:
            print(f"[encoder] reusing {cached} ({len(c)} items)")
            return c
    print("[encoder] reproducing training sample from ml-20m ...")
    r = pd.read_csv(ml20m / "ratings.csv", usecols=["userId", "movieId"])
    s = r.sample(n=SAMPLE_SIZE, random_state=SEED)
    nu, ni = s.userId.nunique(), s.movieId.nunique()
    if (nu, ni) != (EXPECT_USERS, EXPECT_ITEMS):
        sys.exit(f"[encoder] FAILED to reproduce sample: got {nu} users / {ni} items, "
                 f"expected {EXPECT_USERS}/{EXPECT_ITEMS}. Refusing to emit a wrong mapping.")
    classes = np.sort(s.movieId.unique()).astype(np.int32)
    out.mkdir(parents=True, exist_ok=True)
    np.save(cached, classes)
    print(f"[encoder] verified + saved -> {cached}")
    return classes


_YEAR = re.compile(r"\((\d{4})\)\s*$")
_ART = re.compile(r"^(.*), (The|A|An|La|Le|Les|Il|L'|El|Das|Der|Die|Los|Las)$")


def clean_title(raw: str):
    raw = str(raw).strip()
    m = _YEAR.search(raw)
    year = int(m.group(1)) if m else None
    t = _YEAR.sub("", raw).strip() if m else raw
    a = _ART.match(t)
    if a:
        t = f"{a.group(2)} {a.group(1)}"
        t = t.replace("L' ", "L'")
    return t, year


def build_catalog(classes, ml20m: Path, ml32m: Path, tmdb_cache: dict, credits: dict | None = None):
    """Metadata for every indexed item, from MovieLens + cached TMDB details."""
    mv = pd.read_csv(ml32m / "movies.csv")
    m20 = pd.read_csv(ml20m / "movies.csv")
    mv = pd.concat([mv, m20[~m20.movieId.isin(mv.movieId)]], ignore_index=True).set_index("movieId")
    lk = pd.read_csv(ml32m / "links.csv")
    l20 = pd.read_csv(ml20m / "links.csv")
    lk = pd.concat([lk, l20[~l20.movieId.isin(lk.movieId)]], ignore_index=True).set_index("movieId")

    print("[catalog] rating stats from ml-32m ...")
    st = (pd.read_csv(ml32m / "ratings.csv", usecols=["movieId", "rating"])
            .groupby("movieId")["rating"].agg(["mean", "count"]))

    print("[catalog] tags ...")
    keep = set(int(x) for x in classes)
    t = pd.read_csv(ml32m / "tags.csv", usecols=["movieId", "tag"])
    t = t[t.movieId.isin(keep)]
    t["tag"] = t.tag.astype(str).str.lower().str.strip()
    t = t[t.tag.str.len().between(2, 30)]
    tc = (t.groupby(["movieId", "tag"]).size().reset_index(name="n")
            .sort_values(["movieId", "n"], ascending=[True, False])
            .groupby("movieId").head(8))
    tags = tc.groupby("movieId")["tag"].apply(list).to_dict()

    N = len(classes)
    cols = {k: [None] * N for k in
            ("title", "year", "genres", "tags", "tmdbId", "movieId", "poster",
             "backdrop", "overview", "runtime", "lang", "media", "rc", "ra", "tv",
             "directors", "cast")}
    for i, mid in enumerate(classes):
        mid = int(mid)
        row = mv.loc[mid] if mid in mv.index else None
        title, year = clean_title(row["title"]) if row is not None else (f"Movie {mid}", None)
        g = [x for x in (str(row["genres"]).split("|") if row is not None
             and pd.notna(row["genres"]) else []) if x and x != "(no genres listed)"]
        tm = None
        if mid in lk.index and pd.notna(lk.loc[mid, "tmdbId"]):
            tm = int(lk.loc[mid, "tmdbId"])
        d = tmdb_cache.get(str(i)) or {}
        s = st.loc[mid] if mid in st.index else None
        cols["title"][i] = title
        cols["year"][i] = year or (int(d["rd"][:4]) if d.get("rd") else None)
        cols["genres"][i] = g
        cols["tags"][i] = tags.get(mid, [])
        cols["tmdbId"][i] = tm
        cols["movieId"][i] = mid
        cols["poster"][i] = d.get("p")
        cols["backdrop"][i] = d.get("b")
        cols["overview"][i] = d.get("o") or ""
        cols["runtime"][i] = d.get("rt")
        cols["lang"][i] = d.get("lang")
        cols["media"][i] = d.get("mt", "movie")
        cols["rc"][i] = int(s["count"]) if s is not None else 0
        cols["ra"][i] = round(float(s["mean"]), 2) if s is not None else None
        cols["tv"][i] = d.get("tv")
        cr = (credits or {}).get(str(i)) or {}
        cols["directors"][i] = cr.get("d", [])
        cols["cast"][i] = cr.get("c", [])
    print(f"[catalog] {N} items, {sum(1 for p in cols['poster'] if p)} with posters")
    return cols


def build_similarity(cols, classes, index_path: Path, ml32m: Path, out: Path):
    """Hybrid item-item similarity: co-occurrence CF + two-tower + content."""
    import faiss
    N = len(classes)
    ix = faiss.read_index(str(index_path))
    assert ix.ntotal == N, f"index has {ix.ntotal} vectors, expected {N}"
    V = ix.reconstruct_n(0, N).astype(np.float32)

    # ---- signal 1: item-item collaborative filtering over the full 32M ratings
    print("[sim] item-item CF from ml-32m ...")
    pos = {int(m): i for i, m in enumerate(classes)}
    r = pd.read_csv(ml32m / "ratings.csv", usecols=["userId", "movieId", "rating"],
                    dtype={"userId": np.int32, "movieId": np.int32, "rating": np.float32})
    r = r[r.movieId.isin(pos.keys()) & (r.rating >= 3.5)]
    it = r.movieId.map(pos).values.astype(np.int32)
    _, ur = np.unique(r.userId.values, return_inverse=True)
    nU = ur.max() + 1
    del r
    X = sp.csr_matrix((np.ones(len(it), np.float32), (ur, it)), shape=(nU, N))
    X.sum_duplicates(); X.data[:] = 1.0
    ul = np.asarray(X.sum(1)).ravel(); ul[ul == 0] = 1
    X = sp.diags((1 / np.sqrt(ul)).astype(np.float32)) @ X          # damp power users
    df = np.asarray((X > 0).sum(0)).ravel().astype(np.float32); df[df == 0] = 1
    X = (X @ sp.diags(np.log(1 + nU / df).astype(np.float32)))      # damp blockbusters
    cn = np.sqrt(np.asarray(X.multiply(X).sum(0)).ravel()); cn[cn == 0] = 1
    X = (X @ sp.diags((1 / cn).astype(np.float32))).tocsr().astype(np.float32)

    # ---- content signals
    GEN = sorted({g for gs in cols["genres"] for g in gs})
    gi = {g: i for i, g in enumerate(GEN)}
    G = np.zeros((N, len(GEN)), np.float32)
    for i, gs in enumerate(cols["genres"]):
        for g in gs: G[i, gi[g]] = 1
    Gw = G * np.log(N / np.maximum(G.sum(0), 1)).astype(np.float32)   # rare genres count more
    Gn = Gw / np.maximum(np.linalg.norm(Gw, axis=1, keepdims=True), 1e-6)
    TAG = sorted({t for ts in cols["tags"] for t in ts})
    ti = {t: i for i, t in enumerate(TAG)}
    rr, cc = [], []
    for i, ts in enumerate(cols["tags"]):
        for t in ts: rr.append(i); cc.append(ti[t])
    T = sp.csr_matrix((np.ones(len(rr), np.float32), (rr, cc)), shape=(N, len(TAG)))
    tn = np.sqrt(T.multiply(T).sum(1)).A.ravel(); tn[tn == 0] = 1
    Tn = (sp.diags(1 / tn).tocsr() @ T).tocsr()

    rc = np.array(cols["rc"], np.float32)
    thin = np.maximum(0, 1 - np.log10(rc + 1) / np.log10(300))        # demote thin data

    def norm(s):
        s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
        s = re.sub(r"\b(the|a|an|part|episode|[ivx]+|\d+)\b", " ", s)
        return re.sub(r"\s+", " ", s).strip()
    NT = [norm(t) for t in cols["title"]]

    W = dict(icf=1.35, tt=0.75, gen=0.60, tag=0.50, thin=0.70, fran=0.40)
    nbr = np.zeros((N, TOPK), np.int32); sco = np.zeros((N, TOPK), np.float32)
    Xc = X.tocsc(); B = 512
    print("[sim] blending + ranking ...")
    for s0 in range(0, N, B):
        e = min(s0 + B, N)
        blk = (Xc[:, s0:e].T @ X).toarray()
        tt_blk = V[s0:e] @ V.T
        for j in range(e - s0):
            i = s0 + j
            icf = blk[j].copy(); icf[i] = 0
            cand = np.argpartition(-icf, 200)[:200]
            tt_row = tt_blk[j]
            cand = np.union1d(cand, np.argpartition(-tt_row, 200)[:200])
            cand = cand[cand != i]
            a = icf[cand]; a = a / a.max() if a.max() > 0 else a
            b = tt_row[cand]; b = (b - b.min()) / max(float(np.ptp(b)), 1e-6)
            gs = Gn[cand] @ Gn[i]
            ts = np.asarray((Tn[cand] @ Tn[i].T).todense()).ravel()
            fr = np.zeros(len(cand), np.float32)
            na = NT[i]
            if len(na) >= 4:
                for n, c in enumerate(cand):
                    nb = NT[c]
                    if nb and (na == nb or na in nb or nb in na): fr[n] = 1.0
            sc = (W["icf"] * a + W["tt"] * b + W["gen"] * gs + W["tag"] * ts
                  - W["thin"] * thin[cand] + W["fran"] * fr)
            o = np.argsort(-sc)[:TOPK]
            nbr[i] = cand[o]; sco[i] = sc[o]
        if s0 % 4096 == 0: print(f"  {s0}/{N}")
    np.savez_compressed(out / "neighbors.npz", ids=nbr, scores=sco.astype(np.float16))
    np.save(out / "item_vectors.npy", V.astype(np.float16))
    print(f"[sim] wrote neighbors.npz + item_vectors.npy")
    return nbr, sco


def build_browse(cols, out: Path):
    """Prebuild everything the home page shows, so serving is a dict lookup."""
    N = len(cols["title"])
    rc = np.array(cols["rc"], np.float32)
    ra = np.array([x if x is not None else 0 for x in cols["ra"]], np.float32)
    # Bayesian average: a 4.6 from 40 voters must not outrank a 4.4 from 90,000
    C, m = ra[rc > 0].mean(), 2500.0
    bayes = (rc * ra + m * C) / (rc + m)
    has_art = np.array([bool(p) for p in cols["poster"]])

    def pack(ids):
        return [int(i) for i in ids]

    top50 = pack(np.argsort(-np.where(has_art & (rc >= 1000), bayes, -1))[:50])
    genres = {}
    for g in sorted({g for gs in cols["genres"] for g in gs}):
        mask = np.array([g in gs for gs in cols["genres"]]) & has_art & (rc >= 300)
        ids = np.argsort(-np.where(mask, bayes, -1))[:40]
        ids = [int(i) for i in ids if mask[i]]
        if len(ids) >= 12:
            genres[g] = ids
    tv = pack([i for i in np.argsort(-bayes) if cols["media"][i] == "tv" and has_art[i]][:40])
    hero = pack(np.argsort(-np.where(has_art & np.array([bool(b) for b in cols["backdrop"]])
                                     & (rc >= 20000), bayes, -1))[:12])
    payload = {"top50": top50, "genres": genres, "tv": tv, "hero": hero,
               "bayes": [round(float(x), 3) for x in bayes]}
    (out / "browse.json").write_text(json.dumps(payload, separators=(",", ":")))
    print(f"[browse] top50 + {len(genres)} genre rows + {len(tv)} tv + {len(hero)} hero")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ml20m", required=True, type=Path)
    ap.add_argument("--ml32m", required=True, type=Path)
    ap.add_argument("--index", default=Path("models/production.index"), type=Path)
    ap.add_argument("--tmdb-cache", type=Path, default=None)
    ap.add_argument("--credits-cache", type=Path, default=None)
    ap.add_argument("--out", default=Path("data/artifacts"), type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    classes = recover_encoder(a.ml20m, a.out)
    cache = json.loads(a.tmdb_cache.read_text()) if a.tmdb_cache and a.tmdb_cache.exists() else {}
    print(f"[tmdb] {len(cache)} cached detail records")
    credits = (json.loads(a.credits_cache.read_text())
               if a.credits_cache and a.credits_cache.exists() else {})
    print(f"[tmdb] {len(credits)} cached credit records")
    cols = build_catalog(classes, a.ml20m, a.ml32m, cache, credits)
    (a.out / "catalog.json").write_text(json.dumps(cols, separators=(",", ":")))
    print(f"[catalog] wrote {(a.out/'catalog.json').stat().st_size/1e6:.1f} MB")
    build_similarity(cols, classes, a.index, a.ml32m, a.out)
    build_browse(cols, a.out)
    print("\n✅ artifacts ready in", a.out)


if __name__ == "__main__":
    main()
