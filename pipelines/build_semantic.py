"""
Build the semantic layer for natural-language discovery.

    python pipelines/build_semantic.py --model-dir data/raw/potion-base-8M

1. Converts Model2Vec's potion-base-8M into two serving artifacts: the
   WordPiece vocabulary in id order, and the token table as float16.
2. Verifies api/_semantic.py against the model2vec library on real catalogue
   text and on edge cases (accents, punctuation, CJK, text past 512 tokens,
   words it has never seen). Token ids must match exactly and every vector
   must agree to a cosine of 0.9995 or better — the gap is float16 storage.
   If either check fails, nothing is written.
3. Embeds every film's description into semantic_items.npy, row i for
   item_id i. Vectors are cached by a hash of the text they came from
   (data/raw/build/semantic_cache.npz), so a rebuild encodes only films whose
   text is new or changed, and the cache is saved every few batches so an
   interrupted run resumes. The result is checked — one finite 256-d row per
   film, zero only where a film has no usable text — and recorded in
   manifest.json before the API will load it.

The text recipe is the same for every film (see doc_text), so films added by a
catalogue expansion land in the same space as the ones already served.

Get the model with:
    huggingface-cli download minishlab/potion-base-8M --local-dir data/raw/potion-base-8M
(requires model2vec, safetensors — see pipelines/requirements-build.txt)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "data" / "artifacts"
CACHE = ROOT / "data" / "raw" / "build" / "semantic_cache.npz"
MODEL = "minishlab/potion-base-8M"
RECIPE = "title. genres. tags. tags. overview"
sys.path.insert(0, str(ROOT / "api"))
sys.path.insert(0, str(ROOT / "pipelines"))

from _artifacts import record, save_npy, save_npz, write_atomic  # noqa: E402
from _engine import load_catalog  # noqa: E402
from _semantic import SemanticModel  # noqa: E402


def display_title(t: str) -> str:
    """'Postman, The (Postino, Il)' -> 'The Postman'."""
    t2 = re.sub(r"\s*\(.*\)\s*$", "", t).strip() or t
    m = re.match(r"^(.*), (The|A|An)$", t2)
    return f"{m.group(2)} {m.group(1)}" if m else t2


def doc_text(c: dict, i: int) -> str:
    """
    What a film is about, as one passage: title, genres, the themes viewers
    tagged it with, and the synopsis. Tags carry most of the mood and theme
    ("twist ending", "feel-good", "space"), so they are written twice to keep a
    long synopsis from drowning them out in the mean.
    """
    tags = ", ".join(c["tags"][i] or [])
    genres = ", ".join(c["genres"][i] or [])
    parts = [display_title(c["title"][i]), genres, tags, tags, c["overview"][i] or ""]
    return ". ".join(p for p in parts if p)


EDGE_CASES = [
    "", "   ", "a", "Amélie", "naïve café — déjà vu!", "SCI-FI/FANTASY (1999)", "don't stop",
    "日本の映画 とても 良い", "Ωμέγα αλφα", "x" * 150, "supercalifragilisticexpialidocious",
    "psychological thrillers with unreliable narrators and a shocking ending",
    "Something uplifting under two hours for tonight.", "\t\nnew\r\nlines​ and spaces",
    " ".join(["word"] * 700),            # past the 512-token truncation
]


def text_hash(t: str) -> str:
    return hashlib.sha1(f"{MODEL}|{t}".encode()).hexdigest()


def embed_all(model: SemanticModel, texts: list[str], batch: int = 2000) -> np.ndarray:
    """Every text's vector, reusing cached ones and checkpointing new ones."""
    cache: dict[str, np.ndarray] = {}
    if CACHE.exists():
        z = np.load(CACHE)
        cache = dict(zip(z["keys"].tolist(), z["vecs"]))
    hashes = [text_hash(t) for t in texts]
    todo = sorted({h: t for h, t in zip(hashes, texts) if h not in cache}.items())
    print(f"[semantic] {len(texts) - sum(1 for h in hashes if h not in cache):,} of {len(texts):,} "
          f"films cached; encoding {len(todo):,} new texts", flush=True)

    def checkpoint():
        keys = sorted(cache)
        save_npz(CACHE, compressed=False, keys=np.array(keys), vecs=np.stack([cache[k] for k in keys]).astype(np.float16))

    for s0 in range(0, len(todo), batch):
        chunk = todo[s0:s0 + batch]
        vecs = model.encode_many([t for _, t in chunk]).astype(np.float16)
        for (h, _), v in zip(chunk, vecs):
            cache[h] = v
        if (s0 // batch) % 5 == 4:
            checkpoint()
        print(f"[semantic]   {min(s0 + batch, len(todo)):,}/{len(todo):,}", flush=True)
    if todo:
        checkpoint()
    return np.stack([cache[h] for h in hashes]).astype(np.float16)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--art", type=Path, default=ART, help="artifact directory: reads catalog.json, writes here")
    ap.add_argument("--samples", type=int, default=3000, help="catalogue texts to verify against")
    args = ap.parse_args()
    args.out = args.art

    from model2vec import StaticModel
    from safetensors.numpy import load_file

    tok = json.loads((args.model_dir / "tokenizer.json").read_text())
    vocab_map: dict[str, int] = tok["model"]["vocab"]
    vocab = [None] * len(vocab_map)
    for t, i in vocab_map.items():
        vocab[i] = t
    assert all(v is not None for v in vocab), "vocabulary ids are not contiguous"
    assert not any("\n" in v for v in vocab), "a token contains a newline"
    table32 = load_file(str(args.model_dir / "model.safetensors"))["embeddings"]
    assert table32.shape[0] == len(vocab), (table32.shape, len(vocab))
    table = table32.astype(np.float16)

    ours = SemanticModel(vocab, table)
    ref = StaticModel.from_pretrained(str(args.model_dir))
    assert ref.weights is None and ref.token_mapping is None, "model has weights/mapping this port ignores"

    c = load_catalog(args.art)
    n = len(c["title"])
    rng = random.Random(7)
    sample = [doc_text(c, i) for i in rng.sample(range(n), min(args.samples, n))]
    texts = EDGE_CASES + sample

    # 1. Token ids, exactly. model2vec truncates inside the tokenizer (512 by
    #    default) and only then drops [UNK]; the reference is built the same way.
    ref.tokenizer.enable_truncation(512)
    enc = ref.tokenizer.encode_batch_fast(texts, add_special_tokens=False)
    ref_ids = [[i for i in e.ids if i != ref.unk_token_id] for e in enc]
    assert ref_ids == ref.tokenize(texts), "reference tokenisation is not what encode() uses"
    bad = [(t[:60], a[:8], b[:8]) for t, a, b in zip(texts, ref_ids, (ours.tokenise(t) for t in texts)) if a != b]
    if bad:
        print(f"[semantic] TOKEN MISMATCH on {len(bad)} texts, e.g. {bad[:3]}")
        sys.exit(1)

    # 2. Vectors, to float16 precision.
    a = ref.encode(texts, max_length=512)
    b = ours.encode_many(texts)
    nz = np.linalg.norm(a, axis=1) > 0
    assert (np.linalg.norm(b, axis=1) > 0).tolist() == nz.tolist(), "empty-text handling differs"
    cos = (a[nz] * b[nz]).sum(1) / (np.linalg.norm(a[nz], axis=1) * np.linalg.norm(b[nz], axis=1))
    print(f"[semantic] verified {len(texts)} texts: token ids identical, cosine min {cos.min():.6f}")
    if cos.min() < 0.9995:
        print("[semantic] VECTOR MISMATCH — not writing artifacts")
        sys.exit(1)

    # 3. Every film, from the cache where the text is unchanged.
    texts = [doc_text(c, i) for i in range(n)]
    items = embed_all(ours, texts)
    f = items.astype(np.float32)
    norms = np.linalg.norm(f, axis=1)
    empty = norms == 0
    assert items.shape == (n, table.shape[1]), f"expected {n} rows of {table.shape[1]}, got {items.shape}"
    assert np.isfinite(f).all(), "non-finite values in the item vectors"
    assert np.allclose(norms[~empty], 1.0, atol=5e-3), "item vectors are not unit length"
    assert all(not t.strip() for t, e in zip(texts, empty) if e), "a film with text got a zero vector"

    args.out.mkdir(parents=True, exist_ok=True)
    write_atomic(args.out / "semantic_vocab.txt", ("\n".join(vocab) + "\n").encode("utf-8"))
    save_npy(args.out / "semantic_tokens.npy", table)
    save_npy(args.out / "semantic_items.npy", items)
    record(args.out, "semantic_items.npy", rows=n, dim=int(items.shape[1]), dtype="float16",
           model=MODEL, recipe=RECIPE, empty_rows=int(empty.sum()))
    record(args.out, "semantic_tokens.npy", vocab=int(table.shape[0]), dim=int(table.shape[1]), model=MODEL)
    print(f"[semantic] wrote vocab {len(vocab)}, tokens {table.shape} f16, items {items.shape} f16 "
          f"({int(empty.sum())} films with no usable text)")


if __name__ == "__main__":
    main()
