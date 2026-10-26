"""
Sentence embeddings for natural-language discovery, in NumPy alone.

The model is Model2Vec's potion-base-8M: a static sentence-embedding model
distilled from a sentence transformer (BAAI/bge-base-en-v1.5). Encoding a
sentence is WordPiece tokenisation, a row lookup per token and a mean, so it
runs in well under a millisecond without PyTorch, ONNX or a GPU — which keeps
the serving function NumPy-only, as it has always been.

This module reproduces the reference implementation exactly:
  - BERT normalisation: clean control characters, space out CJK characters,
    lowercase, strip accents;
  - BERT pre-tokenisation: split on whitespace and on every punctuation mark;
  - greedy longest-match WordPiece with "##" continuations, a word over 100
    characters or with no match becoming [UNK];
  - no special tokens, truncation at 512 tokens, [UNK] dropped, then the mean
    of the token vectors, L2-normalised.

pipelines/build_semantic.py checks it against the model2vec library on a
sample of real catalogue text before any artifact is written.
"""
from __future__ import annotations

import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np

ART = Path(__file__).resolve().parent.parent / "data" / "artifacts"

MAX_TOKENS = 512
MAX_WORD_CHARS = 100


def _is_punct(ch: str) -> bool:
    cp = ord(ch)
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True
    return unicodedata.category(ch).startswith("P")


def _is_cjk(cp: int) -> bool:
    return (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0x20000 <= cp <= 0x2A6DF
            or 0x2A700 <= cp <= 0x2B73F or 0x2B740 <= cp <= 0x2B81F or 0x2B820 <= cp <= 0x2CEAF
            or 0xF900 <= cp <= 0xFAFF or 0x2F800 <= cp <= 0x2FA1F)


def _normalise(text: str) -> str:
    out = []
    for ch in text:
        cp = ord(ch)
        if cp == 0 or cp == 0xFFFD:
            continue
        cat = unicodedata.category(ch)
        if ch in ("\t", "\n", "\r") or cat == "Zs":
            out.append(" ")
            continue
        if cat.startswith("C"):
            continue
        if _is_cjk(cp):
            out.append(f" {ch} ")
            continue
        out.append(ch)
    s = "".join(out).lower()
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def _pre_tokenise(text: str) -> list[str]:
    words, cur = [], []
    for ch in text:
        if ch.isspace():
            if cur:
                words.append("".join(cur))
                cur = []
        elif _is_punct(ch):
            if cur:
                words.append("".join(cur))
                cur = []
            words.append(ch)
        else:
            cur.append(ch)
    if cur:
        words.append("".join(cur))
    return words


class SemanticModel:
    def __init__(self, vocab: list[str], table: np.ndarray):
        self.vocab = {t: i for i, t in enumerate(vocab)}
        self.unk = self.vocab.get("[UNK]")
        self.table = table                      # (vocab, dim), unit rows are NOT assumed
        self.dim = table.shape[1]

    def tokenise(self, text: str) -> list[int]:
        ids: list[int] = []
        for word in _pre_tokenise(_normalise(text)):
            if len(word) > MAX_WORD_CHARS:
                ids.append(self.unk)
                continue
            start, pieces = 0, []
            while start < len(word):
                end, found = len(word), None
                while start < end:
                    sub = word[start:end] if start == 0 else "##" + word[start:end]
                    if sub in self.vocab:
                        found = self.vocab[sub]
                        break
                    end -= 1
                if found is None:
                    pieces = [self.unk]
                    break
                pieces.append(found)
                start = end
            ids.extend(pieces)
        ids = ids[:MAX_TOKENS]                  # truncation happens before [UNK] is dropped
        return [i for i in ids if i != self.unk]

    def encode(self, text: str) -> np.ndarray:
        """A unit vector for the text; all zeros when no token is known."""
        ids = self.tokenise(text)
        if not ids:
            return np.zeros(self.dim, np.float32)
        v = self.table[ids].astype(np.float32).mean(axis=0)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    def encode_many(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.encode(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)


CHUNK = 8192


def dot(matrix: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """
    matrix @ vec for unit-vector tables. Some BLAS builds (Accelerate on recent
    macOS) raise spurious divide/overflow flags on perfectly finite input; the
    result is correct, so the flags are silenced here and nowhere else.

    The tables are kept as float16 to halve their memory; they are multiplied in
    float32, a chunk of rows at a time, so no full float32 copy is ever made.
    """
    vec = np.asarray(vec, np.float32)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        if matrix.dtype != np.float16:
            out = matrix @ vec
        else:
            out = np.empty((matrix.shape[0],) + vec.shape[1:], np.float32)
            for s in range(0, matrix.shape[0], CHUNK):
                out[s:s + CHUNK] = matrix[s:s + CHUNK].astype(np.float32) @ vec
    return np.nan_to_num(out, nan=-1.0, posinf=1.0, neginf=-1.0)


def mm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a @ b with the same treatment as dot(), in float32: for small matrices and single pairs."""
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        out = a @ b
    return np.nan_to_num(out, nan=-1.0, posinf=1.0, neginf=-1.0)


@lru_cache(maxsize=1)
def get_model(root: Path = ART) -> SemanticModel:
    # One token per line, in id order (written from tokenizer.json by the build).
    vocab = (root / "semantic_vocab.txt").read_text(encoding="utf-8").rstrip("\n").split("\n")
    return SemanticModel(vocab, np.load(root / "semantic_tokens.npy"))


@lru_cache(maxsize=1)
def item_matrix(root: Path = ART) -> np.ndarray:
    """(items, dim) unit vectors for every film, kept as float16 (dot() multiplies in float32)."""
    return np.load(root / "semantic_items.npy").astype(np.float16, copy=False)
