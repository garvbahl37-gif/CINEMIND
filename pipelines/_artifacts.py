"""
Shared by the build steps: atomic writes, checksums and the artifact manifest.

Every step writes into an artifact directory (data/artifacts, or a staging copy
of it) through write_atomic, so an interrupted build never leaves a half-written
file, and records what it wrote in manifest.json — row counts, dimensions,
checksums and the versions it was built from. The API reads the manifest at
startup and refuses artifacts that disagree with the catalogue.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def write_atomic(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def save_npy(path: Path, arr: np.ndarray):
    buf = io.BytesIO()
    np.save(buf, arr)
    write_atomic(path, buf.getvalue())


def save_npz(path: Path, compressed: bool = True, **arrays):
    buf = io.BytesIO()
    (np.savez_compressed if compressed else np.savez)(buf, **arrays)
    write_atomic(path, buf.getvalue())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def forget(art: Path, name: str):
    """Drop an artifact that is no longer built from manifest.json."""
    path = art / "manifest.json"
    if path.exists():
        m = json.loads(path.read_text())
        if m["artifacts"].pop(name, None) is not None:
            write_atomic(path, json.dumps(m, indent=1, sort_keys=True).encode())


def record(art: Path, name: str, **meta):
    """Add or replace one artifact's entry in manifest.json, with its checksum."""
    path = art / "manifest.json"
    m = json.loads(path.read_text()) if path.exists() else {"artifacts": {}}
    entry = {"sha256": sha256(art / name), "bytes": (art / name).stat().st_size,
             "built": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), **meta}
    m["artifacts"][name] = entry
    write_atomic(path, json.dumps(m, indent=1, sort_keys=True).encode())
