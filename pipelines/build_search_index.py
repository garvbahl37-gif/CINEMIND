"""
Prebuild the API's search index, so a cold start loads it instead of building it.

    python pipelines/build_search_index.py [--art DIR]

The engine's search structures — folded titles, each film's title keys, and
posting lists by title key, title word, tag and person — come from the
catalogue alone. Building them took most of the engine's start-up: about 1.8 s
of 2.3 s on a laptop, and several times that on a serverless cold start.

This builds them with the engine's own code (api/_engine.py, build_search) and
writes them to search_index.npz as flat arrays, stamped with the SHA-256 of the
catalogue file they came from. The engine loads them only when that stamp
matches the catalogue it serves, and otherwise builds them itself, so an index
left behind by a catalogue change is never used: it costs only the time it was
meant to save. pipelines/validate_artifacts.py fails a build whose index is
missing, stale, or different from a fresh build.

rebuild.py runs this after build_catalog.py; run it by hand after anything else
that rewrites the catalogue.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "api"))
sys.path.insert(0, str(ROOT / "pipelines"))
from _artifacts import record, save_npz  # noqa: E402
from _engine import (ART, POSTINGS, SEARCH_FORMAT, SEARCH_INDEX, aliases, build_search,  # noqa: E402
                     catalog_path, file_sha256, load_catalog, pack_search)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--art", type=Path, default=ART, help="artifact directory: reads the catalogue, writes here")
    args = ap.parse_args()
    t0 = time.time()
    cat = catalog_path(args.art)
    c = load_catalog(args.art)
    ix = build_search(c, *aliases(c))
    sha = file_sha256(cat)
    save_npz(args.art / SEARCH_INDEX, **pack_search(ix, sha))
    sizes = {name: len(ix[name]) for name in POSTINGS}
    record(args.art, SEARCH_INDEX, format=SEARCH_FORMAT, catalog=cat.name, catalog_sha256=sha,
           films=len(ix["fold"]), keys=sizes)
    print(f"[search] {len(ix['fold']):,} films; keys {sizes}; wrote {SEARCH_INDEX} "
          f"({(args.art / SEARCH_INDEX).stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
