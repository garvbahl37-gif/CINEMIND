"""
Write the home page's data as a static file, so the site opens without waiting
for the API to start.

    python pipelines/build_home.py [--art DIR] [--out frontend-app/public/home.json]

The home page is one prebuilt response — the hero picks, the top 50, series
and every genre shelf — that depends on the served artifacts alone. The API
serves it at /api/browse, but on a cold start that waits for the Python
function to load; as a file next to the frontend it comes from Vercel's CDN
with the rest of the site. It is written by the engine's own code
(api/_engine.py, Engine.home), and tests/test_discovery_and_adaptive.py fails
if it differs from what /api/browse answers, so the two cannot drift. The
frontend falls back to /api/browse if the file is missing.

rebuild.py runs this after promoting a build; run it by hand after anything
else that changes the served artifacts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "api"))
sys.path.insert(0, str(ROOT / "pipelines"))
from _artifacts import write_atomic  # noqa: E402
from _engine import ART, Engine  # noqa: E402

OUT = ROOT / "frontend-app" / "public" / "home.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--art", type=Path, default=ART)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    home = Engine(args.art).home()
    write_atomic(args.out, json.dumps(home, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    n = len(home["hero"]) + len(home["top50"]) + len(home["tv"]) + sum(len(r["items"]) for r in home["rows"])
    print(f"[home] {n} film cards, {len(home['rows'])} shelves; wrote {args.out} "
          f"({args.out.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
