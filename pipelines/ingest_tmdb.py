"""
Fetch TMDB details for MovieLens films, in rate-limited, resumable batches.

    TMDB_API_KEY=... python pipelines/ingest_tmdb.py --ml32m data/raw/ml-32m [--scope added|served|all]
        [--refresh-days 180]

The key is read from the environment and never written anywhere.

WHAT IT FETCHES
One request per film: /movie/{tmdbId}?append_to_response=credits,keywords,
reduced to the fields the catalogue uses (titles, release date, runtime,
language, countries, genres, overview, artwork paths, vote average and count,
popularity, keywords, directors, top-billed cast) plus the time it was fetched.
Only the movie namespace is asked: falling back to /tv/{id} for a missing film
is what once gave "Raiders of the Lost Ark: The Adaptation" another title's
poster (see repair_tv_metadata.py).

WHICH FILMS
  added   the MovieLens films beyond the 17,719 served before the expansion
  served  those 17,719 (their films, not their series: series ids are TMDB's
          TV namespace, and this script only asks the movie one)
  all     both (the default)
Each scope is batched on its own, so a rerun finds its finished batches.

TERMS
TMDB's API terms (2023-10-20) limit caching to six months (§1.C). Every record
carries `fetched`; --refresh-days 180 refetches the ones older than that, so a
monthly run keeps the cache inside the limit. They also require the TMDB logo
and notice (§3). They also forbid use "in connection with ... a
machine learning ... based Application" without a separate agreement (§1.C,
§2.A); the catalogue keeps every TMDB-derived field attributable to TMDB so it
can be withheld from the models. See docs/AUDIT.md.

BATCHES
Ids are fetched in batches (default 500) at a capped request rate (default
20/s across all threads). Each finished batch is one file in
data/raw/tmdb/movie/, written atomically. A rerun skips finished batches and,
within a batch, refetches only the films that could not be reached. "Not
found" (404) is recorded as such; a network failure is never recorded as "not
found".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
API = "https://api.themoviedb.org/3"
sys.path.insert(0, str(ROOT / "api"))
from _engine import load_catalog  # noqa: E402


class Unreachable(Exception):
    """TMDB could not be reached — not the same thing as "not found"."""


class Pace:
    """A shared limit on requests per second, across every thread."""

    def __init__(self, per_second: float):
        self.gap = 1.0 / per_second
        self.next = time.monotonic()
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next)
            self.next = t + self.gap
        if t > now:
            time.sleep(t - now)


def get(path: str, key: str, pace: Pace, **params) -> dict | None:
    params["api_key"] = key
    url = f"{API}{path}?{urllib.parse.urlencode(params)}"
    for attempt in range(7):
        pace.wait()
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                body = r.read()
            if not body:
                raise ConnectionError("empty response")      # a glitch, not "no film"
            return json.loads(body)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code == 401:
                raise SystemExit("TMDB rejected the API key (401). Check TMDB_API_KEY.")
            time.sleep(float(e.headers.get("Retry-After") or 0) or 0.5 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ConnectionError, OSError):
            time.sleep(0.5 * (attempt + 1))
    raise Unreachable(path)


def compact(d: dict, fetched: str) -> dict:
    """The fields the catalogue uses, and when they were fetched."""
    credits = d.get("credits") or {}
    kw = (d.get("keywords") or {}).get("keywords") or []
    return {
        "status": "ok", "fetched": fetched,
        "id": d.get("id"), "imdb_id": d.get("imdb_id"),
        "title": d.get("title"), "original_title": d.get("original_title"),
        "release_date": d.get("release_date") or None,
        "runtime": d.get("runtime") or None,
        "original_language": d.get("original_language") or None,
        "origin_country": d.get("origin_country") or [],
        "production_countries": [c.get("iso_3166_1") for c in d.get("production_countries") or [] if c.get("iso_3166_1")],
        "genres": [g.get("name") for g in d.get("genres") or [] if g.get("name")],
        "overview": d.get("overview") or "",
        "poster_path": d.get("poster_path"), "backdrop_path": d.get("backdrop_path"),
        "vote_average": d.get("vote_average"), "vote_count": d.get("vote_count"),
        "popularity": d.get("popularity"), "adult": d.get("adult"), "status_text": d.get("status"),
        "keywords": [k.get("name") for k in kw[:15] if k.get("name")],
        "directors": [p["name"] for p in credits.get("crew", []) if p.get("job") == "Director"][:3],
        "cast": [p["name"] for p in sorted(credits.get("cast", []), key=lambda p: p.get("order", 999))[:8]],
    }


def batch_path(out: Path, i: int, ids: list[int]) -> Path:
    h = hashlib.sha1(" ".join(map(str, ids)).encode()).hexdigest()[:10]
    return out / f"batch-{i:05d}-{h}.json"


def write_atomic(path: Path, payload: dict):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
    os.replace(tmp, path)


def run_batch(i: int, ids: list[int], out: Path, key: str, pace: Pace, threads: int,
              stale_before: str | None = None) -> tuple[int, int, int]:
    path = batch_path(out, i, ids)
    have = json.loads(path.read_text()) if path.exists() else {"ids": ids, "results": {}, "unreachable": ids}
    unreachable = set(map(str, have.get("unreachable", [])))
    todo = [t for t in ids if str(t) in unreachable or str(t) not in have["results"]
            or (stale_before and (have["results"][str(t)].get("fetched") or "") < stale_before)]
    if not todo:
        return 0, 0, 0

    def one(tid: int):
        fetched = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            d = get(f"/movie/{tid}", key, pace, append_to_response="credits,keywords", language="en-US")
        except Unreachable:
            return tid, None
        if d is None:
            return tid, {"status": "not_found", "fetched": fetched}
        if d.get("id") != tid:   # never file one film's data under another's id
            return tid, {"status": "mismatch", "fetched": fetched, "returned_id": d.get("id")}
        return tid, compact(d, fetched)

    with ThreadPoolExecutor(threads) as pool:
        got = list(pool.map(one, todo))
    unreachable = [tid for tid, rec in got if rec is None]
    for tid, rec in got:
        if rec is not None:
            have["results"][str(tid)] = rec
    have["unreachable"] = unreachable
    write_atomic(path, have)
    ok = sum(1 for _, r in got if r and r["status"] == "ok")
    return ok, sum(1 for _, r in got if r and r["status"] != "ok"), len(unreachable)


def target_ids(ml32m: Path, scope: str) -> list[int]:
    """TMDB movie ids for the films the scope names (see WHICH FILMS)."""
    art = ROOT / "data" / "artifacts"
    legacy = [int(m) for m in np.load(art / "item_classes.npy")]
    if scope == "added":
        links = pd.read_csv(ml32m / "links.csv", dtype={"tmdbId": "Int64"})
        links = links[~links.movieId.isin(set(legacy))]
        return sorted({int(t) for t in links.tmdbId.dropna()})
    c = load_catalog(art)
    return sorted({int(c["tmdbId"][i]) for i in range(len(legacy))
                   if c["media"][i] == "movie" and c["tmdbId"][i] is not None})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ml32m", type=Path, required=True)
    ap.add_argument("--scope", choices=["added", "served", "all"], default="all",
                    help="added: films beyond the original 17,719; served: those 17,719; all: both")
    ap.add_argument("--refresh-days", type=float, default=None,
                    help="also refetch records fetched more than this many days ago (TMDB allows six months)")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "raw" / "tmdb" / "movie")
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--rate", type=float, default=20.0, help="requests per second, all threads together")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only the first N ids (for a trial run)")
    args = ap.parse_args()
    key = os.environ.get("TMDB_API_KEY")
    if not key:
        sys.exit("Set TMDB_API_KEY (it is read from the environment and never stored).")

    stale_before = None
    if args.refresh_days is not None:
        stale_before = datetime.fromtimestamp(time.time() - args.refresh_days * 86400, timezone.utc) \
            .strftime("%Y-%m-%dT%H:%M:%SZ")
    args.out.mkdir(parents=True, exist_ok=True)
    pace = Pace(args.rate)
    t0, tot = time.time(), [0, 0, 0]
    for scope in (("added", "served") if args.scope == "all" else (args.scope,)):
        ids = target_ids(args.ml32m, scope)
        if args.limit:
            ids = ids[:args.limit]
        groups = [ids[i:i + args.batch] for i in range(0, len(ids), args.batch)]
        print(f"[tmdb] {scope}: {len(ids):,} TMDB ids in {len(groups)} batches of {args.batch}, "
              f"at most {args.rate:g} requests/s"
              + (f", refetching records from before {stale_before}" if stale_before else ""), flush=True)
        for i, g in enumerate(groups):
            ok, other, unreachable = run_batch(i, g, args.out, key, pace, args.threads, stale_before)
            tot[0] += ok; tot[1] += other; tot[2] += unreachable
            if ok or other or unreachable:
                print(f"[tmdb] {scope} batch {i + 1}/{len(groups)}: {ok} fetched, {other} not found or "
                      f"mismatched, {unreachable} unreachable  ({time.time() - t0:.0f}s)", flush=True)
    print(f"[tmdb] done: {tot[0]:,} fetched, {tot[1]:,} not found/mismatched, {tot[2]:,} unreachable"
          + (" — rerun to retry them" if tot[2] else ""), flush=True)
    return 1 if tot[2] else 0


if __name__ == "__main__":
    sys.exit(main())
