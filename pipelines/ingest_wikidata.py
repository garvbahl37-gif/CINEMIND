"""
Fetch film metadata from Wikidata for every MovieLens film, by IMDb id.

    python pipelines/ingest_wikidata.py --ml32m data/raw/ml-32m

WHY WIKIDATA
Wikidata is CC0 (public domain), so its facts may be used anywhere, including
in the models. TMDB's API terms forbid using TMDB content "in connection with
... a machine learning (ML) or artificial intelligence (AI) based Application"
(§1.C, 2023-10-20), so new films take no content from TMDB. IMDb's datasets are
for personal, non-commercial use and not for republishing, so only IMDb
*identifiers* are used — the ones MovieLens ships in links.csv.

WHAT IT FETCHES (per film, matched on P345 "IMDb ID")
  core     item, English label and description, instance-of, earliest release
           date (P577), duration with its unit (P2047), sitelink count
  people   directors (P57) and cast (P161), truthy statements, each with an
           English label and sitelink count
  facets   country of origin (P495 → ISO 3166-1), original language
           (P364 → ISO 639-1), genre (P136, English label)

RESUMABLE
Each batch of IMDb ids is one query per family, cached as
data/raw/wikidata/<family>/<batch>.json with the query and its fetch time,
written atomically. A rerun skips every batch already on disk, so an
interrupted run picks up where it stopped and nothing is downloaded twice.
Failed batches are retried with backoff and, if they still fail, reported and
left for the next run — never written as empty.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ENDPOINT = "https://query.wikidata.org/sparql"
# Wikimedia asks every client to identify itself.
UA = "CINEMIND-catalog/1.0 (https://github.com/garvbahl37-gif/CINEMIND; catalogue ingestion)"

QUERIES = {
    "core": """
SELECT ?imdb ?film (SAMPLE(?lab) AS ?label) (SAMPLE(?ds) AS ?desc) (SAMPLE(?sl) AS ?sitelinks)
       (GROUP_CONCAT(DISTINCT STRAFTER(STR(?inst), "entity/"); separator="|") AS ?instances)
       (MIN(?date) AS ?released)
       (GROUP_CONCAT(DISTINCT CONCAT(STR(?amt), "@", STRAFTER(STR(?unit), "entity/")); separator="|") AS ?durations)
WHERE {
  VALUES ?imdb { %s }
  ?film wdt:P345 ?imdb .
  OPTIONAL { ?film rdfs:label ?lab FILTER(LANG(?lab) = "en") }
  OPTIONAL { ?film schema:description ?ds FILTER(LANG(?ds) = "en") }
  OPTIONAL { ?film wikibase:sitelinks ?sl }
  OPTIONAL { ?film wdt:P31 ?inst }
  OPTIONAL { ?film wdt:P577 ?date }
  OPTIONAL { ?film p:P2047/psv:P2047 [ wikibase:quantityAmount ?amt ; wikibase:quantityUnit ?unit ] }
}
GROUP BY ?imdb ?film""",
    # Truthy statements only (wdt:), which already leave out deprecated ones.
    # Walking full statements to read ordinal qualifiers timed out at 60 s per
    # batch; cast is ordered by each person's sitelink count instead.
    "people": """
SELECT ?imdb ?film ?role ?person ?name ?sl WHERE {
  VALUES ?imdb { %s }
  ?film wdt:P345 ?imdb .
  { ?film wdt:P57 ?person . BIND("d" AS ?role) } UNION { ?film wdt:P161 ?person . BIND("c" AS ?role) }
  OPTIONAL { ?person rdfs:label ?name FILTER(LANG(?name) = "en") }
  OPTIONAL { ?person wikibase:sitelinks ?sl }
}""",
    "facets": """
SELECT ?imdb ?film ?kind ?value WHERE {
  VALUES ?imdb { %s }
  ?film wdt:P345 ?imdb .
  { ?film wdt:P495 ?c . ?c wdt:P297 ?value . BIND("country" AS ?kind) }
  UNION
  { ?film wdt:P364 ?l . ?l wdt:P218 ?value . BIND("lang" AS ?kind) }
  UNION
  { ?film wdt:P136 ?g . ?g rdfs:label ?value FILTER(LANG(?value) = "en") BIND("genre" AS ?kind) }
}""",
}


def imdb_ids(ml32m: Path) -> list[str]:
    """Every film's IMDb id as tt-prefixed, zero-padded to seven digits, sorted."""
    links = pd.read_csv(ml32m / "links.csv", dtype={"imdbId": str})
    ids = {f"tt{int(x):07d}" for x in links.imdbId.dropna() if str(x).isdigit()}
    return sorted(ids)


def batches(ids: list[str], size: int) -> list[list[str]]:
    return [ids[i:i + size] for i in range(0, len(ids), size)]


def batch_name(i: int, ids: list[str]) -> str:
    # The hash ties a cached file to its exact ids, so a changed id list
    # can never be mistaken for one already fetched.
    return f"{i:05d}-{hashlib.sha1(' '.join(ids).encode()).hexdigest()[:10]}"


def query(sparql: str, tries: int = 6) -> list[dict]:
    body = urllib.parse.urlencode({"query": sparql, "format": "json"}).encode()
    for attempt in range(tries):
        try:
            req = urllib.request.Request(ENDPOINT, data=body, headers={
                "User-Agent": UA, "Accept": "application/sparql-results+json",
                "Content-Type": "application/x-www-form-urlencoded"})
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.load(r)["results"]["bindings"]
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504):
                raise
            wait = float(e.headers.get("Retry-After") or 0) or 2 ** attempt + random.random()
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError):
            wait = 2 ** attempt + random.random()
        time.sleep(min(wait, 60))
    raise RuntimeError("gave up after retries")


def write_atomic(path: Path, payload: dict):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":")))
    os.replace(tmp, path)


def fetch(family: str, i: int, ids: list[str], out: Path) -> tuple[str, int, str]:
    path = out / family / f"{batch_name(i, ids)}.json"
    if path.exists():
        return family, i, "cached"
    sparql = QUERIES[family] % " ".join(f'"{x}"' for x in ids)
    rows = query(sparql)
    write_atomic(path, {"family": family, "ids": ids,
                        "fetched": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "query": QUERIES[family].strip(), "rows": rows})
    return family, i, f"{len(rows)} rows"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ml32m", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "raw" / "wikidata")
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--workers", type=int, default=3, help="Wikidata allows a few parallel queries per client")
    ap.add_argument("--families", default="core,people,facets")
    ap.add_argument("--limit", type=int, default=0, help="only the first N ids (for a dry run)")
    args = ap.parse_args()

    ids = imdb_ids(args.ml32m)
    if args.limit:
        ids = ids[:args.limit]
    families = [f for f in args.families.split(",") if f]
    for f in families:
        (args.out / f).mkdir(parents=True, exist_ok=True)
    jobs = [(f, i, b) for f in families for i, b in enumerate(batches(ids, args.batch))]
    print(f"[wikidata] {len(ids):,} IMDb ids, {len(jobs):,} batch queries ({args.batch} ids each)", flush=True)

    t0, done, failed = time.time(), 0, []
    with ThreadPoolExecutor(args.workers) as pool:
        futs = {pool.submit(fetch, f, i, b, args.out): (f, i) for f, i, b in jobs}
        for fut in as_completed(futs):
            f, i = futs[fut]
            try:
                fut.result()
            except Exception as e:  # reported, and left for the next run
                failed.append((f, i, repr(e)[:120]))
            done += 1
            if done % 50 == 0 or done == len(jobs):
                print(f"[wikidata] {done:,}/{len(jobs):,} batches, {len(failed)} failed, "
                      f"{time.time() - t0:.0f}s", flush=True)
    for f, i, err in failed[:20]:
        print(f"[wikidata] FAILED {f} batch {i}: {err}", flush=True)
    print(f"[wikidata] finished: {len(jobs) - len(failed):,} of {len(jobs):,} batches on disk"
          + (" — rerun to fetch the rest" if failed else ""), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
