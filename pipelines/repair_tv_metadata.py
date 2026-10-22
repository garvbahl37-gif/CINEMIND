"""
Repair the catalogue entries whose TMDB metadata belongs to a different title.

The TMDB cache the catalogue was built from looked each film up by the tmdbId
in MovieLens' links.csv, and when the /movie/{id} lookup failed — a dead ID, or
an empty, rate-limited response — it retried /tv/{id} with the same number. On
TMDB that is a different namespace, so the retry "succeeded" with an unrelated
show: Raiders of the Lost Ark: The Adaptation came back as The Kitchen, a
cooking programme, and every such item was marked media=tv.

For each of those items this script takes the first of these that verifies:

  1. /movie/{tmdbId}, if the title matches what MovieLens calls the film;
  2. /tv/{tmdbId}, if the name matches exactly — for a real miniseries
     (Olive Kitteridge, Cosmos) MovieLens' ID *is* the TMDB series ID;
  3. a search of films and series by title and year: exact titles beat near
     ones, and among those the work more people have voted on wins. TMDB holds
     obscure namesakes — a 7-minute short also called Alice (2009) — and the
     one MovieLens viewers rated is the miniseries everyone voted on, not that.
     A candidate nobody on TMDB has voted for is not evidence of anything;

and accepts a result only when its title matches and its year is within one of
MovieLens'. An item nothing verifies keeps its title and ratings but loses the
borrowed artwork, synopsis and credits: showing nothing is better than showing
another film's.

The Series shelf in browse.json is rebuilt from the items still marked tv.

    TMDB_API_KEY=... python pipelines/repair_tv_metadata.py [--dry-run] [--root DIR] [--out DIR]

Items TMDB can't be reached for are skipped, never cleared; run it again (with
--root pointing at the previous output) to retry just those.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ART = Path(__file__).resolve().parent.parent / "data" / "artifacts"
API = "https://api.themoviedb.org/3"


def key(s: str) -> str:
    """A title reduced for comparison: accents, articles-last and punctuation gone."""
    s = unicodedata.normalize("NFKD", (s or "").lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\s*\(.*\)\s*$", "", s)
    m = re.match(r"^(.*), (the|a|an|les|la|le|il|el|der|die|das)$", s)
    if m:
        s = f"{m.group(2)} {m.group(1)}"
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def variants(title: str) -> set[str]:
    """The MovieLens title plus any alternates it carries in brackets."""
    out = {key(title)}
    for alt in re.findall(r"\(([^()]*)\)", title):
        out.add(key(re.sub(r"^a\.k\.a\.\s*", "", alt.strip(), flags=re.I)))
    return {v for v in out if v}


def match_quality(ml_title: str, *candidates: str) -> int | None:
    """
    0 for an exact title match, 1 for a near one, None for no match.

    A near match must also have the same number of words: "Helen of Troy" is
    a 2003 miniseries, and must not be taken for a film called "Helen of
    Troy's" just because the two strings are 93% alike.
    """
    ours = variants(ml_title)
    best = None
    for c in candidates:
        k = key(c)
        if not k:
            continue
        if k in ours:
            return 0
        if any(len(k.split()) == len(v.split()) and difflib.SequenceMatcher(None, k, v).ratio() >= 0.9
               for v in ours):
            best = 1
    return best


class Unreachable(RuntimeError):
    """TMDB could not be reached — not the same thing as "not found"."""


class TMDB:
    def __init__(self, api_key: str):
        self.k = api_key
        self.calls = 0

    def get(self, path: str, **params):
        params["api_key"] = self.k
        url = f"{API}{path}?{urllib.parse.urlencode(params)}"
        for attempt in range(6):
            self.calls += 1
            try:
                with urllib.request.urlopen(url, timeout=20) as r:
                    body = r.read()
                if not body:
                    raise ConnectionError("empty response")   # a glitch, not "no results"
                return json.loads(body)
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return None
                time.sleep(0.5 * (attempt + 1))         # 429 or 5xx: back off
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ConnectionError):
                time.sleep(0.5 * (attempt + 1))
        # Never let a network failure read as "no such title": that would clear
        # the metadata of an item that is fine. The caller skips the item instead.
        raise Unreachable(path)


def year_of(d: dict, field: str) -> int | None:
    v = d.get(field) or ""
    return int(v[:4]) if v[:4].isdigit() else None


def resolve(t: TMDB, title: str, year: int | None, tmdb_id: int | None, taken: set[int]):
    """
    (media, details) for the best candidate that verifies, else (None, None).

    `taken` holds the TMDB film IDs other catalogue entries already own. A film
    can only be one entry: Joan of Arc (1999) is the CBS miniseries, and must not
    become Besson's film just because TMDB lists that one's original title as
    "Joan of Arc" — the catalogue already has it, as The Messenger.
    """
    def close(y):
        return year is None or y is None or abs(y - year) <= 1

    if tmdb_id and tmdb_id not in taken:
        d = t.get(f"/movie/{tmdb_id}", append_to_response="credits")
        if d and match_quality(title, d.get("title"), d.get("original_title")) == 0 \
                and close(year_of(d, "release_date")):
            return "movie", d
    if tmdb_id:
        d = t.get(f"/tv/{tmdb_id}", append_to_response="credits")
        if d and match_quality(title, d.get("name"), d.get("original_name")) == 0 \
                and close(year_of(d, "first_air_date")):
            return "tv", d

    # Search films and series alike and weigh every candidate: exact titles
    # first, then the one with the most votes.
    cands = {}                                      # (media, id) -> (quality, -votes)
    query = key(title)
    for media, path, title_f, date_f, year_p in (
            ("movie", "/search/movie", ("title", "original_title"), "release_date", "year"),
            ("tv", "/search/tv", ("name", "original_name"), "first_air_date", "first_air_date_year")):
        for params in ({"query": query, year_p: year} if year else {"query": query}, {"query": query}):
            for r in ((t.get(path, **params) or {}).get("results") or [])[:10]:
                q = match_quality(title, *(r.get(f) for f in title_f))
                if q is None or not close(year_of(r, date_f)):
                    continue
                if media == "movie" and r["id"] in taken:
                    continue
                if not r.get("vote_count"):
                    continue                        # nobody voted: no evidence it's this work
                cands[(media, r["id"])] = min(cands.get((media, r["id"]), (9, 0)), (q, -r["vote_count"]))
    if not cands:
        return None, None
    (media, rid), _ = min(cands.items(), key=lambda kv: kv[1])
    d = t.get(f"/{media}/{rid}", append_to_response="credits")
    return (media, d) if d else (None, None)


def apply(c: dict, i: int, media: str | None, d: dict | None):
    if d is None:
        for f, empty in (("poster", None), ("backdrop", None), ("overview", ""),
                         ("runtime", None), ("tv", None), ("directors", []), ("cast", [])):
            c[f][i] = empty
        c["media"][i] = "movie"
        return
    credits = d.get("credits") or {}
    c["tmdbId"][i] = d["id"]
    c["media"][i] = media
    c["poster"][i] = d.get("poster_path")
    c["backdrop"][i] = d.get("backdrop_path")
    c["overview"][i] = d.get("overview") or ""
    if media == "movie":
        c["runtime"][i] = d.get("runtime") or None
        c["directors"][i] = [p["name"] for p in credits.get("crew", []) if p.get("job") == "Director"][:3]
    else:
        rt = d.get("episode_run_time") or []
        c["runtime"][i] = rt[0] if rt else None
        c["directors"][i] = [p["name"] for p in d.get("created_by", [])][:3]
    c["cast"][i] = [p["name"] for p in credits.get("cast", [])[:8]]
    c["lang"][i] = d.get("original_language") or c["lang"][i]
    c["tv"][i] = round(float(d["vote_average"]), 1) if d.get("vote_average") else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="report, but write nothing")
    ap.add_argument("--root", type=Path, default=ART, help="artifacts to read")
    ap.add_argument("--out", type=Path, default=None, help="where to write (default: --root)")
    args = ap.parse_args()
    out = args.out or args.root
    out.mkdir(parents=True, exist_ok=True)
    api_key = os.environ.get("TMDB_API_KEY")
    if not api_key:
        sys.exit("Set TMDB_API_KEY.")

    c = json.loads((args.root / "catalog.json").read_text())
    b = json.loads((args.root / "browse.json").read_text())
    # Items already settled by an earlier run into the same --out are not asked again.
    state = out / "repair_state.json"
    done = set(json.loads(state.read_text())) if state.exists() else set()
    todo = [i for i, m in enumerate(c["media"]) if m == "tv" and i not in done]
    print(f"[repair] {len(todo)} items marked tv still to check ({len(done)} settled earlier)")

    # TMDB films already owned by an entry outside this repair.
    todo_set = set(todo)
    taken = {c["tmdbId"][j] for j in range(len(c["title"]))
             if c["media"][j] == "movie" and c["tmdbId"][j] and j not in todo_set}

    t = TMDB(api_key)

    def lookup(i):
        try:
            return i, resolve(t, c["title"][i], c["year"][i], c["tmdbId"][i], taken)
        except Unreachable:
            return i, None

    # Lookups run in parallel; results are applied in catalogue order, so the
    # output is the same however the requests interleave.
    with ThreadPoolExecutor(max_workers=8) as pool:
        resolved = dict(pool.map(lookup, todo))

    tally = {"movie": 0, "tv": 0, "cleared": 0, "skipped": 0}
    for n, i in enumerate(todo, 1):
        title, year = c["title"][i], c["year"][i]
        if resolved[i] is None:
            tally["skipped"] += 1
            print(f"  {n:3}/{len(todo)} skipped {title[:48]:48} {year}  (TMDB unreachable; re-run to retry)")
            continue
        media, d = resolved[i]
        if media == "movie" and d and d["id"] in taken:
            media, d = None, None             # another entry in this run claimed it first
        apply(c, i, media, d)
        if media == "movie" and d:
            taken.add(d["id"])
        done.add(i)
        outcome = media or "cleared"
        tally[outcome] += 1
        found = (d.get("title") or d.get("name")) if d else "-"
        print(f"  {n:3}/{len(todo)} {outcome:7} {title[:48]:48} {year}  ->  {found}")

    # Series shelf: what is still TV, best first, with artwork.
    bayes = b["bayes"]
    b["tv"] = sorted((i for i, m in enumerate(c["media"]) if m == "tv" and c["poster"][i]),
                     key=lambda i: -bayes[i])[:40]
    # Nothing on the home page may point at an item that just lost its artwork.
    for k in ("hero", "top50"):
        b[k] = [i for i in b[k] if c["poster"][i]]
    b["genres"] = {g: [i for i in ids if c["poster"][i]] for g, ids in b["genres"].items()}

    print(f"[repair] movie {tally['movie']}, tv {tally['tv']}, cleared {tally['cleared']}, "
          f"skipped {tally['skipped']} "
          f"({t.calls} TMDB calls); series shelf now {len(b['tv'])}")
    if args.dry_run:
        print("[repair] dry run: nothing written")
        return
    # Same serialisation as build_artifacts.py, so the diff is only the repair.
    (out / "catalog.json").write_text(json.dumps(c, separators=(",", ":")))
    (out / "browse.json").write_text(json.dumps(b, separators=(",", ":")))
    if out.resolve() != ART.resolve():
        state.write_text(json.dumps(sorted(done)))
    print(f"[repair] wrote catalog.json and browse.json to {out}")


if __name__ == "__main__":
    main()
