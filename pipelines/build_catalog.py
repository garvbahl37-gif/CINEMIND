"""
Build the expanded catalogue: every valid MovieLens 32M film, with stable ids.

    python pipelines/build_catalog.py --ml32m data/raw/ml-32m

INPUTS
  data/artifacts/catalog.json.gz   the catalogue being extended; its rows keep
  data/artifacts/item_classes.npy  their item_ids (checked against the encoder)
  data/raw/ml-32m                  films, IMDb/TMDB links, ratings, tags
  data/raw/tmdb/movie/             ingest_tmdb.py's cache (optional)
  data/raw/wikidata/               ingest_wikidata.py's cache (optional)

IDS
`item_id` is the canonical id. The films already served keep theirs (0–17,718)
— browsers store them for likes and watchlists — and every new film is
appended in MovieLens-id order. `id_map.json` maps every MovieLens id to its
item_id, including ids merged as duplicates, so ratings for either land on the
same film.

DUPLICATES
Two MovieLens ids are one film only when they share a TMDB id *and* their years
agree (within one) *and* their titles match closely. A shared TMDB id without
that agreement is a conflict: both films stay, and neither takes the shared
TMDB record unless it matches that record's own title and year. Titles alone
never merge anything. When both duplicates were already served, both ids may
be stored in browsers, so neither row can go: the second gets `alias_of`, the
first's item_id, and is never listed — search, matches, home rows and
recommendations skip it — while its own id still resolves.

FIELDS AND SOURCES (first that is present and validated)
  title, year     MovieLens title  → TMDB release date → Wikidata release date
  genres          MovieLens        → TMDB (mapped)     → Wikidata (mapped)
  tags            MovieLens (top 8 by frequency)
  ratings         MovieLens 32M count and mean, duplicates folded in
  artwork, synopsis, keywords, vote average      TMDB
  runtime, language, countries, directors, cast  TMDB → Wikidata
A TMDB record is used only if its IMDb id matches MovieLens' (when both exist)
and its year is within two of the MovieLens year; a Wikidata record only if its
year is within two. Nothing is filled in from guesswork: missing stays missing.

The films already served keep their titles, years, genres and tags. Their
artwork, synopsis, runtime, language and credits come from a fresh TMDB record
when one validates (fetched with --scope served), field by field, and from the
old cache — marked as of unknown date — where it doesn't. Series keep their
old data. Every served film gains an IMDb id, countries and folded-in ratings.

OUTPUTS (written atomically)
  data/artifacts/catalog.json.gz     columnar, the serving schema plus imdbId,
                                     countries, released
  data/artifacts/id_map.json         MovieLens id → item_id
  data/artifacts/provenance.json.gz  per film: sources, field sources, fetch
                                     times, Wikidata id, merged ids, flags
  data/raw/build/features.json       build-time text features (keywords,
                                     descriptions) for the embedding step
  data/artifacts/data_quality.json   and docs/DATA_QUALITY.md
"""
from __future__ import annotations

import argparse
import difflib
import glob
import gzip
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipelines"))
sys.path.insert(0, str(ROOT / "api"))
from _artifacts import forget, record  # noqa: E402
from build_artifacts import clean_title  # noqa: E402
from _engine import _key, load_catalog, title_variants  # noqa: E402

ART = ROOT / "data" / "artifacts"
RAW = ROOT / "data" / "raw"
LEGACY_N = 17_719
YEAR_MIN, YEAR_MAX = 1874, 2026

TMDB_GENRES = {
    "Action": "Action", "Adventure": "Adventure", "Animation": "Animation", "Comedy": "Comedy",
    "Crime": "Crime", "Documentary": "Documentary", "Drama": "Drama", "Family": "Children",
    "Fantasy": "Fantasy", "Horror": "Horror", "Mystery": "Mystery", "Romance": "Romance",
    "Science Fiction": "Sci-Fi", "Thriller": "Thriller", "War": "War", "Western": "Western",
    # History, Music and TV Movie have no MovieLens counterpart and are left out.
}
# Wikidata genre labels ("comedy-drama film", "science fiction film") to MovieLens genres.
WD_GENRES = [
    (r"\bdrama\b", "Drama"), (r"\bcomedy\b", "Comedy"), (r"\bhorror\b", "Horror"),
    (r"\bthriller\b", "Thriller"), (r"\bdocumentary\b", "Documentary"), (r"\banimat", "Animation"),
    (r"\bscience fiction\b", "Sci-Fi"), (r"\bfantasy\b", "Fantasy"), (r"\bromanc|\bromantic\b", "Romance"),
    (r"\bcrime\b", "Crime"), (r"\bwar film\b", "War"), (r"\bwestern\b", "Western"),
    (r"\bmusical\b", "Musical"), (r"\bmystery\b", "Mystery"), (r"\baction\b", "Action"),
    (r"\badventure\b", "Adventure"), (r"\bfilm noir\b", "Film-Noir"), (r"\bchildren'?s\b|\bfamily film\b", "Children"),
]
TV_CLASSES = {"Q5398426", "Q1259759", "Q15416", "Q21191270", "Q63952888"}   # series, miniseries, programme, episode
UNIT_MIN = {"Q7727": 1.0, "Q11574": 1 / 60, "Q25235": 60.0}                   # minute, second, hour
POSTER = re.compile(r"^/[A-Za-z0-9_-]+\.(?:jpg|jpeg|png|svg)$")


def log(msg):
    print(f"[catalog] {msg}", flush=True)


def write_atomic(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def dump(obj) -> bytes:
    # ASCII with \u escapes: 0.2% of characters are non-ASCII, but one in a file makes
    # Python hold the whole text two bytes a character while the API parses it.
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=True).encode()


# -- sources --------------------------------------------------------------------

def movielens(ml32m: Path):
    mv = pd.read_csv(ml32m / "movies.csv")
    ln = pd.read_csv(ml32m / "links.csv", dtype={"imdbId": str, "tmdbId": "Int64"})
    mv = mv.merge(ln, on="movieId", how="left").set_index("movieId")
    cache = RAW / "build" / "ml32m_stats.npz"
    if cache.exists():
        z = np.load(cache)
        stats = pd.DataFrame({"count": z["count"], "sum": z["sum"]}, index=z["movieId"])
    else:
        log("rating statistics from ml-32m (once; cached) ...")
        r = pd.read_csv(ml32m / "ratings.csv", usecols=["movieId", "rating"],
                        dtype={"movieId": np.int32, "rating": np.float32})
        g = r.groupby("movieId")["rating"]
        stats = pd.DataFrame({"count": g.count(), "sum": g.sum()})
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, movieId=stats.index.values, count=stats["count"].values, sum=stats["sum"].values)
    t = pd.read_csv(ml32m / "tags.csv", usecols=["movieId", "tag"])
    t["tag"] = t.tag.astype(str).str.lower().str.strip()
    t = t[t.tag.str.len().between(2, 30)]
    tc = (t.groupby(["movieId", "tag"]).size().reset_index(name="n")
            .sort_values(["movieId", "n", "tag"], ascending=[True, False, True])
            .groupby("movieId").head(8))
    tags = tc.groupby("movieId")["tag"].apply(list).to_dict()
    return mv, stats, tags


def tmdb_cache() -> dict[int, dict]:
    """TMDB id → its record. An id can sit in more than one batch (a film merged as a
    duplicate, a refresh): a found record beats a failed one, then the newest wins."""
    out: dict[int, dict] = {}
    for f in sorted(glob.glob(str(RAW / "tmdb" / "movie" / "batch-*.json"))):
        for k, v in json.loads(Path(f).read_text())["results"].items():
            old = out.get(int(k))
            if old is None or ((v.get("status") == "ok", v.get("fetched") or "") >=
                               (old.get("status") == "ok", old.get("fetched") or "")):
                out[int(k)] = v
    return out


def wikidata_cache() -> dict[str, dict]:
    """IMDb id → merged Wikidata facts; ids that match several items are marked ambiguous."""
    films: dict[str, dict] = {}
    qids: dict[str, set] = defaultdict(set)
    for fam in ("core", "people", "facets"):
        for f in sorted(glob.glob(str(RAW / "wikidata" / fam / "*.json"))):
            batch = json.loads(Path(f).read_text())
            for b in batch["rows"]:
                v = {k: x["value"] for k, x in b.items()}
                imdb = v["imdb"]
                qids[imdb].add(v["film"].rsplit("/", 1)[-1])
                d = films.setdefault(imdb, {"fetched": batch["fetched"], "directors": {}, "cast": {},
                                            "countries": set(), "langs": set(), "genres": set()})
                d["fetched"] = min(d["fetched"], batch["fetched"])
                if fam == "core":
                    d.update(qid=v["film"].rsplit("/", 1)[-1], label=v.get("label"), desc=v.get("desc"),
                             sitelinks=int(v.get("sitelinks") or 0),
                             instances=set((v.get("instances") or "").split("|")) - {""},
                             released=(v.get("released") or "")[:10] or None,
                             durations=[x for x in (v.get("durations") or "").split("|") if x])
                elif fam == "people":
                    who = d["directors"] if v["role"] == "d" else d["cast"]
                    if v.get("name"):
                        who[v["name"]] = max(who.get(v["name"], 0), int(v.get("sl") or 0))
                else:
                    {"country": d["countries"], "lang": d["langs"], "genre": d["genres"]}[v["kind"]].add(v["value"])
    for imdb, q in qids.items():
        films[imdb]["ambiguous"] = len(q) > 1
    return films


# -- helpers --------------------------------------------------------------------

def imdb_of(raw) -> str | None:
    s = str(raw).strip() if raw is not None and not pd.isna(raw) else ""
    return f"tt{int(s):07d}" if s.isdigit() else None


def year_of(date: str | None) -> int | None:
    if date and re.match(r"^\d{4}", date):
        y = int(date[:4])
        return y if YEAR_MIN <= y <= YEAR_MAX else None
    return None


def same_title(a: str, b: str) -> bool:
    ka, kb = _key(a), _key(b)
    return bool(ka and kb) and (ka == kb or ka in kb or kb in ka or difflib.SequenceMatcher(None, ka, kb).ratio() >= 0.75)


def same_film(title_a: str, year_a, title_b: str, year_b) -> tuple[bool, str]:
    """Whether two records that share a TMDB id are one film: years within one
    (when both are known) and titles that match. A shared id alone is not enough."""
    if year_a is not None and year_b is not None and abs(int(year_a) - int(year_b)) > 1:
        return False, "years differ"
    if not same_title(title_a, title_b):
        return False, "titles differ"
    return True, ""


def title_forms(title: str) -> set[str]:
    """A title as written and in MovieLens' alternate forms ("Pokemon 4 Ever (a.k.a.
    Pokémon 4: The Movie)" is also "pokemon 4 ever")."""
    return {title} | title_variants(title)


def main_title(title: str) -> str:
    return re.split(r"\s*:\s+|\s+[-–—]\s+", title)[0]


def record_titles_match(ml_title: str, names) -> bool:
    """Whether a source's record carries this film's title, in any of its forms — or the
    same title with one side's subtitle dropped ("Nightmares in Red White & Blue: The
    Evolution of ..." is TMDB's "Nightmares in Red, White and Blue"). Only one side, and
    closely: "Louis Theroux: Law & Disorder" is not "Louis Theroux: Law and Disorder in
    Johannesburg". For checking a record, never for merging two films."""
    ours = title_forms(ml_title)
    theirs = {f for x in names if x for f in title_forms(x)}
    if any(same_title(a, b) for a in ours for b in theirs):
        return True
    close = lambda a, b: difflib.SequenceMatcher(None, _key(a), _key(b)).ratio() >= 0.85  # noqa: E731
    return any(close(main_title(a), b) or close(a, main_title(b)) for a in ours for b in theirs)


def tmdb_mismatch(ml_title: str, ml_year, imdb: str | None, t: dict) -> str | None:
    """Why TMDB record t is not this MovieLens film — or None when it is.

    Matching IMDb ids settle it whatever the years say: production and release years
    differ (Andrei Rublev: 1966, 1969). A different IMDb id is accepted only with the
    same title (in any of MovieLens' alternate forms) and year — an outdated duplicate
    id on one side. With no IMDb id to compare, the year must be within two."""
    ty = year_of(t.get("release_date"))
    if imdb and t.get("imdb_id"):
        if t["imdb_id"] == imdb:
            return None
        close = ml_year is None or ty is None or abs(ml_year - ty) <= 1
        if close and record_titles_match(ml_title, (t.get("title"), t.get("original_title"))):
            return None
        return "imdb_mismatch"
    if ml_year and ty and abs(ml_year - ty) > 2:
        return "year_conflict"
    return None


def wd_runtime(durations: list[str]) -> float | None:
    vals = []
    for d in durations:
        amt, _, unit = d.partition("@")
        try:
            if unit in UNIT_MIN:
                vals.append(float(amt) * UNIT_MIN[unit])
        except ValueError:
            pass
    return round(float(np.median(vals))) if vals else None


def wd_genres(labels) -> list[str]:
    out = []
    for lab in sorted(labels):
        for pat, g in WD_GENRES:
            if re.search(pat, lab.lower()) and g not in out:
                out.append(g)
    return out


def top_people(d: dict, k: int) -> list[str]:
    """Wikidata lists cast unordered; the best-known (most sitelinks) come first."""
    return [n for n, _ in sorted(d.items(), key=lambda x: (-x[1], x[0]))[:k]]


# -- the build ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ml32m", type=Path, required=True)
    ap.add_argument("--base", type=Path, default=ART,
                    help="the catalogue being extended (its file, or the artifact directory holding it): "
                         "every row keeps its item_id")
    ap.add_argument("--out", type=Path, default=ART)
    args = ap.parse_args()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    base = load_catalog(args.base)
    classes = np.load(ART / "item_classes.npy")
    if [int(m) for m in base["movieId"][:LEGACY_N]] != [int(m) for m in classes]:
        sys.exit("The base catalogue's first 17,719 rows do not match item_classes.npy — "
                 "refusing to build on a catalogue whose ids have moved.")
    legacy = {k: v[:LEGACY_N] for k, v in base.items() if isinstance(v, list)}

    mv, stats, tags = movielens(args.ml32m)
    tm = tmdb_cache()
    wd = wikidata_cache()
    log(f"MovieLens {len(mv):,} films · TMDB cache {len(tm):,} · Wikidata {len(wd):,} IMDb ids")

    flags: dict[int, set] = defaultdict(set)
    legacy_ids = [int(m) for m in classes]
    legacy_set = set(legacy_ids)
    LEGACY_POS = {m: i for i, m in enumerate(legacy_ids)}
    # Ids are append-only: every film the base catalogue already serves keeps its
    # row, so a later build can only add films at the end, never shift one.
    served_ids = [int(m) for m in base["movieId"]]
    served_set = set(served_ids)
    if any(m not in mv.index for m in served_ids[LEGACY_N:]):
        sys.exit("A film added by an earlier build is missing from MovieLens 32M — refusing to drop a served id.")

    def fallback_title(mid: int) -> tuple[str | None, str | None]:
        """A title from a source matched by IMDb id, for films MovieLens leaves untitled."""
        row = mv.loc[mid]
        imdb = imdb_of(row["imdbId"])
        t = tm.get(int(row["tmdbId"])) if pd.notna(row["tmdbId"]) else None
        if t and t.get("status") == "ok" and imdb and t.get("imdb_id") == imdb and t.get("title"):
            return t["title"], "tmdb"
        w = wd.get(imdb) if imdb else None
        if w and not w.get("ambiguous") and w.get("label"):
            return w["label"], "wikidata"
        return None, None

    # 1. Candidates: films already served (fixed ids), then every other MovieLens film by id.
    #    A film with no title anywhere is not a valid record and takes no id.
    excluded = []
    new_ids = []
    for m in sorted(int(x) for x in mv.index if int(x) not in served_set):
        if clean_title(mv.loc[m, "title"])[0] or fallback_title(m)[0]:
            new_ids.append(m)
        else:
            excluded.append({"movieId": m, "raw_title": str(mv.loc[m, "title"]), "reason": "no title in any source"})

    # 2. Duplicates through shared TMDB ids.
    def info(mid: int):
        if mid in legacy_set:
            i = LEGACY_POS[mid]
            return legacy["title"][i], legacy["year"][i], legacy["tmdbId"][i]
        row = mv.loc[mid]
        t, y = clean_title(row["title"])
        return t or fallback_title(mid)[0] or "", y, (int(row["tmdbId"]) if pd.notna(row["tmdbId"]) else None)

    by_tmdb: dict[int, list[int]] = defaultdict(list)
    for mid in served_ids + new_ids:
        t = info(mid)[2]
        if t is not None:
            by_tmdb[int(t)].append(mid)
    alias: dict[int, int] = {}            # merged movieId -> canonical movieId
    served_alias: dict[int, int] = {}     # served movieId -> the served film it duplicates
    conflicts: list[dict] = []
    merges: list[dict] = []
    served_dupes: list[dict] = []
    def rc_of(m):
        return int(stats.loc[m, "count"]) if m in stats.index else 0
    for t, mids in sorted(by_tmdb.items()):
        if len(mids) < 2:
            continue
        # Canonical: a film already served, else the one with most ratings, else the lowest id.
        mids = sorted(mids, key=lambda m: (m not in served_set, -rc_of(m), m))
        head = mids[0]
        ht, hy, _ = info(head)
        for m in mids[1:]:
            mt_, my, _ = info(m)
            same, why = same_film(ht, hy, mt_, my)
            if same and m not in served_set:
                alias[m] = head
                merges.append({"tmdbId": t, "kept": head, "merged": m, "titles": [ht, mt_], "years": [hy, my]})
            elif same and head in served_set:
                # Both already served, so both ids may be stored in browsers: the second stays
                # addressable but is marked as the first's alias and never listed.
                served_alias[m] = head
                served_dupes.append({"tmdbId": t, "kept": head, "alias": m, "titles": [ht, mt_], "years": [hy, my]})
            else:
                conflicts.append({"tmdbId": t, "movieIds": [head, m], "titles": [ht, mt_], "years": [hy, my],
                                  "reason": why})
    canon_new = [m for m in new_ids if m not in alias]
    conflict_ids = {m for c in conflicts for m in c["movieIds"]}
    log(f"{len(merges)} duplicates merged by TMDB id, {len(served_dupes)} served duplicates kept as aliases, "
        f"{len(conflicts)} shared-id conflicts kept apart")

    # 3. Rows.
    order = served_ids + canon_new
    n = len(order)
    pos = {m: i for i, m in enumerate(order)}
    id_map = {str(m): pos[m] for m in order}
    id_map.update({str(m): pos[c] for m, c in alias.items()})
    members: dict[int, list[int]] = defaultdict(list)
    for m in order:
        members[pos[m]].append(m)
    for m, c in alias.items():
        members[pos[c]].append(m)

    keys = list(base.keys())
    for k in ("imdbId", "countries", "released", "alias_of"):
        if k not in keys:
            keys.append(k)
    cols = {k: [None] * n for k in keys}
    prov = {k: [None] * n for k in ("movieIds", "sources", "fields", "tmdb_fetched", "wikidata", "wikidata_fetched", "flags")}
    feats = {"keywords": [[] for _ in range(n)], "description": [None] * n, "original_title": [None] * n}

    for i, m in enumerate(order):
        old = i < LEGACY_N
        row = mv.loc[m] if m in mv.index else None
        imdb = imdb_of(row["imdbId"]) if row is not None else None
        tmdb_id = (int(legacy["tmdbId"][i]) if old and legacy["tmdbId"][i] is not None else
                   (int(row["tmdbId"]) if row is not None and pd.notna(row["tmdbId"]) else None))
        ml_title, ml_year = clean_title(row["title"]) if row is not None else (legacy["title"][i], legacy["year"][i])
        if not ml_title and not old:
            ml_title, tsrc = fallback_title(m)
            flags[i].add(f"title_from_{tsrc}")
        fs: dict[str, str] = {}
        src = []

        # Ratings and tags: every MovieLens id that is this film.
        cnt = sum(rc_of(x) for x in members[i])
        tot = sum(float(stats.loc[x, "sum"]) for x in members[i] if x in stats.index)
        if row is not None or cnt:
            src.append("movielens-32m")
        if old and m not in mv.index:
            src.append("movielens-20m")

        # TMDB, validated against MovieLens. Films served before take a fresh record too —
        # their old data has no fetch date, so TMDB's six-month limit can't be shown to
        # hold — but not series, whose ids are TMDB's TV namespace.
        t = tm.get(tmdb_id) if (tmdb_id is not None and (not old or legacy["media"][i] == "movie")) else None
        if t is not None and t.get("status") != "ok":
            flags[i].add("tmdb_" + t.get("status", "missing"))
            t = None
        another_film = False
        if t is not None:
            why = tmdb_mismatch(ml_title, ml_year, imdb, t)
            if why is None and m in conflict_ids and \
                    not record_titles_match(ml_title, (t.get("title"), t.get("original_title"))):
                why = "shared_id_not_this_film"
            if why:
                flags[i].add("tmdb_" + why)
                # The film's own TMDB id is another film's — so was the data it was served with.
                another_film = old and why in ("imdb_mismatch", "shared_id_not_this_film")
                t = None
        if not old and tmdb_id is None:
            flags[i].add("no_tmdb_id")
        if t is not None:
            src.append("tmdb")

        # Wikidata, validated by year.
        w = wd.get(imdb) if imdb else None
        if w is not None and w.get("ambiguous"):
            flags[i].add("wikidata_ambiguous"); w = None
        if w is not None and "qid" not in w:
            w = None
        if w is not None:
            wy = year_of(w.get("released"))
            if ml_year and wy and abs(ml_year - wy) > 2:
                flags[i].add("wikidata_year_conflict"); w = None
        if w is not None:
            src.append("wikidata")

        if old:
            for k in base:
                cols[k][i] = legacy[k][i] if k in legacy else None
            fs.update(title="movielens", genres="movielens", tags="movielens")
            stale = [k for k in ("poster", "backdrop", "overview", "runtime", "lang", "directors", "cast")
                     if legacy[k][i]]
            if another_film:
                # Artwork, synopsis and credits that belonged to another film are cleared, and
                # what Wikidata knows about this one is used instead.
                for k in stale + ["tv"]:
                    cols[k][i] = [] if k in ("directors", "cast") else "" if k == "overview" else None
                stale = []
                flags[i].add("legacy_tmdb_data_was_another_film")
                if w is not None:
                    rt = wd_runtime(w.get("durations", []))
                    if rt:
                        cols["runtime"][i], fs["runtime"] = rt, "wikidata"
                    if len(w["langs"]) == 1:
                        cols["lang"][i], fs["lang"] = next(iter(w["langs"])), "wikidata"
                    if w["directors"]:
                        cols["directors"][i], fs["directors"] = top_people(w["directors"], 3), "wikidata"
                    if w["cast"]:
                        cols["cast"][i], fs["cast"] = top_people(w["cast"], 8), "wikidata"
            if t is not None:
                # The fresh record replaces the old cache field by field; a field it lacks
                # keeps the old value, and says so.
                for col, val in (("poster", t.get("poster_path")), ("backdrop", t.get("backdrop_path")),
                                 ("overview", t.get("overview")), ("runtime", t.get("runtime")),
                                 ("lang", t.get("original_language")), ("directors", t.get("directors")),
                                 ("cast", t.get("cast"))):
                    if val:
                        cols[col][i] = val
                        fs[col] = "tmdb"
                        if col in stale:
                            stale.remove(col)
                if t.get("vote_average") and (t.get("vote_count") or 0) >= 10:
                    cols["tv"][i] = round(float(t["vote_average"]), 1)
            for col in stale:
                fs[col] = "tmdb (legacy cache, fetch date unknown)"
            if stale:
                src.append("tmdb-legacy")
        else:
            cols["title"][i] = ml_title
            year, ysrc = ml_year, "movielens"
            if year is None and t is not None:
                year, ysrc = year_of(t.get("release_date")), "tmdb"
            if year is None and w is not None:
                year, ysrc = year_of(w.get("released")), "wikidata"
            cols["year"][i] = year
            if year is not None:
                fs["year"] = ysrc
            g = [x for x in str(row["genres"]).split("|") if x and x != "(no genres listed)"]
            gsrc = "movielens"
            if not g and t is not None:
                g = list(dict.fromkeys(TMDB_GENRES[x] for x in t.get("genres", []) if x in TMDB_GENRES)); gsrc = "tmdb"
            if not g and w is not None:
                g = wd_genres(w["genres"]); gsrc = "wikidata"
            cols["genres"][i] = g
            if g:
                fs["genres"] = gsrc
            cols["tags"][i] = []
            cols["tmdbId"][i] = tmdb_id
            cols["movieId"][i] = m
            cols["poster"][i] = t.get("poster_path") if t else None
            cols["backdrop"][i] = t.get("backdrop_path") if t else None
            cols["overview"][i] = (t.get("overview") or "") if t else ""
            rt, rsrc = (t.get("runtime"), "tmdb") if t and t.get("runtime") else (None, None)
            if rt is None and w is not None:
                rt, rsrc = wd_runtime(w.get("durations", [])), "wikidata"
            cols["runtime"][i] = rt
            if rt:
                fs["runtime"] = rsrc
            lang, lsrc = (t.get("original_language"), "tmdb") if t and t.get("original_language") else (None, None)
            if lang is None and w is not None and len(w["langs"]) == 1:
                lang, lsrc = next(iter(w["langs"])), "wikidata"
            cols["lang"][i] = lang
            if lang:
                fs["lang"] = lsrc
            media = "movie"
            if t is None and w is not None and w.get("instances", set()) & TV_CLASSES:
                media = "tv"
            cols["media"][i] = media
            cols["tv"][i] = round(float(t["vote_average"]), 1) if t and t.get("vote_average") and (t.get("vote_count") or 0) >= 10 else None
            dirs, csrc = (t.get("directors") or [], "tmdb") if t else ([], None)
            cast = t.get("cast") or [] if t else []
            if not dirs and w is not None and w["directors"]:
                dirs, csrc = top_people(w["directors"], 3), "wikidata"
            if not cast and w is not None and w["cast"]:
                cast = top_people(w["cast"], 8); fs["cast"] = "wikidata"
            elif cast:
                fs["cast"] = "tmdb"
            cols["directors"][i], cols["cast"][i] = dirs, cast
            if dirs:
                fs["directors"] = csrc
            if t is not None:
                fs.update(poster="tmdb", overview="tmdb")

        # For every film: tags and ratings from MovieLens, ids, countries, release date.
        tg = []
        for x in members[i]:
            for tag in tags.get(x, []):
                if tag not in tg:
                    tg.append(tag)
        if not old:
            cols["tags"][i] = tg[:8]
        cols["rc"][i] = cnt
        cols["ra"][i] = round(tot / cnt, 2) if cnt else None
        cols["imdbId"][i] = imdb
        ctry, csrc2 = ([], None)
        if t is not None and (t.get("production_countries") or t.get("origin_country")):
            ctry, csrc2 = list(dict.fromkeys((t.get("production_countries") or []) + (t.get("origin_country") or []))), "tmdb"
        elif w is not None and w["countries"]:
            ctry, csrc2 = sorted(w["countries"]), "wikidata"
        cols["countries"][i] = ctry
        if ctry:
            fs["countries"] = csrc2
        rel = (t.get("release_date") if t else None) or (w.get("released") if w else None)
        cols["released"][i] = rel if rel and year_of(rel) else None

        # Build-time text features.
        feats["keywords"][i] = (t.get("keywords") or []) if t else []
        feats["description"][i] = w.get("desc") if w else None
        feats["original_title"][i] = t.get("original_title") if t else None

        # A runtime of 0 is TMDB's "unknown" stored as a number; filters would
        # read it as a film of no length. Missing is missing.
        if cols["runtime"][i] is not None and cols["runtime"][i] <= 0:
            cols["runtime"][i] = None
            flags[i].add("runtime_zero_cleared")

        # Validation flags.
        if cols["year"][i] is None:
            flags[i].add("missing_year")
        if not cols["genres"][i]:
            flags[i].add("missing_genres")
        if not cols["runtime"][i]:
            flags[i].add("missing_runtime")
        elif not 1 <= cols["runtime"][i] <= 600:
            flags[i].add("runtime_outlier")
        if not cols["overview"][i]:
            flags[i].add("missing_overview")
        if not cols["poster"][i]:
            flags[i].add("missing_poster")
        elif not POSTER.match(cols["poster"][i]):
            flags[i].add("malformed_poster_path")
        if not cols["lang"][i]:
            flags[i].add("missing_language")
        if imdb is None:
            flags[i].add("missing_imdb_id")
        if cnt == 0:
            flags[i].add("no_ratings")
        if len(members[i]) > 1:
            flags[i].add("has_merged_duplicate")
        cols["alias_of"][i] = pos[served_alias[m]] if m in served_alias else None
        if m in served_alias:
            flags[i].add("alias_of_served_film")

        prov["movieIds"][i] = members[i]
        prov["sources"][i] = src
        prov["fields"][i] = fs
        prov["tmdb_fetched"][i] = t.get("fetched") if t else None
        prov["wikidata"][i] = w.get("qid") if w else None
        prov["wikidata_fetched"][i] = w.get("fetched") if w else None
        prov["flags"][i] = sorted(flags[i])

    # 4. Checks that must hold before anything is written.
    assert cols["movieId"][:LEGACY_N] == [int(c) for c in classes], "legacy ids moved"
    assert len(set(cols["movieId"])) == n, "duplicate canonical MovieLens ids"
    assert all(cols["title"][i] for i in range(n)), "a film without a title"
    imdbs = [x for x in cols["imdbId"] if x]
    assert len(imdbs) == len(set(imdbs)), "two films share an IMDb id"
    assert all(0 <= v < n for v in id_map.values())
    assert all(a is None or cols["alias_of"][a] is None for a in cols["alias_of"]), "an alias of an alias"

    report = quality(cols, prov, merges, conflicts, n, len(mv), tm, wd, now)
    report["served_duplicates"] = served_dupes
    report["excluded"] = excluded
    # Compressed: 57 MB of JSON is 22 MB this way — under GitHub's 50 MB warning, with room to grow.
    write_atomic(args.out / "catalog.json.gz", gzip.compress(dump(cols), compresslevel=9, mtime=0))
    (args.out / "catalog.json").unlink(missing_ok=True)
    forget(args.out, "catalog.json")
    write_atomic(args.out / "id_map.json", dump(id_map))
    write_atomic(args.out / "provenance.json.gz", gzip.compress(dump(prov), mtime=0))
    write_atomic(RAW / "build" / "features.json", dump(feats))
    write_atomic(args.out / "data_quality.json", json.dumps(report, indent=1).encode())
    write_atomic(ROOT / "docs" / "DATA_QUALITY.md", markdown(report).encode())
    record(args.out, "catalog.json.gz", rows=n, served_before=LEGACY_N, added=n - LEGACY_N,
           rating_backed=report["rating_backed"], aliases=len(served_dupes))
    record(args.out, "id_map.json", movielens_ids=len(id_map))
    record(args.out, "provenance.json.gz", rows=n)
    record(args.out, "data_quality.json")
    log(f"wrote {n:,} films ({n - LEGACY_N:,} new), catalog.json.gz "
        f"{(args.out / 'catalog.json.gz').stat().st_size / 1e6:.1f} MB")


# -- the data quality report ------------------------------------------------------

def quality(cols, prov, merges, conflicts, n, ml_total, tm, wd, now) -> dict:
    new = range(LEGACY_N, n)
    flags = prov["flags"]
    fc = Counter(f for fl in flags for f in fl)
    fc_new = Counter(f for i in new for f in flags[i])
    rated = sum(1 for x in cols["rc"] if x)
    movies = sum(1 for x in cols["media"] if x == "movie")
    tm_status = Counter(v.get("status") for v in tm.values())
    src = Counter(s for ss in prov["sources"] for s in ss)
    def share(field, rng):
        rng = list(rng)
        return round(sum(1 for i in rng if cols[field][i] not in (None, "", [])) / max(len(rng), 1), 4)
    fields = ["year", "genres", "tags", "runtime", "lang", "overview", "poster", "directors", "cast", "countries", "imdbId", "tmdbId"]
    return {
        "generated": now,
        "films": n, "movies": movies, "tv": n - movies, "already_served": LEGACY_N, "added": n - LEGACY_N,
        "rating_backed": rated, "metadata_only": n - rated,
        "rating_backed_min5": sum(1 for x in cols["rc"] if x >= 5),
        "movielens_ids": ml_total, "duplicates_merged": len(merges), "shared_id_conflicts": len(conflicts),
        "merges": merges, "conflicts": conflicts,
        "source_coverage": dict(src),
        "tmdb_cache": dict(tm_status), "wikidata_ids_matched": len(wd),
        "field_coverage": {"all": {f: share(f, range(n)) for f in fields},
                           "already_served": {f: share(f, range(LEGACY_N)) for f in fields},
                           "added": {f: share(f, new) for f in fields}},
        "flags": dict(sorted(fc.items())), "flags_added": dict(sorted(fc_new.items())),
        "invalid": {"year_out_of_range_or_missing": fc.get("missing_year", 0),
                    "malformed_poster_path": fc.get("malformed_poster_path", 0),
                    "missing_imdb_id": fc.get("missing_imdb_id", 0),
                    "runtime_outlier": fc.get("runtime_outlier", 0)},
        "sources": {
            "movielens-32m": "GroupLens MovieLens 32M (2023-10). Research use; transformations may be redistributed under the same terms; no commercial use without permission.",
            "movielens-20m": "GroupLens MovieLens 20M, for films served before the expansion that ML-32M no longer lists.",
            "tmdb": "TMDB API, fetched per film with the project's key; see docs/AUDIT.md for the terms.",
            "tmdb-legacy": "TMDB data cached before this build, fetch date unknown.",
            "wikidata": "Wikidata (CC0), matched by IMDb id.",
        },
    }


def markdown(r: dict) -> str:
    pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731
    fc = r["field_coverage"]
    rows = "\n".join(f"| {f} | {pct(fc['all'][f])} | {pct(fc['already_served'][f])} | {pct(fc['added'][f])} |" for f in fc["all"])
    flags = "\n".join(f"| `{k}` | {v:,} | {r['flags_added'].get(k, 0):,} |" for k, v in r["flags"].items())
    merges = "\n".join(f"| {m['tmdbId']} | {m['kept']} | {m['merged']} | {m['titles'][0]} / {m['titles'][1]} | {m['years'][0]} / {m['years'][1]} |" for m in r["merges"]) or "| – | – | – | – | – |"
    conf = "\n".join(f"| {c['tmdbId']} | {c['movieIds'][0]}, {c['movieIds'][1]} | {c['titles'][0]} / {c['titles'][1]} | {c['years'][0]} / {c['years'][1]} | {c['reason']} |" for c in r["conflicts"]) or "| – | – | – | – | – |"
    dupes = "\n".join(f"| {d['tmdbId']} | {d['kept']} | {d['alias']} | {d['titles'][0]} / {d['titles'][1]} | {d['years'][0]} / {d['years'][1]} |" for d in r.get("served_duplicates", [])) or "| – | – | – | – | – |"
    src = "\n".join(f"- **{k}** — {v}" for k, v in r["sources"].items())
    return f"""# Catalogue data quality

Generated {r['generated']} by `pipelines/build_catalog.py`. Machine-readable:
`data/artifacts/data_quality.json`.

## Size

| | |
|---|---|
| Films | **{r['films']:,}** ({r['movies']:,} films, {r['tv']:,} series) |
| Served before this build (ids unchanged) | {r['already_served']:,} |
| Added | {r['added']:,} |
| With MovieLens ratings | {r['rating_backed']:,} ({r['rating_backed_min5']:,} with five or more) |
| Metadata only (no ratings) | {r['metadata_only']:,} |
| MovieLens 32M ids read | {r['movielens_ids']:,} |
| Duplicates merged (shared TMDB id, matching year and title) | {r['duplicates_merged']} |
| Duplicates among films served before, kept as aliases | {len(r.get('served_duplicates', []))} |
| Shared TMDB ids kept apart (conflicts) | {r['shared_id_conflicts']} |
| Excluded as invalid | {len(r.get('excluded', []))} |

## Field coverage

| Field | All | Served before | Added |
|---|---|---|---|
{rows}

## Flags

| Flag | All films | Added films |
|---|---|---|
{flags}

`no_ratings` films are metadata-only: recommended by content, never by
co-rating. `tmdb_imdb_mismatch` and the `*_year_conflict` flags mark a record
from that source that disagreed with MovieLens and was **not used** (a TMDB
record with MovieLens' own IMDb id is used whatever its year).
`legacy_tmdb_data_was_another_film` marks a film served before whose own TMDB
id turned out to be another film's: the artwork and synopsis it was shown with
were that film's, and are cleared.

## Duplicates merged

| TMDB id | Kept movieId | Merged movieId | Titles | Years |
|---|---|---|---|---|
{merges}

## Duplicates among films served before

Both ids may be stored in browsers, so neither row is removed: the alias keeps
its item_id and resolves, but is never listed — search, matches, home rows and
recommendations skip it, and a like of it counts for the film it duplicates.

| TMDB id | Listed movieId | Alias movieId | Titles | Years |
|---|---|---|---|---|
{dupes}

## Shared TMDB ids kept apart

| TMDB id | movieIds | Titles | Years | Why not merged |
|---|---|---|---|---|
{conf}

## Sources

{src}

TMDB cache: {', '.join(f'{k}: {v:,}' for k, v in r['tmdb_cache'].items()) or 'empty'}.
Wikidata: {r['wikidata_ids_matched']:,} IMDb ids matched.
"""


if __name__ == "__main__":
    main()
