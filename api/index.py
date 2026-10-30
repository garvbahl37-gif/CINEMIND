"""
CINEMIND API — FastAPI on Vercel Python Functions (Fluid Compute).

Serverless, so there is no instance to spin down: no cold-start sleep of the kind
Hugging Face Spaces imposes. Artifacts are memory-mapped/parsed once per warm
instance and reused across requests.
"""
from __future__ import annotations

import json
import os
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

# Vercel imports this file with /var/task as the working directory, so the
# function's own directory is not on sys.path by default.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field

from _adaptive import Adaptive, Signal
from _chat import Assistant
from _discover import Discover
from _engine import ART, get_engine

app = FastAPI(
    title="CINEMIND API",
    description="Hybrid film recommendations: item-item CF + two-tower embeddings + content reranking.",
    version="3.0.0",
)

# An explicit allowlist. The previous config paired allow_origins=["*"] with
# allow_credentials=True, which browsers reject outright.
_origins = [o for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o]
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_origins=_origins + ["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=86400,
)
app.add_middleware(GZipMiddleware, minimum_size=800)


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, exc: RequestValidationError):
    # The default handler echoes each bad value back, and a NaN or Infinity
    # can't be written as JSON — so a 422 turned into a 500. Say where and why.
    return JSONResponse(status_code=422, content={"detail": [
        {"loc": [str(x) for x in err.get("loc", ())], "msg": err.get("msg", ""), "type": err.get("type", "")}
        for err in exc.errors()]})

CACHE = "public, max-age=300, s-maxage=86400, stale-while-revalidate=604800"
# For answers built from what one person sent — their likes, their messages.
# No shared cache may keep them, so one person's picks are never served to another.
PERSONAL = "private, no-store, max-age=0"


def _cache(r: Response, v: str = CACHE):
    r.headers["Cache-Control"] = v


@app.get("/api/health")
async def health(response: Response):
    e = get_engine()
    _cache(response, "public, max-age=30")
    return {"status": "healthy", "items": e.n, "neighbors_per_item": int(e.nbr.shape[1]),
            "rating_backed": e.rated, "search_index": "prebuilt" if e.search_prebuilt else "built at start-up"}


@app.get("/api/browse")
async def browse(response: Response):
    """Everything the home page needs, in one prebuilt round trip."""
    t0 = time.perf_counter()
    home = get_engine().home()
    _cache(response)
    return {**home, "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}


@app.get("/api/similar/{item_id}")
async def similar(item_id: int, response: Response, k: int = Query(12, ge=1, le=24)):
    t0 = time.perf_counter()
    e = get_engine()
    if not 0 <= item_id < e.n:
        raise HTTPException(404, f"Unknown item {item_id}. Valid range 0-{e.n - 1}.")
    res = e.similar(item_id, k)
    _cache(response)
    return {
        "item_id": item_id,
        "source": e.item(item_id),
        # collaborative: from co-ratings; hybrid: thin co-rating evidence, leaning
        # on content; content: no positive ratings, matched on what it is about.
        "match_basis": e.basis_of(item_id),
        "similar_items": res,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


@app.get("/api/movie/{item_id}")
async def movie(item_id: int, response: Response):
    e = get_engine()
    if not 0 <= item_id < e.n:
        raise HTTPException(404, f"Unknown item {item_id}.")
    _cache(response)
    return e.item(item_id)


SORTS = {"relevance", "rating", "newest", "oldest", "title"}


def _csv(v: Optional[str]) -> Optional[list]:
    """`?genres=Action,Sci-Fi` -> ["Action", "Sci-Fi"]. Empty string means
    "no genres", which is different from the parameter being absent."""
    if v is None:
        return None
    return [x.strip() for x in v.split(",") if x.strip()]


def filters(
    q: str = Query("", description="Free text; may also carry structure like "
                                   "'korean thrillers' or '90s sci-fi'"),
    genres: Optional[str] = Query(None, description="Comma-separated genre names; "
                                                    "overrides any the query implied"),
    year_min: Optional[int] = Query(None, ge=1874, le=2100),
    year_max: Optional[int] = Query(None, ge=1874, le=2100),
    lang: Optional[str] = Query(None, min_length=2, max_length=3),
    media: Optional[str] = Query(None, pattern="^(movie|tv)$"),
    sort: str = Query("relevance"),
    limit: int = Query(40, ge=1, le=120),
    offset: int = Query(0, ge=0, le=5000),
    facets: bool = Query(False),
) -> dict:
    """
    The filter parameters every search surface accepts.

    A function rather than a class: this module uses postponed annotations, and
    FastAPI resolves a class dependency's __init__ annotations against an empty
    namespace, so `Optional[str]` never resolves and every request carrying one
    of these parameters fails inside Pydantic.
    """
    return dict(q=q or "", genres=_csv(genres), year_min=year_min,
                year_max=year_max, lang=lang, media=media,
                sort=sort if sort in SORTS else "relevance",
                limit=limit, offset=offset, facets=facets)


@app.get("/api/search")
async def search(response: Response, f: dict = Depends(filters)):
    t0 = time.perf_counter()
    r = get_engine().search(**f)
    _cache(response, "public, max-age=120, s-maxage=3600")
    return {"query": f["q"], **r, "count": len(r["results"]),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}


@app.get("/api/categories")
async def categories(response: Response):
    """The filter vocabulary: every genre, decade and language, with counts."""
    _cache(response)
    return get_engine().categories()


@app.get("/api/suggest")
async def suggest(response: Response, q: str = Query(..., min_length=1)):
    e = get_engine()
    _cache(response, "public, max-age=300, s-maxage=86400")
    return {"categories": e.suggest_categories(q), "results": e.suggest(q)}


class BlendRequest(BaseModel):
    seeds: List[int]
    k: Optional[int] = 20


@app.post("/api/blend")
async def blend(req: BlendRequest):
    """Recommendations from several films at once — a taste profile."""
    t0 = time.perf_counter()
    e = get_engine()
    seeds = [s for s in req.seeds if 0 <= s < e.n][:12]
    if not seeds:
        raise HTTPException(400, "Provide at least one valid item_id in 'seeds'.")
    return {"seeds": seeds, "results": e.blend(seeds, min(req.k or 20, 40)),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}


@app.get("/api/top50")
async def top50(response: Response):
    e = get_engine()
    _cache(response)
    return {"results": e.items(e.browse["top50"])}


# ============================================================
# Compatibility layer for the original frontend
#
# The first version of this app fetched a dictionary of every film and did its
# grouping, suggestions and filtering in the browser. These endpoints keep that
# shape — but keyed by the model's item_id rather than raw MovieLens movieIds,
# which is the mismatch that made every recommendation wrong.
# ============================================================

def _legacy(e, i: int) -> dict:
    """The field names the original components read."""
    d = e.item(i)
    d["vote_average"] = d.pop("rating_avg", None)
    d["vote_count"] = d.pop("rating_count", 0)
    d["releaseDate"] = f"{d['year']}-01-01" if d.get("year") else None
    p = d.get("poster_path")
    d["poster"] = f"https://image.tmdb.org/t/p/w500{p}" if p else None
    b = d.get("backdrop_path")
    d["backdrop"] = f"https://image.tmdb.org/t/p/w780{b}" if b else None
    return d


@app.get("/api/movies", tags=["Compat"])
async def all_movies(response: Response, limit: int = Query(1600, ge=100, le=4000)):
    """
    The browsable catalogue, keyed by item_id.

    Capped rather than complete: all 17,719 films is ~11 MB of JSON, which is a
    slow first paint on a phone. These are the best-rated films that have
    artwork, which is what the genre rows actually draw from.
    """
    e = get_engine()
    order = sorted(range(e.n),
                   key=lambda i: -e.bayes[i] if e.c["poster"][i] else 1)[:limit]
    _cache(response)
    return {"count": len(order), "movies": {str(i): _legacy(e, i) for i in order}}


@app.get("/api/movies/top50", tags=["Compat"])
async def legacy_top50(response: Response):
    e = get_engine()
    _cache(response)
    res = [_legacy(e, i) for i in e.browse["top50"]]
    return {"count": len(res), "results": res, "source": "precomputed"}


@app.get("/api/movies/tv", tags=["Compat"])
async def legacy_tv(response: Response):
    e = get_engine()
    _cache(response)
    res = [_legacy(e, i) for i in e.browse["tv"]]
    return {"count": len(res), "results": res, "source": "precomputed"}


@app.get("/api/movies/search", tags=["Compat"])
async def legacy_search(response: Response, f: dict = Depends(filters)):
    t0 = time.perf_counter()
    e = get_engine()
    r = e.search(**f)
    _cache(response, "public, max-age=120, s-maxage=3600")
    out = []
    for item in r["results"]:
        d = _legacy(e, item["item_id"])
        d["score"] = item.get("score")
        out.append(d)
    return {"query": f["q"], "filters": r["filters"], "facets": r["facets"],
            "total": r["total"], "count": len(out), "results": out,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}


# ============================================================
# Natural-language discovery and the adaptive engine
# ============================================================

@lru_cache(maxsize=1)
def get_discover() -> Discover:
    return Discover(get_engine())


@lru_cache(maxsize=1)
def get_adaptive() -> Adaptive:
    return Adaptive(get_engine())


@app.get("/api/discover", tags=["Discover"])
async def discover(response: Response, q: str = Query(..., min_length=1, max_length=300),
                   limit: int = Query(24, ge=1, le=60)):
    """
    Describe a film in plain words — "something uplifting under two hours",
    "feels like Interstellar but more emotional" — and get the films that fit,
    each with the reasons it is there and a reading of how the description was
    understood. Sentence embeddings rank by meaning; runtime, era, language and
    genre apply as filters.
    """
    t0 = time.perf_counter()
    out = get_discover().search(q, limit)
    _cache(response, "public, max-age=300, s-maxage=86400")
    out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return out


SIGNAL_KINDS = {"like", "skip", "watchlist", "view", "rating"}


# Python's JSON parser accepts NaN and Infinity; one of them in a rating or a
# timestamp would quietly erase the signal, and in `explore` it was a 500.
class SignalIn(BaseModel):
    item_id: int
    kind: str
    # A star rating, for kind == "rating"; clamped to the 0.5-5 scale.
    value: Optional[float] = Field(None, allow_inf_nan=False)
    # Seconds or milliseconds since the epoch; omitted means "now".
    ts: Optional[float] = Field(None, allow_inf_nan=False)


class SearchIn(BaseModel):
    q: str
    ts: Optional[float] = Field(None, allow_inf_nan=False)


class ForYouRequest(BaseModel):
    signals: List[SignalIn] = Field(default_factory=list)
    searches: List[SearchIn] = Field(default_factory=list)
    explore: float = Field(0.3, allow_inf_nan=False)
    k: int = 24
    exclude: List[int] = Field(default_factory=list)


def _seconds(ts: Optional[float], now: float) -> float:
    if ts is None or ts <= 0:
        return now
    return min(ts / 1000.0 if ts > 1e11 else ts, now)


@app.post("/api/foryou", tags=["Adaptive"])
async def for_you(req: ForYouRequest, response: Response):
    """
    Personal recommendations from the signals this browser has collected —
    likes, skips, watchlist, ratings, films opened, searches. Taste lives in the
    two-tower space, with recent signals blended in to follow drift; candidates
    come from that space, the co-rating graph and search intent; the list is
    reranked for relevance, quality and diversity, with a share of slots spent
    on exploration. Each film carries its evidence, a confidence, and — for
    exploration slots — the propensity it was shown with, for offline
    evaluation. The server keeps nothing.
    """
    e = get_engine()
    now = time.time()
    signals = [Signal(s.item_id, s.kind, None if s.value is None else min(max(s.value, 0.5), 5.0), _seconds(s.ts, now))
               for s in req.signals[-500:] if s.kind in SIGNAL_KINDS and 0 <= s.item_id < e.n]
    searches = [(q.q[:200], _seconds(q.ts, now)) for q in req.searches[-20:] if q.q.strip()]
    out = get_adaptive().recommend(
        signals, searches, k=max(1, min(req.k, 48)), explore=min(max(req.explore, 0.0), 1.0),
        exclude={i for i in req.exclude[:500] if 0 <= i < e.n}, now=now)
    out["ignored_signals"] = len(req.signals[-500:]) - len(signals)
    _cache(response, PERSONAL)
    return out


@app.get("/api/evaluation", tags=["Adaptive"])
async def evaluation(response: Response):
    """Offline evaluation of the recommenders — see pipelines/evaluate.py."""
    path = ART / "eval.json"
    if not path.exists():
        raise HTTPException(404, "No evaluation has been run yet: python pipelines/evaluate.py")
    _cache(response)
    return json.loads(path.read_text())


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    # The film the conversation is about, as the last reply's `focus` named it,
    # so "who directed it?" knows what "it" is. The server keeps no sessions.
    context: Optional[int] = None


@lru_cache(maxsize=1)
def get_assistant() -> Assistant:
    return Assistant(get_engine())


@app.post("/api/chat/message", tags=["Chat"])
async def chat(req: ChatRequest, response: Response):
    """
    The film assistant: questions about a film, films like one (or between
    several), a person's films, or a category. Answers come from the catalogue
    and the recommendation graph rather than a hosted LLM, so every fact it
    states is one this API would return, and it answers in milliseconds.
    """
    t0 = time.perf_counter()
    out = get_assistant().reply((req.message or "")[:500], req.context)
    out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    _cache(response, PERSONAL)
    return out
