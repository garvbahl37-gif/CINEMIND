"""
CINEMIND API — FastAPI on Vercel Python Functions (Fluid Compute).

Serverless, so there is no instance to spin down: no cold-start sleep of the kind
Hugging Face Spaces imposes. Artifacts are memory-mapped/parsed once per warm
instance and reused across requests.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List, Optional

# Vercel imports this file with /var/task as the working directory, so the
# function's own directory is not on sys.path by default.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel

from _engine import get_engine

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

CACHE = "public, max-age=300, s-maxage=86400, stale-while-revalidate=604800"


def _cache(r: Response, v: str = CACHE):
    r.headers["Cache-Control"] = v


@app.get("/api/health")
async def health(response: Response):
    e = get_engine()
    _cache(response, "public, max-age=30")
    return {"status": "healthy", "items": e.n, "neighbors_per_item": int(e.nbr.shape[1])}


@app.get("/api/browse")
async def browse(response: Response):
    """Everything the home page needs, in one prebuilt round trip."""
    t0 = time.perf_counter()
    e = get_engine()
    b = e.browse
    rows = [{"genre": g, "items": e.items(ids[:24])} for g, ids in b["genres"].items()]
    rows.sort(key=lambda r: -len(r["items"]))
    _cache(response)
    return {
        "hero": e.items(b["hero"]),
        "top50": e.items(b["top50"]),
        "tv": e.items(b["tv"]),
        "rows": rows,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


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


@app.get("/api/search")
async def search(response: Response,
                 q: str = Query(..., min_length=1),
                 limit: int = Query(40, ge=1, le=60)):
    t0 = time.perf_counter()
    e = get_engine()
    res, filters = e.search(q, limit)
    _cache(response, "public, max-age=120, s-maxage=3600")
    return {"query": q, "filters": filters, "count": len(res), "results": res,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}


@app.get("/api/suggest")
async def suggest(response: Response, q: str = Query(..., min_length=1)):
    e = get_engine()
    _cache(response, "public, max-age=300, s-maxage=86400")
    return {"results": e.suggest(q)}


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
async def legacy_search(response: Response,
                        q: str = Query(..., min_length=1),
                        limit: int = Query(40, ge=1, le=60)):
    t0 = time.perf_counter()
    e = get_engine()
    res, filters = e.search(q, limit)
    _cache(response, "public, max-age=120, s-maxage=3600")
    out = []
    for r in res:
        d = _legacy(e, r["item_id"])
        d["score"] = r.get("score")
        out.append(d)
    return {"query": q, "filters": filters, "count": len(out), "results": out,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


@app.post("/api/chat/message", tags=["Compat"])
async def chat(req: ChatRequest):
    """
    Conversational search. Intent parsing is deterministic rather than a call
    out to a hosted LLM, so it costs nothing, cannot rate-limit, and answers in
    milliseconds instead of seconds.
    """
    e = get_engine()
    msg = (req.message or "").strip()
    if not msg:
        return {"response": "Tell me a film you liked, or a genre and a decade.",
                "recommendations": []}

    res, filters = e.search(msg, 8)

    # "more like X" — if the query names one film, answer with its neighbours
    if res and (res[0].get("score") or 0) > 150 and len(res) < 4:
        src = res[0]
        near = e.similar(src["item_id"], 8)
        body = ", ".join(f"{f['title']} ({f['year']})" for f in near[:3])
        return {
            "response": f"If you liked {src['title']}, the closest films in the "
                        f"model's space are {body}. They share "
                        f"{', '.join(near[0].get('shared_genres') or ['a lot']) }.",
            "recommendations": [_legacy(e, f["item_id"]) | {"score": f.get("score")}
                                for f in near],
        }

    if not res:
        return {"response": "Nothing in the catalogue matches that. It covers 17,719 "
                            "films from MovieLens — try a genre, a decade like “90s”, "
                            "a director, or a title.",
                "recommendations": []}

    LANG_NAMES = {"en": "English", "hi": "Hindi", "fr": "French", "es": "Spanish",
                  "ko": "Korean", "ja": "Japanese", "de": "German", "it": "Italian",
                  "zh": "Chinese", "ru": "Russian", "sv": "Swedish", "da": "Danish"}
    bits = []
    if filters.get("lang"):
        bits.append(LANG_NAMES.get(filters["lang"], ""))
    if filters.get("genres"):
        bits.append(" and ".join(filters["genres"]))
    if filters.get("year_min"):
        lo, hi = filters["year_min"], filters.get("year_max", filters["year_min"])
        bits.append(f"from {lo}" if lo == hi else f"between {lo} and {hi}")
    lead = f"Here are the best {' '.join(bits)} films I have" if bits \
        else f"Here's what matches “{msg}”"
    top = ", ".join(f"{f['title']} ({f['year']})" for f in res[:3])
    return {"response": f"{lead}. Start with {top}.",
            "recommendations": [_legacy(e, f["item_id"]) | {"score": f.get("score")}
                                for f in res]}
