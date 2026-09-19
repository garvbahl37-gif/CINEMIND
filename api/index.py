"""
CINEMIND API — FastAPI on Vercel Python Functions (Fluid Compute).

Serverless, so there is no instance to spin down: no cold-start sleep of the kind
Hugging Face Spaces imposes. Artifacts are memory-mapped/parsed once per warm
instance and reused across requests.
"""
from __future__ import annotations

import os
import time
from typing import List, Optional

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
