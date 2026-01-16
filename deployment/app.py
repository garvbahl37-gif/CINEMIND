"""
MovieRec AI - Production FastAPI Backend for Hugging Face Spaces
================================================================
Uses trained embeddings from Kaggle for real recommendations.
"""

import os
import json
import logging
from typing import List
from datetime import datetime
import time

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import faiss

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="MovieRec AI",
    description="FAANG-Grade Movie Recommendation API - Trained on MovieLens",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# Load Trained Embeddings & FAISS Index
# ============================================

# Paths to trained files (upload these to HF Space)
ITEM_EMB_PATH = "item_embeddings.npy"
USER_EMB_PATH = "user_embeddings.npy"
INDEX_PATH = "production.index"
METADATA_PATH = "metadata.json"
MOVIES_PATH = "movies.json"

# Global variables
item_embeddings = None
user_embeddings = None
faiss_index = None
metadata = {}
movies_data = {}


def load_embeddings():
    """Load embeddings and FAISS index at startup."""
    global item_embeddings, user_embeddings, faiss_index, metadata, movies_data
    
    try:
        # Load metadata
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Loaded metadata: {metadata}")
        
        # Load movies metadata
        if os.path.exists(MOVIES_PATH):
            with open(MOVIES_PATH, 'r', encoding='utf-8') as f:
                movies_data = json.load(f)
            logger.info(f"Loaded {len(movies_data)} movie metadata entries")
        
        # Load item embeddings
        if os.path.exists(ITEM_EMB_PATH):
            item_embeddings = np.load(ITEM_EMB_PATH, mmap_mode='r').astype(np.float32)
            logger.info(f"Loaded item embeddings: {item_embeddings.shape}")
        
        # Load user embeddings
        if os.path.exists(USER_EMB_PATH):
            user_embeddings = np.load(USER_EMB_PATH, mmap_mode='r').astype(np.float32)
            logger.info(f"Loaded user embeddings: {user_embeddings.shape}")
        
        # Load FAISS index
        if os.path.exists(INDEX_PATH):
            faiss_index = faiss.read_index(INDEX_PATH)
            logger.info(f"Loaded FAISS index with {faiss_index.ntotal} vectors")
        elif item_embeddings is not None:
            # Build index if not provided
            logger.info("Building FAISS index from embeddings...")
            dim = item_embeddings.shape[1]
            faiss_index = faiss.IndexFlatIP(dim)
            normalized = item_embeddings.copy()
            faiss.normalize_L2(normalized)
            faiss_index.add(normalized)
            logger.info(f"Built FAISS index with {faiss_index.ntotal} vectors")
            
        return True
    except Exception as e:
        logger.error(f"Error loading embeddings: {e}")
        return False


# ============================================
# Response Models
# ============================================

class ItemRecommendation(BaseModel):
    item_id: int
    score: float


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[ItemRecommendation]
    num_candidates: int
    latency_ms: float


class SimilarItemsResponse(BaseModel):
    item_id: int
    similar_items: List[ItemRecommendation]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    num_items: int
    num_users: int
    embedding_dim: int
    index_loaded: bool


# ============================================
# API Endpoints
# ============================================

@app.get("/", tags=["Info"])
async def root():
    """API information."""
    return {
        "name": "MovieRec AI",
        "version": "2.0.0 (Production)",
        "description": "FAANG-Grade Movie Recommendations with trained embeddings",
        "endpoints": {
            "health": "/health",
            "recommend": "/recommend/{user_id}",
            "similar": "/similar/{item_id}",
            "docs": "/docs"
        },
        "model": {
            "num_items": metadata.get("num_items", 0),
            "num_users": metadata.get("num_users", 0),
            "embedding_dim": metadata.get("embedding_dim", 64)
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if faiss_index is not None else "degraded",
        timestamp=datetime.now().isoformat(),
        num_items=item_embeddings.shape[0] if item_embeddings is not None else 0,
        num_users=user_embeddings.shape[0] if user_embeddings is not None else 0,
        embedding_dim=metadata.get("embedding_dim", 64),
        index_loaded=faiss_index is not None
    )


@app.get("/recommend/{user_id}", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations(
    user_id: int,
    k: int = Query(default=10, ge=1, le=100, description="Number of recommendations")
):
    """Get personalized movie recommendations for a user."""
    start = time.time()
    
    # Validate user_id
    if user_embeddings is None:
        raise HTTPException(status_code=503, detail="User embeddings not loaded")
    
    if user_id < 0 or user_id >= user_embeddings.shape[0]:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found. Valid range: 0-{user_embeddings.shape[0]-1}")
    
    if faiss_index is None:
        raise HTTPException(status_code=503, detail="FAISS index not loaded")
    
    # Get user embedding and normalize
    user_emb = user_embeddings[user_id:user_id+1].copy()
    faiss.normalize_L2(user_emb)
    
    # Search FAISS index
    distances, indices = faiss_index.search(user_emb, k)
    
    latency_ms = (time.time() - start) * 1000
    
    return RecommendationResponse(
        user_id=user_id,
        recommendations=[
            ItemRecommendation(item_id=int(idx), score=round(float(dist), 4))
            for idx, dist in zip(indices[0], distances[0])
        ],
        num_candidates=faiss_index.ntotal,
        latency_ms=round(latency_ms, 2)
    )


@app.get("/similar/{item_id}", response_model=SimilarItemsResponse, tags=["Recommendations"])
async def get_similar_items(
    item_id: int,
    k: int = Query(default=10, ge=1, le=100, description="Number of similar items")
):
    """Get movies similar to a given movie."""
    start = time.time()
    
    # Validate item_id
    if item_embeddings is None:
        raise HTTPException(status_code=503, detail="Item embeddings not loaded")
    
    if item_id < 0 or item_id >= item_embeddings.shape[0]:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found. Valid range: 0-{item_embeddings.shape[0]-1}")
    
    if faiss_index is None:
        raise HTTPException(status_code=503, detail="FAISS index not loaded")
    
    # Get item embedding and normalize
    item_emb = item_embeddings[item_id:item_id+1].copy()
    faiss.normalize_L2(item_emb)
    
    # Search FAISS index (k+1 because the item itself will be first result)
    distances, indices = faiss_index.search(item_emb, k + 1)
    
    # Filter out the query item itself
    results = [(idx, dist) for idx, dist in zip(indices[0], distances[0]) if idx != item_id][:k]
    
    latency_ms = (time.time() - start) * 1000
    
    return SimilarItemsResponse(
        item_id=item_id,
        similar_items=[
            ItemRecommendation(item_id=int(idx), score=round(float(dist), 4))
            for idx, dist in results
        ],
        latency_ms=round(latency_ms, 2)
    )


@app.get("/stats", tags=["Info"])
async def get_stats():
    """Get system statistics."""
    return {
        "embeddings": {
            "items": item_embeddings.shape if item_embeddings is not None else None,
            "users": user_embeddings.shape if user_embeddings is not None else None
        },
        "index": {
            "type": type(faiss_index).__name__ if faiss_index else None,
            "total_vectors": faiss_index.ntotal if faiss_index else 0
        },
        "metadata": metadata,
        "movies_loaded": len(movies_data)
    }


@app.get("/movies", tags=["Movies"])
async def get_all_movies():
    """Get all movie metadata. Use for frontend caching."""
    return {
        "count": len(movies_data),
        "movies": movies_data
    }


@app.get("/movie/{item_id}", tags=["Movies"])
async def get_movie(item_id: int):
    """Get metadata for a specific movie by encoded ID."""
    item_key = str(item_id)
    if item_key not in movies_data:
        return {
            "item_id": item_id,
            "title": f"Movie #{item_id}",
            "year": None,
            "genres": [],
            "tmdbId": None,
            "found": False
        }
    
    movie = movies_data[item_key]
    return {
        "item_id": item_id,
        **movie,
        "found": True
    }


@app.get("/movies/search", tags=["Movies"])
async def search_movies(
    q: str = Query(..., min_length=2, description="Search query"),
    limit: int = Query(default=20, ge=1, le=50)
):
    """Search movies by title, tags, and genres with weighted scoring."""
    q_lower = q.lower()
    scored_results = []
    
    for item_id, movie in movies_data.items():
        score = 0
        title = movie.get("title", "").lower()
        tags = [t.lower() for t in movie.get("tags", [])]
        genres = [g.lower() for g in movie.get("genres", [])]
        
        # 1. Title Match (Highest Priority)
        if q_lower == title:
            score += 100
        elif q_lower in title:
            score += 50
            
        # 2. Tag Match (Medium Priority)
        # Check if query is in tags OR tag is in query (e.g. "hindi movies" -> tag "hindi")
        query_tokens = set(q_lower.split())
        for tag in tags:
            if q_lower == tag:
                score += 30
            elif q_lower in tag:
                score += 20
            elif tag in q_lower and len(tag) > 2: # Avoid matching short words like "in"
                score += 25
            
            # Check for token overlap
            if tag in query_tokens:
                 score += 15

        # 3. Genre Match (Lowest Priority)
        for genre in genres:
            if q_lower == genre:
                score += 20
            elif genre in q_lower:
                score += 20
        
        if score > 0:
            scored_results.append({
                "item_id": int(item_id),
                "score": score,
                **movie
            })
    
    # Sort by score descending
    scored_results.sort(key=lambda x: x["score"], reverse=True)
    
    return {
        "query": q,
        "count": len(scored_results),
        "results": scored_results[:limit]
    }


# ============================================
# Redis Configuration
# ============================================
import redis

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
redis_client = None

try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
except Exception as e:
    logger.warning(f"Redis connection failed: {e}. Caching will be disabled.")


# ============================================
# Startup Event
# ============================================

@app.on_event("startup")
async def startup():
    logger.info("🎬 MovieRec AI starting...")
    success = load_embeddings()
    if success:
        logger.info("✅ API ready with trained embeddings!")
    else:
        logger.warning("⚠️ Running without embeddings - upload trained files!")
        
    # Check Redis
    if redis_client:
        try:
            redis_client.ping()
            logger.info("✅ Redis connected successfully!")
        except Exception as e:
            logger.warning(f"⚠️ Redis ping failed: {e}")

# ... (Rest of the file remains similar, but we update endpoints)

# Cache for top 50 movies to avoid re-sorting every request
# Replaced global list with Redis pattern
@app.get("/movies/top50", tags=["Movies"])
async def get_top_50_movies():
    """Get top 50 movies by average rating (min 1000 votes). Cached in Redis for 1 hour."""
    
    # Try Redis first
    if redis_client:
        try:
            cached = redis_client.get("top_50_movies")
            if cached:
                results = json.loads(cached)
                return {"count": len(results), "results": results, "source": "redis"}
        except Exception as e:
            logger.error(f"Redis get error: {e}")
    
    # Filter and sort movies (Fallthrough logic)
    valid_movies = []
    for item_id, movie in movies_data.items():
        if movie.get("vote_count", 0) >= 1000:
            valid_movies.append({
                "item_id": int(item_id),
                **movie
            })
    
    # Sort by rating descending
    valid_movies.sort(key=lambda x: x.get("vote_average", 0), reverse=True)
    
    # Top 50
    top_50 = valid_movies[:50]
    
    # Save to Redis
    if redis_client:
        try:
            redis_client.setex("top_50_movies", 3600, json.dumps(top_50)) # Cache for 1 hour
        except Exception as e:
            logger.error(f"Redis set error: {e}")
    
    return {
        "count": len(top_50),
        "results": top_50,
        "source": "database"
    }


@app.get("/movies/tv", tags=["Movies"])
async def get_tv_movies():
    """Get popular TV shows from the dataset. Cached in Redis for 1 hour."""
    
    # Try Redis first
    if redis_client:
        try:
            cached = redis_client.get("tv_movies")
            if cached:
                results = json.loads(cached)
                return {"count": len(results), "results": results, "source": "redis"}
        except Exception as e:
            logger.error(f"Redis get error: {e}")
    
    # Filter for TV shows
    tv_shows = []
    for item_id, movie in movies_data.items():
        if movie.get("media_type") == "tv":
            tv_shows.append({
                "item_id": int(item_id),
                **movie
            })
    
    # Sort by rating descending
    tv_shows.sort(key=lambda x: x.get("vote_average", 0), reverse=True)
    
    # Save to Redis
    if redis_client:
        try:
            redis_client.setex("tv_movies", 3600, json.dumps(tv_shows))
        except Exception as e:
            logger.error(f"Redis set error: {e}")
    
    return {
        "count": len(tv_shows),
        "results": tv_shows,
        "source": "database"
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
