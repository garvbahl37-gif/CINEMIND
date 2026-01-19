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

# Initialize LLM Engine (Global)
from llm_engine import LLMEngine
llm_engine = LLMEngine()

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
    
    # Search FAISS index (Get more candidates to account for filtering)
    search_k = max(50, k * 5)
    distances, indices = faiss_index.search(item_emb, search_k)
    
    # Filter out query item AND ensure item exists in metadata
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx != item_id and str(idx) in movies_data:
            results.append((idx, dist))
            if len(results) >= k:
                break
    
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
    
    # Log search event to Kafka (Production Event)
    from kafka_utils import kafka_producer
    kafka_producer.send_event("user_searches", "SEARCH_QUERY", {"query": q})
    
    # Keyword Expansion Map
    KEYWORD_MAP = {
        "hindi": ["bollywood", "india", "indian"],
        "sci-fi": ["science fiction", "scifi", "futuristic"],
        "romance": ["romantic", "love"],
        "animated": ["animation", "cartoon", "anime"]
    }
    
    # Language Map for Metadata Matching
    LANGUAGE_MAP = {
        "english": "en",
        "hindi": "hi",
        "french": "fr",
        "spanish": "es",
        "korean": "ko",
        "japanese": "ja"
    }
    
    # Detect target language from query
    target_lang = None
    for lang, code in LANGUAGE_MAP.items():
        if lang in q_lower:
            target_lang = code
            break

    expanded_tokens = set(q_lower.split())
    for token in q_lower.split():
        if token in KEYWORD_MAP:
            expanded_tokens.update(KEYWORD_MAP[token])
            
    # 🌟 Intelligent Intent Parsing (LLM)
    # If query is > 1 word, try to extract intent (Genre, Year, etc.)
    intent_filters = {}
    if len(q.split()) > 1:
        try:
            intent_filters = llm_engine.parse_intent(q)
            if intent_filters:
                logger.info(f"🧠 Smart Search Filters: {intent_filters}")
        except Exception as e:
            logger.warning(f"Intent parsing failed: {e}")
            
    for item_id, movie in movies_data.items():
        score = 0
        title = movie.get("title", "").lower()
        tags = [t.lower() for t in movie.get("tags", [])]
        genres = [g.lower() for g in movie.get("genres", [])]
        
        # 0. Language Match (Critical Priority)
        if target_lang and movie.get("original_language") == target_lang:
            score += 100
        
        # 0. Intelligent Intent Match (High Priority)
        # Boost if movie matches LLM-extracted genre
        if "genres" in intent_filters:
             target_genres = [g.lower() for g in intent_filters["genres"]]
             common_genres = set(target_genres).intersection(genres)
             if common_genres:
                 score += 40 * len(common_genres)
        
        # Boost if matches year range
        if "year_min" in intent_filters:
            m_year = movie.get("year", 0)
            if m_year >= intent_filters["year_min"]:
                if "year_max" in intent_filters and m_year <= intent_filters["year_max"]:
                     score += 30 # Perfect range match
                elif "year_max" not in intent_filters:
                     score += 15 # Open ended match
        
        # 1. Title Match (Highest Priority)
        if q_lower == title:
            score += 100
        elif q_lower in title:
            score += 50
            
        # 2. Tag Match (Medium Priority)
        for tag in tags:
            # Check original query
            if q_lower == tag:
                score += 30
            elif q_lower in tag:
                score += 20
            elif tag in q_lower and len(tag) > 2:
                score += 25
            
            # Check expanded tokens
            if tag in expanded_tokens:
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
    global redis_client
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
            logger.warning(f"⚠️ Redis ping failed: {e}. Disabling Redis.")
            redis_client = None

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


# ============================================
# Chat & LLM Integration
# ============================================

class ChatRequest(BaseModel):
    message: str
    session_id: str

@app.post("/chat/message", tags=["Chat"])
async def chat_message(request: ChatRequest):
    """
    Handle chat interactions:
    1. Retrieve history from Redis
    2. Parse intent (filters) with LLM
    3. Search movies with filters
    4. Generate response with LLM
    5. Update history
    """
    
    # 1. Get History
    history = []
    if redis_client:
        try:
            cached_history = redis_client.get(f"chat:{request.session_id}")
            if cached_history:
                history = json.loads(cached_history)
        except Exception as e:
            logger.error(f"Redis get chat history error: {e}")

    # Add user message to history
    history.append({"role": "user", "content": request.message})
    
    # 2. Parse Intent
    filters = llm_engine.parse_intent(request.message)
    logger.info(f"Extracted filters: {filters}")
    
    # 3. Search Movies logic (simplified version of search_movies)
    # We will score movies based on filters + simple text match if relevant
    # For now, just simplistic filter application on top of popularity or relevance
    
    candidates = []
    
    # Simple candidate generation strategy:
    # If filters exist, filter all movies. If not, maybe just use top 20 popular?
    # Or actually run the search logic if query is present?
    # Let's simple reuse search logic manually or call internal function?
    # We'll do a custom filter pass:
    
    for item_id, movie in movies_data.items():
        score = 0
        
        # Filter checks
        if "year_min" in filters and movie.get("year", 0) < filters["year_min"]: continue
        if "year_max" in filters and movie.get("year", 0) > filters["year_max"]: continue
        
        # Genre filter (any match)
        if "genres" in filters:
            movie_genres = [g.lower() for g in movie.get("genres", [])]
            target_genres = [g.lower() for g in filters["genres"]]
            if not any(g in movie_genres for g in target_genres):
                continue
                
        # Duration check
        # (Assuming we had runtime in metadata, if not, skip)
        
        # Scoring: Text match or Popularity
        # If the user query has "action", we filtered. 
        # So now we just want good movies.
        score = movie.get("vote_average", 0) + (movie.get("vote_count", 0) / 10000)
        
        candidates.append({
            "item_id": int(item_id),
            "score": score,
            **movie
        })
        
    # Sort and take top 5
    candidates.sort(key=lambda x: x["score"], reverse=True)
    top_candidates = candidates[:5]
    
    # 4. Generate Response with LLM (RAG)
    # detailed_response = "Here are some movies you might like..." # Placeholder
    try:
        current_history = [{"role": m["role"], "content": m["content"]} for m in history] # Clean history
        detailed_response = llm_engine.generate_response(
            user_query=request.message,
            candidates=top_candidates,
            history=current_history
        )
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        detailed_response = "I found some movies for you, but I'm having trouble analyzing them in detail right now."

    # 5. Update History
    history.append({"role": "assistant", "content": detailed_response})
    
    if redis_client:
        try:
             # Keep last 10 messages
            redis_client.setex(f"chat:{request.session_id}", 3600, json.dumps(history[-10:]))
        except Exception as e:
            logger.error(f"Redis set chat history error: {e}")

    # Format response for frontend
    # Frontend expects: { response: str, recommendations: List[Movie] }
    
    return {
        "response": detailed_response,
        "recommendations": top_candidates
    }
    
    # 4. Generate Response
    response_text = llm_engine.generate_response(request.message, top_candidates, history)
    
    # Add assistant response to history
    history.append({"role": "assistant", "content": response_text})
    
    # 5. Save History (limit to last 10 turns)
    if redis_client:
        try:
            redis_client.setex(f"chat:{request.session_id}", 3600, json.dumps(history[-10:]))
        except Exception as e:
            logger.error(f"Redis set chat history error: {e}")
            
    return {
        "response": response_text,
        "recommendations": top_candidates
    }

@app.delete("/chat/history/{session_id}", tags=["Chat"])
async def clear_history(session_id: str):
    if redis_client:
        redis_client.delete(f"chat:{session_id}")
    return {"status": "cleared"}
