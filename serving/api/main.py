"""
FastAPI Recommendation API
===========================
Production-grade REST API for movie recommendations.

Endpoints:
- GET /recommend/{user_id} - Get recommendations
- GET /similar/{item_id} - Get similar items
- POST /batch-recommend - Batch recommendations
- POST /chat/message - Intelligent Chatbot
- GET /health - Health check
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Set, Dict, Any
from datetime import datetime
import json
import requests
import re
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load env
base_path = Path(__file__).parent.parent.parent
load_dotenv(base_path / ".env")

# Initialize FastAPI app
app = FastAPI(
    title="Movie Recommender API",
    description="FAANG-grade movie recommendation service",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service instance
recommendation_service = None
movies_metadata = {} # For Chat RAG
hf_client = None # For LLM

# Constants
HF_TOKEN = os.getenv("HF_TOKEN")
# Using Qwen2.5-7B-Instruct as it is free and reliable
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# Request/Response models
class RecommendationItem(BaseModel):
    """Single recommendation item."""
    item_id: int
    score: float
    # Optional: movie_title, genres, etc.
    

class RecommendationResponse(BaseModel):
    """Response for recommendation request."""
    user_id: int
    recommendations: List[RecommendationItem]
    num_candidates: int
    latency_ms: float


class BatchRecommendRequest(BaseModel):
    """Batch recommendation request."""
    user_ids: List[int]
    k: int = Field(default=20, ge=1, le=100)
    exclude_watched: bool = True


class BatchRecommendResponse(BaseModel):
    """Batch recommendation response."""
    results: List[RecommendationResponse]
    total_latency_ms: float


class SimilarItemsResponse(BaseModel):
    """Similar items response."""
    item_id: int
    similar_items: List[RecommendationItem]
    latency_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    num_users: int
    num_items: int
    index_loaded: bool

class ChatRequest(BaseModel):
    message: str
    session_id: str

class ChatResponse(BaseModel):
    response: str
    recommendations: List[Dict[str, Any]] = []


@app.on_event("startup")
async def startup_event():
    """Load recommendation service on startup."""
    global recommendation_service, movies_metadata, hf_client
    
    # 1. Load Recommendation Service
    try:
        import sys
        sys.path.insert(0, str(base_path))
        from serving.recommendation_service import RecommendationService
        
        recommendation_service = RecommendationService(base_path)
        recommendation_service.load()
        logger.info("Recommendation service loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load recommendation service: {e}")
        logger.warning("Recommendation API will return errors.")

    # 2. Load Chat Service (Independent)
    try:
        from huggingface_hub import InferenceClient
        
        # Load Rich Metadata
        movies_json_path = base_path / "deployment" / "movies.json"
        if movies_json_path.exists():
             with open(movies_json_path, encoding='utf-8') as f:
                 movies_metadata = json.load(f)
             logger.info(f"Loaded {len(movies_metadata)} movies for Chat RAG")
        else:
             logger.warning("deployment/movies.json not found. Chat will be limited.")
        
        # Initialize HF Client
        if HF_TOKEN:
            hf_client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)
            logger.info(f"Initialized HF Inference Client with {MODEL_ID}")
        else:
            logger.warning("HF_TOKEN not found in env. Chat will not work.")
            logger.warning(f"Env vars keys: {list(os.environ.keys())[:5]}...")
            
    except Exception as e:
        logger.error(f"Failed to load Chat service: {e}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if recommendation_service is None or not recommendation_service._loaded:
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now().isoformat(),
            num_users=0,
            num_items=0,
            index_loaded=False
        )
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        num_users=recommendation_service._metadata.get('num_users', 0),
        num_items=recommendation_service._metadata.get('num_items', 0),
        index_loaded=recommendation_service._faiss_index is not None
    )


@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
async def get_recommendations(
    user_id: int,
    k: int = Query(default=20, ge=1, le=100, description="Number of recommendations"),
    exclude_watched: bool = Query(default=True, description="Exclude already watched items")
):
    """
    Get personalized recommendations for a user.
    """
    if recommendation_service is None or not recommendation_service._loaded:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    import time
    start = time.time()
    
    try:
        result = recommendation_service.recommend_by_user_id(
            user_id=user_id,
            k=k
        )
        
        if 'error' in result:
            raise HTTPException(status_code=404, detail=result['error'])
        
        latency_ms = (time.time() - start) * 1000
        
        return RecommendationResponse(
            user_id=user_id,
            recommendations=[
                RecommendationItem(**rec) for rec in result['recommendations']
            ],
            num_candidates=result['num_candidates'],
            latency_ms=latency_ms
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-recommend", response_model=BatchRecommendResponse)
async def batch_recommendations(request: BatchRecommendRequest):
    """
    Get recommendations for multiple users.
    """
    if recommendation_service is None or not recommendation_service._loaded:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    import time
    start = time.time()
    
    results = []
    for user_id in request.user_ids:
        try:
            user_start = time.time()
            result = recommendation_service.recommend_by_user_id(
                user_id=user_id,
                k=request.k
            )
            user_latency = (time.time() - user_start) * 1000
            
            if 'error' not in result:
                results.append(RecommendationResponse(
                    user_id=user_id,
                    recommendations=[
                        RecommendationItem(**rec) for rec in result['recommendations']
                    ],
                    num_candidates=result['num_candidates'],
                    latency_ms=user_latency
                ))
        except Exception as e:
            logger.error(f"Error for user {user_id}: {e}")
    
    total_latency = (time.time() - start) * 1000
    
    return BatchRecommendResponse(
        results=results,
        total_latency_ms=total_latency
    )


@app.get("/similar/{item_id}", response_model=SimilarItemsResponse)
async def get_similar_items(
    item_id: int,
    k: int = Query(default=20, ge=1, le=100)
):
    """
    Get similar items using item-to-item similarity.
    """
    if recommendation_service is None or not recommendation_service._loaded:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    import time
    start = time.time()
    
    try:
        # Get item embedding
        item_idx = item_id  # Would need mapping in production
        
        if item_idx >= len(recommendation_service._item_embeddings):
            raise HTTPException(status_code=404, detail=f"Item {item_id} not found")
        
        item_embedding = recommendation_service._item_embeddings[item_idx:item_idx+1]
        
        # Find similar items
        distances, indices = recommendation_service._faiss_index.search(item_embedding, k=k+1)
        
        # Exclude the query item itself
        similar = [
            RecommendationItem(item_id=int(idx), score=float(dist))
            for idx, dist in zip(indices[0], distances[0])
            if idx != item_idx
        ][:k]
        
        latency_ms = (time.time() - start) * 1000
        
        return SimilarItemsResponse(
            item_id=item_id,
            similar_items=similar,
            latency_ms=latency_ms
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Similar items error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Intelligent Chatbot Logic ---

def search_movies(query: str, k: int = 5) -> List[Dict]:
    """Simple keyword/ranking search for RAG context."""
    if not movies_metadata:
        return []

    q_lower = query.lower()
    scored = []
    
    # Priority Keywords
    is_hindi = "hindi" in q_lower or "bollywood" in q_lower
    
    for mid, movie in movies_metadata.items():
        score = 0
        title = movie.get("title", "").lower()
        overview = movie.get("overview", "").lower()
        cast = [c.lower() for c in movie.get("cast", [])]
        genres = [g.lower() for g in movie.get("genres", [])]
        
        # Keyword Matching (Heuristic)
        # Check if title is in query (Entity Recognition equivalent)
        if title and title in q_lower: score += 150
        
        # Check if query is in title (Search equivalent)
        if q_lower in title: score += 50
        
        # Exact title match
        if q_lower == title: score += 100
        
        # Cast match
        for actor in cast:
             if actor in q_lower: score += 40
        
        # Genre match
        for genre in genres:
            if genre in q_lower: score += 20
        
        # Description match (weak)
        if q_lower in overview: score += 10

        # Language Boost
        if is_hindi:
             # If we had language field... we added 'original_language' in create_movies_json!
             if movie.get('original_language') == 'hi':
                 score += 30
    
        if score > 0:
            movie['id'] = mid # Ensure ID is passed
            scored.append((score, movie))
            
    # Sort by score
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s[1] for s in scored[:k]]

def query_llm(messages: List[Dict[str, str]]) -> str:
    """Query Hugging Face Inference API via Client."""
    if not hf_client:
        return "I'm sorry, I can't chat right now because the AI service is not initialized."
    
    try:
        response = hf_client.chat_completion(
            messages=messages,
            max_tokens=400,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM Call failed: {e}")
        return f"I'm having trouble thinking right now. (Error: {str(e)[:50]})"

@app.post("/chat/message", response_model=ChatResponse)
async def chat_message(req: ChatRequest):
    """
    Intelligent RAG Chat Endpoint.
    """
    query = req.message
    
    # 1. Retrieve Context
    context_movies = search_movies(query, k=5)
    
    # 2. Build Prompt
    context_str = ""
    for m in context_movies:
        title = m.get('title', 'Unknown')
        year = m.get('year', 'N/A')
        overview = m.get('overview', 'No description found')[:150]
        cast = ', '.join(m.get('cast', [])[:3])
        context_str += f"- {title} ({year}) [Cast: {cast}]: {overview}...\n"
        
    system_message = {
        "role": "system",
        "content": f"""You are a helpful movie assistant called CineMind. 
Your goal is to answer user queries using the provided Movie Database Context.

CONTEXT FROM LOCAL DATABASE:
{context_str}

INSTRUCTIONS:
1. Use the Context above to answer questions about specific movies (cast, plot, year).
2. If the user asks for recommendations, prioritize movies from the Context.
3. If the answer is NOT in the Context, you may use your general knowledge, but verify if the movie exists in the context first.
4. Correctly identify Indian/Hindi movies if asked.
5. Be concise, friendly, and engaging.
"""
    }
    
    user_message = {"role": "user", "content": query}
    
    # 3. Generate Response
    # Pass full message history if we had sessions, currently just system + user one-shot
    ai_response = query_llm([system_message, user_message])
    
    return ChatResponse(
        response=ai_response,
        recommendations=context_movies
    )


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
