"""
FastAPI Recommendation API
===========================
Production-grade REST API for movie recommendations.

Endpoints:
- GET /recommend/{user_id} - Get recommendations
- GET /similar/{item_id} - Get similar items
- POST /batch-recommend - Batch recommendations
- GET /health - Health check
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Set
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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


@app.on_event("startup")
async def startup_event():
    """Load recommendation service on startup."""
    global recommendation_service
    
    try:
        import sys
        base_path = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(base_path))
        
        from serving.recommendation_service import RecommendationService
        
        recommendation_service = RecommendationService(base_path)
        recommendation_service.load()
        
        logger.info("Recommendation service loaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load recommendation service: {e}")
        logger.warning("API will return errors until service is loaded")


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
    
    Args:
        user_id: User ID
        k: Number of recommendations to return
        exclude_watched: Whether to exclude already watched items
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


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
