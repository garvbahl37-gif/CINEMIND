"""
Recommendation Service
=======================
Complete inference pipeline for production serving.

Pipeline:
1. Load user features from feature store
2. Generate user embedding
3. FAISS retrieval (top-k candidates)
4. Ranking model scoring
5. Re-ranking with business rules
6. Return final recommendations
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RecommendationService:
    """
    Production recommendation service.
    
    Combines all stages:
    - Candidate retrieval (FAISS)
    - Ranking (LightGBM/Neural)
    - Re-ranking (diversity, business rules)
    """
    
    def __init__(self, base_path: str = None, config: Dict = None):
        if base_path is None:
            base_path = Path(__file__).parent.parent
        self.base_path = Path(base_path)
        
        self.config = config or {
            'num_candidates': 100,
            'final_k': 20,
            'use_ranking': True,
            'use_reranking': True
        }
        
        # Components (lazy loaded)
        self._faiss_index = None
        self._model = None
        self._user_embeddings = None
        self._item_embeddings = None
        self._user_features = None
        self._item_features = None
        self._ranker = None
        self._reranker = None
        self._metadata = None
        
        self._loaded = False
        
    def load(self):
        """Load all model components."""
        import torch
        
        logger.info("Loading recommendation service components...")
        
        # Load metadata
        with open(self.base_path / "data" / "processed" / "metadata.json") as f:
            self._metadata = json.load(f)
        
        # Load embeddings
        embeddings_path = self.base_path / "embeddings"
        self._user_embeddings = np.load(embeddings_path / "user_embeddings.npy")
        self._item_embeddings = np.load(embeddings_path / "item_embeddings.npy")
        
        logger.info(f"Loaded embeddings: users={self._user_embeddings.shape}, items={self._item_embeddings.shape}")
        
        # Load FAISS index
        from candidate_generation.ann.faiss_index import FAISSIndexManager
        
        index_path = self.base_path / "indices" / "production.index"
        if index_path.exists():
            self._faiss_index = FAISSIndexManager()
            self._faiss_index.load(index_path)
            logger.info(f"Loaded FAISS index: {self._faiss_index.num_vectors} vectors")
        else:
            # Build index on the fly
            logger.info("Building FAISS index...")
            from candidate_generation.ann.faiss_index import create_optimized_index
            self._faiss_index = create_optimized_index(self._item_embeddings)
        
        # Load reranker
        if self.config.get('use_reranking'):
            from reranking.diversity import ReRanker
            self._reranker = ReRanker()
        
        # Load encoders for ID mapping
        encoder_path = self.base_path / "data" / "processed" / "encoders.npz"
        if encoder_path.exists():
            encoders = np.load(encoder_path, allow_pickle=True)
            self._user_classes = encoders['user_classes']
            self._item_classes = encoders['item_classes']
        
        self._loaded = True
        logger.info("Recommendation service loaded successfully!")
        
    def get_user_embedding(self, user_idx: int) -> np.ndarray:
        """Get user embedding by index."""
        if not self._loaded:
            self.load()
        return self._user_embeddings[user_idx:user_idx+1]
    
    def retrieve_candidates(
        self, 
        user_embedding: np.ndarray, 
        k: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve top-k candidates using FAISS."""
        distances, indices = self._faiss_index.search(user_embedding, k=k)
        return indices[0], distances[0]
    
    def rank_candidates(
        self,
        user_idx: int,
        candidate_ids: np.ndarray,
        candidate_scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Rank candidates using ranking model."""
        if self._ranker is None:
            # Use retrieval scores as ranking scores
            return candidate_ids, candidate_scores
        
        # Build ranking features and score
        # (Placeholder - would use RankingFeatureBuilder)
        return candidate_ids, candidate_scores
    
    def rerank(
        self,
        candidate_ids: np.ndarray,
        candidate_scores: np.ndarray,
        k: int = 20,
        watched_items: Optional[Set[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply re-ranking for diversity and business rules."""
        if self._reranker is None:
            return candidate_ids[:k], candidate_scores[:k]
        
        # Get embeddings for candidates
        candidate_embeddings = self._item_embeddings[candidate_ids]
        
        return self._reranker.rerank(
            item_ids=candidate_ids,
            scores=candidate_scores,
            embeddings=candidate_embeddings,
            k=k,
            watched_items=watched_items
        )
    
    def recommend(
        self,
        user_idx: int,
        k: int = None,
        watched_items: Optional[Set[int]] = None,
        exclude_items: Optional[List[int]] = None
    ) -> Dict:
        """
        Generate recommendations for a user.
        
        Args:
            user_idx: Internal user index
            k: Number of recommendations
            watched_items: Items to exclude (already watched)
            exclude_items: Additional items to exclude
            
        Returns:
            Dict with recommendations and metadata
        """
        if not self._loaded:
            self.load()
        
        k = k or self.config['final_k']
        num_candidates = self.config['num_candidates']
        
        # 1. Get user embedding
        user_embedding = self.get_user_embedding(user_idx)
        
        # 2. Retrieve candidates
        candidate_ids, candidate_scores = self.retrieve_candidates(
            user_embedding, k=num_candidates
        )
        
        # 3. Rank candidates
        if self.config.get('use_ranking'):
            candidate_ids, candidate_scores = self.rank_candidates(
                user_idx, candidate_ids, candidate_scores
            )
        
        # 4. Re-rank for diversity
        if self.config.get('use_reranking'):
            candidate_ids, candidate_scores = self.rerank(
                candidate_ids, candidate_scores, k=k, watched_items=watched_items
            )
        else:
            candidate_ids = candidate_ids[:k]
            candidate_scores = candidate_scores[:k]
        
        # 5. Map back to original IDs if needed
        if hasattr(self, '_item_classes'):
            original_ids = self._item_classes[candidate_ids]
        else:
            original_ids = candidate_ids
        
        return {
            'user_idx': user_idx,
            'recommendations': [
                {'item_id': int(item_id), 'score': float(score)}
                for item_id, score in zip(original_ids, candidate_scores)
            ],
            'num_candidates': len(candidate_ids)
        }
    
    def recommend_by_user_id(
        self,
        user_id: int,
        k: int = None,
        **kwargs
    ) -> Dict:
        """
        Recommend by original user ID.
        """
        if not self._loaded:
            self.load()
            
        # Map to internal index
        if hasattr(self, '_user_classes'):
            user_idx = np.where(self._user_classes == user_id)[0]
            if len(user_idx) == 0:
                return {'error': f'User {user_id} not found', 'recommendations': []}
            user_idx = user_idx[0]
        else:
            user_idx = user_id
        
        return self.recommend(user_idx, k=k, **kwargs)


# Singleton instance
_service = None

def get_recommendation_service(base_path: str = None) -> RecommendationService:
    """Get singleton recommendation service instance."""
    global _service
    if _service is None:
        _service = RecommendationService(base_path)
        _service.load()
    return _service


def test_service():
    """Test the recommendation service."""
    service = RecommendationService()
    
    try:
        service.load()
        
        # Test recommendation
        result = service.recommend(user_idx=0, k=10)
        print(f"Recommendations for user 0:")
        for rec in result['recommendations']:
            print(f"  Item {rec['item_id']}: {rec['score']:.4f}")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run training and index building first.")


if __name__ == "__main__":
    test_service()
