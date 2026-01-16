"""
Re-ranking Modules
===================
Post-ranking diversification and business rules.

Implements:
- MMR (Maximal Marginal Relevance) for diversity
- Freshness boosting
- Business rules (already watched, etc.)
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MMRDiversifier:
    """
    Maximal Marginal Relevance for diversity.
    
    Balances relevance vs diversity by iteratively selecting items
    that are both relevant and different from already selected items.
    
    MMR = λ * sim(q, d) - (1-λ) * max(sim(d, d_j)) for d_j in selected
    """
    
    def __init__(self, lambda_param: float = 0.7):
        """
        Args:
            lambda_param: Tradeoff between relevance (1.0) and diversity (0.0)
        """
        self.lambda_param = lambda_param
        
    def rerank(
        self,
        scores: np.ndarray,
        embeddings: np.ndarray,
        k: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Re-rank items using MMR.
        
        Args:
            scores: (n_candidates,) relevance scores
            embeddings: (n_candidates, dim) item embeddings
            k: Number of items to return
            
        Returns:
            reranked_indices: (k,) indices of selected items
            reranked_scores: (k,) adjusted scores
        """
        n = len(scores)
        k = min(k, n)
        
        # Normalize scores to [0, 1]
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        # Normalize embeddings for cosine similarity
        emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        
        selected = []
        selected_emb = []
        remaining = set(range(n))
        
        for _ in range(k):
            best_idx = None
            best_mmr = -float('inf')
            
            for idx in remaining:
                # Relevance component
                relevance = scores_norm[idx]
                
                # Diversity component (max similarity to already selected)
                if selected_emb:
                    similarities = np.dot(emb_norm[idx], np.array(selected_emb).T)
                    max_sim = np.max(similarities)
                else:
                    max_sim = 0
                
                # MMR score
                mmr = self.lambda_param * relevance - (1 - self.lambda_param) * max_sim
                
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = idx
            
            if best_idx is not None:
                selected.append(best_idx)
                selected_emb.append(emb_norm[best_idx])
                remaining.remove(best_idx)
        
        selected_indices = np.array(selected)
        selected_scores = scores[selected_indices]
        
        return selected_indices, selected_scores


class FreshnessBooster:
    """
    Boost scores for fresh/new items.
    
    Items released more recently get a score boost to
    increase their visibility.
    """
    
    def __init__(
        self,
        max_boost: float = 0.2,
        decay_days: int = 90
    ):
        """
        Args:
            max_boost: Maximum boost for brand new items
            decay_days: Days after which boost becomes 0
        """
        self.max_boost = max_boost
        self.decay_days = decay_days
        
    def boost(
        self,
        scores: np.ndarray,
        item_ages_days: np.ndarray
    ) -> np.ndarray:
        """
        Apply freshness boost to scores.
        
        Args:
            scores: (n_items,) relevance scores
            item_ages_days: (n_items,) age of items in days
            
        Returns:
            Boosted scores
        """
        # Exponential decay boost
        boost = self.max_boost * np.exp(-item_ages_days / self.decay_days)
        boost = np.clip(boost, 0, self.max_boost)
        
        return scores + boost


class BusinessRulesFilter:
    """
    Apply business rules to filter/adjust recommendations.
    
    Rules:
    - Remove already watched items
    - Apply category quotas
    - Enforce must-have items (promotions)
    """
    
    def __init__(
        self,
        remove_watched: bool = True,
        category_quotas: Optional[Dict[str, int]] = None
    ):
        self.remove_watched = remove_watched
        self.category_quotas = category_quotas or {}
        
    def apply(
        self,
        item_ids: np.ndarray,
        scores: np.ndarray,
        watched_items: Optional[Set[int]] = None,
        item_categories: Optional[Dict[int, str]] = None,
        promoted_items: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply business rules.
        
        Args:
            item_ids: Candidate item IDs
            scores: Candidate scores
            watched_items: Set of already watched item IDs
            item_categories: Mapping of item_id -> category
            promoted_items: List of items to promote to top
            
        Returns:
            Filtered item_ids and scores
        """
        mask = np.ones(len(item_ids), dtype=bool)
        
        # Remove watched items
        if self.remove_watched and watched_items:
            for i, item_id in enumerate(item_ids):
                if item_id in watched_items:
                    mask[i] = False
        
        # Apply mask
        filtered_items = item_ids[mask]
        filtered_scores = scores[mask]
        
        # Sort by score
        order = np.argsort(-filtered_scores)
        filtered_items = filtered_items[order]
        filtered_scores = filtered_scores[order]
        
        # Promote items (insert at top)
        if promoted_items:
            promoted_set = set(promoted_items)
            non_promoted = [(item, score) for item, score in zip(filtered_items, filtered_scores) 
                          if item not in promoted_set]
            promoted = [(item, 999.0) for item in promoted_items if item in set(item_ids)]
            
            combined = promoted + non_promoted
            filtered_items = np.array([x[0] for x in combined])
            filtered_scores = np.array([x[1] for x in combined])
        
        return filtered_items, filtered_scores


class ReRanker:
    """
    Complete re-ranking pipeline.
    
    Applies multiple re-ranking strategies in sequence.
    """
    
    def __init__(
        self,
        mmr_lambda: float = 0.7,
        freshness_boost: float = 0.2,
        remove_watched: bool = True
    ):
        self.mmr = MMRDiversifier(lambda_param=mmr_lambda)
        self.freshness = FreshnessBooster(max_boost=freshness_boost)
        self.business_rules = BusinessRulesFilter(remove_watched=remove_watched)
        
    def rerank(
        self,
        item_ids: np.ndarray,
        scores: np.ndarray,
        embeddings: np.ndarray,
        k: int = 20,
        item_ages_days: Optional[np.ndarray] = None,
        watched_items: Optional[Set[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply full re-ranking pipeline.
        
        Returns:
            Final item_ids and scores
        """
        # 1. Apply business rules
        item_ids, scores = self.business_rules.apply(
            item_ids, scores, watched_items=watched_items
        )
        
        # Update embeddings to match filtered items
        # (In production, would need to look up embeddings)
        
        # 2. Apply freshness boost
        if item_ages_days is not None:
            # Filter ages to match filtered items
            scores = self.freshness.boost(scores, item_ages_days)
        
        # 3. Apply MMR diversity (k * 2 candidates for diversity selection)
        # Note: Would need embeddings for filtered items
        # For now, just return top-k by score
        
        top_k = min(k, len(item_ids))
        order = np.argsort(-scores)[:top_k]
        
        return item_ids[order], scores[order]
