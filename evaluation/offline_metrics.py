"""
Offline Evaluation Metrics
===========================
Comprehensive metrics for recommender system evaluation.

Metrics:
- Ranking: NDCG, MAP, MRR
- Retrieval: Recall@K, Precision@K, Hit Rate
- Beyond-accuracy: Coverage, Diversity (ILD), Novelty
"""

import logging
from typing import Dict, List, Optional, Set
from collections import defaultdict

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RecommenderMetrics:
    """Compute standard recommender system metrics."""
    
    @staticmethod
    def ndcg_at_k(
        predictions: List[int],
        ground_truth: Set[int],
        k: int = 10,
        relevance_scores: Optional[Dict[int, float]] = None
    ) -> float:
        """
        Normalized Discounted Cumulative Gain @ K.
        
        Measures ranking quality considering position bias.
        """
        if not ground_truth:
            return 0.0
            
        predictions = predictions[:k]
        
        # DCG
        dcg = 0.0
        for i, item in enumerate(predictions):
            if relevance_scores:
                rel = relevance_scores.get(item, 0)
            else:
                rel = 1.0 if item in ground_truth else 0.0
            dcg += rel / np.log2(i + 2)
        
        # Ideal DCG
        if relevance_scores:
            ideal_rels = sorted([relevance_scores.get(item, 0) for item in ground_truth], reverse=True)
        else:
            ideal_rels = [1.0] * min(k, len(ground_truth))
        
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels[:k]))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def map_at_k(
        predictions: List[int],
        ground_truth: Set[int],
        k: int = 10
    ) -> float:
        """
        Mean Average Precision @ K.
        
        Measures precision at each relevant position.
        """
        if not ground_truth:
            return 0.0
            
        predictions = predictions[:k]
        
        hits = 0
        sum_precisions = 0.0
        
        for i, item in enumerate(predictions):
            if item in ground_truth:
                hits += 1
                precision_at_i = hits / (i + 1)
                sum_precisions += precision_at_i
        
        return sum_precisions / min(len(ground_truth), k)
    
    @staticmethod
    def mrr(
        predictions: List[int],
        ground_truth: Set[int]
    ) -> float:
        """
        Mean Reciprocal Rank.
        
        Focuses on position of first relevant item.
        """
        for i, item in enumerate(predictions):
            if item in ground_truth:
                return 1.0 / (i + 1)
        return 0.0
    
    @staticmethod
    def recall_at_k(
        predictions: List[int],
        ground_truth: Set[int],
        k: int = 10
    ) -> float:
        """Recall @ K: fraction of relevant items retrieved."""
        if not ground_truth:
            return 0.0
        predictions = set(predictions[:k])
        return len(predictions & ground_truth) / len(ground_truth)
    
    @staticmethod
    def precision_at_k(
        predictions: List[int],
        ground_truth: Set[int],
        k: int = 10
    ) -> float:
        """Precision @ K: fraction of retrieved items that are relevant."""
        predictions = predictions[:k]
        if not predictions:
            return 0.0
        hits = sum(1 for item in predictions if item in ground_truth)
        return hits / len(predictions)
    
    @staticmethod
    def hit_rate(
        predictions: List[int],
        ground_truth: Set[int],
        k: int = 10
    ) -> float:
        """Hit Rate @ K: 1 if any relevant item in top-K, else 0."""
        predictions = set(predictions[:k])
        return 1.0 if predictions & ground_truth else 0.0


class BeyondAccuracyMetrics:
    """Metrics beyond pure accuracy - diversity, coverage, novelty."""
    
    @staticmethod
    def catalog_coverage(
        all_recommendations: List[List[int]],
        catalog_size: int
    ) -> float:
        """
        Catalog Coverage: fraction of catalog items ever recommended.
        
        Higher is better - indicates less popularity bias.
        """
        recommended_items = set()
        for rec_list in all_recommendations:
            recommended_items.update(rec_list)
        return len(recommended_items) / catalog_size
    
    @staticmethod
    def intra_list_diversity(
        predictions: List[int],
        embeddings: np.ndarray
    ) -> float:
        """
        Intra-List Diversity (ILD): average dissimilarity within a list.
        
        Higher diversity = more varied recommendations.
        """
        if len(predictions) < 2:
            return 0.0
            
        # Get embeddings
        item_embs = embeddings[predictions]
        
        # Normalize
        norms = np.linalg.norm(item_embs, axis=1, keepdims=True)
        item_embs = item_embs / (norms + 1e-8)
        
        # Pairwise similarities
        sim_matrix = np.dot(item_embs, item_embs.T)
        
        # Average off-diagonal (dissimilarity = 1 - similarity)
        n = len(predictions)
        total_dissim = 0.0
        count = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                total_dissim += (1 - sim_matrix[i, j])
                count += 1
        
        return total_dissim / count if count > 0 else 0.0
    
    @staticmethod
    def novelty(
        predictions: List[int],
        item_popularity: Dict[int, float]
    ) -> float:
        """
        Novelty: average of -log2(popularity).
        
        Recommending unpopular items = higher novelty.
        """
        if not predictions:
            return 0.0
            
        novelties = []
        for item in predictions:
            pop = item_popularity.get(item, 1e-6)
            novelties.append(-np.log2(pop + 1e-10))
        
        return np.mean(novelties)


class RecommenderEvaluator:
    """
    Complete evaluation pipeline.
    """
    
    def __init__(self, k_values: List[int] = [10, 50, 100]):
        self.k_values = k_values
        self.metrics = RecommenderMetrics()
        self.beyond_accuracy = BeyondAccuracyMetrics()
        
    def evaluate(
        self,
        predictions: Dict[int, List[int]],
        ground_truth: Dict[int, Set[int]],
        item_embeddings: Optional[np.ndarray] = None,
        item_popularity: Optional[Dict[int, float]] = None,
        catalog_size: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate recommendations.
        
        Args:
            predictions: {user_id: [item_ids]} predictions
            ground_truth: {user_id: {item_ids}} ground truth
            item_embeddings: For diversity calculation
            item_popularity: For novelty calculation
            catalog_size: For coverage calculation
            
        Returns:
            Dictionary of metric_name -> value
        """
        results = defaultdict(list)
        
        for user_id, pred_items in predictions.items():
            gt_items = ground_truth.get(user_id, set())
            
            if not gt_items:
                continue
            
            for k in self.k_values:
                results[f'ndcg@{k}'].append(
                    self.metrics.ndcg_at_k(pred_items, gt_items, k)
                )
                results[f'recall@{k}'].append(
                    self.metrics.recall_at_k(pred_items, gt_items, k)
                )
                results[f'precision@{k}'].append(
                    self.metrics.precision_at_k(pred_items, gt_items, k)
                )
                results[f'map@{k}'].append(
                    self.metrics.map_at_k(pred_items, gt_items, k)
                )
            
            results['mrr'].append(self.metrics.mrr(pred_items, gt_items))
            results['hit_rate@10'].append(self.metrics.hit_rate(pred_items, gt_items, 10))
            
            # Beyond-accuracy metrics
            if item_embeddings is not None:
                results['ild'].append(
                    self.beyond_accuracy.intra_list_diversity(pred_items[:20], item_embeddings)
                )
            
            if item_popularity is not None:
                results['novelty'].append(
                    self.beyond_accuracy.novelty(pred_items[:20], item_popularity)
                )
        
        # Aggregate
        aggregated = {name: np.mean(values) for name, values in results.items()}
        
        # Coverage
        if catalog_size:
            all_recs = list(predictions.values())
            aggregated['coverage'] = self.beyond_accuracy.catalog_coverage(all_recs, catalog_size)
        
        return aggregated
    
    def print_results(self, results: Dict[str, float]):
        """Pretty print evaluation results."""
        logger.info("\n" + "="*60)
        logger.info("Evaluation Results")
        logger.info("="*60)
        
        # Group by metric type
        ranking = {k: v for k, v in results.items() if k.startswith(('ndcg', 'map', 'mrr'))}
        retrieval = {k: v for k, v in results.items() if k.startswith(('recall', 'precision', 'hit'))}
        beyond = {k: v for k, v in results.items() if k in ['coverage', 'ild', 'novelty']}
        
        logger.info("\nRanking Metrics:")
        for name, value in sorted(ranking.items()):
            logger.info(f"  {name}: {value:.4f}")
        
        logger.info("\nRetrieval Metrics:")
        for name, value in sorted(retrieval.items()):
            logger.info(f"  {name}: {value:.4f}")
        
        if beyond:
            logger.info("\nBeyond-Accuracy Metrics:")
            for name, value in sorted(beyond.items()):
                logger.info(f"  {name}: {value:.4f}")
        
        logger.info("="*60)
