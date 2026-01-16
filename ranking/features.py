"""
Ranking Feature Engineering
============================
Build features for the ranking/scoring stage.

Features:
- User-item embedding similarity
- Position bias features
- Historical interaction features
- Cross features
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RankingFeatureBuilder:
    """Build features for ranking model."""
    
    def __init__(self, base_path: str = None):
        if base_path is None:
            base_path = Path(__file__).parent.parent
        self.base_path = Path(base_path)
        self.features_path = self.base_path / "data" / "features"
        self.embeddings_path = self.base_path / "embeddings"
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load features and embeddings."""
        logger.info("Loading data for ranking features...")
        
        self.user_features = pd.read_parquet(self.features_path / "user_features.parquet")
        self.item_features = pd.read_parquet(self.features_path / "item_features.parquet")
        
        if (self.embeddings_path / "user_embeddings.npy").exists():
            self.user_embeddings = np.load(self.embeddings_path / "user_embeddings.npy")
            self.item_embeddings = np.load(self.embeddings_path / "item_embeddings.npy")
        else:
            self.user_embeddings = None
            self.item_embeddings = None
            
        logger.info(f"Loaded {len(self.user_features)} users, {len(self.item_features)} items")
        
    def build_pairwise_features(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray
    ) -> np.ndarray:
        """
        Build features for user-item pairs.
        
        Features:
        1. Embedding similarity
        2. User statistics
        3. Item statistics
        4. Cross features (user_genre_affinity * item_genre)
        """
        features = []
        
        # Embedding similarity
        if self.user_embeddings is not None and self.item_embeddings is not None:
            user_emb = self.user_embeddings[user_ids]
            item_emb = self.item_embeddings[item_ids]
            similarity = (user_emb * item_emb).sum(axis=1)
            features.append(similarity.reshape(-1, 1))
        
        # User features
        user_feat_cols = [c for c in self.user_features.columns if c != 'user_idx']
        user_feats = self.user_features.set_index('user_idx').loc[user_ids][user_feat_cols].values
        features.append(user_feats)
        
        # Item features (numeric only)
        item_numeric_cols = [
            c for c in self.item_features.columns 
            if c not in ['item_idx', 'movieId', 'clean_title', 'tags_text', 'genres_list']
            and self.item_features[c].dtype in [np.float64, np.int64, np.float32, np.int32]
        ]
        item_feats = self.item_features.set_index('item_idx').loc[item_ids][item_numeric_cols].values
        features.append(item_feats)
        
        # Combine
        combined = np.hstack(features)
        
        # Handle NaNs
        combined = np.nan_to_num(combined, nan=0.0)
        
        return combined.astype(np.float32)
    
    def build_training_data(
        self,
        interactions: pd.DataFrame,
        negative_ratio: int = 4
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build labeled training data for ranking."""
        logger.info("Building ranking training data...")
        
        # Positive samples
        pos_users = interactions['user_idx'].values
        pos_items = interactions['item_idx'].values
        pos_labels = np.ones(len(interactions))
        
        # Negative samples
        all_items = self.item_features['item_idx'].unique()
        user_items = interactions.groupby('user_idx')['item_idx'].apply(set).to_dict()
        
        neg_users, neg_items = [], []
        for user_idx in np.unique(pos_users):
            seen_items = user_items.get(user_idx, set())
            neg_candidates = np.setdiff1d(all_items, list(seen_items))
            
            n_neg = min(negative_ratio * len(seen_items), len(neg_candidates))
            if n_neg > 0:
                sampled = np.random.choice(neg_candidates, n_neg, replace=False)
                neg_users.extend([user_idx] * n_neg)
                neg_items.extend(sampled)
        
        neg_labels = np.zeros(len(neg_users))
        
        # Combine
        all_users = np.concatenate([pos_users, np.array(neg_users)])
        all_items = np.concatenate([pos_items, np.array(neg_items)])
        all_labels = np.concatenate([pos_labels, neg_labels])
        
        # Build features
        features = self.build_pairwise_features(all_users, all_items)
        
        # Shuffle
        indices = np.random.permutation(len(all_labels))
        features = features[indices]
        all_labels = all_labels[indices]
        
        logger.info(f"Built {len(features)} samples ({pos_labels.sum():.0f} positive, {neg_labels.sum():.0f} negative)")
        
        return features, all_labels


def main():
    """Test feature building."""
    builder = RankingFeatureBuilder()
    builder.load_data()
    
    # Test with sample data
    user_ids = np.array([0, 1, 2, 3, 4])
    item_ids = np.array([0, 1, 2, 3, 4])
    
    features = builder.build_pairwise_features(user_ids, item_ids)
    print(f"Feature shape: {features.shape}")


if __name__ == "__main__":
    main()
