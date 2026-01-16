"""
LightGBM Ranking Model
=======================
LambdaRank-based ranking model for scoring candidates.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pickle

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LGBMRanker:
    """LightGBM-based ranking model."""
    
    def __init__(
        self,
        objective: str = 'lambdarank',
        num_leaves: int = 31,
        learning_rate: float = 0.05,
        n_estimators: int = 200,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42
    ):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("lightgbm required. Install with: pip install lightgbm")
        
        self.params = {
            'objective': objective,
            'num_leaves': num_leaves,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'min_child_samples': min_child_samples,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'random_state': random_state,
            'metric': 'ndcg',
            'ndcg_eval_at': [10, 50, 100],
            'verbosity': -1
        }
        
        self.model = None
        self.feature_importance = None
        
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        group_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        group_val: Optional[np.ndarray] = None
    ):
        """
        Train ranking model.
        
        Args:
            X_train: (n_samples, n_features) training features
            y_train: (n_samples,) training labels (relevance)
            group_train: (n_groups,) number of samples per group
            X_val, y_val, group_val: Optional validation data
        """
        logger.info(f"Training LightGBM ranker on {len(X_train)} samples...")
        
        train_set = lgb.Dataset(X_train, label=y_train, group=group_train)
        
        valid_sets = [train_set]
        valid_names = ['train']
        
        if X_val is not None:
            val_set = lgb.Dataset(X_val, label=y_val, group=group_val, reference=train_set)
            valid_sets.append(val_set)
            valid_names.append('valid')
        
        callbacks = [
            lgb.early_stopping(stopping_rounds=20),
            lgb.log_evaluation(period=20)
        ]
        
        self.model = lgb.train(
            self.params,
            train_set,
            num_boost_round=self.params['n_estimators'],
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )
        
        # Feature importance
        self.feature_importance = dict(zip(
            range(X_train.shape[1]),
            self.model.feature_importance(importance_type='gain')
        ))
        
        logger.info(f"Training complete. Best iteration: {self.model.best_iteration}")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Score items."""
        if self.model is None:
            raise RuntimeError("Model not trained")
        return self.model.predict(X)
    
    def save(self, path: str):
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save_model(str(path.with_suffix('.txt')))
        
        with open(path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump({'params': self.params, 'feature_importance': self.feature_importance}, f)
            
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model from disk."""
        path = Path(path)
        
        self.model = lgb.Booster(model_file=str(path.with_suffix('.txt')))
        
        with open(path.with_suffix('.pkl'), 'rb') as f:
            meta = pickle.load(f)
            self.params = meta['params']
            self.feature_importance = meta['feature_importance']
            
        logger.info(f"Model loaded from {path}")


def train_lgbm_ranker(
    base_path: str = None,
    val_ratio: float = 0.1
) -> LGBMRanker:
    """Train LightGBM ranking model from processed data."""
    from features import RankingFeatureBuilder
    
    if base_path is None:
        base_path = Path(__file__).parent.parent
    base_path = Path(base_path)
    
    # Load training data
    train_data = pd.read_parquet(base_path / "data" / "processed" / "train.parquet")
    
    # Build features
    builder = RankingFeatureBuilder(base_path)
    builder.load_data()
    X, y = builder.build_training_data(train_data)
    
    # Create groups (group by user for learning to rank)
    # For simplicity, use fixed group size
    group_size = 10
    n_groups = len(y) // group_size
    groups = np.full(n_groups, group_size)
    
    X = X[:n_groups * group_size]
    y = y[:n_groups * group_size]
    
    # Train/val split (by groups)
    n_train_groups = int(n_groups * (1 - val_ratio))
    
    X_train = X[:n_train_groups * group_size]
    y_train = y[:n_train_groups * group_size]
    group_train = groups[:n_train_groups]
    
    X_val = X[n_train_groups * group_size:]
    y_val = y[n_train_groups * group_size:]
    group_val = groups[n_train_groups:]
    
    # Train
    ranker = LGBMRanker()
    ranker.train(X_train, y_train, group_train, X_val, y_val, group_val)
    
    # Save
    model_path = base_path / "checkpoints" / "lgbm_ranker"
    ranker.save(str(model_path))
    
    return ranker


if __name__ == "__main__":
    train_lgbm_ranker()
