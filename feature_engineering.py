"""
Feature Engineering Pipeline
=============================
Build rich features for users, items, and interactions.

Features:
- User behavior statistics
- Item popularity and genre features
- Text embeddings using sentence transformers
- Feature crossing for ranking
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureConfig:
    """Feature engineering configuration."""
    
    def __init__(self, base_path: str = None):
        if base_path is None:
            base_path = Path(__file__).parent
        self.base_path = Path(base_path)
        
        self.processed_path = self.base_path / "data" / "processed"
        self.features_path = self.base_path / "data" / "features"
        self.embeddings_path = self.base_path / "embeddings"
        
        # Feature parameters
        self.user_embedding_dim = 128
        self.item_embedding_dim = 128
        self.text_embedding_dim = 384  # sentence-transformers default
        self.genre_embedding_dim = 20
        self.tfidf_max_features = 1000
        
        # Create directories
        self.features_path.mkdir(parents=True, exist_ok=True)
        self.embeddings_path.mkdir(parents=True, exist_ok=True)


class UserFeatureBuilder:
    """
    Build user-side features from interaction history.
    
    Features:
    - Interaction statistics (count, avg rating, std)
    - Genre affinity scores
    - Temporal patterns (avg hour, day distribution)
    - Activity recency
    """
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        
    def build(
        self, 
        interactions: pd.DataFrame,
        movies: pd.DataFrame
    ) -> pd.DataFrame:
        """Build user features from interactions."""
        logger.info("Building user features...")
        
        # Merge movie info
        data = interactions.merge(
            movies[['movieId', 'genres_list']], 
            left_on='item_idx', 
            right_on='movieId',
            how='left'
        )
        
        features = []
        
        # Group by user
        user_groups = data.groupby('user_idx')
        
        for user_idx, group in tqdm(user_groups, desc="Building user features"):
            user_feat = {'user_idx': user_idx}
            
            # Basic statistics
            user_feat['user_rating_count'] = len(group)
            user_feat['user_avg_rating'] = group['rating'].mean()
            user_feat['user_std_rating'] = group['rating'].std() if len(group) > 1 else 0
            user_feat['user_min_rating'] = group['rating'].min()
            user_feat['user_max_rating'] = group['rating'].max()
            
            # Rating distribution
            for r in [1, 2, 3, 4, 5]:
                user_feat[f'user_rating_{r}_ratio'] = (
                    (group['rating'] >= r - 0.5) & (group['rating'] < r + 0.5)
                ).mean()
            
            # Temporal features
            if 'hour_of_day' in group.columns:
                user_feat['user_avg_hour'] = group['hour_of_day'].mean()
                user_feat['user_weekend_ratio'] = (group['day_of_week'] >= 5).mean()
            
            # Activity span
            if 'timestamp' in group.columns:
                user_feat['user_activity_days'] = (
                    group['timestamp'].max() - group['timestamp'].min()
                ) / 86400
                user_feat['user_rating_velocity'] = (
                    len(group) / max(user_feat['user_activity_days'], 1)
                )
            
            # Genre preferences (will be computed separately)
            features.append(user_feat)
        
        user_features = pd.DataFrame(features)
        
        # Add genre affinity scores
        user_features = self._add_genre_affinity(user_features, data)
        
        logger.info(f"Built {len(user_features)} user feature vectors with {len(user_features.columns)} features")
        return user_features
    
    def _add_genre_affinity(
        self, 
        user_features: pd.DataFrame,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate per-user genre affinity scores."""
        logger.info("Computing genre affinity scores...")
        
        # Get all unique genres
        all_genres = set()
        for genres in data['genres_list'].dropna():
            if isinstance(genres, list):
                all_genres.update(genres)
        all_genres = sorted(all_genres - {'(no genres listed)'})
        
        # Initialize genre columns
        for genre in all_genres:
            user_features[f'genre_affinity_{genre}'] = 0.0
        
        # Calculate weighted genre affinity per user
        genre_affinity = defaultdict(lambda: defaultdict(float))
        genre_counts = defaultdict(lambda: defaultdict(int))
        
        for _, row in tqdm(data.iterrows(), total=len(data), desc="Computing genre affinity"):
            user_idx = row['user_idx']
            rating = row['rating']
            genres = row['genres_list']
            
            if isinstance(genres, list):
                for genre in genres:
                    if genre in all_genres:
                        genre_affinity[user_idx][genre] += rating
                        genre_counts[user_idx][genre] += 1
        
        # Normalize and assign
        for user_idx in genre_affinity:
            for genre in genre_affinity[user_idx]:
                count = genre_counts[user_idx][genre]
                if count > 0:
                    avg_rating = genre_affinity[user_idx][genre] / count
                    user_features.loc[
                        user_features['user_idx'] == user_idx,
                        f'genre_affinity_{genre}'
                    ] = avg_rating
        
        return user_features


class ItemFeatureBuilder:
    """
    Build item-side features.
    
    Features:
    - Popularity statistics
    - Genre multi-hot encoding
    - Rating distribution
    - Age (years since release)
    - Tag TF-IDF features
    - Text embeddings (optional)
    """
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.genre_encoder = MultiLabelBinarizer()
        self.tfidf = TfidfVectorizer(max_features=config.tfidf_max_features)
        
    def build(
        self,
        interactions: pd.DataFrame,
        movies: pd.DataFrame,
        include_text_embeddings: bool = False
    ) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        """Build item features."""
        logger.info("Building item features...")
        
        # Aggregate interaction statistics per item
        item_stats = interactions.groupby('item_idx').agg({
            'rating': ['count', 'mean', 'std', 'min', 'max'],
            'user_idx': 'nunique'
        }).reset_index()
        
        item_stats.columns = [
            'item_idx', 'item_rating_count', 'item_avg_rating', 
            'item_std_rating', 'item_min_rating', 'item_max_rating',
            'item_unique_users'
        ]
        item_stats['item_std_rating'] = item_stats['item_std_rating'].fillna(0)
        
        # Rating distribution per item
        rating_dist = interactions.groupby('item_idx')['rating'].apply(
            lambda x: pd.cut(x, bins=[0, 1.5, 2.5, 3.5, 4.5, 5.5], 
                           labels=['1', '2', '3', '4', '5']).value_counts(normalize=True)
        ).unstack(fill_value=0).reset_index()
        rating_dist.columns = ['item_idx'] + [f'item_rating_{i}_ratio' for i in ['1', '2', '3', '4', '5']]
        
        item_stats = item_stats.merge(rating_dist, on='item_idx', how='left')
        
        # Merge with movie metadata
        # First, create item_idx to movieId mapping
        item_movie_map = interactions[['item_idx', 'movieId']].drop_duplicates()
        movies_mapped = movies.merge(item_movie_map, on='movieId', how='inner')
        
        item_features = item_stats.merge(
            movies_mapped[['item_idx', 'movieId', 'clean_title', 'year', 'genres_list', 'tags_text']],
            on='item_idx',
            how='left'
        )
        
        # Item age
        current_year = 2024
        item_features['item_age'] = current_year - item_features['year'].fillna(current_year)
        
        # Popularity percentile
        item_features['item_popularity_percentile'] = (
            item_features['item_rating_count'].rank(pct=True)
        )
        
        # Genre encoding
        genres_clean = item_features['genres_list'].apply(
            lambda x: x if isinstance(x, list) else []
        )
        genre_matrix = self.genre_encoder.fit_transform(genres_clean)
        genre_df = pd.DataFrame(
            genre_matrix,
            columns=[f'genre_{g}' for g in self.genre_encoder.classes_]
        )
        item_features = pd.concat([item_features.reset_index(drop=True), genre_df], axis=1)
        
        # Tag TF-IDF features
        tags_text = item_features['tags_text'].fillna('')
        if tags_text.str.len().sum() > 0:
            tfidf_matrix = self.tfidf.fit_transform(tags_text)
            # Keep top features as dense
            tfidf_dense = tfidf_matrix.toarray()[:, :50]  # Top 50 TF-IDF features
            tfidf_df = pd.DataFrame(
                tfidf_dense,
                columns=[f'tfidf_{i}' for i in range(tfidf_dense.shape[1])]
            )
            item_features = pd.concat([item_features.reset_index(drop=True), tfidf_df], axis=1)
        
        # Text embeddings (optional - requires sentence-transformers)
        text_embeddings = None
        if include_text_embeddings:
            text_embeddings = self._generate_text_embeddings(item_features)
        
        logger.info(f"Built {len(item_features)} item feature vectors with {len(item_features.columns)} features")
        
        return item_features, text_embeddings
    
    def _generate_text_embeddings(self, item_features: pd.DataFrame) -> np.ndarray:
        """Generate text embeddings using sentence transformers."""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info("Generating text embeddings with sentence-transformers...")
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Combine title and tags
            texts = (
                item_features['clean_title'].fillna('') + ' ' +
                item_features['tags_text'].fillna('')
            ).tolist()
            
            embeddings = model.encode(texts, show_progress_bar=True, batch_size=256)
            
            logger.info(f"Generated text embeddings: {embeddings.shape}")
            return embeddings
            
        except ImportError:
            logger.warning("sentence-transformers not installed, skipping text embeddings")
            return None


class FeatureEngineer:
    """
    Main feature engineering orchestrator.
    """
    
    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()
        self.user_builder = UserFeatureBuilder(self.config)
        self.item_builder = ItemFeatureBuilder(self.config)
        self.scaler = StandardScaler()
        
    def process(self, include_text_embeddings: bool = False) -> Dict[str, str]:
        """Run full feature engineering pipeline."""
        logger.info("=" * 60)
        logger.info("Starting Feature Engineering Pipeline")
        logger.info("=" * 60)
        
        # Load processed data
        logger.info("Loading processed data...")
        interactions = pd.read_parquet(self.config.processed_path / "train.parquet")
        movies = pd.read_parquet(self.config.processed_path / "movies.parquet")
        
        # Build user features
        user_features = self.user_builder.build(interactions, movies)
        
        # Build item features
        item_features, text_embeddings = self.item_builder.build(
            interactions, movies, include_text_embeddings
        )
        
        # Save features
        logger.info("Saving features...")
        user_features.to_parquet(self.config.features_path / "user_features.parquet", index=False)
        item_features.to_parquet(self.config.features_path / "item_features.parquet", index=False)
        
        if text_embeddings is not None:
            np.save(self.config.embeddings_path / "text_embeddings.npy", text_embeddings)
        
        # Save genre encoder
        import pickle
        with open(self.config.features_path / "genre_encoder.pkl", 'wb') as f:
            pickle.dump(self.item_builder.genre_encoder, f)
        
        output_paths = {
            'user_features': str(self.config.features_path / "user_features.parquet"),
            'item_features': str(self.config.features_path / "item_features.parquet"),
        }
        
        if text_embeddings is not None:
            output_paths['text_embeddings'] = str(self.config.embeddings_path / "text_embeddings.npy")
        
        logger.info("=" * 60)
        logger.info("Feature Engineering Complete!")
        logger.info(f"User features: {user_features.shape}")
        logger.info(f"Item features: {item_features.shape}")
        logger.info("=" * 60)
        
        return output_paths


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Feature Engineering")
    parser.add_argument(
        '--text-embeddings',
        action='store_true',
        help='Generate text embeddings (requires sentence-transformers)'
    )
    parser.add_argument(
        '--base-path',
        type=str,
        default=None,
        help='Base path for data directories'
    )
    args = parser.parse_args()
    
    config = FeatureConfig(base_path=args.base_path)
    engineer = FeatureEngineer(config)
    paths = engineer.process(include_text_embeddings=args.text_embeddings)
    
    return paths


if __name__ == "__main__":
    main()
