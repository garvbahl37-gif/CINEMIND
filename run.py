#!/usr/bin/env python
"""
FAANG-Grade Movie Recommender System
======================================
Main pipeline script to run the full system.

Usage:
    python run.py preprocess [--sample N]
    python run.py features [--text-embeddings]
    python run.py train [--epochs N] [--batch-size N]
    python run.py build-index [--benchmark]
    python run.py serve [--port N]
    python run.py full [--sample N]  # Run everything
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_preprocessing(args):
    """Run data preprocessing."""
    from preprocessing import MovieLensPreprocessor, DataConfig
    
    logger.info("="*60)
    logger.info("STEP 1: Data Preprocessing")
    logger.info("="*60)
    
    config = DataConfig()
    preprocessor = MovieLensPreprocessor(config, sample_size=args.sample)
    stats = preprocessor.process()
    
    return stats


def run_feature_engineering(args):
    """Run feature engineering."""
    from feature_engineering import FeatureEngineer, FeatureConfig
    
    logger.info("="*60)
    logger.info("STEP 2: Feature Engineering")
    logger.info("="*60)
    
    config = FeatureConfig()
    engineer = FeatureEngineer(config)
    paths = engineer.process(include_text_embeddings=args.text_embeddings)
    
    return paths


def run_training(args):
    """Run model training."""
    logger.info("="*60)
    logger.info("STEP 3: Two-Tower Model Training")
    logger.info("="*60)
    
    # Import here to avoid circular imports
    import sys
    sys.path.insert(0, str(Path(__file__).parent / "candidate_generation" / "two_tower"))
    
    from candidate_generation.two_tower.train import Trainer, TrainingConfig
    
    config = TrainingConfig()
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    
    trainer = Trainer(config)
    trainer.setup()
    trainer.train()
    trainer.export_embeddings()
    
    return trainer


def run_index_building(args):
    """Build FAISS index."""
    logger.info("="*60)
    logger.info("STEP 4: FAISS Index Building")
    logger.info("="*60)
    
    from candidate_generation.ann.build_index import IndexBuilder, IndexBuildConfig
    
    config = IndexBuildConfig()
    builder = IndexBuilder(config)
    
    manager, results = builder.run(
        benchmark=args.benchmark,
        production_type=args.index_type
    )
    
    return manager, results


def run_serving(args):
    """Start API server."""
    logger.info("="*60)
    logger.info("STEP 5: Starting API Server")
    logger.info("="*60)
    
    from serving.api.main import run_server
    run_server(host="0.0.0.0", port=args.port)


def run_evaluation(args):
    """Run offline evaluation."""
    logger.info("="*60)
    logger.info("Offline Evaluation")
    logger.info("="*60)
    
    import numpy as np
    import pandas as pd
    from serving.recommendation_service import RecommendationService
    from evaluation.offline_metrics import RecommenderEvaluator
    
    base_path = Path(__file__).parent
    
    # Load service
    service = RecommendationService(base_path)
    service.load()
    
    # Load test data
    test_data = pd.read_parquet(base_path / "data" / "processed" / "test.parquet")
    
    # Build ground truth
    ground_truth = test_data.groupby('user_idx')['item_idx'].apply(set).to_dict()
    
    # Generate predictions
    predictions = {}
    test_users = list(ground_truth.keys())[:1000]  # Sample for speed
    
    for user_idx in test_users:
        try:
            result = service.recommend(user_idx, k=100)
            predictions[user_idx] = [r['item_id'] for r in result['recommendations']]
        except:
            pass
    
    # Evaluate
    evaluator = RecommenderEvaluator()
    results = evaluator.evaluate(
        predictions=predictions,
        ground_truth=ground_truth,
        item_embeddings=service._item_embeddings,
        catalog_size=service._metadata['num_items']
    )
    
    evaluator.print_results(results)
    
    return results


def run_full_pipeline(args):
    """Run the complete pipeline."""
    logger.info("="*60)
    logger.info("FAANG-Grade Recommender System - Full Pipeline")
    logger.info("="*60)
    
    # Step 1: Preprocessing
    run_preprocessing(args)
    
    # Step 2: Feature Engineering
    args.text_embeddings = False  # Skip for speed
    run_feature_engineering(args)
    
    # Step 3: Training
    run_training(args)
    
    # Step 4: Index Building
    args.benchmark = False
    args.index_type = 'ivf_pq'
    run_index_building(args)
    
    logger.info("="*60)
    logger.info("Pipeline Complete!")
    logger.info("="*60)
    logger.info("To start the API server, run: python run.py serve")


def main():
    parser = argparse.ArgumentParser(
        description="FAANG-Grade Movie Recommender System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Preprocess
    preprocess_parser = subparsers.add_parser('preprocess', help='Run data preprocessing')
    preprocess_parser.add_argument('--sample', type=int, default=None, help='Sample size for testing')
    
    # Features
    features_parser = subparsers.add_parser('features', help='Run feature engineering')
    features_parser.add_argument('--text-embeddings', action='store_true', help='Generate text embeddings')
    
    # Train
    train_parser = subparsers.add_parser('train', help='Train two-tower model')
    train_parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, default=2048, help='Batch size')
    
    # Build index
    index_parser = subparsers.add_parser('build-index', help='Build FAISS index')
    index_parser.add_argument('--benchmark', action='store_true', help='Benchmark all index types')
    index_parser.add_argument('--index-type', type=str, default='ivf_pq', help='Index type')
    
    # Serve
    serve_parser = subparsers.add_parser('serve', help='Start API server')
    serve_parser.add_argument('--port', type=int, default=8000, help='Server port')
    
    # Evaluate
    eval_parser = subparsers.add_parser('evaluate', help='Run offline evaluation')
    
    # Full pipeline
    full_parser = subparsers.add_parser('full', help='Run full pipeline')
    full_parser.add_argument('--sample', type=int, default=None, help='Sample size for testing')
    full_parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    full_parser.add_argument('--batch-size', type=int, default=2048, help='Batch size')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Change to project directory
    import os
    os.chdir(Path(__file__).parent)
    
    # Run command
    if args.command == 'preprocess':
        run_preprocessing(args)
    elif args.command == 'features':
        run_feature_engineering(args)
    elif args.command == 'train':
        run_training(args)
    elif args.command == 'build-index':
        run_index_building(args)
    elif args.command == 'serve':
        run_serving(args)
    elif args.command == 'evaluate':
        run_evaluation(args)
    elif args.command == 'full':
        run_full_pipeline(args)


if __name__ == "__main__":
    main()
