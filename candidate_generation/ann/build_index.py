"""
FAISS Index Building Pipeline
==============================
Build and benchmark optimized FAISS indices for production.

Features:
- Load embeddings from trained model
- Build multiple index types for comparison
- Benchmark recall vs latency
- Save production-ready index
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional
import time

import numpy as np

from faiss_index import (
    FAISSIndexManager, 
    IndexConfig, 
    IndexType,
    create_optimized_index
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IndexBuildConfig:
    """Configuration for index building."""
    
    def __init__(self, base_path: str = None):
        if base_path is None:
            base_path = Path(__file__).parent.parent.parent
        self.base_path = Path(base_path)
        
        self.embeddings_path = self.base_path / "embeddings"
        self.index_path = self.base_path / "indices"
        
        # Index configurations to benchmark
        self.index_configs = {
            'flat': IndexConfig(index_type=IndexType.FLAT),
            'ivf_flat': IndexConfig(
                index_type=IndexType.IVF_FLAT,
                nlist=256,
                nprobe=16
            ),
            'ivf_pq': IndexConfig(
                index_type=IndexType.IVF_PQ,
                nlist=256,
                m=8,
                nbits=8,
                nprobe=16
            ),
            'hnsw': IndexConfig(
                index_type=IndexType.HNSW,
                M=32,
                ef_construction=200,
                ef_search=64
            ),
        }
        
        # Benchmark settings
        self.num_test_queries = 1000
        self.k_values = [10, 50, 100]
        self.nprobe_values = [8, 16, 32, 64]
        
        # Target metrics
        self.target_recall_100 = 0.95
        self.target_latency_ms = 5.0
        
        # Create directories
        self.index_path.mkdir(parents=True, exist_ok=True)


class IndexBuilder:
    """Build and benchmark FAISS indices."""
    
    def __init__(self, config: IndexBuildConfig = None):
        self.config = config or IndexBuildConfig()
        self.embeddings = None
        self.test_queries = None
        self.ground_truth = None
        
    def load_embeddings(self):
        """Load item embeddings from disk."""
        embeddings_file = self.config.embeddings_path / "item_embeddings.npy"
        
        if not embeddings_file.exists():
            raise FileNotFoundError(
                f"Item embeddings not found at {embeddings_file}. "
                "Run training with --export flag first."
            )
        
        self.embeddings = np.load(embeddings_file).astype(np.float32)
        logger.info(f"Loaded embeddings: {self.embeddings.shape}")
        
        # Create test queries (sample from embeddings + random perturbations)
        np.random.seed(42)
        query_indices = np.random.choice(
            len(self.embeddings), 
            self.config.num_test_queries, 
            replace=False
        )
        self.test_queries = self.embeddings[query_indices].copy()
        
        # Add small noise to queries for realistic testing
        noise = np.random.randn(*self.test_queries.shape).astype(np.float32) * 0.01
        self.test_queries += noise
        
        # Normalize queries
        norms = np.linalg.norm(self.test_queries, axis=1, keepdims=True)
        self.test_queries /= norms
        
        return self
    
    def compute_ground_truth(self, k: int = 100):
        """Compute exact nearest neighbors for benchmarking."""
        logger.info(f"Computing ground truth (exact search) for {len(self.test_queries)} queries...")
        
        # Build flat index for exact search
        flat_config = IndexConfig(index_type=IndexType.FLAT)
        flat_manager = FAISSIndexManager(flat_config)
        flat_manager.build_index(self.embeddings.copy())
        
        # Get exact neighbors
        _, self.ground_truth = flat_manager.search(self.test_queries, k=k)
        logger.info(f"Ground truth computed: {self.ground_truth.shape}")
        
        return self
    
    def build_and_benchmark(self) -> Dict[str, Dict]:
        """Build all index types and benchmark."""
        results = {}
        
        for name, config in self.config.index_configs.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Building index: {name}")
            logger.info(f"{'='*60}")
            
            try:
                # Build index
                start = time.time()
                manager = FAISSIndexManager(config)
                manager.build_index(self.embeddings.copy())
                build_time = time.time() - start
                
                # Benchmark
                if config.index_type in [IndexType.IVF_FLAT, IndexType.IVF_PQ, IndexType.IVF_SQ]:
                    benchmark_results = manager.benchmark(
                        self.test_queries,
                        self.ground_truth,
                        self.config.k_values,
                        self.config.nprobe_values
                    )
                else:
                    # Single config benchmark for non-IVF indices
                    start = time.time()
                    _, indices = manager.search(self.test_queries, k=100)
                    latency_ms = (time.time() - start) / len(self.test_queries) * 1000
                    
                    recall_100 = self._compute_recall(indices, self.ground_truth)
                    
                    benchmark_results = {
                        'default': {
                            'latency_ms': latency_ms,
                            'recall@100': recall_100
                        }
                    }
                
                results[name] = {
                    'build_time_s': build_time,
                    'memory_mb': manager._get_memory_usage(),
                    'benchmarks': benchmark_results
                }
                
                # Save the index
                index_file = self.config.index_path / f"{name}.index"
                manager.save(index_file)
                results[name]['index_file'] = str(index_file)
                
            except Exception as e:
                logger.error(f"Failed to build {name}: {e}")
                results[name] = {'error': str(e)}
        
        return results
    
    def _compute_recall(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """Compute recall."""
        recalls = []
        for pred, gt in zip(predictions, ground_truth):
            recall = len(set(pred) & set(gt)) / len(gt) if len(gt) > 0 else 0
            recalls.append(recall)
        return np.mean(recalls)
    
    def select_best_index(self, results: Dict) -> str:
        """
        Select best index based on recall/latency tradeoff.
        
        Priority:
        1. Meet target recall@100 >= 0.95
        2. Minimize latency
        3. Minimize memory
        """
        candidates = []
        
        for name, result in results.items():
            if 'error' in result:
                continue
                
            benchmarks = result.get('benchmarks', {})
            
            for config_name, metrics in benchmarks.items():
                recall = metrics.get('recall@100', 0)
                latency = metrics.get('latency_ms', float('inf'))
                memory = result.get('memory_mb', float('inf'))
                
                if recall >= self.config.target_recall_100:
                    candidates.append({
                        'name': name,
                        'config': config_name,
                        'recall@100': recall,
                        'latency_ms': latency,
                        'memory_mb': memory
                    })
        
        if not candidates:
            # Fallback: select highest recall
            logger.warning("No index met target recall. Selecting highest recall.")
            for name, result in results.items():
                if 'error' in result:
                    continue
                for config_name, metrics in result.get('benchmarks', {}).items():
                    candidates.append({
                        'name': name,
                        'config': config_name,
                        'recall@100': metrics.get('recall@100', 0),
                        'latency_ms': metrics.get('latency_ms', float('inf')),
                        'memory_mb': result.get('memory_mb', float('inf'))
                    })
        
        if not candidates:
            return None
        
        # Sort by latency (prefer low latency among those meeting recall)
        candidates.sort(key=lambda x: (x['latency_ms'], -x['recall@100']))
        
        best = candidates[0]
        logger.info(f"\nBest index: {best['name']} ({best['config']})")
        logger.info(f"  Recall@100: {best['recall@100']:.4f}")
        logger.info(f"  Latency: {best['latency_ms']:.2f}ms")
        logger.info(f"  Memory: {best['memory_mb']:.2f}MB")
        
        return best['name']
    
    def build_production_index(self, index_type: str = None):
        """Build and save production index."""
        logger.info("\n" + "="*60)
        logger.info("Building Production Index")
        logger.info("="*60)
        
        if index_type is None:
            # Use IVF+PQ as default for production
            config = self.config.index_configs.get('ivf_pq')
            if config is None:
                config = IndexConfig.for_catalog_size(len(self.embeddings))
        else:
            config = self.config.index_configs.get(index_type)
            if config is None:
                raise ValueError(f"Unknown index type: {index_type}")
        
        # Build optimized index
        manager = FAISSIndexManager(config)
        manager.build_index(self.embeddings.copy())
        
        # Tune nprobe for optimal recall
        if hasattr(manager.index, 'nprobe'):
            logger.info("Tuning nprobe...")
            best_nprobe = 16
            best_recall = 0
            
            for nprobe in [8, 16, 32, 64, 128]:
                manager.set_nprobe(nprobe)
                _, indices = manager.search(self.test_queries[:100], k=100)
                recall = self._compute_recall(indices, self.ground_truth[:100])
                
                if recall >= 0.95 and nprobe < best_nprobe * 2:
                    best_nprobe = nprobe
                    best_recall = recall
                    break
                    
                if recall > best_recall:
                    best_nprobe = nprobe
                    best_recall = recall
            
            manager.set_nprobe(best_nprobe)
            logger.info(f"Selected nprobe={best_nprobe}, recall@100={best_recall:.4f}")
        
        # Save production index
        prod_index_path = self.config.index_path / "production.index"
        manager.save(prod_index_path)
        
        # Save config
        config_dict = {
            'index_type': config.index_type.value,
            'nlist': config.nlist,
            'nprobe': getattr(manager.index, 'nprobe', None),
            'm': config.m,
            'nbits': config.nbits,
            'num_vectors': len(self.embeddings),
            'dim': self.embeddings.shape[1]
        }
        
        with open(self.config.index_path / "production_config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Production index saved to: {prod_index_path}")
        
        return manager
    
    def run(self, benchmark: bool = True, production_type: str = 'ivf_pq'):
        """Run full index building pipeline."""
        logger.info("="*60)
        logger.info("FAISS Index Building Pipeline")
        logger.info("="*60)
        
        # Load embeddings
        self.load_embeddings()
        
        # Compute ground truth
        self.compute_ground_truth(k=100)
        
        results = {}
        best_index = None
        
        if benchmark:
            # Build and benchmark all indices
            results = self.build_and_benchmark()
            
            # Select best
            best_index = self.select_best_index(results)
            
            # Save benchmark results
            results_file = self.config.index_path / "benchmark_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Benchmark results saved to: {results_file}")
        
        # Build production index
        production_type = best_index or production_type
        manager = self.build_production_index(production_type)
        
        logger.info("\n" + "="*60)
        logger.info("Index Building Complete!")
        logger.info("="*60)
        
        return manager, results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build FAISS Index")
    parser.add_argument(
        '--benchmark', 
        action='store_true',
        help='Run full benchmark across index types'
    )
    parser.add_argument(
        '--index-type',
        type=str,
        default='ivf_pq',
        choices=['flat', 'ivf_flat', 'ivf_pq', 'hnsw'],
        help='Index type to build'
    )
    parser.add_argument(
        '--base-path',
        type=str,
        default=None,
        help='Base path for data directories'
    )
    args = parser.parse_args()
    
    config = IndexBuildConfig(base_path=args.base_path)
    builder = IndexBuilder(config)
    
    try:
        manager, results = builder.run(
            benchmark=args.benchmark,
            production_type=args.index_type
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.info("Please run training first: python -m candidate_generation.two_tower.train --export")
        return
    
    return manager, results


if __name__ == "__main__":
    main()
