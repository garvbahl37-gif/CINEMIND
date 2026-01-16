"""
FAANG-Grade FAISS Index Manager
================================
Production-ready approximate nearest neighbor search with:
- Multiple index types (Flat, IVF, IVF+PQ, HNSW, ScalarQuantizer)
- Automatic index selection based on catalog size
- Multi-probe search optimization
- GPU acceleration support
- Recall vs latency benchmarking
- Index persistence and loading
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("WARNING: faiss not installed. Install with: pip install faiss-cpu")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IndexType(Enum):
    """Supported FAISS index types."""
    FLAT = "flat"                    # Exact search (baseline)
    IVF_FLAT = "ivf_flat"            # Inverted file index
    IVF_PQ = "ivf_pq"                # IVF with Product Quantization
    IVF_SQ = "ivf_sq"                # IVF with Scalar Quantization
    HNSW = "hnsw"                    # Hierarchical Navigable Small World
    HNSW_PQ = "hnsw_pq"              # HNSW with Product Quantization


@dataclass
class IndexConfig:
    """Configuration for FAISS index."""
    
    # Index type
    index_type: IndexType = IndexType.IVF_PQ
    
    # IVF parameters
    nlist: int = 256                 # Number of clusters
    nprobe: int = 16                 # Number of clusters to search
    
    # PQ parameters
    m: int = 8                       # Number of subquantizers
    nbits: int = 8                   # Bits per subquantizer code
    
    # HNSW parameters
    M: int = 32                      # Connections per layer
    ef_construction: int = 200       # Search depth during construction
    ef_search: int = 64              # Search depth during query
    
    # SQ parameters
    sq_type: str = "QT_8bit"         # Quantization type
    
    # Hardware
    use_gpu: bool = False
    gpu_id: int = 0
    
    # Training
    train_size: int = 100000         # Max vectors for training
    
    @classmethod
    def for_catalog_size(cls, num_items: int) -> 'IndexConfig':
        """
        Auto-configure based on catalog size.
        
        Guidelines:
        - < 10K items: Use Flat (exact)
        - 10K - 100K: Use IVF_Flat or HNSW
        - 100K - 1M: Use IVF_PQ
        - > 1M: Use IVF_PQ with more clusters
        """
        if num_items < 10_000:
            return cls(index_type=IndexType.FLAT)
        elif num_items < 100_000:
            return cls(
                index_type=IndexType.IVF_FLAT,
                nlist=int(np.sqrt(num_items)),
                nprobe=16
            )
        elif num_items < 1_000_000:
            return cls(
                index_type=IndexType.IVF_PQ,
                nlist=int(np.sqrt(num_items)),
                nprobe=32,
                m=8
            )
        else:
            return cls(
                index_type=IndexType.IVF_PQ,
                nlist=int(np.sqrt(num_items)),
                nprobe=64,
                m=16
            )


class FAISSIndexManager:
    """
    Production FAISS index manager.
    
    Supports multiple index types with automatic optimization
    based on recall/latency requirements.
    """
    
    def __init__(self, config: IndexConfig = None):
        if not FAISS_AVAILABLE:
            raise ImportError("faiss is required. Install with: pip install faiss-cpu")
            
        self.config = config or IndexConfig()
        self.index = None
        self.embeddings = None
        self.dim = None
        self.num_vectors = None
        self.is_trained = False
        
    def build_index(
        self, 
        embeddings: np.ndarray,
        normalize: bool = True
    ) -> 'FAISSIndexManager':
        """
        Build FAISS index from embeddings.
        
        Args:
            embeddings: (num_items, dim) float32 embeddings
            normalize: Whether to L2 normalize embeddings
            
        Returns:
            self for chaining
        """
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
        
        embeddings = embeddings.astype(np.float32)
        
        if normalize:
            faiss.normalize_L2(embeddings)
            
        self.embeddings = embeddings
        self.num_vectors, self.dim = embeddings.shape
        
        logger.info(f"Building {self.config.index_type.value} index for {self.num_vectors:,} vectors of dim {self.dim}")
        
        # Create index based on type
        if self.config.index_type == IndexType.FLAT:
            self.index = self._build_flat_index()
        elif self.config.index_type == IndexType.IVF_FLAT:
            self.index = self._build_ivf_flat_index()
        elif self.config.index_type == IndexType.IVF_PQ:
            self.index = self._build_ivf_pq_index()
        elif self.config.index_type == IndexType.IVF_SQ:
            self.index = self._build_ivf_sq_index()
        elif self.config.index_type == IndexType.HNSW:
            self.index = self._build_hnsw_index()
        elif self.config.index_type == IndexType.HNSW_PQ:
            self.index = self._build_hnsw_pq_index()
        else:
            raise ValueError(f"Unknown index type: {self.config.index_type}")
        
        # Move to GPU if requested
        if self.config.use_gpu:
            self._to_gpu()
        
        self.is_trained = True
        logger.info(f"Index built successfully. Total memory: {self._get_memory_usage():.2f} MB")
        
        return self
    
    def _build_flat_index(self) -> faiss.Index:
        """Build exact search index (baseline)."""
        index = faiss.IndexFlatIP(self.dim)  # Inner product for cosine similarity
        index.add(self.embeddings)
        return index
    
    def _build_ivf_flat_index(self) -> faiss.Index:
        """Build IVF index with flat quantization."""
        quantizer = faiss.IndexFlatIP(self.dim)
        index = faiss.IndexIVFFlat(
            quantizer, 
            self.dim, 
            self.config.nlist,
            faiss.METRIC_INNER_PRODUCT
        )
        
        # Train on subset
        train_data = self._get_train_data()
        index.train(train_data)
        index.add(self.embeddings)
        index.nprobe = self.config.nprobe
        
        return index
    
    def _build_ivf_pq_index(self) -> faiss.Index:
        """
        Build IVF+PQ index - best for large catalogs.
        
        Memory efficient with good recall/latency tradeoff.
        """
        quantizer = faiss.IndexFlatIP(self.dim)
        index = faiss.IndexIVFPQ(
            quantizer,
            self.dim,
            self.config.nlist,
            self.config.m,
            self.config.nbits,
            faiss.METRIC_INNER_PRODUCT
        )
        
        # Train on subset
        train_data = self._get_train_data()
        index.train(train_data)
        index.add(self.embeddings)
        index.nprobe = self.config.nprobe
        
        return index
    
    def _build_ivf_sq_index(self) -> faiss.Index:
        """Build IVF+Scalar Quantizer index."""
        quantizer = faiss.IndexFlatIP(self.dim)
        
        # Map string to FAISS constant
        sq_map = {
            "QT_8bit": faiss.ScalarQuantizer.QT_8bit,
            "QT_4bit": faiss.ScalarQuantizer.QT_4bit,
            "QT_fp16": faiss.ScalarQuantizer.QT_fp16,
        }
        sq_type = sq_map.get(self.config.sq_type, faiss.ScalarQuantizer.QT_8bit)
        
        index = faiss.IndexIVFScalarQuantizer(
            quantizer,
            self.dim,
            self.config.nlist,
            sq_type,
            faiss.METRIC_INNER_PRODUCT
        )
        
        train_data = self._get_train_data()
        index.train(train_data)
        index.add(self.embeddings)
        index.nprobe = self.config.nprobe
        
        return index
    
    def _build_hnsw_index(self) -> faiss.Index:
        """Build HNSW index - best latency for medium catalogs."""
        index = faiss.IndexHNSWFlat(self.dim, self.config.M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self.config.ef_construction
        index.hnsw.efSearch = self.config.ef_search
        index.add(self.embeddings)
        return index
    
    def _build_hnsw_pq_index(self) -> faiss.Index:
        """Build HNSW+PQ index for memory efficiency."""
        # Create HNSW index
        index = faiss.IndexHNSWPQ(self.dim, self.config.m, self.config.M)
        index.hnsw.efConstruction = self.config.ef_construction
        index.hnsw.efSearch = self.config.ef_search
        
        # Train and add
        train_data = self._get_train_data()
        index.train(train_data)
        index.add(self.embeddings)
        
        return index
    
    def _get_train_data(self) -> np.ndarray:
        """Get training data subset."""
        if self.num_vectors <= self.config.train_size:
            return self.embeddings
        
        indices = np.random.choice(
            self.num_vectors, 
            self.config.train_size, 
            replace=False
        )
        return self.embeddings[indices]
    
    def _to_gpu(self):
        """Move index to GPU."""
        try:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, self.config.gpu_id, self.index)
            logger.info(f"Index moved to GPU {self.config.gpu_id}")
        except Exception as e:
            logger.warning(f"Failed to move to GPU: {e}")
    
    def _get_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        if self.config.index_type == IndexType.FLAT:
            return self.num_vectors * self.dim * 4 / 1e6
        elif self.config.index_type in [IndexType.IVF_PQ, IndexType.HNSW_PQ]:
            # PQ compressed
            return self.num_vectors * self.config.m * self.config.nbits / 8 / 1e6
        else:
            return self.num_vectors * self.dim * 4 / 1e6
    
    def search(
        self, 
        query: np.ndarray, 
        k: int = 100,
        nprobe: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for nearest neighbors.
        
        Args:
            query: (num_queries, dim) query vectors
            k: Number of neighbors to return
            nprobe: Override nprobe for this query
            
        Returns:
            distances: (num_queries, k) similarity scores
            indices: (num_queries, k) neighbor indices
        """
        if not self.is_trained:
            raise RuntimeError("Index not trained. Call build_index first.")
        
        query = np.ascontiguousarray(query.astype(np.float32))
        faiss.normalize_L2(query)
        
        # Temporarily update nprobe if specified
        if nprobe is not None and hasattr(self.index, 'nprobe'):
            old_nprobe = self.index.nprobe
            self.index.nprobe = nprobe
        
        distances, indices = self.index.search(query, k)
        
        # Restore nprobe
        if nprobe is not None and hasattr(self.index, 'nprobe'):
            self.index.nprobe = old_nprobe
        
        return distances, indices
    
    def search_batch(
        self,
        queries: np.ndarray,
        k: int = 100,
        batch_size: int = 1024
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Batch search for memory efficiency."""
        num_queries = len(queries)
        all_distances = []
        all_indices = []
        
        for i in range(0, num_queries, batch_size):
            batch = queries[i:i+batch_size]
            distances, indices = self.search(batch, k)
            all_distances.append(distances)
            all_indices.append(indices)
        
        return np.vstack(all_distances), np.vstack(all_indices)
    
    def set_nprobe(self, nprobe: int):
        """Update nprobe for IVF indices."""
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = nprobe
            logger.info(f"Set nprobe = {nprobe}")
    
    def set_ef_search(self, ef_search: int):
        """Update efSearch for HNSW indices."""
        if hasattr(self.index, 'hnsw'):
            self.index.hnsw.efSearch = ef_search
            logger.info(f"Set efSearch = {ef_search}")
    
    def save(self, path: Union[str, Path]):
        """Save index to disk."""
        path = Path(path)
        faiss.write_index(self.index, str(path))
        logger.info(f"Saved index to {path}")
    
    def load(self, path: Union[str, Path]) -> 'FAISSIndexManager':
        """Load index from disk."""
        path = Path(path)
        self.index = faiss.read_index(str(path))
        self.is_trained = True
        self.num_vectors = self.index.ntotal
        
        # Infer dimension
        if hasattr(self.index, 'd'):
            self.dim = self.index.d
        
        logger.info(f"Loaded index from {path} ({self.num_vectors:,} vectors)")
        return self
    
    def benchmark(
        self,
        query_vectors: np.ndarray,
        ground_truth: np.ndarray,
        k_values: List[int] = [10, 50, 100],
        nprobe_values: List[int] = None
    ) -> Dict[str, Dict]:
        """
        Benchmark recall and latency.
        
        Args:
            query_vectors: (num_queries, dim) test queries
            ground_truth: (num_queries, k) exact nearest neighbors
            k_values: K values to evaluate
            nprobe_values: nprobe values to test (for IVF)
            
        Returns:
            Dict with recall and latency metrics
        """
        if nprobe_values is None:
            nprobe_values = [8, 16, 32, 64, 128]
        
        results = {}
        
        for nprobe in nprobe_values:
            if hasattr(self.index, 'nprobe'):
                self.index.nprobe = nprobe
            
            # Warmup
            self.search(query_vectors[:10], k=100)
            
            # Time search
            start = time.time()
            distances, indices = self.search(query_vectors, k=max(k_values))
            elapsed = time.time() - start
            latency_ms = elapsed / len(query_vectors) * 1000
            
            # Compute recall at different K
            recalls = {}
            for k in k_values:
                recall = self._compute_recall(indices[:, :k], ground_truth[:, :k])
                recalls[f'recall@{k}'] = recall
            
            results[f'nprobe_{nprobe}'] = {
                'latency_ms': latency_ms,
                **recalls
            }
            
            logger.info(f"nprobe={nprobe}: latency={latency_ms:.2f}ms, recall@100={recalls.get('recall@100', 0):.4f}")
        
        return results
    
    def _compute_recall(
        self, 
        predictions: np.ndarray, 
        ground_truth: np.ndarray
    ) -> float:
        """Compute recall between predictions and ground truth."""
        recalls = []
        for pred, gt in zip(predictions, ground_truth):
            recall = len(set(pred) & set(gt)) / len(gt) if len(gt) > 0 else 0
            recalls.append(recall)
        return np.mean(recalls)


def create_optimized_index(
    embeddings: np.ndarray,
    index_type: str = "auto",
    use_gpu: bool = False
) -> FAISSIndexManager:
    """
    Factory function to create optimized index.
    
    Args:
        embeddings: Item embeddings
        index_type: 'auto', 'flat', 'ivf_flat', 'ivf_pq', 'hnsw'
        use_gpu: Whether to use GPU
        
    Returns:
        Configured FAISSIndexManager
    """
    num_items = len(embeddings)
    
    if index_type == "auto":
        config = IndexConfig.for_catalog_size(num_items)
    else:
        index_type_enum = IndexType(index_type)
        config = IndexConfig(index_type=index_type_enum)
    
    config.use_gpu = use_gpu
    
    manager = FAISSIndexManager(config)
    manager.build_index(embeddings)
    
    return manager
