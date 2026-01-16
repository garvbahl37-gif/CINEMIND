# FAANG-Grade Movie Recommender System
## Complete Technical Documentation

---

# Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Deep Dive](#2-architecture-deep-dive)
3. [Data Pipeline](#3-data-pipeline)
4. [Feature Engineering](#4-feature-engineering)
5. [Two-Tower Model](#5-two-tower-model)
6. [FAISS Index Optimization](#6-faiss-index-optimization)
7. [Ranking System](#7-ranking-system)
8. [Re-ranking & Business Logic](#8-re-ranking--business-logic)
9. [Serving Infrastructure](#9-serving-infrastructure)
10. [Evaluation Metrics](#10-evaluation-metrics)
11. [Performance Optimization](#11-performance-optimization)
12. [Production Deployment](#12-production-deployment)

---

# 1. System Overview

## 1.1 What is This System?

This is a **production-grade movie recommendation system** built using the same architectural patterns employed by FAANG companies (Netflix, YouTube, Spotify, Pinterest, etc.). The system processes the **MovieLens-32M dataset** containing:

- **32 Million+ ratings**
- **200,000+ users**
- **87,000+ movies**

## 1.2 Key Capabilities

| Capability | Description |
|------------|-------------|
| **Personalized Recommendations** | Generate top-K movies tailored to each user |
| **Similar Items** | Find movies similar to a given movie |
| **Real-time Serving** | Sub-10ms latency at scale |
| **Diverse Results** | MMR-based diversity to avoid filter bubbles |
| **Cold-start Handling** | Popularity fallbacks for new users |

## 1.3 Technology Stack

```
┌─────────────────────────────────────────────────────────┐
│                    TECHNOLOGY STACK                      │
├─────────────────────────────────────────────────────────┤
│  Deep Learning     │ PyTorch 2.x, Mixed Precision (AMP) │
│  Vector Search     │ FAISS (IVF+PQ, HNSW)              │
│  Ranking           │ LightGBM (LambdaRank)              │
│  API               │ FastAPI + Uvicorn                  │
│  Data Processing   │ Pandas, PyArrow (Parquet)          │
│  Text Embeddings   │ Sentence-Transformers              │
│  Feature Store     │ Redis (optional)                   │
│  Configuration     │ YAML                               │
└─────────────────────────────────────────────────────────┘
```

---

# 2. Architecture Deep Dive

## 2.1 High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     RECOMMENDATION PIPELINE                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  User Request                                                     │
│       │                                                           │
│       ▼                                                           │
│  ┌─────────────────┐                                             │
│  │  Feature Store  │ ← User features (genre affinity, stats)    │
│  └────────┬────────┘                                             │
│           │                                                       │
│           ▼                                                       │
│  ┌─────────────────┐                                             │
│  │   User Tower    │ → 64-dim L2-normalized embedding           │
│  │   (Deep MLP)    │                                             │
│  └────────┬────────┘                                             │
│           │                                                       │
│           ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │                    FAISS Index                           │     │
│  │  ┌─────────┐   ┌─────────┐   ┌─────────┐               │     │
│  │  │ IVF+PQ  │   │  HNSW   │   │  Flat   │   (87K items) │     │
│  │  │ 256 cls │   │  M=32   │   │(baseline)│               │     │
│  │  └─────────┘   └─────────┘   └─────────┘               │     │
│  └────────────────────────┬────────────────────────────────┘     │
│                           │                                       │
│                           ▼ (Top-100 candidates, <5ms)           │
│  ┌─────────────────┐                                             │
│  │  Ranking Model  │ LightGBM LambdaRank                        │
│  │  (Score + Sort) │ Features: similarity, popularity, etc.     │
│  └────────┬────────┘                                             │
│           │                                                       │
│           ▼ (Top-20 scored)                                      │
│  ┌─────────────────┐                                             │
│  │   Re-ranking    │                                             │
│  │  ┌───────────┐  │                                             │
│  │  │MMR Divers.│  │ ← Balance relevance vs. novelty            │
│  │  │Freshness  │  │ ← Boost recent releases                    │
│  │  │Biz Rules  │  │ ← Remove watched, apply promotions         │
│  │  └───────────┘  │                                             │
│  └────────┬────────┘                                             │
│           │                                                       │
│           ▼                                                       │
│  ┌─────────────────┐                                             │
│  │  Final Results  │ → Top-K personalized recommendations       │
│  └─────────────────┘                                             │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

## 2.2 Why This Architecture?

This **retrieval-ranking** pattern is the industry standard because:

1. **Scalability**: FAISS retrieval is O(log n), not O(n)
2. **Flexibility**: Each stage can be optimized independently
3. **Latency**: Sub-10ms by reducing candidate set early
4. **Accuracy**: Deep ranking model on filtered candidates

---

# 3. Data Pipeline

## 3.1 Data Source: MovieLens-32M

| File | Size | Contents |
|------|------|----------|
| `ratings.csv` | 877 MB | 32M ratings (userId, movieId, rating, timestamp) |
| `movies.csv` | 4 MB | 87K movies (movieId, title, genres) |
| `tags.csv` | 72 MB | 2M user tags |
| `links.csv` | 2 MB | IMDb/TMDb links |

## 3.2 Preprocessing Pipeline

```python
# preprocessing.py workflow
1. Chunked Loading (1M rows/chunk) → Prevent OOM on 877MB file
2. Cold-Start Filtering          → Remove users/items with <5 interactions
3. ID Encoding                   → LabelEncoder for contiguous indices
4. Temporal Features             → hour, day_of_week, month, recency
5. Time-Based Split              → Train (80%) / Val (10%) / Test (10%)
6. Negative Sampling             → 4:1 ratio for contrastive learning
7. Parquet Export                → Efficient columnar storage
```

## 3.3 Output Files

```
data/processed/
├── interactions.parquet      # Full preprocessed ratings
├── train.parquet             # Training split
├── val.parquet               # Validation split
├── test.parquet              # Test split
├── train_interactions.parquet# With negative samples
├── movies.parquet            # Movies with tags
├── encoders.npz              # User/item ID mappings
└── metadata.json             # Dataset statistics
```

---

# 4. Feature Engineering

## 4.1 User Features

| Feature Category | Examples |
|------------------|----------|
| **Rating Statistics** | count, mean, std, min, max |
| **Rating Distribution** | ratio of 1-star, 2-star, etc. |
| **Temporal Patterns** | avg hour, weekend ratio |
| **Activity Span** | days active, rating velocity |
| **Genre Affinity** | weighted avg rating per genre |

## 4.2 Item Features

| Feature Category | Examples |
|------------------|----------|
| **Popularity** | rating count, unique users, percentile |
| **Rating Stats** | avg rating, std, distribution |
| **Genre Encoding** | Multi-hot vector (20 genres) |
| **Tag TF-IDF** | Top-50 TF-IDF features |
| **Temporal** | Item age (years) |
| **Text Embeddings** | Sentence-transformer (384-dim) |

---

# 5. Two-Tower Model

## 5.1 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        TWO-TOWER MODEL                           │
├───────────────────────────┬─────────────────────────────────────┤
│       USER TOWER          │         ITEM TOWER                  │
├───────────────────────────┼─────────────────────────────────────┤
│                           │                                      │
│  user_id ──┐              │  item_id ──┐                        │
│            ▼              │            ▼                        │
│  ┌─────────────────┐      │  ┌─────────────────┐                │
│  │ Embedding(128)  │      │  │ Embedding(128)  │                │
│  └────────┬────────┘      │  └────────┬────────┘                │
│           │               │           │                          │
│  behavior_features ─┐     │  genre_vector ──┐                   │
│                     ▼     │                 ▼                    │
│  ┌─────────────────────┐  │  ┌─────────────────────┐            │
│  │   Concatenate       │  │  │   Concatenate       │            │
│  └──────────┬──────────┘  │  └──────────┬──────────┘            │
│             │             │             │                        │
│             ▼             │             ▼                        │
│  ┌─────────────────────┐  │  ┌─────────────────────┐            │
│  │  MLP (256→128)      │  │  │  MLP (256→128)      │            │
│  │  + LayerNorm        │  │  │  + LayerNorm        │            │
│  │  + Residual         │  │  │  + Residual         │            │
│  │  + GELU             │  │  │  + GELU             │            │
│  └──────────┬──────────┘  │  └──────────┬──────────┘            │
│             │             │             │                        │
│             ▼             │             ▼                        │
│  ┌─────────────────────┐  │  ┌─────────────────────┐            │
│  │  Linear(128→64)     │  │  │  Linear(128→64)     │            │
│  │  + L2 Normalize     │  │  │  + L2 Normalize     │            │
│  └──────────┬──────────┘  │  └──────────┬──────────┘            │
│             │             │             │                        │
│             ▼             │             ▼                        │
│       User Embedding      │       Item Embedding                │
│         (64-dim)          │         (64-dim)                    │
│                           │                                      │
└───────────────────────────┴─────────────────────────────────────┘
                    │                    │
                    └────────┬───────────┘
                             ▼
                    Dot Product / Temperature
                             │
                             ▼
                     Cosine Similarity
```

## 5.2 Loss Function: InfoNCE

```python
# In-batch negatives: all items in batch are negatives for each user
logits = user_embeddings @ item_embeddings.T / temperature
labels = torch.arange(batch_size)  # Diagonal is positive
loss = F.cross_entropy(logits, labels)
```

**Why InfoNCE?**
- Efficient: No separate negative sampling needed
- Effective: Hard negatives emerge naturally in large batches
- Scalable: Batch size = number of negatives

## 5.3 Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Batch Size | 2048 | Large batch for more in-batch negatives |
| Learning Rate | 1e-3 | Standard for Adam on embeddings |
| Warmup Epochs | 2 | Prevent early divergence |
| Scheduler | Cosine Decay | Smooth convergence |
| Temperature | 0.07 | Controls similarity distribution sharpness |
| Early Stopping | Recall@100 | Retrieval-focused metric |
| Mixed Precision | AMP | 2x speedup, 50% memory |

---

# 6. FAISS Index Optimization

## 6.1 Index Types Comparison

| Index Type | Memory | Build Time | Query Time | Recall@100 | Best For |
|------------|--------|------------|------------|------------|----------|
| **Flat** | 100% | O(1) | O(n) | 100% | <10K items (baseline) |
| **IVF_Flat** | 100% | O(n) | O(√n) | 99% | 10K-100K items |
| **IVF_PQ** | ~10% | O(n) | O(√n) | 95%+ | **100K-10M items** |
| **HNSW** | 100%+ | O(n log n) | O(log n) | 98% | Low latency priority |

## 6.2 Recommended Configuration (87K items)

```yaml
# IVF+PQ (Production Default)
index_type: ivf_pq
nlist: 256          # √87000 ≈ 295, use 256
nprobe: 16          # Search 16/256 = 6% of clusters
m: 8                # 8 subquantizers
nbits: 8            # 8 bits per code

# Memory: 87K × 8 bytes = 696 KB (vs 22 MB for Flat)
# Latency: ~3ms
# Recall@100: 95%+
```

## 6.3 Tuning nprobe

```
nprobe  | Latency | Recall@100
--------|---------|------------
8       | 2ms     | 90%
16      | 3ms     | 95%  ← Production default
32      | 5ms     | 98%
64      | 8ms     | 99%
```

---

# 7. Ranking System

## 7.1 LightGBM LambdaRank

After FAISS retrieves 100 candidates, the ranking model scores them:

```python
Features:
1. Embedding similarity (user @ item)
2. User rating count, avg, std
3. Item popularity percentile
4. Genre match score
5. Item age
6. Position bias
```

## 7.2 Training Objective

**LambdaRank** optimizes NDCG directly:
- Learns pairwise ranking preferences
- Weighted by NDCG gain of swapping pairs
- Handles position-biased evaluation

---

# 8. Re-ranking & Business Logic

## 8.1 MMR Diversity

```python
# Maximal Marginal Relevance
MMR(d) = λ × Relevance(d) - (1-λ) × max(Similarity(d, d_j))

# λ = 0.7: 70% relevance, 30% diversity
```

**Why MMR?**
- Prevents filter bubbles
- Improves user exploration
- Increases catalog coverage

## 8.2 Business Rules

| Rule | Purpose |
|------|---------|
| Remove Watched | Don't recommend already-seen items |
| Freshness Boost | Surface new releases |
| Promotion Slots | Insert sponsored content |
| Category Quotas | Ensure genre diversity |

---

# 9. Serving Infrastructure

## 9.1 FastAPI Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/recommend/{user_id}` | GET | Personalized recommendations |
| `/similar/{item_id}` | GET | Similar items |
| `/batch-recommend` | POST | Batch requests |
| `/health` | GET | Health check |

## 9.2 Response Format

```json
{
  "user_id": 1,
  "recommendations": [
    {"item_id": 1234, "score": 0.95},
    {"item_id": 5678, "score": 0.92}
  ],
  "num_candidates": 100,
  "latency_ms": 4.5
}
```

---

# 10. Evaluation Metrics

## 10.1 Ranking Metrics

| Metric | Formula | What it Measures |
|--------|---------|------------------|
| **NDCG@K** | DCG/IDCG | Ranking quality with position discounting |
| **MAP@K** | Avg precision at each relevant position | Precision-focused ranking |
| **MRR** | 1/rank of first hit | First-result quality |

## 10.2 Retrieval Metrics

| Metric | Formula | What it Measures |
|--------|---------|------------------|
| **Recall@K** | Hits@K / Total Relevant | Coverage of relevant items |
| **Precision@K** | Hits@K / K | Accuracy of top-K |
| **Hit Rate@K** | 1 if any hit else 0 | Binary success |

## 10.3 Beyond-Accuracy Metrics

| Metric | What it Measures |
|--------|------------------|
| **Coverage** | Fraction of catalog ever recommended |
| **ILD** | Intra-list diversity (dissimilarity) |
| **Novelty** | Avg -log(popularity) of recommendations |

---

# 11. Performance Optimization

## 11.1 Training Optimizations

- **Mixed Precision (AMP)**: 2x training speedup
- **Gradient Accumulation**: Larger effective batch
- **Gradient Checkpointing**: Reduce memory 40%

## 11.2 Inference Optimizations

- **Batch Queries**: Process multiple users together
- **Index Caching**: Keep FAISS index in memory
- **Async Pipeline**: Parallel feature fetching

## 11.3 Latency Breakdown (Target: <10ms)

```
1. Feature lookup:    1ms
2. User embedding:    1ms
3. FAISS search:      3ms
4. Ranking model:     2ms
5. Re-ranking:        1ms
─────────────────────────
Total:                8ms
```

---

# 12. Production Deployment

## 12.1 Running the System

```bash
# Full pipeline
python run.py full --sample 1000000 --epochs 10

# Individual stages
python run.py preprocess --sample 1000000
python run.py features
python run.py train --epochs 10
python run.py build-index
python run.py evaluate
python run.py serve --port 8000
```

## 12.2 Docker Deployment

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "run.py", "serve", "--port", "8000"]
```

## 12.3 Scaling Considerations

| Scale | Recommendation |
|-------|----------------|
| <100K items | Single node, Flat/IVF index |
| 100K-1M items | Single node, IVF+PQ |
| 1M-10M items | Index sharding, multiple replicas |
| >10M items | Distributed FAISS, GPU inference |

---

# Appendix: Project Structure

```
faang-recommender-system/
├── preprocessing.py           # Data loading & preprocessing
├── feature_engineering.py     # Feature extraction
├── run.py                     # Main CLI
├── requirements.txt           # Dependencies
├── README.md                  # Quick start
│
├── candidate_generation/
│   ├── two_tower/
│   │   ├── model.py           # Two-tower architecture
│   │   ├── train.py           # Training with AMP
│   │   ├── loss.py            # InfoNCE, BPR losses
│   │   └── inference.py       # Embedding generation
│   └── ann/
│       ├── faiss_index.py     # FAISS index manager
│       └── build_index.py     # Index building
│
├── ranking/
│   ├── features.py            # Ranking features
│   └── train_lgbm.py          # LightGBM ranker
│
├── reranking/
│   └── diversity.py           # MMR, freshness, rules
│
├── evaluation/
│   └── offline_metrics.py     # NDCG, MAP, coverage
│
├── serving/
│   ├── api/main.py            # FastAPI server
│   └── recommendation_service.py
│
├── frontend/
│   └── index.html             # Web UI
│
├── configs/
│   └── model.yaml             # Configuration
│
└── docs/
    └── SYSTEM_DOCUMENTATION.md
```

---

**Document Version**: 1.0  
**Last Updated**: December 2024  
**Author**: FAANG-Grade Recommender System
