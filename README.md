# FAANG-Grade Movie Recommender System

A production-ready movie recommendation system built on the MovieLens-32M dataset (32M ratings, 200K users, 87K movies) with optimized FAISS indexing for sub-millisecond retrieval.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     RECOMMENDATION PIPELINE                      │
├─────────────────────────────────────────────────────────────────┤
│  User Request                                                    │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────┐                                                │
│  │ User Tower  │ → User Embedding (64-dim)                      │
│  └─────────────┘                                                │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────┐                    │
│  │  FAISS Index (IVF+PQ)                   │                    │
│  │  • 87K items indexed                    │                    │
│  │  • < 5ms latency                        │                    │
│  │  • 95%+ Recall@100                      │                    │
│  └─────────────────────────────────────────┘                    │
│       │                                                          │
│       ▼ (Top-100 Candidates)                                    │
│  ┌─────────────┐                                                │
│  │  Ranking    │ LightGBM / Neural Ranker                       │
│  └─────────────┘                                                │
│       │                                                          │
│       ▼ (Top-20 Scored)                                         │
│  ┌─────────────┐                                                │
│  │ Re-ranking  │ MMR Diversity + Business Rules                 │
│  └─────────────┘                                                │
│       │                                                          │
│       ▼                                                          │
│  Final Recommendations                                           │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Full Pipeline

```bash
# With full dataset (requires ~16GB RAM)
python run.py full

# With sample for testing
python run.py full --sample 1000000 --epochs 10
```

### 3. Start API Server

```bash
python run.py serve --port 8000
```

### 4. Get Recommendations

```bash
curl http://localhost:8000/recommend/1?k=10
```

## Pipeline Stages

### Data Preprocessing
```bash
python run.py preprocess [--sample N]
```
- Loads ML-32M dataset (877MB ratings)
- Filters cold-start users/items
- Creates time-based train/val/test splits
- Generates negative samples

### Feature Engineering
```bash
python run.py features [--text-embeddings]
```
- User features: rating stats, genre affinity
- Item features: popularity, genres, tags
- Optional text embeddings (requires sentence-transformers)

### Two-Tower Training
```bash
python run.py train --epochs 50 --batch-size 2048
```
- Deep user and item towers (128→256→128→64)
- InfoNCE loss with in-batch negatives
- Mixed precision training (AMP)
- Early stopping on Recall@100

### FAISS Index Building
```bash
python run.py build-index [--benchmark]
```
- Builds IVF+PQ index for fast retrieval
- Benchmarks recall vs latency
- Auto-tunes nprobe parameter

### Evaluation
```bash
python run.py evaluate
```
- NDCG@K, MAP@K, MRR
- Recall@K, Precision@K
- Coverage, Diversity (ILD), Novelty

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/recommend/{user_id}` | GET | Get recommendations |
| `/similar/{item_id}` | GET | Get similar items |
| `/batch-recommend` | POST | Batch recommendations |
| `/health` | GET | Health check |

## Project Structure

```
├── preprocessing.py          # Data loading & preprocessing
├── feature_engineering.py    # Feature extraction
├── run.py                    # Main pipeline CLI
│
├── candidate_generation/
│   ├── two_tower/
│   │   ├── model.py          # Two-tower architecture
│   │   ├── train.py          # Training pipeline
│   │   ├── loss.py           # Contrastive losses
│   │   └── inference.py      # Embedding generation
│   │
│   └── ann/
│       ├── faiss_index.py    # FAISS index manager
│       └── build_index.py    # Index building
│
├── ranking/
│   ├── features.py           # Ranking features
│   └── train_lgbm.py         # LightGBM ranker
│
├── reranking/
│   └── diversity.py          # MMR + business rules
│
├── evaluation/
│   └── offline_metrics.py    # NDCG, MAP, etc.
│
├── serving/
│   ├── api/main.py           # FastAPI server
│   └── recommendation_service.py
│
└── configs/
    └── model.yaml            # Configuration
```

## Performance Targets

| Metric | Target | Achieved |
|--------|--------|----------|
| Recall@100 | ≥ 0.95 | - |
| Latency (p99) | < 10ms | - |
| NDCG@10 | ≥ 0.15 | - |
| Index Memory | < 100MB | - |

## License

This project uses the MovieLens dataset. See [ML-32M README](ml-32m/ml-32m/README.txt) for citation requirements.
