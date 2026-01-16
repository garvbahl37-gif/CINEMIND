# Interview Q&A: FAANG-Grade Recommender System

Complete guide for technical interviews covering recommender system design.

---

## 1. System Design Questions

### Q: Walk me through how you would design a movie recommendation system at scale.

**Answer:**

I would use a **multi-stage retrieval-ranking architecture**:

1. **Candidate Generation** (FAISS):
   - Train a Two-Tower model (separate user/item encoders)
   - Generate 64-dim embeddings for all users and items
   - Build IVF+PQ FAISS index for O(log n) retrieval
   - Retrieve top-100 candidates in <5ms

2. **Ranking** (LightGBM):
   - Score candidates with rich features
   - LambdaRank objective optimizes NDCG
   - Reduces 100 → 20 candidates

3. **Re-ranking** (Business Logic):
   - MMR for diversity (λ=0.7)
   - Freshness boost for recency
   - Remove already-watched items

**Why this architecture?**
- Scalability: FAISS handles millions of items
- Latency: Each stage filters progressively
- Accuracy: Deep ranking on manageable candidate set

---

### Q: Why Two-Tower instead of a single model?

**Answer:**

Two-Tower (Dual Encoder) is optimal for **retrieval** because:

1. **Decoupling**: Item embeddings computed offline, user embeddings at runtime
2. **Efficiency**: Just one forward pass per tower + dot product
3. **Scalability**: FAISS indexes pre-computed item embeddings
4. **Serving**: No cross-attention = O(1) not O(n) per query

Single models (like cross-encoders) are better for **ranking** where you score a small candidate set.

---

### Q: Explain your choice of FAISS index.

**Answer:**

For **87K movies**, I chose **IVF+PQ**:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| nlist | 256 | √87K ≈ 295, power of 2 |
| nprobe | 16 | 6% clusters, 95% recall |
| m | 8 | 8 subquantizers |
| nbits | 8 | 1 byte per subvector |

**Memory**: 87K × 8 bytes = 696 KB (vs 22 MB for Flat)
**Latency**: ~3ms for top-100
**Recall@100**: 95%+

For <10K items, I'd use Flat. For >1M, I'd consider HNSW or GPU FAISS.

---

## 2. Machine Learning Questions

### Q: Why InfoNCE loss for Two-Tower training?

**Answer:**

InfoNCE (Contrastive Loss) is ideal because:

```python
# Simplified InfoNCE
logits = user_emb @ item_emb.T / temperature
labels = diagonal  # Positive pairs
loss = CrossEntropy(logits, labels)
```

1. **In-batch negatives**: All items in batch are negatives (free sampling)
2. **Hard negatives**: Large batches naturally include hard examples
3. **Embedding space**: Pushes positives together, negatives apart
4. **Temperature**: τ=0.07 sharpens the distribution

Alternatives: BPR (pairwise), Sampled Softmax (explicit negatives)

---

### Q: How do you handle cold-start users?

**Answer:**

Multi-strategy approach:

1. **Feature-based fallback**: If user has features (demographics), use content-based
2. **Popularity baseline**: Recommend globally popular items
3. **Contextual**: Use session context (time, device)
4. **Exploration**: Include diverse items to gather signals

In this system, users with <5 ratings are filtered during training, but production would use popularity fallback.

---

### Q: How do you evaluate the system?

**Answer:**

**Offline Metrics:**
- NDCG@10: Ranking quality (target: 0.15+)
- Recall@100: Retrieval coverage (target: 95%+)
- MAP@K: Precision at relevant positions
- Coverage: Catalog diversity (avoid popularity bias)
- ILD: Intra-list diversity

**Online Metrics (A/B):**
- CTR: Click-through rate
- Watch time: Engagement
- Diversity: Genre distribution
- Conversion: Sign-ups, subscriptions

---

## 3. Implementation Questions

### Q: Why mixed precision training?

**Answer:**

**torch.cuda.amp** benefits:
- **2x faster**: FP16 Tensor Cores
- **50% memory**: Smaller gradients
- **Same accuracy**: Loss scaling prevents underflow

```python
with autocast():
    loss = model(batch)
scaler.scale(loss).backward()
scaler.step(optimizer)
```

---

### Q: How do you ensure diversity in recommendations?

**Answer:**

I use **MMR (Maximal Marginal Relevance)**:

```
MMR(d) = λ × Relevance(d) - (1-λ) × max(Sim(d, selected))
```

- λ=1.0: Pure relevance (repetitive)
- λ=0.0: Pure diversity (random)
- λ=0.7: Balanced (production)

Iteratively select items that are relevant AND different from already-selected.

---

### Q: How do you deploy this to production?

**Answer:**

1. **Training Pipeline**: Daily/weekly retraining on new data
2. **Embedding Export**: Generate and store in feature store
3. **Index Building**: Rebuild FAISS index with new embeddings
4. **Serving**: FastAPI with Redis caching
5. **Monitoring**: Track latency, recall drift, A/B metrics

```
┌─────────┐    ┌─────────┐    ┌─────────┐
│ Train   │───▶│ Export  │───▶│ Index   │
│ Daily   │    │ Embeddings   │ FAISS   │
└─────────┘    └─────────┘    └────┬────┘
                                   ▼
                              ┌─────────┐
                              │ FastAPI │
                              │ Serving │
                              └─────────┘
```

---

## 4. Trade-off Questions

### Q: How do you balance latency vs. accuracy?

**Answer:**

| Trade-off | Lower Latency | Higher Accuracy |
|-----------|---------------|-----------------|
| Index | Flat → IVF+PQ | More nprobe |
| Candidates | Fewer (50) | More (200) |
| Ranking | Simpler model | Deeper model |
| Features | Fewer | Richer |

Production target: **P99 < 10ms at Recall@100 ≥ 95%**

Tune nprobe until you hit the recall target, then optimize other components.

---

### Q: What would you do with 10x more data?

**Answer:**

1. **Index**: Switch to sharded FAISS or GPU index
2. **Training**: Distributed training (DDP/FSDP)
3. **Features**: More sophisticated (GNN, sequences)
4. **Serving**: Multiple replicas, load balancing
5. **Caching**: Aggressive pre-computation for top users

---

## 5. Code Snippets for Interviews

### Two-Tower Forward Pass
```python
def forward(self, user_ids, item_ids):
    user_emb = self.user_tower(user_ids)  # (B, 64)
    item_emb = self.item_tower(item_ids)  # (B, 64)
    
    # L2 normalize for cosine similarity
    user_emb = F.normalize(user_emb, p=2, dim=-1)
    item_emb = F.normalize(item_emb, p=2, dim=-1)
    
    return user_emb, item_emb
```

### FAISS Search
```python
def search(self, query, k=100):
    query = np.ascontiguousarray(query.astype(np.float32))
    faiss.normalize_L2(query)
    distances, indices = self.index.search(query, k)
    return indices, distances
```

### MMR Re-ranking
```python
def mmr(scores, embeddings, k, λ=0.7):
    selected = []
    for _ in range(k):
        mmr_scores = λ * scores - (1-λ) * max_sim_to_selected
        best = argmax(mmr_scores)
        selected.append(best)
    return selected
```
