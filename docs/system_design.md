# CSAO System Design — Cart Super Add-On Recommendation Engine

## 1. Problem Framing

We frame CSAO recommendation as a **Learning-to-Rank** (LTR) problem using LambdaRank:

- **Query**: A user's current cart state at a specific restaurant (user_id, restaurant_id, cart_items[], timestamp)
- **Candidates**: ~200 potential add-on items from the restaurant's menu
- **Labels**: Binary — did the user actually add this item next? (from historical order sequences)
- **Objective**: Rank candidates so that the items the user is most likely to add appear at the top

### Why LTR over Classification or Collaborative Filtering?

| Approach | Limitation for CSAO |
|----------|-------------------|
| Binary classification | Only predicts "will add / won't add" — doesn't optimise viewing order |
| Collaborative filtering (ALS, SVD) | Ignores cart context; treats recommendations as user-level, not session-level |
| Sequence models (RNN/Transformer) | Requires large-scale sequential data; latency-heavy for real-time |
| **LambdaRank (our choice)** | Directly optimises NDCG; handles position bias; fast inference via gradient-boosted trees |

The LambdaRank loss computes gradients weighted by the **NDCG swap delta** between pair (i, j):

$$\lambda_{ij} = \frac{-\sigma}{1 + e^{\sigma(s_i - s_j)}} \cdot |\Delta NDCG_{ij}|$$

This ensures the model focuses on improving rankings where it matters most — the top of the list where users actually see recommendations.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CSAO Serving Pipeline                          │
│                         (Target: < 200 ms e2e)                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐    ┌─────────────┐    ┌───────────┐    ┌──────────────┐  │
│  │ Feature   │──▶│  Candidate   │──▶│ LightGBM  │──▶│   Neural     │  │
│  │ Fetch     │    │ Generation   │    │ LTR       │    │ Reranker     │  │
│  │ (20 ms)   │    │ (40-60 ms)   │    │ (30-50 ms)│    │ (<10 ms)     │  │
│  └──────────┘    └─────────────┘    └───────────┘    └──────────────┘  │
│       │                │                  │                  │          │
│       ▼                ▼                  ▼                  ▼          │
│  Pre-indexed      5 Retrievers        73 features       Cross-attention│
│  O(1) lookup    (co-occurrence,      (cart, user,        α=0.3 blend  │
│  user+item       session covisit,    item, CSAO          with LightGBM│
│  features        meal-gap,           intelligence,                     │
│                  category,           is_veg,                            │
│                  popularity)         user_veg_ratio)                   │
│                                                                         │
│  ┌──────────────┐    ┌────────────┐                                    │
│  │ MMR Diversity │──▶│    LLM     │──▶  Top-10 recommendations         │
│  │ Reranking     │    │ Explainer  │     with explanations              │
│  │ (λ=0.7)      │    │ (template  │                                    │
│  │               │    │+OpenRouter)│                                    │
│  └──────────────┘    └────────────┘                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Two-Stage Ranking Design

**Stage 1 — LightGBM LambdaRank** (primary scorer):
- Scores all ~200 candidates simultaneously via vectorised NumPy feature matrix
- 73 features across 7 groups: cart context (incl. `cart_has_addon`), user RFM (incl. `user_veg_ratio`), item properties (incl. `is_veg`), CSAO intelligence, complementarity, embeddings (PCA-8), cuisine shares
- Single `model.predict()` call on full batch — no per-candidate loop
- Outputs raw relevance scores; top-30 passed to Stage 2

**Stage 2 — Neural Cross-Attention Reranker** (AI edge):
- Lightweight ~50K params model: cart embeddings attend to candidate embedding
- Captures non-linear cart-candidate interactions that gradient-boosted features miss
- α-blended with LightGBM: `final_score = 0.3 × neural + 0.7 × lgbm`
- <10 ms on CPU for 30 candidates

**Stage 3 — MMR Diversity Reranking**:
- Maximal Marginal Relevance with λ=0.7 prevents recommendation fatigue
- Ensures category diversity in the final top-10

---

## 3. Offline Pipeline (Training & Feature Engineering)

```
Raw Data Sources                Feature Build               Model Training
─────────────────               ──────────────              ──────────────

┌──────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
│ Synthetic     │──▶│ Schema Unification    │──▶│ Temporal Train/Val   │
│ Indian Food   │    │ (users, orders,       │    │ Split (80/20)        │
│ Orders (877K) │    │  order_items, items,  │    │                      │
├──────────────┤    │  restaurants)          │    │ Leakage validation:  │
│ Mendeley      │──▶│                        │    │ max(train_ts) <      │
│ Takeaway      │    └──────────────────────┘    │ min(val_ts)          │
│ Orders (75K)  │              │                  │                      │
└──────────────┘              ▼                  │ Unseen item analysis │
                    ┌──────────────────────┐    └──────────────────────┘
                    │ Feature Pipelines     │              │
                    │                        │              ▼
                    │ • User: RFM, cuisine   │    ┌──────────────────────┐
                    │   share, price         │    │ LightGBM LambdaRank  │
                    │   sensitivity, segment │    │                      │
                    │ • Item: popularity,    │    │ • lambdarank obj     │
                    │   price band, category │    │ • 63 leaves, 200 est│
                    │ • Cart: completeness,  │    │ • Positive + 6 hard  │
                    │   meal gaps, category  │    │   negatives per query│
                    │   shares               │    │ • Smart neg sampling │
                    │ • Complementarity:     │    │   (popularity +      │
                    │   PMI, lift, co-occ    │    │    same-category +   │
                    │ • LLM Embeddings:      │    │    co-occurrence     │
                    │   8-dim PCA from       │    │    weighted)         │
                    │   sentence-transformers│    └──────────────────────┘
                    │ • Category affinity    │              │
                    └──────────────────────┘              ▼
                                                 ┌──────────────────────┐
                                                 │ Artifacts Saved       │
                                                 │ • lgbm_ranker.joblib  │
                                                 │ • feature_columns.json│
                                                 │ • neural_reranker.pt  │
                                                 │ • All feature parquets│
                                                 └──────────────────────┘
```

### Sequential Cart Training

The training dataset is built by iterating through each order's item sequence:

```
Order: [Biryani, Raita, Gulab Jamun, Lassi]

Position 1: Cart=[Biryani]        → Positive=Raita         + 6 negatives
Position 2: Cart=[Biryani, Raita] → Positive=Gulab Jamun   + 6 negatives
Position 3: Cart=[B, R, GJ]       → Positive=Lassi         + 6 negatives
```

This directly models the problem statement's requirement: **"Biryani → recommend salan → added → recommend gulab jamun → added → recommend drinks."**

---

## 4. Feature Architecture

| Group | Count | Key Features | Real-time? |
|-------|-------|-------------|-----------|
| **Cart Context** | 13 | cart_size, cart_value, completeness, missing_categories, has_main/beverage/dessert/starter, category shares | Yes — computed per request |
| **User RFM** | 5+ | order_count, order_frequency, avg_order_value, recency_days, total_spend, user_segment, price_sensitivity | Yes — pre-indexed O(1) |
| **User Cuisine Shares** | ~8 | user_cuisine_share for each cuisine | Yes — pre-indexed O(1) |
| **Item Properties** | 5 | item_popularity, item_price, price_band, category OHE | Yes — pre-indexed O(1) |
| **CSAO Intelligence** | 4 | completeness_delta, fills_meal_gap, complement_confidence, candidate_new_category | Yes — computed per candidate |
| **Complementarity** | 5 | comp_max_lift, comp_mean_lift, comp_max_pmi, comp_mean_pmi, candidate_score | Yes — vectorised lookup |
| **LLM Embeddings** | 8 | emb_0..emb_7 (sentence-transformer → PCA) | Yes — cached |
| **Temporal** | 6 | hour_of_day, day_of_week, is_weekend, is_lunch, is_dinner, is_late_night | Yes — from request time |
| **Geographic** | 4 | city_hash, cuisine_hash, restaurant_hash, user_hash | Yes — from request |

### Feature Store Design (Production)

```
┌────────────────────────────────────────────┐
│              Feature Store                  │
├────────────────┬───────────────────────────┤
│ Offline Store  │ Online Store (Redis)       │
│ (S3/Parquet)   │                           │
│                │ • User features: O(1) GET │
│ • Batch        │ • Item features: O(1) GET │
│   compute      │ • Complementarity pairs   │
│   daily        │ • Restaurant menu cache   │
│                │ • TTL: 1h user, 24h item  │
│ • Full         │                           │
│   retrain      │ • Write-through on        │
│   weekly       │   feature pipeline runs   │
└────────────────┴───────────────────────────┘
```

---

## 5. Cold-Start Strategy

Five-strategy cascade for handling new users and restaurants:

| Scenario | Strategy | Fallback Signal |
|----------|----------|----------------|
| New user + known restaurant | Restaurant-level popularity | Menu's best-sellers for this meal time |
| New user + new restaurant | Cuisine-level popularity | Popular items for this cuisine across all restaurants |
| Warm user + empty cart | User's historical favourites | Items frequently ordered by similar users |
| Warm user + non-empty cart | Full pipeline (reduced features) | Complementarity + popularity hybrid |
| All else | Global top-N | Most popular items platform-wide |

---

## 6. LLM Integration (The AI Edge)

### 6a. LLM-Powered Explanations (OpenRouter)
- Template engine (0ms latency) generates baseline explanations
- OpenRouter API enriches with food-specific reasoning when available
- Fallback chain: Nemotron → Gemma → Llama → Mistral → Trinity (5 free models)
- 5-second timeout prevents latency impact

### 6b. LLM Embeddings as Features
- sentence-transformers (all-MiniLM-L6-v2) embeds item names
- PCA-reduced to 8 dimensions → fed as features to LightGBM
- Captures semantic similarity missed by category-only features

### 6c. LLM-as-Judge Evaluation
- Embedding-based semantic coherence scoring
- Category coverage analysis
- Optional API-based quality scoring for deeper evaluation

---

## 7. Failure Modes & Mitigation

| Failure | Detection | Mitigation |
|---------|-----------|-----------|
| LLM API timeout | 5s timeout | Template fallback (0ms) |
| Feature store unavailable | Health check | Use cached/default features |
| Model file corrupt | Load-time validation | Keep N-1 model as backup |
| Latency > 300ms | LatencyTracker | Truncate results to top-3 |
| Cold-start user | User not in index | 5-strategy cascade |
| Empty candidate pool | len(candidates)==0 | Fallback to restaurant popularity |

---

## 8. Deployment Topology (Production)

```
                    ┌──────────────┐
                    │ Load Balancer │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ CSAO API │ │ CSAO API │ │ CSAO API │  (14 pods, HPA)
        │  Pod 1   │ │  Pod 2   │ │  Pod N   │  2 vCPU, 4 GiB
        └────┬─────┘ └────┬─────┘ └────┬─────┘
             │             │             │
             └─────────────┼─────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │  Redis   │ │  Model   │ │ Prom /   │
        │ Features │ │  (S3)    │ │ Grafana  │
        └──────────┘ └──────────┘ └──────────┘
```

- **Auto-scaling**: HPA targets 60% CPU, min=7 pods, max=28 pods
- **Canary rollout**: 5% → 25% → 50% → 100% with automated rollback
- **Cache hit rate**: 86.1% (simulated), saving ~4.2ms per request
- **Estimated cost**: $1,120/month for 500 QPS peak capacity

---

## 9. Monitoring Plan

| Dashboard | Key Metrics | Alert Threshold |
|-----------|------------|-----------------|
| **Real-time** | p50/p95/p99 latency, QPS, error rate | p95 > 200ms, error > 1% |
| **Business** | Attach rate, AOV uplift, CTR | Attach rate drops > 2pp |
| **Model Health** | Feature drift, prediction distribution, NDCG decay | KL divergence > 0.1 |
