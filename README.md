# CSAO — Cart Super Add-On Recommender for Zomato

**Real-time, cart-aware add-on recommendations for food delivery.**  
LightGBM LambdaRank → Neural Cross-Attention Reranker → LLM-powered Explanations — all under 200ms.

---

## Results at a Glance

| Metric | Value | vs Baselines |
|--------|-------|-------------|
| **NDCG@10** | **0.711** | +16.1% vs Popularity, +40.9% vs Co-occurrence |
| **Precision@10** | 0.100 | +37% vs Random |
| **Coverage@10** | 57.6% | Broad catalog exposure |
| **Diversity@10** | 0.750 | High recommendation diversity |
| **Latency (p95)** | <200ms | Within 300ms budget |
| **Cold Start** | 5/5 PASS | Unseen user, restaurant, item, triple-cold |

## Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌──────────────┐
│  5-Retriever    │───>│  LightGBM    │───>│  Neural         │───>│  LLM         │
│  Candidate Gen  │    │  LambdaRank  │    │  Reranker       │    │  Explainer   │
│  (200 cands)    │    │  (71 feats)  │    │  (cross-attn)   │    │  (OpenRouter)│
└─────────────────┘    └──────────────┘    └─────────────────┘    └──────────────┘
  Stage 0: Retrieve      Stage 1: Rank      Stage 2: Rerank       Stage 3: Explain
  ~5ms                   ~15ms              ~8ms                   <1ms (template)
```

### Stage 0: Multi-Strategy Candidate Generation
Five retrievers with restaurant-menu gating to prevent cross-cuisine contamination:

1. **Co-occurrence** — PMI/lift from 200K+ order co-purchase pairs (min_support=10)
2. **Session Co-visit** — Sequential add-to-cart transition graph (vectorized)
3. **Meal-Gap Analysis** — Structural completeness vs 6 meal archetypes
4. **Category Complement** — Cuisine-aware category diversification
5. **Popularity Fallback** — Restaurant-level → global (cold-start safety net)

### Stage 1: LightGBM LambdaRank (71 Features)
- **Temporal train/val split** — no future data leakage
- **Hard negative sampling** — popularity-weighted + same-category negatives
- **Key feature groups**: complementarity (lift/PMI), CSAO intelligence (`completeness_delta`, `fills_meal_gap`), sentence-transformer embeddings (PCA-8), item/cart/user features
- **MMR diversity reranking** post-LightGBM (λ=0.7)

### Stage 2: Neural Cross-Attention Reranker
- `CartCandidateAttention`: Multi-head attention (4 heads) over cart↔candidate interactions
- Trained on **real sentence-transformer embeddings** (all-MiniLM-L6-v2 → PCA-8)
- BPR loss, α=0.3 blend with LightGBM scores
- ~33K parameters, <8ms inference on CPU

### Stage 3: LLM-Powered Explanations
- **Template engine** (default): Rule-based, <1ms, context-aware (meal gaps, pairings)
- **OpenRouter LLM overlay**: 5-model fallback chain (Nemotron → Gemma → Llama → Mistral → Trinity)
- Graceful degradation: missing/bad API key → templates, never crashes

## AI/ML Components

| Component | Type | Where Used | Value |
|-----------|------|-----------|-------|
| LightGBM LambdaRank | Core ranker | Serving (Stage 1) | NDCG: 0.711 |
| Neural Reranker | Cross-attention | Serving (Stage 2) | Cart-item interaction modeling |
| Sentence-Transformers | Offline embeddings | Feature store → LightGBM + Reranker | 8-dim PCA, semantic similarity |
| LLM Explainer | OpenRouter API | Serving (Stage 3) | Generative explanations |
| LLM-as-a-Judge | Offline eval | Evaluation only | Coherence: 0.47, Diversity: 0.78 |

## Cold Start Robustness (All 5 Tests PASS)

| Scenario | Behavior | Latency |
|----------|----------|---------|
| Unseen User | User features → 0.0; ranking uses cart/item signals | 49ms |
| Unseen Restaurant | Menu gate bypassed; global popularity fills | 62ms |
| Unseen Item in Cart | Co-occurrence → empty; meal-gap + popularity fill | 25ms |
| Triple Cold Start | All defaults; global popularity only | 34ms |

## Business Impact

- **Projected incremental GMV**: ₹21–111 Cr/year (sensitivity range)
- **Per-order uplift**: ₹0.3–1.0 (one beverage or side dish)
- **A/B test plan**: 2-week test, 80% power to detect ₹0.66 AOV lift
- See [business_impact_validation.md](docs/business_impact_validation.md) for full formula chain and assumptions audit

## Quickstart

```bash
# Install
python -m pip install -e .

# Generate & process data
python scripts/build_unified_data.py
python scripts/build_features.py

# Train & evaluate
python scripts/train_ranker.py
python scripts/run_offline_eval.py

# Full evaluation (baselines, ablation, HP tuning, neural reranker)
python scripts/run_full_evaluation.py

# Live demo (FastAPI at localhost:8000)
python scripts/live_demo.py

# Tests
python tests/test_llm_integration.py    # 7 AI/LLM tests
python tests/test_cold_start.py          # 5 cold-start scenarios
```

## Repository Layout

```
├── candidate_generation/    # 5 retrievers + restaurant-menu gate
├── ranking/
│   ├── training/            # LightGBM training, dataset builder, HP tuning
│   └── inference/           # Ranker, neural reranker, LLM explainer
├── features/                # Cart/user/item/complementarity feature pipelines
├── evaluation/              # NDCG/Precision, baselines, business impact, LLM judge
├── serving/                 # FastAPI pipeline, latency tracking
├── data/                    # Data loaders, schemas, processed parquets
├── models/                  # Trained LightGBM + neural reranker weights
├── configs/                 # YAML configs (base, ranking, serving)
├── tests/                   # LLM integration tests, cold-start tests
├── docs/                    # System design, latency analysis, data defense
└── scripts/                 # Entry points (build, train, eval, demo)
```

## Documentation

- [System Design](docs/system_design.md) — Full architecture and design decisions
- [Latency Analysis](docs/latency_analysis.md) — p50/p95/p99 breakdown
- [Business Impact Validation](docs/business_impact_validation.md) — Formula chain audit
- [Data Realism Defense](docs/data_realism_defense.md) — Synthetic data methodology
- [A/B Testing Plan](docs/ab_testing_plan.md) — Power analysis and guardrails
- [Architecture Diagram](docs/architecture_diagram.md) — Visual system overview
