# Cart Super Add-On (CSAO) Recommender

Production-style hackathon scaffold for real-time cart-aware add-on recommendations in food delivery.

## Objectives

- Recommend complementary add-ons from an evolving cart
- Support cart-aware + session-aware ranking
- Keep serving path compatible with 200-300 ms latency budgets
- Separate offline training from online inference

## Quickstart

```bash
python -m pip install -e .
python scripts/build_unified_data.py
python scripts/build_features.py
python scripts/train_ranker.py
python scripts/run_offline_eval.py
python scripts/simulate_serving.py
```

## Key Entry Points

- `scripts/build_unified_data.py`: data ingestion + unification
- `scripts/build_features.py`: feature pipelines
- `scripts/train_ranker.py`: LightGBM ranker training
- `scripts/run_offline_eval.py`: Precision/Recall/NDCG/Coverage/Diversity
- `scripts/simulate_serving.py`: end-to-end request simulation

## Repository Layout

- `data/`: ingestion, normalization, unified data tables
- `features/`: cart/user/item/complementarity feature pipelines
- `candidate_generation/`: retrieval and fallback logic
- `ranking/`: LightGBM ranker training and inference wrappers
- `evaluation/`: ranking quality, coverage, diversity, segment analysis
- `serving/`: low-latency recommendation pipeline simulation
- `configs/`: centralized YAML configs
- `experiments/`: ablation and comparison placeholders
- `docs/`: system design and production notes

## Data Contracts

Unified tables produced in `data/processed/`:

- `users.parquet`
- `orders.parquet`
- `order_items.parquet`
- `items.parquet`
- `restaurants.parquet`

## Training vs Inference Separation

- Training: `ranking/training/`, `evaluation/`, `experiments/`
- Inference: `candidate_generation/`, `ranking/inference/`, `serving/`

## Notes

- Synthetic fallback data is enabled by default for hackathon speed.
- Replace synthetic data with real datasets by updating `configs/base.yaml`.
- TODO markers in code indicate production hardening gaps.
