# Cart Super Add-On (CSAO) Recommender

Cart-aware, real-time add-on recommendation engine for food delivery — built with LightGBM LambdaRank, complementarity features, LLM-powered explanations, and a FastAPI serving layer.

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
- `experiments/`: ablation studies, model comparisons, A/B simulation, SHAP analysis
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

- Synthetic data is used for development and demonstration.
- Replace with real datasets by updating `configs/base.yaml`.
- See `docs/` for system design, latency analysis, and A/B testing plan.

## Additional Scripts

```bash
# SHAP feature importance
python -m experiments.feature_importance.shap_analysis

# Ablation studies
python -m experiments.ablations.no_complementarity
python -m experiments.ablations.no_session_evolution

# Model comparison (LightGBM vs XGBoost vs MLP vs Popularity)
python -m experiments.model_comparisons.compare_rankers

# A/B test offline simulation
python -m experiments.ab_test_simulator

# Error analysis (bottom-20% queries)
python -m evaluation.error_analysis

# Diversity & fairness analysis
python -m evaluation.fairness_analysis

# Cold-start validation
python -m experiments.cold_start_validation

# HP tuning (Optuna, 100 trials)
python -m experiments.ablations.tuning_and_ablation

# Serve API
python -m serving.api.main
```
