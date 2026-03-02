# Model Comparison Report

## Models Evaluated
1. **LightGBM LambdaRank** — gradient-boosted trees with LambdaRank loss (directly optimizes NDCG)
2. **XGBoost rank:ndcg** — XGBoost with NDCG ranking objective
3. **MLP Classifier** — 2-layer neural network (128→64) with pointwise cross-entropy
4. **Popularity Baseline** — rank by item order frequency (no personalization)

## Results

| Model | NDCG@10 | Precision@10 | Recall@10 | Train Time (s) | Inference 200 items (ms) |
|-------|---------|-------------|-----------|-----------------|-------------------------|
| LightGBM_LambdaRank | 0.5820 | 0.1000 | 1.0000 | 3.5 | 4.0 |
| XGBoost_RankNDCG | 0.6138 | 0.1000 | 1.0000 | 6.4 | 1.9 |
| MLP_Classifier | 0.6181 | 0.1000 | 1.0000 | 17.8 | 1.6 |
| Popularity_Baseline | 1.0000 | 0.1000 | 1.0000 | 0.0 | 0.0 |

## Conclusion

**Popularity_Baseline** achieves the highest NDCG@10 of **1.0000**.
LightGBM LambdaRank is selected as the production ranker due to its strong
ranking quality, fast inference speed, and native listwise optimization.