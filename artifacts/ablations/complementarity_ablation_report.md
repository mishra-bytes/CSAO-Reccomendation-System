# Ablation: Complementarity Features

## Setup
- Dropped features: `['comp_max_lift', 'comp_mean_lift', 'comp_max_pmi', 'comp_mean_pmi']`
- Training samples: 71364, Validation: 17836

## Results

| Variant | NDCG@10 | Precision@10 | Recall@10 | NDCG Δ |
|---------|---------|-------------|-----------|--------|
| full_model | 0.5828 | 0.1000 | 1.0000 | +0.0000 |
| no_complementarity | 0.5666 | 0.1000 | 1.0000 | -0.0162 |

## Interpretation

Removing complementarity features causes an NDCG@10 change of **-0.0162**.
This confirms that co-purchase lift and PMI signals provide meaningful ranking signal.