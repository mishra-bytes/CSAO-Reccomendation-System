# Ablation: Session Evolution Features

## Setup
- Dropped features: `['session_position', 'cart_size', 'cart_completeness', 'cart_missing_cats', 'cart_missing_cat_ratio', 'completeness_delta', 'fills_meal_gap']`
- Training samples: 71364, Validation: 17836

## Results

| Variant | NDCG@10 | Precision@10 | Recall@10 | NDCG Δ |
|---------|---------|-------------|-----------|--------|
| full_model | 0.5864 | 0.1000 | 1.0000 | +0.0000 |
| no_session_evolution | 0.5826 | 0.1000 | 1.0000 | -0.0037 |

## Interpretation

Removing session evolution features causes an NDCG@10 change of **-0.0037**.
These features model how the cart grows within an order, helping the ranker
understand which items are needed at different stages of meal assembly.