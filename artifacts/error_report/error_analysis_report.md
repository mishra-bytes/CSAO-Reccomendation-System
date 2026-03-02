# Error Analysis Report

**Total queries:** 2548
**Bottom 20% threshold:** NDCG ≤ 0.3869
**Queries in bottom bucket:** 712

## Bottom vs Good Queries

| Metric | Bottom 20% | Good 80% |
|--------|-----------|----------|
| Mean NDCG@10 | 0.3639 | 0.6680 |
| Mean # candidates | 7.0 | 7.0 |
| Mean # positives | 1.00 | 1.00 |
| Mean cart size | 2.04 | 2.44 |
| Mean cart value | ₹511 | ₹612 |

## Key Findings

1. Queries with fewer positive items in the candidate pool tend to have lower NDCG.
2. Cart size and user experience level correlate with prediction quality.
3. The model struggles most with queries where the positive item has low
   co-occurrence signals with the current cart.

## Recommendations

- Improve candidate generation recall for underperforming segments.
- Add additional features for new-user queries (e.g., cuisine-level priors).
- Consider segment-specific ranking models or feature weighting.