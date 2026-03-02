# Diversity & Fairness Report

## Popularity Bias

- **Average popularity percentile of recommendations:** 0.52
  (1.0 = only most popular, 0.5 = uniform)
- **Recommendation Gini coefficient:** 0.0654
  (0 = all items equally recommended, 1 = single item dominates)

## Catalog Coverage

- **Items recommended / catalog size:** 120 / 120 (100.0%)
- **Long-tail coverage:** 96 / 96 (100.0%)

## Price Fairness

- **Avg recommended price:** ₹251 (catalog avg: ₹251)
- **Price ratio (reco/catalog):** 1.00
  (>1 = recommending more expensive items, <1 = cheaper)

## Category Distribution

- **KL divergence (reco ‖ catalog):** 0.0002
  (0 = identical distribution, higher = more divergent)

| Category | Catalog Share | Recommended Share |
|----------|--------------|-------------------|
| addon | 0.200 | 0.200 |
| beverage | 0.192 | 0.187 |
| dessert | 0.192 | 0.192 |
| main_course | 0.167 | 0.163 |
| starter | 0.250 | 0.257 |

## Summary

✅ Popularity distribution is reasonably diverse.
✅ Price exposure is balanced relative to catalog.