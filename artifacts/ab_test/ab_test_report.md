# A/B Test Offline Replay Report

## Setup
- **Control:** Popularity baseline (1274 sessions)
- **Treatment:** Full CSAO model (1274 sessions)
- **Assignment:** Random 50/50 query-level split
- **Bootstrap samples:** 1000

## Results

| Metric | Control | Treatment | Lift |
|--------|---------|-----------|------|
| NDCG@10 | 0.5166 | 0.5786 | +0.0621 |
| Precision@10 | 0.1000 | 0.1000 | +0.0000 |
| Attach Rate | 1.0000 | 1.0000 | +0.0 pp |

## Statistical Significance

- **Z-statistic (attach rate):** 0.000
- **p-value:** 1.0000
- **Significant at α=0.05:** ❌ NO
- **NDCG lift 95% CI:** [+0.0439, +0.0779]

## Interpretation

The difference did not reach statistical significance at α=0.05. This may be due to insufficient session count or small effect size.