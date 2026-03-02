# Business Impact Validation

## Formula chain
Let:
- `orders_per_day`
- `addon_attach_rate` (baseline and treatment)
- `avg_addon_value`
- `contribution_margin`

Then:
1. `incremental_addons_per_day = orders_per_day * (attach_rate_treatment - attach_rate_baseline)`
2. `incremental_gmv_per_day = incremental_addons_per_day * avg_addon_value`
3. `incremental_margin_per_day = incremental_gmv_per_day * contribution_margin`

## Assumption table
| Assumption | Conservative | Baseline | Optimistic |
|---|---:|---:|---:|
| Orders/day | 10,000 | 10,000 | 10,000 |
| Attach-rate uplift | 0.5 pp | 1.5 pp | 3.0 pp |
| Avg add-on value (₹) | 60 | 75 | 90 |
| Margin | 18% | 22% | 25% |

## Sensitivity output (example)
| Scenario | Incremental GMV/day (₹) | Incremental margin/day (₹) |
|---|---:|---:|
| Conservative | 30,000 | 5,400 |
| Baseline | 112,500 | 24,750 |
| Optimistic | 270,000 | 67,500 |

Key drivers: attach-rate uplift and average add-on value.
