# SHAP Feature Importance Report

**Samples analysed:** 200
**Total features:** 67

## Top 20 Features by Mean |SHAP|

| Rank | Feature | Mean |SHAP| |
|------|---------|-------------|
| 1 | `comp_mean_lift` | 0.179856 |
| 2 | `comp_mean_pmi` | 0.129555 |
| 3 | `complement_confidence` | 0.089330 |
| 4 | `comp_max_lift` | 0.069481 |
| 5 | `comp_max_pmi` | 0.069305 |
| 6 | `candidate_score` | 0.046292 |
| 7 | `item_item_popularity` | 0.020871 |
| 8 | `completeness_delta` | 0.018073 |
| 9 | `item_emb_0` | 0.016091 |
| 10 | `item_emb_3` | 0.015394 |
| 11 | `item_emb_2` | 0.015162 |
| 12 | `item_item_price` | 0.014599 |
| 13 | `item_emb_7` | 0.014366 |
| 14 | `item_emb_4` | 0.013405 |
| 15 | `item_emb_1` | 0.011706 |
| 16 | `item_emb_5` | 0.010434 |
| 17 | `item_emb_6` | 0.009464 |
| 18 | `fills_meal_gap` | 0.007516 |
| 19 | `candidate_new_category` | 0.006998 |
| 20 | `item_item_price_band_high` | 0.001832 |

## Feature Group Importance

| Group | Sum Mean |SHAP| |
|-------|-------------|
| csao_intelligence | 0.168209 |
| llm_embeddings | 0.106022 |
| cart_context | 0.000000 |
| cart_category_shares | 0.000000 |
| user_features | 0.000000 |
| item_features | 0.000000 |
| complementarity | 0.000000 |

## Plots

- Bar chart: `artifacts/shap/shap_summary_bar.png`
- Beeswarm: `artifacts/shap/shap_beeswarm.png`
- Dependence: `artifacts/shap/shap_dependence_1_comp_mean_lift.png`
- Dependence: `artifacts/shap/shap_dependence_2_comp_mean_pmi.png`
- Dependence: `artifacts/shap/shap_dependence_3_complement_confidence.png`

## Interpretation

The SHAP analysis reveals which features the LightGBM LambdaRank model
relies on most heavily when scoring add-on candidates. Features with high
mean |SHAP| values have the greatest influence on the model's ranking
decisions. The beeswarm plot shows the distribution and direction of each
feature's impact (red = high feature value).

Group-level aggregation shows which conceptual feature groups contribute
most to the model's predictions, informing future feature engineering.