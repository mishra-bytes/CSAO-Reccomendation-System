"""
Diversity & Fairness Analysis for CSAO Recommendations
========================================================
Computes popularity bias, price exposure distribution, long-tail coverage,
and category fairness metrics.

Outputs:
  - artifacts/fairness/bias_metrics.csv
  - artifacts/fairness/fairness_report.md
  - artifacts/fairness/fairness_plots.png

Usage:
    python -m evaluation.fairness_analysis
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _gini_coefficient(values: np.ndarray) -> float:
    """Compute Gini coefficient (0 = perfect equality, 1 = max inequality)."""
    if len(values) == 0:
        return 0.0
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * sorted_vals) - (n + 1) * np.sum(sorted_vals)) / (n * np.sum(sorted_vals) + 1e-9))


def run_fairness_analysis(
    output_dir: str = "artifacts/fairness",
    k: int = 10,
) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    from scripts._utils import load_project_config, load_unified_tables, load_feature_tables
    from features.complementarity import build_complementarity_lookup
    from ranking.training.dataset import build_training_dataset
    from ranking.training.train import _temporal_train_valid_split, _group_from_query_ids

    config = load_project_config()
    ranking_cfg = config.get("ranking", {})
    ranking_params = config.get("lightgbm", {})

    processed_dir = config.get("paths", {}).get("processed_dir", "data/processed")
    unified = load_unified_tables(processed_dir)
    features = load_feature_tables(processed_dir)
    comp_lookup = build_complementarity_lookup(features["complementarity"])

    print("[fairness] Building training dataset …")
    data = build_training_dataset(
        unified=unified,
        user_features=features["user_features"],
        item_features=features["item_features"],
        comp_lookup=comp_lookup,
        config=config,
    )

    orders = unified.get("orders", pd.DataFrame())
    val_frac = float(config.get("train", {}).get("validation_fraction", 0.2))
    seed = int(ranking_cfg.get("random_state", 42))
    train_idx, val_idx = _temporal_train_valid_split(data, orders, val_frac, seed)

    from lightgbm import LGBMRanker
    q_train = _group_from_query_ids([data.query_ids[i] for i in train_idx])
    q_val = _group_from_query_ids([data.query_ids[i] for i in val_idx])

    model = LGBMRanker(**ranking_params)
    model.fit(data.X.iloc[train_idx], data.y[train_idx], group=q_train,
              eval_set=[(data.X.iloc[val_idx], data.y[val_idx])],
              eval_group=[q_val], eval_at=[10])
    scores = model.predict(data.X.iloc[val_idx])

    predictions = pd.DataFrame({
        "query_id": [data.query_ids[i] for i in val_idx],
        "item_id": [data.candidate_items[i] for i in val_idx],
        "label": data.y[val_idx].astype(int),
        "score": scores.astype(float),
    })

    # Get top-k recommendations per query
    predictions["rank"] = predictions.groupby("query_id")["score"].rank(ascending=False, method="first")
    top_k = predictions[predictions["rank"] <= k].copy()

    # Build item metadata lookup
    items_df = unified["items"].drop_duplicates("item_id")
    item_meta = {}
    for _, row in items_df.iterrows():
        iid = str(row["item_id"])
        item_meta[iid] = {
            "category": str(row.get("item_category", "unknown")),
            "price": float(row.get("item_price", 0)),
        }
    top_k["category"] = top_k["item_id"].map(lambda x: item_meta.get(x, {}).get("category", "unknown"))
    top_k["price"] = top_k["item_id"].map(lambda x: item_meta.get(x, {}).get("price", 0))

    metrics = {}

    # ── 1. Popularity Bias ────────────────────────────────────────────────
    # How much the model favours popular items
    item_order_counts = unified["order_items"]["item_id"].astype(str).value_counts()
    total_orders = item_order_counts.sum()

    reco_items = top_k["item_id"].value_counts()
    reco_total = reco_items.sum()

    # Average popularity percentile of recommended items
    pop_pctile = item_order_counts.rank(pct=True)
    reco_pop_pctiles = top_k["item_id"].map(pop_pctile).dropna()
    metrics["avg_popularity_percentile"] = float(reco_pop_pctiles.mean()) if len(reco_pop_pctiles) > 0 else 0.0
    metrics["popularity_gini"] = _gini_coefficient(reco_items.values.astype(float))

    # ── 2. Long-tail Coverage ─────────────────────────────────────────────
    # What % of the catalog appears in recommendations?
    catalog_size = len(items_df)
    unique_reco = top_k["item_id"].nunique()
    metrics["catalog_coverage_pct"] = unique_reco / max(catalog_size, 1) * 100

    # Long-tail = items in bottom 80% by popularity
    pop_threshold = item_order_counts.quantile(0.8)
    long_tail_items = set(item_order_counts[item_order_counts <= pop_threshold].index)
    reco_long_tail = set(top_k["item_id"].unique()) & long_tail_items
    metrics["long_tail_coverage_pct"] = len(reco_long_tail) / max(len(long_tail_items), 1) * 100

    # ── 3. Price Exposure Distribution ────────────────────────────────────
    reco_prices = top_k["price"]
    catalog_prices = items_df["item_price"].astype(float)
    metrics["reco_avg_price"] = float(reco_prices.mean())
    metrics["catalog_avg_price"] = float(catalog_prices.mean())
    metrics["reco_median_price"] = float(reco_prices.median())
    metrics["catalog_median_price"] = float(catalog_prices.median())
    metrics["price_ratio"] = metrics["reco_avg_price"] / max(metrics["catalog_avg_price"], 1)

    # ── 4. Category Distribution ──────────────────────────────────────────
    reco_cat_dist = top_k["category"].value_counts(normalize=True)
    catalog_cat_dist = items_df["item_category"].astype(str).value_counts(normalize=True)

    # KL divergence (reco || catalog)
    all_cats = set(reco_cat_dist.index) | set(catalog_cat_dist.index)
    kl = 0.0
    for cat in all_cats:
        p = reco_cat_dist.get(cat, 1e-6)
        q = catalog_cat_dist.get(cat, 1e-6)
        kl += p * math.log(max(p, 1e-9) / max(q, 1e-9))
    metrics["category_kl_divergence"] = kl

    # Save metrics CSV
    metrics_df = pd.DataFrame([metrics])
    csv_path = out / "bias_metrics.csv"
    metrics_df.to_csv(csv_path, index=False)
    print(f"[fairness] Metrics saved → {csv_path}")

    # ── Visualizations ────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Popularity distribution: catalog vs recommended
    axes[0, 0].hist(catalog_prices, bins=20, alpha=0.5, label="Catalog", color="#4C72B0", density=True)
    axes[0, 0].hist(reco_prices, bins=20, alpha=0.5, label="Recommended", color="#DD8452", density=True)
    axes[0, 0].set_xlabel("Price (₹)")
    axes[0, 0].set_title("Price Distribution: Catalog vs Recommended")
    axes[0, 0].legend()

    # 2. Category distribution comparison
    cat_compare = pd.DataFrame({
        "catalog": catalog_cat_dist,
        "recommended": reco_cat_dist,
    }).fillna(0)
    cat_compare.plot(kind="bar", ax=axes[0, 1], rot=30)
    axes[0, 1].set_title("Category Share: Catalog vs Recommended")
    axes[0, 1].set_ylabel("Fraction")

    # 3. Item recommendation frequency (popularity bias)
    top_items = reco_items.head(20)
    axes[1, 0].barh(range(len(top_items)), top_items.values, color="#55A868")
    axes[1, 0].set_yticks(range(len(top_items)))
    axes[1, 0].set_yticklabels(top_items.index, fontsize=7)
    axes[1, 0].set_xlabel("Times Recommended")
    axes[1, 0].set_title("Top 20 Most Recommended Items")
    axes[1, 0].invert_yaxis()

    # 4. Popularity percentile of recommendations
    axes[1, 1].hist(reco_pop_pctiles, bins=20, color="#C44E52", edgecolor="white")
    axes[1, 1].set_xlabel("Popularity Percentile")
    axes[1, 1].set_title("Popularity Percentile of Recommended Items")

    fig.tight_layout()
    plot_path = out / "fairness_plots.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[fairness] Plots saved → {plot_path}")

    # ── Markdown report ───────────────────────────────────────────────────
    md = [
        "# Diversity & Fairness Report",
        "",
        "## Popularity Bias",
        "",
        f"- **Average popularity percentile of recommendations:** {metrics['avg_popularity_percentile']:.2f}",
        f"  (1.0 = only most popular, 0.5 = uniform)",
        f"- **Recommendation Gini coefficient:** {metrics['popularity_gini']:.4f}",
        f"  (0 = all items equally recommended, 1 = single item dominates)",
        "",
        "## Catalog Coverage",
        "",
        f"- **Items recommended / catalog size:** {unique_reco} / {catalog_size} ({metrics['catalog_coverage_pct']:.1f}%)",
        f"- **Long-tail coverage:** {len(reco_long_tail)} / {len(long_tail_items)} "
        f"({metrics['long_tail_coverage_pct']:.1f}%)",
        "",
        "## Price Fairness",
        "",
        f"- **Avg recommended price:** ₹{metrics['reco_avg_price']:.0f} "
        f"(catalog avg: ₹{metrics['catalog_avg_price']:.0f})",
        f"- **Price ratio (reco/catalog):** {metrics['price_ratio']:.2f}",
        "  (>1 = recommending more expensive items, <1 = cheaper)",
        "",
        "## Category Distribution",
        "",
        f"- **KL divergence (reco ‖ catalog):** {metrics['category_kl_divergence']:.4f}",
        "  (0 = identical distribution, higher = more divergent)",
        "",
        "| Category | Catalog Share | Recommended Share |",
        "|----------|--------------|-------------------|",
    ]
    for cat in sorted(all_cats):
        p_cat = catalog_cat_dist.get(cat, 0)
        r_cat = reco_cat_dist.get(cat, 0)
        md.append(f"| {cat} | {p_cat:.3f} | {r_cat:.3f} |")

    md += [
        "",
        "## Summary",
        "",
    ]
    if metrics["popularity_gini"] > 0.7:
        md.append("⚠️ High popularity Gini — the model concentrates on a few items. "
                   "Consider increasing MMR λ or adding diversity constraints.")
    else:
        md.append("✅ Popularity distribution is reasonably diverse.")

    if metrics["price_ratio"] > 1.2:
        md.append("⚠️ Recommendations skew toward expensive items. Consider price-aware reranking.")
    elif metrics["price_ratio"] < 0.8:
        md.append("ℹ️ Recommendations skew toward cheaper items (potentially good for value perception).")
    else:
        md.append("✅ Price exposure is balanced relative to catalog.")

    report_path = out / "fairness_report.md"
    report_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[fairness] Report saved → {report_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Diversity & fairness analysis")
    parser.add_argument("--output-dir", default="artifacts/fairness")
    args = parser.parse_args()
    run_fairness_analysis(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
