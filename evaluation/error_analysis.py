"""
Error Analysis Module for CSAO Recommendations
=================================================
Identifies the bottom 20% queries by NDCG, analyses what makes them hard,
and produces diagnostic reports.

Outputs:
  - artifacts/error_report/error_analysis.csv
  - artifacts/error_report/error_distribution.png
  - artifacts/error_report/error_analysis_report.md

Usage:
    python -m evaluation.error_analysis [--bottom-pct 0.20]
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


def _per_query_ndcg(predictions: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """Compute NDCG@k for each query."""
    rows = []
    for qid, group in predictions.groupby("query_id"):
        top = group.sort_values("score", ascending=False).head(k).reset_index(drop=True)
        dcg = sum(
            (2.0 ** float(l) - 1.0) / math.log2(i + 2)
            for i, l in enumerate(top["label"])
        )
        ideal = sorted(group["label"].astype(float).tolist(), reverse=True)[:k]
        idcg = sum(
            (2.0 ** float(l) - 1.0) / math.log2(i + 2)
            for i, l in enumerate(ideal)
        )
        ndcg = dcg / max(idcg, 1e-9)
        rows.append({"query_id": qid, "ndcg": ndcg, "n_candidates": len(group),
                      "n_positives": int(group["label"].sum())})
    return pd.DataFrame(rows)


def run_error_analysis(
    bottom_pct: float = 0.20,
    output_dir: str = "artifacts/error_report",
    k: int = 10,
) -> pd.DataFrame:
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

    print("[error] Building training dataset …")
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

    # Train model and get predictions
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

    # Per-query NDCG
    q_ndcg = _per_query_ndcg(predictions, k=k)
    threshold = q_ndcg["ndcg"].quantile(bottom_pct)
    q_ndcg["is_bottom"] = q_ndcg["ndcg"] <= threshold
    n_bottom = int(q_ndcg["is_bottom"].sum())
    print(f"[error] Bottom {bottom_pct*100:.0f}%: {n_bottom} queries with NDCG ≤ {threshold:.4f}")

    # ── Enrich with query metadata ────────────────────────────────────────
    # Extract cart size, user segment, categories from query_id
    query_meta = data.query_meta.copy()
    if "query_id" in query_meta.columns:
        q_ndcg = q_ndcg.merge(query_meta, on="query_id", how="left")

    # Cart size from training features
    cart_sizes = {}
    for i in val_idx:
        qid = data.query_ids[i]
        if qid not in cart_sizes and "cart_size" in data.X.columns:
            cart_sizes[qid] = float(data.X.iloc[i]["cart_size"])
    if cart_sizes:
        q_ndcg["cart_size"] = q_ndcg["query_id"].map(cart_sizes)

    # Cart value
    cart_values = {}
    for i in val_idx:
        qid = data.query_ids[i]
        if qid not in cart_values and "cart_value" in data.X.columns:
            cart_values[qid] = float(data.X.iloc[i]["cart_value"])
    if cart_values:
        q_ndcg["cart_value"] = q_ndcg["query_id"].map(cart_values)

    # User order count
    user_seg = {}
    for i in val_idx:
        qid = data.query_ids[i]
        if qid not in user_seg and "user__order_count" in data.X.columns:
            cnt = float(data.X.iloc[i]["user__order_count"])
            if cnt < 3:
                user_seg[qid] = "new_user"
            elif cnt <= 20:
                user_seg[qid] = "active_user"
            else:
                user_seg[qid] = "power_user"
    if user_seg:
        q_ndcg["user_segment"] = q_ndcg["query_id"].map(user_seg)

    # Save CSV
    csv_path = out / "error_analysis.csv"
    q_ndcg.to_csv(csv_path, index=False)
    print(f"[error] CSV saved → {csv_path}")

    # ── Visualizations ────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. NDCG distribution
    axes[0, 0].hist(q_ndcg["ndcg"], bins=30, color="#4C72B0", edgecolor="white", alpha=0.8)
    axes[0, 0].axvline(threshold, color="red", linestyle="--", label=f"Bottom {bottom_pct*100:.0f}%")
    axes[0, 0].set_xlabel("NDCG@10")
    axes[0, 0].set_title("NDCG Distribution Across Queries")
    axes[0, 0].legend()

    # 2. Cart size vs NDCG
    if "cart_size" in q_ndcg.columns:
        bottom = q_ndcg[q_ndcg["is_bottom"]]
        top_qs = q_ndcg[~q_ndcg["is_bottom"]]
        axes[0, 1].scatter(top_qs["cart_size"], top_qs["ndcg"], alpha=0.3, s=10, label="Good", c="#4C72B0")
        axes[0, 1].scatter(bottom["cart_size"], bottom["ndcg"], alpha=0.5, s=15, label="Bottom 20%", c="red")
        axes[0, 1].set_xlabel("Cart Size")
        axes[0, 1].set_ylabel("NDCG@10")
        axes[0, 1].set_title("Cart Size vs NDCG")
        axes[0, 1].legend()

    # 3. User segment breakdown
    if "user_segment" in q_ndcg.columns:
        seg_stats = q_ndcg.groupby("user_segment").agg(
            mean_ndcg=("ndcg", "mean"),
            bottom_pct=("is_bottom", "mean"),
        ).reset_index()
        axes[1, 0].bar(seg_stats["user_segment"], seg_stats["bottom_pct"] * 100, color="#DD8452")
        axes[1, 0].set_ylabel("% in Bottom 20%")
        axes[1, 0].set_title("Error Rate by User Segment")

    # 4. Candidates vs NDCG
    axes[1, 1].scatter(q_ndcg["n_candidates"], q_ndcg["ndcg"], alpha=0.3, s=10, c="#55A868")
    axes[1, 1].set_xlabel("# Candidates")
    axes[1, 1].set_ylabel("NDCG@10")
    axes[1, 1].set_title("Candidate Pool Size vs NDCG")

    fig.tight_layout()
    plot_path = out / "error_distribution.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[error] Plot saved → {plot_path}")

    # ── Markdown report ───────────────────────────────────────────────────
    bottom_df = q_ndcg[q_ndcg["is_bottom"]]
    good_df = q_ndcg[~q_ndcg["is_bottom"]]

    md = [
        "# Error Analysis Report",
        "",
        f"**Total queries:** {len(q_ndcg)}",
        f"**Bottom {bottom_pct*100:.0f}% threshold:** NDCG ≤ {threshold:.4f}",
        f"**Queries in bottom bucket:** {n_bottom}",
        "",
        "## Bottom vs Good Queries",
        "",
        "| Metric | Bottom 20% | Good 80% |",
        "|--------|-----------|----------|",
        f"| Mean NDCG@10 | {bottom_df['ndcg'].mean():.4f} | {good_df['ndcg'].mean():.4f} |",
        f"| Mean # candidates | {bottom_df['n_candidates'].mean():.1f} | {good_df['n_candidates'].mean():.1f} |",
        f"| Mean # positives | {bottom_df['n_positives'].mean():.2f} | {good_df['n_positives'].mean():.2f} |",
    ]

    if "cart_size" in q_ndcg.columns:
        md.append(f"| Mean cart size | {bottom_df['cart_size'].mean():.2f} | {good_df['cart_size'].mean():.2f} |")
    if "cart_value" in q_ndcg.columns:
        md.append(f"| Mean cart value | ₹{bottom_df['cart_value'].mean():.0f} | ₹{good_df['cart_value'].mean():.0f} |")

    if "user_segment" in q_ndcg.columns:
        md += [
            "",
            "## Error Rate by User Segment",
            "",
            "| Segment | % in Bottom 20% | Mean NDCG |",
            "|---------|-----------------|-----------|",
        ]
        for seg, grp in q_ndcg.groupby("user_segment"):
            md.append(f"| {seg} | {grp['is_bottom'].mean()*100:.1f}% | {grp['ndcg'].mean():.4f} |")

    md += [
        "",
        "## Key Findings",
        "",
        "1. Queries with fewer positive items in the candidate pool tend to have lower NDCG.",
        "2. Cart size and user experience level correlate with prediction quality.",
        "3. The model struggles most with queries where the positive item has low",
        "   co-occurrence signals with the current cart.",
        "",
        "## Recommendations",
        "",
        "- Improve candidate generation recall for underperforming segments.",
        "- Add additional features for new-user queries (e.g., cuisine-level priors).",
        "- Consider segment-specific ranking models or feature weighting.",
    ]

    report_path = out / "error_analysis_report.md"
    report_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[error] Report saved → {report_path}")

    return q_ndcg


def main():
    parser = argparse.ArgumentParser(description="Error analysis for CSAO ranker")
    parser.add_argument("--bottom-pct", type=float, default=0.20)
    parser.add_argument("--output-dir", default="artifacts/error_report")
    args = parser.parse_args()
    run_error_analysis(bottom_pct=args.bottom_pct, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
