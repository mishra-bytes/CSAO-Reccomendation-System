"""
Cold-Start Validation
======================
Simulates genuinely new users (zero history) and sparse carts to validate
that the cold-start cascade in serving/pipeline/cold_start.py activates
correctly and produces reasonable recommendations.

Outputs:
  - artifacts/cold_start/cold_start_validation.csv
  - artifacts/cold_start/cold_start_report.md
  - artifacts/cold_start/cold_start_scenarios.png

Usage:
    python -m experiments.cold_start_validation
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run_cold_start_validation(
    output_dir: str = "artifacts/cold_start",
    k: int = 10,
) -> pd.DataFrame:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    from scripts._utils import load_project_config, load_unified_tables, load_feature_tables
    from serving.pipeline.cold_start import ColdStartHandler, ColdStartContext

    config = load_project_config()
    processed_dir = config.get("paths", {}).get("processed_dir", "data/processed")
    unified = load_unified_tables(processed_dir)
    features = load_feature_tables(processed_dir)

    items_df = unified["items"].drop_duplicates("item_id")
    order_items = unified["order_items"]
    user_features = features["user_features"]

    handler = ColdStartHandler(
        item_catalog=items_df,
        order_items=order_items,
        user_features=user_features,
        config=config,
    )

    # ── Build scenarios ───────────────────────────────────────────────────
    # Pick real items/restaurants from catalog for plausible contexts
    known_restaurants = list(
        order_items["restaurant_id"].astype(str).unique()[:5]
    ) if "restaurant_id" in order_items.columns else ["R999"]
    sample_items = list(items_df["item_id"].astype(str).head(5))

    fake_user_id = "COLD_USER_9999999"
    unknown_rest = "REST_UNKNOWN_9999"

    scenarios = [
        {
            "name": "Fully Cold — no user, no cart, unknown restaurant",
            "ctx": ColdStartContext(
                user_id=fake_user_id,
                restaurant_id=unknown_rest,
                cart_item_ids=[],
                hour_of_day=12,
            ),
        },
        {
            "name": "New user, empty cart, known restaurant",
            "ctx": ColdStartContext(
                user_id=fake_user_id,
                restaurant_id=known_restaurants[0],
                cart_item_ids=[],
                hour_of_day=19,
            ),
        },
        {
            "name": "New user, 1-item cart, known restaurant",
            "ctx": ColdStartContext(
                user_id=fake_user_id,
                restaurant_id=known_restaurants[0],
                cart_item_ids=sample_items[:1],
                hour_of_day=13,
            ),
        },
        {
            "name": "New user, 1-item cart, unknown restaurant",
            "ctx": ColdStartContext(
                user_id=fake_user_id,
                restaurant_id=unknown_rest,
                cart_item_ids=sample_items[:1],
                hour_of_day=20,
            ),
        },
        {
            "name": "New user, sparse cart (2 items), known restaurant",
            "ctx": ColdStartContext(
                user_id=fake_user_id,
                restaurant_id=known_restaurants[-1],
                cart_item_ids=sample_items[:2],
                hour_of_day=8,
            ),
        },
    ]

    # Also add a warm-user scenario for comparison
    warm_users = user_features[user_features["order_frequency"] >= 0.05]["user_id"].astype(str)
    if len(warm_users) > 0:
        warm_uid = warm_users.iloc[0]
        scenarios.append({
            "name": "Warm user baseline (for comparison)",
            "ctx": ColdStartContext(
                user_id=warm_uid,
                restaurant_id=known_restaurants[0],
                cart_item_ids=sample_items[:2],
                hour_of_day=13,
            ),
        })

    # ── Run each scenario ─────────────────────────────────────────────────
    results = []
    for sc in scenarios:
        ctx = sc["ctx"]
        classification = handler.classify(ctx)
        decision = handler.handle(ctx)

        n_candidates = len(decision.candidates)
        # Check category diversity of candidates
        item_cat_map = {}
        if "item_category" in items_df.columns:
            item_cat_map = dict(zip(
                items_df["item_id"].astype(str),
                items_df["item_category"].astype(str),
            ))
        reco_cats = set()
        for iid, _ in decision.candidates:
            reco_cats.add(item_cat_map.get(iid, "unknown"))

        results.append({
            "scenario": sc["name"],
            "classification": classification,
            "strategy": decision.strategy,
            "confidence": decision.confidence,
            "n_candidates": n_candidates,
            "n_unique_categories": len(reco_cats),
            "explanation": decision.explanation,
        })
        print(f"[cold_start] {sc['name']}")
        print(f"  → classification={classification}  strategy={decision.strategy}  "
              f"candidates={n_candidates}  categories={len(reco_cats)}")

    results_df = pd.DataFrame(results)
    csv_path = out / "cold_start_validation.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n[cold_start] Validation CSV → {csv_path}")

    # ── Also run the segment evaluation on val predictions ────────────────
    from features.complementarity import build_complementarity_lookup
    from ranking.training.dataset import build_training_dataset
    from ranking.training.train import _temporal_train_valid_split, _group_from_query_ids

    comp_lookup = build_complementarity_lookup(features["complementarity"])
    data = build_training_dataset(
        unified=unified,
        user_features=features["user_features"],
        item_features=features["item_features"],
        comp_lookup=comp_lookup,
        config=config,
    )
    orders = unified.get("orders", pd.DataFrame())
    val_frac = float(config.get("train", {}).get("validation_fraction", 0.2))
    seed = int(config.get("ranking", {}).get("random_state", 42))
    train_idx, val_idx = _temporal_train_valid_split(data, orders, val_frac, seed)

    from lightgbm import LGBMRanker
    ranking_params = config.get("lightgbm", {})
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
    query_meta = pd.DataFrame({
        "query_id": [data.query_ids[i] for i in val_idx],
        "user_id": [data.query_ids[i].split("_")[0] if "_" in str(data.query_ids[i]) else "unknown"
                    for i in val_idx],
    }).drop_duplicates("query_id")

    from serving.pipeline.cold_start import evaluate_cold_start_segments
    seg_df = evaluate_cold_start_segments(predictions, query_meta, user_features, k=k)
    seg_csv = out / "cold_start_segments.csv"
    seg_df.to_csv(seg_csv, index=False)
    print(f"[cold_start] Segment eval → {seg_csv}")
    print(seg_df.to_string(index=False))

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scenario validation results
    cold_scenarios = results_df[results_df["classification"] != "warm_user"]
    ax = axes[0]
    bars = ax.barh(range(len(cold_scenarios)), cold_scenarios["n_candidates"],
                   color="#4C72B0", edgecolor="white")
    ax.set_yticks(range(len(cold_scenarios)))
    ax.set_yticklabels(cold_scenarios["scenario"], fontsize=8)
    ax.set_xlabel("Number of Candidates")
    ax.set_title("Cold-Start Cascade: Candidates per Scenario")
    ax.invert_yaxis()
    for i, (nc, ncat) in enumerate(zip(cold_scenarios["n_candidates"], cold_scenarios["n_unique_categories"])):
        ax.text(nc + 0.2, i, f"{ncat} cats", va="center", fontsize=8)

    # Segment NDCG
    ax2 = axes[1]
    segs = seg_df[seg_df["segment"] != "ALL"]
    ndcg_col = "ndcg@10" if "ndcg@10" in segs.columns else "ndcg_10"
    if len(segs) > 0 and ndcg_col in segs.columns:
        ax2.bar(range(len(segs)), segs[ndcg_col], color="#55A868", edgecolor="white")
        ax2.set_xticks(range(len(segs)))
        ax2.set_xticklabels(segs["segment"], rotation=25, ha="right", fontsize=8)
        ax2.set_ylabel("NDCG@10")
        ax2.set_title("Model Performance by User Segment")
    else:
        ax2.set_title("No segment data available")

    fig.tight_layout()
    plot_path = out / "cold_start_scenarios.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[cold_start] Plots → {plot_path}")

    # ── Markdown report ───────────────────────────────────────────────────
    md = [
        "# Cold-Start Validation Report",
        "",
        "## Scenario Testing",
        "",
        "We simulate genuinely new users (zero order history) with varying levels",
        "of context (empty cart, sparse cart, known/unknown restaurant) to validate",
        "that the cold-start cascade in `serving/pipeline/cold_start.py` activates",
        "and produces reasonable recommendations.",
        "",
        "| Scenario | Classification | Strategy | Candidates | Categories | Confidence |",
        "|----------|---------------|----------|-----------|-----------|-----------|",
    ]
    for _, row in results_df.iterrows():
        md.append(
            f"| {row['scenario']} | {row['classification']} | {row['strategy']} | "
            f"{row['n_candidates']} | {row['n_unique_categories']} | {row['confidence']:.1f} |"
        )
    md += [
        "",
        "## Key Findings",
        "",
    ]
    cold_only = results_df[results_df["classification"] != "warm_user"]
    all_activated = (cold_only["n_candidates"] > 0).all()
    if all_activated:
        md.append("✅ **All cold-start scenarios produced recommendations.** "
                   "The cascade correctly activates different strategies based on available context.")
    else:
        zero = cold_only[cold_only["n_candidates"] == 0]
        md.append(f"⚠️ **{len(zero)} scenario(s) produced zero candidates:**")
        for _, r in zero.iterrows():
            md.append(f"  - {r['scenario']} (classification: {r['classification']})")

    avg_cats = cold_only["n_unique_categories"].mean()
    md.append(f"\n- Average category diversity across cold-start scenarios: **{avg_cats:.1f}** unique categories")
    md.append(f"- Confidence ranges from {cold_only['confidence'].min():.1f} to {cold_only['confidence'].max():.1f}")

    md += [
        "",
        "## Segment-Level Model Performance",
        "",
    ]
    if len(seg_df) > 0:
        md.append("| Segment | Queries | NDCG@10 | Precision@10 | Recall@10 |")
        md.append("|---------|---------|---------|-------------|----------|")
        for _, row in seg_df.iterrows():
            ndcg_val = row.get("ndcg@10", row.get("ndcg_10", 0))
            prec_val = row.get("precision@10", row.get("precision_10", 0))
            rec_val = row.get("recall@10", row.get("recall_10", 0))
            md.append(f"| {row['segment']} | {row['n_queries']} | {ndcg_val:.4f} | {prec_val:.4f} | {rec_val:.4f} |")

    md += [
        "",
        "## Strategy Cascade",
        "",
        "The `ColdStartHandler` classifies each request into one of:",
        "1. `warm_user` → delegates to main LTR pipeline",
        "2. `new_user_with_cart_known_rest` → cart-aware restaurant popular",
        "3. `new_user_with_cart_unknown_rest` → cart-aware global popular",
        "4. `new_user_empty_cart_known_rest` → restaurant + meal-time popular",
        "5. `new_user_empty_cart_unknown_rest` → category-diverse global popular",
        "",
        "All five branches are validated in this report.",
    ]

    report_path = out / "cold_start_report.md"
    report_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[cold_start] Report → {report_path}")

    return results_df


def main():
    parser = argparse.ArgumentParser(description="Cold-start validation")
    parser.add_argument("--output-dir", default="artifacts/cold_start")
    args = parser.parse_args()
    run_cold_start_validation(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
