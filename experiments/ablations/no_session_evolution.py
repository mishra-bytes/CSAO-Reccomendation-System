"""
Ablation Study: No Session Evolution Features
===============================================
Trains the LightGBM ranker WITHOUT session/cart-progression features
(session_position, cart_size, cart_completeness, etc.) and compares
NDCG@10 against the full model.

This measures the uplift from modelling the user's within-session journey.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Features to drop — session position and all cart context features
SESSION_EVOLUTION_COLS = [
    "session_position",
    "cart_size", "cart_value", "cart_avg_price", "cart_price_std",
    "cart_completeness", "cart_missing_cats", "cart_missing_cat_ratio",
    "cart_unique_categories",
    "cart_has_main", "cart_has_beverage", "cart_has_dessert",
    "cart_has_starter", "cart_has_addon",
    "cart_cat_share_addon", "cart_cat_share_beverage",
    "cart_cat_share_dessert", "cart_cat_share_main_course",
    "cart_cat_share_starter",
]


def run_no_session_evolution_ablation() -> dict:
    """Train LightGBM without session evolution features and report NDCG delta."""
    from lightgbm import LGBMRanker
    from scripts._utils import load_project_config
    from data.loaders import load_processed
    from ranking.training.dataset import build_training_dataset
    from ranking.training.train import _temporal_train_valid_split, _group_from_query_ids
    from evaluation.metrics.ranking_metrics import ndcg_at_k

    config = load_project_config()
    unified = load_processed()
    user_features = pd.read_parquet(ROOT / "data" / "processed" / "features_user.parquet")
    item_features = pd.read_parquet(ROOT / "data" / "processed" / "features_item.parquet")
    comp = pd.read_parquet(ROOT / "data" / "processed" / "features_complementarity.parquet")

    comp_lookup: dict = {}
    for _, row in comp.iterrows():
        a, b = str(row.get("item_a", "")), str(row.get("item_b", ""))
        if a and b:
            comp_lookup[(a, b)] = row.to_dict()

    ranking_cfg = config.get("ranking", {})
    ranking_params = config.get("lightgbm", {})

    data = build_training_dataset(
        unified=unified, user_features=user_features,
        item_features=item_features, comp_lookup=comp_lookup,
        config={"ranking": ranking_cfg},
    )

    orders = unified.get("orders", pd.DataFrame())
    val_frac = float(config.get("train", {}).get("validation_fraction", 0.2))
    seed = int(ranking_cfg.get("random_state", 42))
    train_idx, val_idx = _temporal_train_valid_split(data, orders, val_frac, seed)
    q_train = _group_from_query_ids([data.query_ids[i] for i in train_idx])
    q_val = _group_from_query_ids([data.query_ids[i] for i in val_idx])

    all_cols = list(data.X.columns)

    def _train_eval(cols: list[str], label: str) -> dict:
        X_tr = data.X.iloc[train_idx][cols]
        X_va = data.X.iloc[val_idx][cols]
        model = LGBMRanker(**ranking_params)
        model.fit(X_tr, data.y[train_idx], group=q_train,
                  eval_set=[(X_va, data.y[val_idx])], eval_group=[q_val], eval_at=[10])
        scores = model.predict(X_va)
        preds = pd.DataFrame({
            "query_id": [data.query_ids[i] for i in val_idx],
            "item_id": [data.candidate_items[i] for i in val_idx],
            "label": data.y[val_idx].astype(int),
            "score": scores.astype(float),
        })
        return {"label": label, "ndcg_10": ndcg_at_k(preds, k=10), "n_features": len(cols)}

    # Full model
    print("[ablation] Training full model...")
    full = _train_eval(all_cols, "ALL_FEATURES")

    # Drop session evolution
    drop_cols = [c for c in SESSION_EVOLUTION_COLS if c in all_cols]
    keep_cols = [c for c in all_cols if c not in drop_cols]
    print(f"[ablation] Training WITHOUT session evolution ({len(drop_cols)} features dropped)...")
    ablated = _train_eval(keep_cols, "NO_SESSION_EVOLUTION")

    delta = ablated["ndcg_10"] - full["ndcg_10"]

    result = {
        "full_model": full,
        "ablated_model": ablated,
        "dropped_features": drop_cols,
        "ndcg_delta": round(delta, 6),
        "relative_change_pct": round(100 * delta / max(full["ndcg_10"], 1e-9), 2),
        "conclusion": (
            "Session evolution features HELP ranking"
            if delta < -0.005
            else "Session evolution features have MINIMAL impact"
            if abs(delta) < 0.005
            else "Session evolution features HURT ranking (investigate)"
        ),
    }

    out_path = ROOT / "experiments" / "ablations" / "no_session_evolution_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[ablation] Results saved to {out_path}")
    return result


def main() -> None:
    result = run_no_session_evolution_ablation()
    print(f"\n=== No-Session-Evolution Ablation ===")
    print(f"  Full model NDCG@10:     {result['full_model']['ndcg_10']:.6f} ({result['full_model']['n_features']} features)")
    print(f"  Ablated model NDCG@10:  {result['ablated_model']['ndcg_10']:.6f} ({result['ablated_model']['n_features']} features)")
    print(f"  Delta:                  {result['ndcg_delta']:+.6f} ({result['relative_change_pct']:+.2f}%)")
    print(f"  Conclusion:             {result['conclusion']}")


if __name__ == "__main__":
    main()

