"""
Model Comparison: LightGBM vs Neural Reranker vs Baselines
============================================================
Compares ranking quality across different model configurations:
  1. LightGBM alone
  2. LightGBM + Neural Reranker (alpha blend)
  3. Popularity baseline
  4. Co-occurrence baseline
  5. Random baseline

Reports NDCG@10, Precision@10, Coverage@10 for each.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def run_model_comparison(top_n: int = 10) -> dict:
    """Compare multiple ranker configurations on the validation set."""
    from scripts._utils import load_project_config
    from data.loaders import load_processed
    from ranking.training.dataset import build_training_dataset
    from ranking.training.train import _temporal_train_valid_split, _group_from_query_ids
    from evaluation.metrics.ranking_metrics import ndcg_at_k, precision_at_k, coverage_at_k

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

    X_train = data.X.iloc[train_idx]
    y_train = data.y[train_idx]
    X_val = data.X.iloc[val_idx]
    y_val = data.y[val_idx]
    val_query_ids = [data.query_ids[i] for i in val_idx]
    val_items = [data.candidate_items[i] for i in val_idx]

    all_item_ids = set(str(i) for i in unified.get("items", pd.DataFrame()).get("item_id", []))

    def _make_preds(scores: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame({
            "query_id": val_query_ids,
            "item_id": val_items,
            "label": y_val.astype(int),
            "score": scores.astype(float),
        })

    def _evaluate(preds_df: pd.DataFrame) -> dict:
        return {
            "ndcg@10": round(ndcg_at_k(preds_df, k=top_n), 6),
            "precision@10": round(precision_at_k(preds_df, k=top_n), 6),
            "coverage@10": round(coverage_at_k(preds_df, k=top_n, n_items=len(all_item_ids)), 6),
        }

    results = {}

    # 1. LightGBM only
    from lightgbm import LGBMRanker
    print("[compare] Training LightGBM ranker...")
    t0 = time.time()
    lgbm = LGBMRanker(**ranking_params)
    lgbm.fit(X_train, y_train, group=q_train,
             eval_set=[(X_val, y_val)], eval_group=[q_val], eval_at=[10])
    lgbm_scores = lgbm.predict(X_val)
    lgbm_time = time.time() - t0
    preds_lgbm = _make_preds(lgbm_scores)
    results["LightGBM"] = {**_evaluate(preds_lgbm), "train_time_s": round(lgbm_time, 1)}

    # 2. LightGBM + Neural Reranker
    try:
        from ranking.inference.neural_reranker import NeuralReranker
        emb_path = ROOT / "data" / "processed" / "recipe_embeddings.parquet"
        item_embs: dict[str, np.ndarray] = {}
        if emb_path.exists():
            emb_df = pd.read_parquet(emb_path)
            emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
            if "item_id" in emb_df.columns and emb_cols:
                for _, row in emb_df.iterrows():
                    item_embs[str(row["item_id"])] = row[emb_cols].values.astype(np.float32)

        model_path = ROOT / "models" / "neural_reranker.pt"
        d_in = len(next(iter(item_embs.values()))) if item_embs else 8

        for alpha in [0.3, 0.5, 0.7]:
            reranker = NeuralReranker(
                model_path=str(model_path) if model_path.exists() else None,
                item_embeddings=item_embs,
                d_in=d_in,
                d_model=32,
                alpha=alpha,
            )
            # For each query group, rerank candidates
            blended_scores = lgbm_scores.copy()
            preds_df = _make_preds(lgbm_scores)
            for qid in preds_df["query_id"].unique():
                mask = preds_df["query_id"] == qid
                q_items = preds_df.loc[mask, "item_id"].tolist()
                q_scores = lgbm_scores[mask.values]
                candidates = list(zip(q_items, q_scores.tolist()))
                # We need cart items — approximate from positive labels
                positives = preds_df.loc[mask & (preds_df["label"] > 0), "item_id"].tolist()
                cart = positives[:3] if positives else q_items[:1]
                reranked = reranker.rerank(cart, candidates, top_n=len(candidates))
                reranked_map = {iid: sc for iid, sc in reranked}
                for idx in preds_df.index[mask]:
                    iid = preds_df.at[idx, "item_id"]
                    blended_scores[preds_df.index.get_loc(idx)] = reranked_map.get(iid, 0.0)

            neural_preds = _make_preds(blended_scores)
            label = f"LightGBM+Neural(α={alpha})"
            results[label] = _evaluate(neural_preds)
            print(f"[compare] {label}: NDCG@10={results[label]['ndcg@10']:.4f}")

    except Exception as e:
        results["LightGBM+Neural"] = {"error": str(e)}
        print(f"[compare] Neural reranker failed: {e}")

    # 3. Popularity baseline
    order_items = unified.get("order_items", pd.DataFrame())
    item_pop = order_items["item_id"].value_counts()
    pop_lookup = item_pop.to_dict()
    pop_scores = np.array([pop_lookup.get(iid, 0) for iid in val_items], dtype=float)
    pop_scores = pop_scores / (pop_scores.max() + 1e-9)
    results["Popularity"] = _evaluate(_make_preds(pop_scores))

    # 4. Co-occurrence baseline
    co_scores = np.zeros(len(val_items), dtype=float)
    for i, iid in enumerate(val_items):
        qid = val_query_ids[i]
        # Find other items in the same query as proxy for cart
        q_mask = [j for j, q in enumerate(val_query_ids) if q == qid and val_items[j] != iid]
        cart = [val_items[j] for j in q_mask[:5]]
        lift_sum = 0.0
        for ci in cart:
            key = (str(ci), str(iid))
            if key in comp_lookup:
                lift_sum += comp_lookup[key].get("lift", 0.0)
        co_scores[i] = lift_sum
    if co_scores.max() > 0:
        co_scores = co_scores / co_scores.max()
    results["CoOccurrence"] = _evaluate(_make_preds(co_scores))

    # 5. Random baseline
    rng = np.random.default_rng(42)
    random_scores = rng.random(len(val_items))
    results["Random"] = _evaluate(_make_preds(random_scores))

    # Summary
    out_path = ROOT / "experiments" / "model_comparisons" / "comparison_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[compare] Results saved to {out_path}")

    return results


def main() -> None:
    results = run_model_comparison()
    print("\n=== Model Comparison ===")
    print(f"{'Model':<30s} {'NDCG@10':>10s} {'Prec@10':>10s} {'Cover@10':>10s}")
    print("-" * 62)
    for model_name, metrics in results.items():
        if "error" in metrics:
            print(f"{model_name:<30s} ERROR: {metrics['error']}")
        else:
            print(
                f"{model_name:<30s} "
                f"{metrics.get('ndcg@10', 0):10.4f} "
                f"{metrics.get('precision@10', 0):10.4f} "
                f"{metrics.get('coverage@10', 0):10.4f}"
            )


if __name__ == "__main__":
    main()

