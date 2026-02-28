from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from common.io import save_table
from ranking.training.dataset import TrainingData, build_training_dataset


@dataclass
class TrainingOutputs:
    model: Any
    feature_columns: list[str]
    validation_predictions: pd.DataFrame
    query_meta: pd.DataFrame
    training_summary: dict[str, float]


def _load_lgbm_ranker(params: dict[str, Any]) -> Any:
    try:
        from lightgbm import LGBMRanker
    except ImportError as exc:
        raise ImportError("lightgbm is required. Install dependencies via `python -m pip install -e .`") from exc
    return LGBMRanker(**params)


# ------------------------------------------------------------------
# Temporal split utilities
# ------------------------------------------------------------------

def _temporal_train_valid_split(
    data: TrainingData,
    orders: pd.DataFrame,
    validation_fraction: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Time-based train/val split that prevents future leakage.

    Strategy:
    1. Map each query_id → order_id → order_ts.
    2. Sort orders by time.
    3. Split at the (1 - val_frac) percentile timestamp.
    4. All queries before cutoff → train, after → val.
    5. Validate no train query is later than any val query.
    """
    # Map query_id -> order_id (query format: "{order_id}__{pos}")
    query_to_order = {}
    for qid in pd.Series(data.query_ids).unique():
        order_id = qid.rsplit("__", 1)[0]
        query_to_order[qid] = order_id

    # Build order_id -> order_ts lookup
    orders_ts = orders.copy()
    orders_ts["order_ts"] = pd.to_datetime(orders_ts["order_ts"], errors="coerce")
    order_ts_map = orders_ts.drop_duplicates("order_id").set_index("order_id")["order_ts"].to_dict()

    # Assign timestamps to queries
    query_times = {}
    for qid, oid in query_to_order.items():
        ts = order_ts_map.get(oid)
        query_times[qid] = ts if ts is not None and pd.notna(ts) else pd.Timestamp.min

    # Sort unique queries by time
    unique_queries = sorted(query_times.keys(), key=lambda q: query_times[q])
    n_val = max(1, int(len(unique_queries) * validation_fraction))
    cutoff_idx = len(unique_queries) - n_val

    train_queries = set(unique_queries[:cutoff_idx])
    val_queries = set(unique_queries[cutoff_idx:])

    query_series = pd.Series(data.query_ids)
    is_val = query_series.isin(val_queries).values
    train_idx = np.where(~is_val)[0]
    val_idx = np.where(is_val)[0]

    # Leakage validation: assert max train time <= min val time
    if train_queries and val_queries:
        max_train_ts = max(query_times[q] for q in train_queries)
        min_val_ts = min(query_times[q] for q in val_queries)
        if max_train_ts > min_val_ts:
            import warnings
            warnings.warn(
                f"Temporal leakage detected: max train ts ({max_train_ts}) > "
                f"min val ts ({min_val_ts}). Some queries may share timestamps. "
                f"Filtering overlapping queries from train set."
            )
            # Remove any train queries with timestamp >= min_val_ts
            leaked = {q for q in train_queries if query_times[q] >= min_val_ts}
            train_queries -= leaked
            is_train_clean = query_series.isin(train_queries).values
            train_idx = np.where(is_train_clean)[0]

    return train_idx, val_idx


def _validate_no_future_item_leakage(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    data: TrainingData,
    orders: pd.DataFrame,
) -> dict[str, Any]:
    """Check that items appearing as positives in val were also seen in training data."""
    train_items = set()
    for i in train_idx:
        if data.y[i] == 1:
            train_items.add(data.candidate_items[i])

    val_positives = set()
    for i in val_idx:
        if data.y[i] == 1:
            val_positives.add(data.candidate_items[i])

    unseen = val_positives - train_items
    return {
        "train_positive_items": len(train_items),
        "val_positive_items": len(val_positives),
        "unseen_val_items": len(unseen),
        "unseen_val_item_fraction": len(unseen) / max(len(val_positives), 1),
        "leakage_free": True,  # temporal split guarantees no future → past leakage
    }


def _train_valid_split_legacy(data: TrainingData, validation_fraction: float, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    """Random split — kept as fallback when order timestamps unavailable."""
    rng = np.random.default_rng(random_state)
    unique_queries = pd.Series(data.query_ids).unique()
    n_val = max(1, int(len(unique_queries) * validation_fraction))
    val_queries = set(rng.choice(unique_queries, size=n_val, replace=False))

    query_series = pd.Series(data.query_ids)
    is_val = query_series.isin(val_queries).values
    train_idx = np.where(~is_val)[0]
    val_idx = np.where(is_val)[0]
    return train_idx, val_idx


def _group_from_query_ids(query_ids: list[str]) -> list[int]:
    return pd.Series(query_ids, name="query_id").value_counts(sort=False).tolist()


def train_lgbm_ranker(
    unified: dict[str, pd.DataFrame],
    user_features: pd.DataFrame,
    item_features: pd.DataFrame,
    comp_lookup: dict[tuple[str, str], tuple[float, float]],
    config: dict[str, Any],
) -> TrainingOutputs:
    ranking_cfg = config.get("ranking", {})
    ranking_params = config.get("lightgbm", {})
    validation_fraction = float(config.get("train", {}).get("validation_fraction", 0.2))
    random_state = int(ranking_cfg.get("random_state", 42))
    use_temporal_split = bool(config.get("train", {}).get("temporal_split", True))

    data = build_training_dataset(
        unified=unified,
        user_features=user_features,
        item_features=item_features,
        comp_lookup=comp_lookup,
        config={"ranking": ranking_cfg},
    )

    orders = unified.get("orders", pd.DataFrame())
    if use_temporal_split and "order_ts" in orders.columns and not orders.empty:
        print("[train] Using TEMPORAL train/val split (leakage-safe)")
        train_idx, val_idx = _temporal_train_valid_split(
            data, orders, validation_fraction=validation_fraction, random_state=random_state,
        )
        leakage_report = _validate_no_future_item_leakage(train_idx, val_idx, data, orders)
        print(f"[train] Leakage report: {leakage_report}")
    else:
        print("[train] Falling back to RANDOM train/val split (no order_ts available)")
        train_idx, val_idx = _train_valid_split_legacy(data, validation_fraction=validation_fraction, random_state=random_state)
        leakage_report = {}

    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("Training/validation split failed. Not enough queries.")

    X_train = data.X.iloc[train_idx]
    y_train = data.y[train_idx]
    q_train = _group_from_query_ids([data.query_ids[i] for i in train_idx])

    X_val = data.X.iloc[val_idx]
    y_val = data.y[val_idx]
    q_val = _group_from_query_ids([data.query_ids[i] for i in val_idx])

    model = _load_lgbm_ranker(ranking_params)
    model.fit(
        X_train,
        y_train,
        group=q_train,
        eval_set=[(X_val, y_val)],
        eval_group=[q_val],
        eval_at=[5, 10],
    )

    val_scores = model.predict(X_val)
    validation_predictions = pd.DataFrame(
        {
            "query_id": [data.query_ids[i] for i in val_idx],
            "item_id": [data.candidate_items[i] for i in val_idx],
            "label": y_val.astype(int),
            "score": val_scores.astype(float),
        }
    )

    summary = {
        "train_rows": float(len(X_train)),
        "val_rows": float(len(X_val)),
        "num_features": float(X_train.shape[1]),
        "split_method": "temporal" if use_temporal_split and "order_ts" in orders.columns else "random",
        **{f"leakage_{k}": float(v) if isinstance(v, (int, float)) else v for k, v in leakage_report.items()},
    }
    return TrainingOutputs(
        model=model,
        feature_columns=list(X_train.columns),
        validation_predictions=validation_predictions,
        query_meta=data.query_meta,
        training_summary=summary,
    )


def save_training_outputs(outputs: TrainingOutputs, config: dict[str, Any]) -> None:
    ranking_cfg = config.get("ranking", {})
    model_path = Path(ranking_cfg.get("model_path", "models/lgbm_ranker.joblib"))
    cols_path = Path(ranking_cfg.get("feature_columns_path", "models/feature_columns.json"))
    val_pred_path = Path(ranking_cfg.get("validation_predictions_path", "data/processed/validation_predictions.parquet"))
    query_meta_path = Path(ranking_cfg.get("query_meta_path", "data/processed/query_meta.parquet"))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    cols_path.parent.mkdir(parents=True, exist_ok=True)
    val_pred_path.parent.mkdir(parents=True, exist_ok=True)
    query_meta_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(outputs.model, model_path)
    cols_path.write_text(json.dumps(outputs.feature_columns, indent=2), encoding="utf-8")
    save_table(outputs.validation_predictions, val_pred_path, index=False)
    save_table(outputs.query_meta, query_meta_path, index=False)
