from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

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


def _train_valid_split(data: TrainingData, validation_fraction: float, random_state: int) -> tuple[np.ndarray, np.ndarray]:
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

    data = build_training_dataset(
        unified=unified,
        user_features=user_features,
        item_features=item_features,
        comp_lookup=comp_lookup,
        config={"ranking": ranking_cfg},
    )

    train_idx, val_idx = _train_valid_split(data, validation_fraction=validation_fraction, random_state=random_state)
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
    outputs.validation_predictions.to_parquet(val_pred_path, index=False)
    outputs.query_meta.to_parquet(query_meta_path, index=False)
