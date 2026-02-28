from __future__ import annotations

import math

import pandas as pd


def _top_k(group: pd.DataFrame, k: int) -> pd.DataFrame:
    return group.sort_values("score", ascending=False).head(k)


def precision_at_k(predictions: pd.DataFrame, k: int = 10) -> float:
    vals = []
    for _, group in predictions.groupby("query_id"):
        top = _top_k(group, k)
        vals.append(float(top["label"].sum()) / k)
    return float(sum(vals) / max(len(vals), 1))


def recall_at_k(predictions: pd.DataFrame, k: int = 10) -> float:
    vals = []
    for _, group in predictions.groupby("query_id"):
        positives = max(int(group["label"].sum()), 1)
        top = _top_k(group, k)
        vals.append(float(top["label"].sum()) / positives)
    return float(sum(vals) / max(len(vals), 1))


def ndcg_at_k(predictions: pd.DataFrame, k: int = 10) -> float:
    vals = []
    for _, group in predictions.groupby("query_id"):
        top = _top_k(group, k).reset_index(drop=True)
        dcg = 0.0
        for idx, label in enumerate(top["label"].astype(float).tolist(), start=1):
            dcg += (2.0**label - 1.0) / math.log2(idx + 1.0)
        ideal = sorted(group["label"].astype(float).tolist(), reverse=True)[:k]
        idcg = 0.0
        for idx, label in enumerate(ideal, start=1):
            idcg += (2.0**label - 1.0) / math.log2(idx + 1.0)
        vals.append(dcg / max(idcg, 1e-9))
    return float(sum(vals) / max(len(vals), 1))


def coverage_at_k(
    ranked_items: pd.DataFrame,
    catalog_items: pd.DataFrame,
    k: int = 10,
) -> float:
    top_items = ranked_items.sort_values(["query_id", "score"], ascending=[True, False]).groupby("query_id").head(k)
    unique_top = set(top_items["item_id"].astype(str).tolist())
    catalog = set(catalog_items["item_id"].astype(str).tolist())
    return float(len(unique_top) / max(len(catalog), 1))

