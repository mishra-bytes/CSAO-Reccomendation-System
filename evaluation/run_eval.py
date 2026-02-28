from __future__ import annotations

from typing import Any

import pandas as pd

from evaluation.metrics.diversity import intra_list_diversity_at_k
from evaluation.metrics.ranking_metrics import coverage_at_k, ndcg_at_k, precision_at_k, recall_at_k
from evaluation.segments.segment_analysis import run_segment_analysis


def evaluate_offline(
    predictions: pd.DataFrame,
    item_catalog: pd.DataFrame,
    query_meta: pd.DataFrame,
    user_features: pd.DataFrame,
    k: int = 10,
) -> dict[str, Any]:
    base = {
        "precision_at_k": precision_at_k(predictions, k=k),
        "recall_at_k": recall_at_k(predictions, k=k),
        "ndcg_at_k": ndcg_at_k(predictions, k=k),
        "coverage_at_k": coverage_at_k(predictions, item_catalog, k=k),
        "diversity_at_k": intra_list_diversity_at_k(predictions, item_catalog, k=k),
    }
    segment = run_segment_analysis(predictions, query_meta=query_meta, user_features=user_features, k=k)
    return {"overall": base, "segments": segment}

