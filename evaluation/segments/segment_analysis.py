from __future__ import annotations

import pandas as pd

from evaluation.metrics.ranking_metrics import ndcg_at_k, precision_at_k, recall_at_k


def run_segment_analysis(
    predictions: pd.DataFrame,
    query_meta: pd.DataFrame,
    user_features: pd.DataFrame,
    k: int = 10,
) -> pd.DataFrame:
    joined = predictions.merge(query_meta[["query_id", "user_id"]], on="query_id", how="left")
    joined = joined.merge(user_features[["user_id", "order_frequency"]], on="user_id", how="left")
    joined["user_segment"] = joined["order_frequency"].fillna(0.0).apply(lambda x: "new_or_low_freq" if x < 0.05 else "repeat")

    rows = []
    for segment, seg_df in joined.groupby("user_segment"):
        rows.append(
            {
                "segment": segment,
                "precision_at_k": precision_at_k(seg_df, k=k),
                "recall_at_k": recall_at_k(seg_df, k=k),
                "ndcg_at_k": ndcg_at_k(seg_df, k=k),
            }
        )
    return pd.DataFrame(rows)

