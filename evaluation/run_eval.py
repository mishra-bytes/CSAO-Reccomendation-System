from __future__ import annotations

from typing import Any

import pandas as pd

from evaluation.metrics.business_impact import evaluate_business_impact
from evaluation.metrics.diversity import intra_list_diversity_at_k
from evaluation.metrics.llm_judge import run_llm_judge
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

    # Business impact proxy metrics
    biz = evaluate_business_impact(predictions, item_catalog, k=k)
    business = {
        "attach_rate": biz.attach_rate,
        "incremental_aov": biz.incremental_aov,
        "revenue_uplift_per_1k_orders": biz.revenue_uplift_per_1k_orders,
        "avg_reco_price": biz.avg_reco_price,
        "fatigue_score": biz.fatigue_score,
        **{f"biz_{k2}": v for k2, v in biz.detail.items()},
    }

    # LLM-as-judge evaluation (semantic coherence)
    try:
        llm_result = run_llm_judge(predictions, item_catalog, k=k, n_samples=50)
        llm_metrics = {
            "llm_semantic_coherence": llm_result.semantic_coherence,
            "llm_embedding_diversity": llm_result.embedding_diversity,
            "llm_category_coverage": llm_result.category_coverage_score,
            "llm_overall_quality": llm_result.overall_quality_score,
        }
        if llm_result.llm_scores:
            llm_metrics.update(llm_result.llm_scores)
    except Exception:
        llm_metrics = {}

    return {"overall": base, "segments": segment, "business_impact": business, "llm_judge": llm_metrics}

