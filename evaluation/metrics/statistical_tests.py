"""
Bootstrap Confidence Intervals & Statistical Significance Tests
================================================================
Provides:
  1. bootstrap_ci() — non-parametric bootstrap CI for any metric
  2. paired_bootstrap_test() — paired bootstrap p-value (model A vs B)
  3. wilcoxon_signed_rank_test() — Wilcoxon signed-rank per-query comparison

These allow us to report not just point estimates but *confidence intervals*
and *p-values*, which is critical for a hackathon to demonstrate statistical
rigour beyond "my NDCG is 0.01 higher".
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd


# ── 1. Bootstrap Confidence Interval ─────────────────────────────────────────

def bootstrap_ci(
    predictions: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame, int], float],
    k: int = 10,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """Compute bootstrap confidence interval for a ranking metric.

    Uses per-query metric values for efficient resampling (avoids
    DataFrame concatenation issues with mixed dtypes).

    Args:
        predictions: DataFrame with query_id, item_id, label, score.
        metric_fn: function(predictions, k) -> float (e.g., ndcg_at_k).
        k: cutoff for the metric.
        n_bootstrap: number of bootstrap resamples.
        confidence: confidence level (e.g., 0.95 for 95% CI).
        seed: random seed.

    Returns:
        dict with point_estimate, ci_lower, ci_upper, std, n_bootstrap.
    """
    rng = np.random.default_rng(seed)

    # Point estimate
    point_est = metric_fn(predictions, k)

    # Compute per-query metric values once
    per_query_values = []
    for qid, group in predictions.groupby("query_id"):
        top = group.sort_values("score", ascending=False).head(k).reset_index(drop=True)
        dcg = 0.0
        for idx, label in enumerate(top["label"].astype(float).tolist(), start=1):
            dcg += (2.0 ** label - 1.0) / math.log2(idx + 1.0)
        ideal = sorted(group["label"].astype(float).tolist(), reverse=True)[:k]
        idcg = 0.0
        for idx, label in enumerate(ideal, start=1):
            idcg += (2.0 ** label - 1.0) / math.log2(idx + 1.0)
        per_query_values.append(dcg / max(idcg, 1e-9))

    per_query_values = np.array(per_query_values)
    n_queries = len(per_query_values)

    # Bootstrap: resample per-query values (much faster than DF concat)
    bootstrap_values = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(per_query_values, size=n_queries, replace=True)
        bootstrap_values.append(float(sampled.mean()))

    bootstrap_values = np.array(bootstrap_values)
    alpha = 1 - confidence
    ci_lower = float(np.percentile(bootstrap_values, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_values, 100 * (1 - alpha / 2)))

    return {
        "point_estimate": round(point_est, 6),
        "ci_lower": round(ci_lower, 6),
        "ci_upper": round(ci_upper, 6),
        "ci_width": round(ci_upper - ci_lower, 6),
        "std": round(float(bootstrap_values.std()), 6),
        "n_bootstrap": n_bootstrap,
        "confidence": confidence,
        "n_queries": n_queries,
    }


# ── 2. Paired Bootstrap Significance Test ────────────────────────────────────

def paired_bootstrap_test(
    predictions_a: pd.DataFrame,
    predictions_b: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame, int], float],
    k: int = 10,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict:
    """Paired bootstrap test: is model A significantly better than model B?

    Uses per-query NDCG differences for efficient resampling.
    p-value = fraction of resamples where mean(delta) <= 0.

    Args:
        predictions_a: predictions from model A (better model).
        predictions_b: predictions from model B (baseline).
        metric_fn: metric function(predictions, k) -> float (unused, kept for API compat).
        k: cutoff.
        n_bootstrap: resamples.
        seed: random seed.

    Returns:
        dict with observed_delta, p_value, significant_at_005, significant_at_001.
    """
    rng = np.random.default_rng(seed)

    # Compute per-query NDCG for both models
    ndcg_a = _per_query_ndcg(predictions_a, k)
    ndcg_b = _per_query_ndcg(predictions_b, k)

    shared_qids = sorted(set(ndcg_a.keys()) & set(ndcg_b.keys()))
    if len(shared_qids) == 0:
        return {"error": "No shared queries between models A and B"}

    n_queries = len(shared_qids)
    diffs = np.array([ndcg_a[q] - ndcg_b[q] for q in shared_qids])
    observed_delta = float(diffs.mean())

    # Bootstrap: resample per-query deltas
    delta_count_le_zero = 0
    for _ in range(n_bootstrap):
        sampled = rng.choice(diffs, size=n_queries, replace=True)
        if sampled.mean() <= 0:
            delta_count_le_zero += 1

    p_value = delta_count_le_zero / n_bootstrap

    obs_a = float(np.mean([ndcg_a[q] for q in shared_qids]))
    obs_b = float(np.mean([ndcg_b[q] for q in shared_qids]))

    return {
        "metric_a": round(obs_a, 6),
        "metric_b": round(obs_b, 6),
        "observed_delta": round(observed_delta, 6),
        "p_value": round(p_value, 4),
        "significant_at_005": p_value < 0.05,
        "significant_at_001": p_value < 0.01,
        "n_bootstrap": n_bootstrap,
        "n_shared_queries": n_queries,
    }


# ── 3. Wilcoxon Signed-Rank Test ─────────────────────────────────────────────

def _per_query_ndcg(predictions: pd.DataFrame, k: int = 10) -> dict[str, float]:
    """Compute NDCG@k for each query individually."""
    result = {}
    for qid, group in predictions.groupby("query_id"):
        top = group.sort_values("score", ascending=False).head(k).reset_index(drop=True)
        dcg = 0.0
        for idx, label in enumerate(top["label"].astype(float).tolist(), start=1):
            dcg += (2.0 ** label - 1.0) / math.log2(idx + 1.0)
        ideal = sorted(group["label"].astype(float).tolist(), reverse=True)[:k]
        idcg = 0.0
        for idx, label in enumerate(ideal, start=1):
            idcg += (2.0 ** label - 1.0) / math.log2(idx + 1.0)
        result[qid] = dcg / max(idcg, 1e-9)
    return result


def wilcoxon_signed_rank_test(
    predictions_a: pd.DataFrame,
    predictions_b: pd.DataFrame,
    k: int = 10,
) -> dict:
    """Wilcoxon signed-rank test on per-query NDCG differences.

    Non-parametric test that doesn't assume normal distribution of deltas.
    """
    from scipy import stats

    ndcg_a = _per_query_ndcg(predictions_a, k)
    ndcg_b = _per_query_ndcg(predictions_b, k)

    shared = sorted(set(ndcg_a.keys()) & set(ndcg_b.keys()))
    if len(shared) < 10:
        return {"error": f"Only {len(shared)} shared queries — need at least 10"}

    diffs = np.array([ndcg_a[q] - ndcg_b[q] for q in shared])

    # Remove zero differences (Wilcoxon requirement)
    nonzero_mask = diffs != 0.0
    diffs_nonzero = diffs[nonzero_mask]

    if len(diffs_nonzero) < 5:
        return {
            "warning": "Too few non-zero differences for Wilcoxon test",
            "n_queries": len(shared),
            "n_nonzero": len(diffs_nonzero),
            "mean_delta": round(float(diffs.mean()), 6),
        }

    stat, p_value = stats.wilcoxon(diffs_nonzero, alternative="greater")

    return {
        "test": "Wilcoxon signed-rank (one-sided: A > B)",
        "statistic": round(float(stat), 4),
        "p_value": round(float(p_value), 4),
        "significant_at_005": p_value < 0.05,
        "significant_at_001": p_value < 0.01,
        "n_queries": len(shared),
        "n_nonzero_diffs": len(diffs_nonzero),
        "mean_delta": round(float(diffs.mean()), 6),
        "median_delta": round(float(np.median(diffs)), 6),
        "pct_a_wins": round(100 * (diffs > 0).mean(), 1),
        "pct_b_wins": round(100 * (diffs < 0).mean(), 1),
        "pct_ties": round(100 * (diffs == 0).mean(), 1),
    }
