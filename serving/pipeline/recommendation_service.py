from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from candidate_generation.candidate_generator import CandidateGenerator
from ranking.inference.ranker import CSAORanker
from serving.api.schemas import RecommendationRequest, RecommendationResponse
from serving.utils.latency import LatencyTracker


@dataclass
class ServingArtifacts:
    candidate_generator: CandidateGenerator
    ranker: CSAORanker
    user_features: pd.DataFrame
    item_features: pd.DataFrame


class RecommendationService:
    def __init__(self, artifacts: ServingArtifacts, config: dict[str, Any]) -> None:
        self.artifacts = artifacts
        serving_cfg = config.get("serving", {})
        self.default_top_n = int(serving_cfg.get("default_top_n", 10))
        self.latency_budget_ms = int(serving_cfg.get("latency_budget_ms", 300))

        # Pre-index user features by user_id for O(1) lookup instead of O(n) scan
        self._user_index = (
            artifacts.user_features.drop_duplicates("user_id")
            .set_index("user_id", drop=False)
        )

        # Latency history for p95/p99 tracking
        self._latency_history: list[float] = []

    def recommend(self, req: RecommendationRequest) -> RecommendationResponse:
        timer = LatencyTracker()

        with timer.track("feature_fetch"):
            # O(1) index lookup instead of O(n) boolean mask
            _ = self._user_index.loc[req.user_id] if req.user_id in self._user_index.index else None

        with timer.track("candidate_generation"):
            candidates = self.artifacts.candidate_generator.generate(
                cart_items=req.cart_item_ids,
                restaurant_id=req.restaurant_id,
            )

        with timer.track("ranking"):
            ranked = self.artifacts.ranker.rank(
                user_id=req.user_id,
                restaurant_id=req.restaurant_id,
                cart_items=req.cart_item_ids,
                candidates=candidates,
                top_n=req.top_n or self.default_top_n,
            )

        lat = timer.finalize()
        total_ms = lat.get("total", 0.0)
        self._latency_history.append(total_ms)

        if total_ms > self.latency_budget_ms:
            ranked = ranked[: max(3, min(len(ranked), req.top_n))]

        return RecommendationResponse(
            user_id=req.user_id,
            session_id=req.session_id,
            recommendations=ranked,
            latency_ms=lat,
        )

    def get_latency_stats(self) -> dict[str, float]:
        """Return p50/p95/p99 latency from request history."""
        import numpy as np
        if not self._latency_history:
            return {}
        arr = np.array(self._latency_history)
        return {
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "mean_ms": float(arr.mean()),
            "n_requests": len(self._latency_history),
        }
