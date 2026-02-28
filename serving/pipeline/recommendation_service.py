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

    def recommend(self, req: RecommendationRequest) -> RecommendationResponse:
        timer = LatencyTracker()

        with timer.track("feature_fetch"):
            # TODO(prod): fetch online features from Redis/feature store.
            _ = self.artifacts.user_features[self.artifacts.user_features["user_id"] == req.user_id]

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
        # TODO(prod): wire structured logs and tracing IDs.
        if lat.get("total", 0.0) > self.latency_budget_ms:
            ranked = ranked[: max(3, min(len(ranked), req.top_n))]

        return RecommendationResponse(
            user_id=req.user_id,
            session_id=req.session_id,
            recommendations=ranked,
            latency_ms=lat,
        )
