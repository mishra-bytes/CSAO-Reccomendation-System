"""FastAPI serving stub for the CSAO recommendation system.

Lightweight production-credible API that wraps RecommendationService.
Supports single and batch recommendation requests, health checks,
and latency stats endpoint.

Usage:
    uvicorn serving.api.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import time
from typing import Any, Optional

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

if HAS_FASTAPI:

    # ---- Pydantic request/response models ----

    class RecoRequest(BaseModel):
        user_id: str
        session_id: str
        restaurant_id: str
        cart_item_ids: list[str]
        top_n: int = Field(default=10, ge=1, le=50)

    class RecoItem(BaseModel):
        item_id: str
        rank_score: float
        candidate_score: float

    class RecoResponse(BaseModel):
        user_id: str
        session_id: str
        recommendations: list[dict[str, Any]]
        latency_ms: dict[str, float]

    class BatchRecoRequest(BaseModel):
        requests: list[RecoRequest]

    class BatchRecoResponse(BaseModel):
        responses: list[RecoResponse]
        total_latency_ms: float

    class HealthResponse(BaseModel):
        status: str
        model_loaded: bool

    class LatencyStatsResponse(BaseModel):
        stats: dict[str, float]

    # ---- App factory ----

    app = FastAPI(
        title="CSAO Recommendation API",
        description="Cart Super Add-On recommendation system",
        version="1.0.0",
    )

    # Global service reference — set by the startup script
    _service = None

    def set_service(service: Any) -> None:
        global _service
        _service = service

    def _get_service():
        if _service is None:
            raise HTTPException(status_code=503, detail="Service not initialised")
        return _service

    # ---- Routes ----

    @app.get("/health", response_model=HealthResponse)
    def health():
        svc = _service
        return HealthResponse(
            status="ok" if svc is not None else "not_ready",
            model_loaded=svc is not None,
        )

    @app.post("/recommend", response_model=RecoResponse)
    def recommend(req: RecoRequest):
        from serving.api.schemas import RecommendationRequest
        svc = _get_service()
        internal_req = RecommendationRequest(
            user_id=req.user_id,
            session_id=req.session_id,
            restaurant_id=req.restaurant_id,
            cart_item_ids=req.cart_item_ids,
            top_n=req.top_n,
        )
        resp = svc.recommend(internal_req)
        return RecoResponse(
            user_id=resp.user_id,
            session_id=resp.session_id,
            recommendations=resp.recommendations,
            latency_ms=resp.latency_ms,
        )

    @app.post("/recommend/batch", response_model=BatchRecoResponse)
    def recommend_batch(batch: BatchRecoRequest):
        """Batch scoring endpoint — processes multiple requests sequentially
        but within a single HTTP round-trip, reducing network overhead."""
        from serving.api.schemas import RecommendationRequest
        svc = _get_service()
        start = time.perf_counter()
        responses = []
        for req in batch.requests:
            internal_req = RecommendationRequest(
                user_id=req.user_id,
                session_id=req.session_id,
                restaurant_id=req.restaurant_id,
                cart_item_ids=req.cart_item_ids,
                top_n=req.top_n,
            )
            resp = svc.recommend(internal_req)
            responses.append(RecoResponse(
                user_id=resp.user_id,
                session_id=resp.session_id,
                recommendations=resp.recommendations,
                latency_ms=resp.latency_ms,
            ))
        total = (time.perf_counter() - start) * 1000
        return BatchRecoResponse(responses=responses, total_latency_ms=total)

    @app.get("/latency-stats", response_model=LatencyStatsResponse)
    def latency_stats():
        svc = _get_service()
        stats = svc.get_latency_stats() if hasattr(svc, "get_latency_stats") else {}
        return LatencyStatsResponse(stats=stats)

    @app.get("/metrics")
    def metrics():
        """Prometheus-style metrics endpoint with real latency stats."""
        svc = _service
        if svc is None:
            return {"status": "not_ready"}
        stats = svc.get_latency_stats() if hasattr(svc, "get_latency_stats") else {}
        return {
            "status": "ok",
            "model_loaded": True,
            "latency": stats,
        }

else:
    # Fallback when FastAPI isn't installed
    app = None

    def set_service(service: Any) -> None:
        pass
