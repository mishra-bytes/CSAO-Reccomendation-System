from __future__ import annotations

from serving.api.schemas import RecommendationRequest
from serving.pipeline.recommendation_service import RecommendationService


def run_simulation(service: RecommendationService) -> None:
    request = RecommendationRequest(
        user_id="u_00010",
        session_id="demo_session_01",
        restaurant_id="r_0001",
        cart_item_ids=["i_001", "i_005"],
        top_n=10,
    )
    response = service.recommend(request)
    print("Latency(ms):", response.latency_ms)
    print("Recommendations:")
    for rec in response.recommendations:
        print(rec)

