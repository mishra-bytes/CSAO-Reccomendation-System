from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RecommendationRequest:
    user_id: str
    session_id: str
    restaurant_id: str
    cart_item_ids: list[str]
    top_n: int = 10
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class RecommendationResponse:
    user_id: str
    session_id: str
    recommendations: list[dict[str, Any]]
    latency_ms: dict[str, float]

