from __future__ import annotations

import time

import numpy as np

from serving.api.schemas import RecommendationRequest
from serving.pipeline.recommendation_service import RecommendationService


def run_simulation(service: RecommendationService, n_warmup: int = 3, n_requests: int = 20) -> dict:
    """Run a realistic serving simulation with warm-up, multiple requests,
    and p50/p95/p99 latency reporting.

    Steps:
    1. Warm-up phase: run a few requests to populate caches.
    2. Benchmark phase: measure latency over n_requests.
    3. Report percentile latencies.
    """

    # Sample requests to simulate realistic traffic
    sample_requests = [
        RecommendationRequest(
            user_id="u_00010",
            session_id="demo_session_01",
            restaurant_id="r_0001",
            cart_item_ids=["i_001", "i_005"],
            top_n=10,
        ),
        RecommendationRequest(
            user_id="u_00042",
            session_id="demo_session_02",
            restaurant_id="r_0010",
            cart_item_ids=["i_010"],
            top_n=10,
        ),
        RecommendationRequest(
            user_id="u_00100",
            session_id="demo_session_03",
            restaurant_id="r_0050",
            cart_item_ids=["i_020", "i_030", "i_040"],
            top_n=10,
        ),
    ]

    # --- Warm-up phase ---
    print(f"[serving] Warm-up: {n_warmup} requests...")
    for i in range(n_warmup):
        req = sample_requests[i % len(sample_requests)]
        _ = service.recommend(req)

    # --- Benchmark phase ---
    print(f"[serving] Benchmarking: {n_requests} requests...")
    latencies = []
    last_response = None
    for i in range(n_requests):
        req = sample_requests[i % len(sample_requests)]
        start = time.perf_counter()
        try:
            response = service.recommend(req)
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies.append(elapsed_ms)
            print(f"  [WARN] Request {i} failed: {e}")
            continue
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies.append(elapsed_ms)
        last_response = response

    arr = np.array(latencies)
    stats = {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(arr.mean()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "n_requests": n_requests,
    }

    print("\n=== Serving Benchmark Results ===")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    if last_response:
        print(f"\nLast response breakdown: {last_response.latency_ms}")
        print(f"Recommendations ({len(last_response.recommendations)}):")
        for rec in last_response.recommendations[:5]:
            print(f"  {rec}")

    # Check SLA compliance
    sla_ms = 300.0
    pct_under_sla = float(np.sum(arr <= sla_ms)) / len(arr) * 100
    print(f"\nSLA compliance ({sla_ms}ms): {pct_under_sla:.1f}% of requests")

    return stats

