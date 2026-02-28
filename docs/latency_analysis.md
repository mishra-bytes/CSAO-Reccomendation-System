# Latency Analysis (Target: 200-300 ms)

## Budget Split (initial)

- Feature fetch: 20-40 ms
- Candidate generation: 40-80 ms
- Rank scoring (200 candidates): 40-100 ms
- Post-processing + response: 10-30 ms
- Network overhead: 30-60 ms

## Instrumentation

- `serving/utils/latency.py` tracks stage-level timing
- `RecommendationService` enforces a budget cap fallback

## Production TODOs

- Add p95/p99 stage metrics
- Add Redis + feature cache hit-rate tracking
- Add circuit-breakers for slow dependency calls
