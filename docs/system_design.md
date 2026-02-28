# CSAO System Design

## Online Flow

1. Client sends current cart + user + restaurant context.
2. Candidate generation retrieves ~200 add-on candidates:
   - item co-occurrence
   - restaurant popularity
   - category complement retrieval
3. Ranker scores candidates with cart/user/item/complementarity features.
4. Top-N recommendations are returned with latency tracking.

## Offline Flow

1. Ingest raw orders from backbone and Mendeley datasets.
2. Normalize and unify into users/orders/order_items/items/restaurants tables.
3. Build feature tables and complementarity artifacts.
4. Train LightGBM ranker with positive/negative query samples.
5. Run offline evaluation and segment analysis.

## Production TODOs

- Add feature store (offline + online parity)
- Add streaming updates for near-real-time co-occurrence refresh
- Add request tracing, structured logs, and monitoring dashboards
- Add online experimentation framework
