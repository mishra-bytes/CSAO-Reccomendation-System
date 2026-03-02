"""
CSAO Recommendation API — Startup Script
==========================================
Loads all artefacts and starts the FastAPI server with uvicorn.

Usage:
    python -m serving.api.main [--host 0.0.0.0] [--port 8000]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def create_serving_artifacts():
    """Load all artefacts needed for serving."""
    import pandas as pd
    from scripts._utils import load_project_config, load_unified_tables, load_feature_tables
    from features.complementarity import build_complementarity_lookup
    from candidate_generation.candidate_generator import CandidateGenerator
    from ranking.inference.ranker import CSAORanker
    from serving.pipeline.recommendation_service import ServingArtifacts

    print("[api] Loading config …")
    config = load_project_config()
    ranking_cfg = config.get("ranking", {})
    processed_dir = config.get("paths", {}).get("processed_dir", "data/processed")

    print("[api] Loading tables …")
    unified = load_unified_tables(processed_dir)
    features = load_feature_tables(processed_dir)
    comp_lookup = build_complementarity_lookup(features["complementarity"])

    print("[api] Building candidate generator …")
    cand_gen = CandidateGenerator(
        complementarity=features["complementarity"],
        category_affinity=features["category_affinity"],
        items=unified["items"],
        orders=unified["orders"],
        order_items=unified["order_items"],
        config=config,
    )

    print("[api] Loading ranker …")
    model_path = ranking_cfg.get("model_path", "models/lgbm_ranker.joblib")
    cols_path = ranking_cfg.get("feature_columns_path", "models/feature_columns.json")
    ranker = CSAORanker(
        model_path=model_path,
        feature_columns_path=cols_path,
        user_features=features["user_features"],
        item_features=features["item_features"],
        items=unified["items"],
        complementarity_lookup=comp_lookup,
        restaurants=unified.get("restaurants"),
    )

    # Build item catalog dict
    items_df = unified["items"].drop_duplicates("item_id")
    item_catalog = {}
    for _, row in items_df.iterrows():
        iid = str(row["item_id"])
        item_catalog[iid] = {
            "item_name": str(row.get("item_name", iid)),
            "item_category": str(row.get("item_category", "unknown")),
            "item_price": float(row.get("item_price", 0)),
        }

    return ServingArtifacts(
        candidate_generator=cand_gen,
        ranker=ranker,
        user_features=features["user_features"],
        item_features=features["item_features"],
        item_catalog=item_catalog,
    ), config


def main():
    parser = argparse.ArgumentParser(description="Start CSAO API server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    t0 = time.perf_counter()
    artifacts, config = create_serving_artifacts()
    load_time = time.perf_counter() - t0
    print(f"[api] Artefacts loaded in {load_time:.1f}s")

    from serving.pipeline.recommendation_service import RecommendationService
    from serving.api.app import app, set_service

    service = RecommendationService(artifacts, config)
    set_service(service)

    print(f"[api] Starting server on {args.host}:{args.port}")
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
