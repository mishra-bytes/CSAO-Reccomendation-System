from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from candidate_generation.candidate_generator import CandidateGenerator
from features.complementarity import build_complementarity_lookup
from ranking.inference.ranker import CSAORanker
from scripts._utils import load_feature_tables, load_project_config, load_unified_tables
from serving.pipeline.recommendation_service import RecommendationService, ServingArtifacts
from serving.simulate import run_simulation


def main() -> None:
    config = load_project_config()
    processed_dir = config.get("paths", {}).get("processed_dir", "data/processed")
    ranking_cfg = config.get("ranking", {})

    try:
        unified = load_unified_tables(processed_dir)
        feats = load_feature_tables(processed_dir)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Serving inputs missing. Run in order:\n"
            "1) python scripts/build_unified_data.py\n"
            "2) python scripts/build_features.py\n"
            "3) python scripts/train_ranker.py\n"
            f"Details: {exc}"
        ) from exc

    comp_lookup = build_complementarity_lookup(feats["complementarity"])
    model_path = Path(ranking_cfg.get("model_path", "models/lgbm_ranker.joblib"))
    feature_cols_path = Path(ranking_cfg.get("feature_columns_path", "models/feature_columns.json"))
    if not model_path.exists() or not feature_cols_path.exists():
        raise FileNotFoundError(
            "Model artifacts missing. Run `python scripts/train_ranker.py` first.\n"
            f"Checked: {model_path}, {feature_cols_path}"
        )

    candidate_generator = CandidateGenerator(
        complementarity=feats["complementarity"],
        category_affinity=feats["category_affinity"],
        items=unified["items"],
        orders=unified["orders"],
        order_items=unified["order_items"],
        config=config,
    )
    ranker = CSAORanker(
        model_path=str(model_path),
        feature_columns_path=str(feature_cols_path),
        user_features=feats["user_features"],
        item_features=feats["item_features"],
        items=unified["items"],
        complementarity_lookup=comp_lookup,
    )
    service = RecommendationService(
        artifacts=ServingArtifacts(
            candidate_generator=candidate_generator,
            ranker=ranker,
            user_features=feats["user_features"],
            item_features=feats["item_features"],
        ),
        config=config,
    )
    run_simulation(service)


if __name__ == "__main__":
    main()
