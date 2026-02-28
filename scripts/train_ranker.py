from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.complementarity import build_complementarity_lookup
from ranking.training.train import save_training_outputs, train_lgbm_ranker
from scripts._utils import load_feature_tables, load_project_config, load_unified_tables


def main() -> None:
    config = load_project_config()
    processed_dir = config.get("paths", {}).get("processed_dir", "data/processed")

    try:
        unified = load_unified_tables(processed_dir)
        features = load_feature_tables(processed_dir)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Training inputs missing. Run in order:\n"
            "1) python scripts/build_unified_data.py\n"
            "2) python scripts/build_features.py\n"
            f"Details: {exc}"
        ) from exc

    if features["user_features"].empty or features["item_features"].empty:
        raise ValueError(
            "Feature tables are empty. Run `python scripts/build_features.py` and ensure source tables are generated."
        )
    comp_lookup = build_complementarity_lookup(features["complementarity"])

    outputs = train_lgbm_ranker(
        unified=unified,
        user_features=features["user_features"],
        item_features=features["item_features"],
        comp_lookup=comp_lookup,
        config=config,
    )
    save_training_outputs(outputs, config=config)
    print("Model trained and saved.")
    print("Training summary:", outputs.training_summary)


if __name__ == "__main__":
    main()
