from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.pipeline import build_feature_artifacts, save_feature_artifacts
from scripts._utils import load_project_config, load_unified_tables


def main() -> None:
    config = load_project_config()
    processed_dir = config.get("paths", {}).get("processed_dir", "data/processed")
    unified = load_unified_tables(processed_dir)
    artifacts = build_feature_artifacts(unified, config=config)
    save_feature_artifacts(artifacts, processed_dir=processed_dir)
    print("Feature tables saved to:", processed_dir)
    print("- user_features:", len(artifacts.user_features))
    print("- item_features:", len(artifacts.item_features))
    print("- complementarity:", len(artifacts.complementarity))
    print("- category_affinity:", len(artifacts.category_affinity))


if __name__ == "__main__":
    main()
