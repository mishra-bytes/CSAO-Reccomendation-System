from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.run_eval import evaluate_offline
from scripts._utils import load_feature_tables, load_project_config, load_unified_tables


def main() -> None:
    config = load_project_config()
    processed_dir = Path(config.get("paths", {}).get("processed_dir", "data/processed"))
    ranking_cfg = config.get("ranking", {})

    pred_path = Path(ranking_cfg.get("validation_predictions_path", processed_dir / "validation_predictions.parquet"))
    meta_path = Path(ranking_cfg.get("query_meta_path", processed_dir / "query_meta.parquet"))
    if not pred_path.exists():
        raise FileNotFoundError(f"Validation predictions not found: {pred_path}. Run train_ranker first.")
    if not meta_path.exists():
        raise FileNotFoundError(f"Query meta not found: {meta_path}. Run train_ranker first.")

    predictions = pd.read_parquet(pred_path)
    query_meta = pd.read_parquet(meta_path)
    unified = load_unified_tables(str(processed_dir))
    features = load_feature_tables(str(processed_dir))

    k = int(config.get("ranking", {}).get("eval_k", 10))
    result = evaluate_offline(
        predictions=predictions,
        item_catalog=unified["items"],
        query_meta=query_meta,
        user_features=features["user_features"],
        k=k,
    )
    print("Overall Metrics:")
    for key, value in result["overall"].items():
        print(f"- {key}: {value:.6f}")
    print("\nSegment Metrics:")
    print(result["segments"].to_string(index=False))


if __name__ == "__main__":
    main()
