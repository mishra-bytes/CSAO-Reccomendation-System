from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.io import load_table
from evaluation.run_eval import evaluate_offline
from scripts._utils import load_feature_tables, load_project_config, load_unified_tables


def main() -> None:
    config = load_project_config()
    processed_dir = Path(config.get("paths", {}).get("processed_dir", "data/processed"))
    ranking_cfg = config.get("ranking", {})

    pred_path = Path(ranking_cfg.get("validation_predictions_path", processed_dir / "validation_predictions.parquet"))
    meta_path = Path(ranking_cfg.get("query_meta_path", processed_dir / "query_meta.parquet"))
    try:
        predictions = load_table(pred_path, required=True)
        query_meta = load_table(meta_path, required=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Offline eval inputs missing. Run in order:\n"
            "1) python scripts/build_unified_data.py\n"
            "2) python scripts/build_features.py\n"
            "3) python scripts/train_ranker.py\n"
            f"Details: {exc}"
        ) from exc
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

    if "business_impact" in result:
        print("\nBusiness Impact Proxy Metrics:")
        for key, value in result["business_impact"].items():
            if isinstance(value, float):
                print(f"- {key}: {value:.4f}")
            else:
                print(f"- {key}: {value}")

    if "llm_judge" in result and result["llm_judge"]:
        print("\nLLM-as-Judge Evaluation:")
        for key, value in result["llm_judge"].items():
            if isinstance(value, float):
                print(f"- {key}: {value:.4f}")
            else:
                print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
