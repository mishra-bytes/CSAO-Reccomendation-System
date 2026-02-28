from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.loaders import load_raw_datasets
from data.unify import build_unified_tables, save_unified_tables
from scripts._utils import load_project_config


def main() -> None:
    config = load_project_config()
    processed_dir = config.get("paths", {}).get("processed_dir", "data/processed")
    Path(processed_dir).mkdir(parents=True, exist_ok=True)

    raw = load_raw_datasets(config)
    unified = build_unified_tables(raw)
    save_unified_tables(unified, processed_dir=processed_dir)
    print("Unified tables saved to:", processed_dir)
    for name in ["users", "orders", "order_items", "items", "restaurants"]:
        print(f"- {name}: {len(unified[name])} rows")


if __name__ == "__main__":
    main()
