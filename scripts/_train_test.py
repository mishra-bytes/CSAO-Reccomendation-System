"""Quick training test script."""
import sys
sys.path.insert(0, r"D:\Work\Hackathons\Zomato")

print("START", flush=True)
from scripts._utils import load_project_config, load_unified_tables, load_feature_tables
from features.complementarity import build_complementarity_lookup

config = load_project_config()
processed_dir = config.get("paths", {}).get("processed_dir", "data/processed")

print("Loading tables...", flush=True)
unified = load_unified_tables(processed_dir)
features = load_feature_tables(processed_dir)
comp_lookup = build_complementarity_lookup(features["complementarity"])

n_orders = len(unified["orders"])
n_items = len(unified["items"])
print(f"Orders: {n_orders}, Items: {n_items}", flush=True)
print(f"Comp lookup size: {len(comp_lookup)}", flush=True)

print("Building training dataset...", flush=True)
from ranking.training.dataset import build_training_dataset
data = build_training_dataset(
    unified=unified,
    user_features=features["user_features"],
    item_features=features["item_features"],
    comp_lookup=comp_lookup,
    config=config,
)
print(f"Dataset: X={data.X.shape}, y={len(data.y)}", flush=True)

print("Training model...", flush=True)
from ranking.training.train import train_lgbm_ranker, save_training_outputs
outputs = train_lgbm_ranker(
    unified=unified,
    user_features=features["user_features"],
    item_features=features["item_features"],
    comp_lookup=comp_lookup,
    config=config,
)
save_training_outputs(outputs, config=config)
print("DONE! Training summary:", outputs.training_summary, flush=True)
