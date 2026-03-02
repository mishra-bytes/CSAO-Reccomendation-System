"""Run Optuna HP tuning."""
import sys; sys.path.insert(0, '.')
import json
from pathlib import Path
from common.io import load_table
from scripts._utils import load_feature_tables, load_project_config, load_unified_tables
from features.complementarity import build_complementarity_lookup

config = load_project_config()
processed_dir = Path(config['paths']['processed_dir'])
unified = load_unified_tables(str(processed_dir))
features = load_feature_tables(str(processed_dir))
comp_lookup = build_complementarity_lookup(features['complementarity'])

print('=== Optuna HP Tuning (15 trials) ===')
from experiments.ablations.tuning_and_ablation import run_optuna_tuning
hp = run_optuna_tuning(
    unified=unified,
    user_features=features['user_features'],
    item_features=features['item_features'],
    comp_lookup=comp_lookup,
    config=config,
    n_trials=15,
    timeout_seconds=300,
)
print("Best NDCG@10:", round(hp["best_ndcg"], 4))
print("Trials completed:", hp["n_trials_completed"])
print("Best params:", json.dumps(hp["best_params"], indent=2))
print()
print("Top 5 trials:")
for t in hp.get("trials_summary", [])[:5]:
    print(f"  Trial {t['number']}: NDCG={t['value']:.4f}")
