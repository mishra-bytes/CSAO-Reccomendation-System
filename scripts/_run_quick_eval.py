"""Quick evaluation runner - runs core metrics without heavy LLM loading."""
import sys, os
sys.path.insert(0, '.')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

from pathlib import Path
from common.io import load_table
from scripts._utils import load_feature_tables, load_project_config, load_unified_tables
from features.complementarity import build_complementarity_lookup

config = load_project_config()
processed_dir = Path(config['paths']['processed_dir'])
unified = load_unified_tables(str(processed_dir))
features = load_feature_tables(str(processed_dir))
comp_lookup = build_complementarity_lookup(features['complementarity'])
ranking_cfg = config.get('ranking', {})
predictions = load_table(Path(ranking_cfg['validation_predictions_path']), required=True)
query_meta = load_table(Path(ranking_cfg['query_meta_path']), required=True)
k = 10

print('=== 1. Standard Eval ===')
from evaluation.metrics.ranking_metrics import ndcg_at_k, precision_at_k, recall_at_k, coverage_at_k
from evaluation.metrics.business_impact import compute_attach_rate
items = unified["items"]
print(f'NDCG@10: {ndcg_at_k(predictions, k=k):.4f}')
print(f'Precision@10: {precision_at_k(predictions, k=k):.4f}')
print(f'Recall@10: {recall_at_k(predictions, k=k):.4f}')
print(f'Coverage@10: {coverage_at_k(predictions, items, k=k):.4f}')
print(f'Attach Rate: {compute_attach_rate(predictions, k=k):.4f}')

print()
print('=== 2. Baselines ===')
from experiments.baselines import run_baseline_comparison
bl = run_baseline_comparison(
    validation_predictions=predictions,
    query_meta=query_meta,
    orders=unified['orders'],
    order_items=unified['order_items'],
    item_catalog=items,
    user_features=features['user_features'],
    k=k,
)
print(bl.to_string())

print()
print('=== 3. Cold Start Segments ===')
from serving.pipeline.cold_start import evaluate_cold_start_segments
cs = evaluate_cold_start_segments(predictions, query_meta, features['user_features'], k=k)
print(cs.to_string(index=False))

print()
print('=== 4. Business Impact ===')
from evaluation.business_impact_model import compute_business_impact, format_executive_summary
biz = compute_business_impact(predictions, items, features['user_features'], k=k)
print(format_executive_summary(biz))
print('A/B Test:', biz.ab_test_plan['recommendation'])

print()
print('=== 5. Scalability ===')
from serving.scalability import simulate_cache_performance, compute_capacity_plan
item_lists = []
for _, grp in predictions.groupby("query_id"):
    top_items = grp.sort_values("score", ascending=False).head(k)["item_id"].astype(str).tolist()
    item_lists.append(top_items)
cache = simulate_cache_performance(item_lists[:500])
print(f'Cache hit rate: {cache.hit_rate:.1%}')
print(f'Latency saving per lookup: {cache.estimated_latency_saving_ms:.1f}ms')
cap = compute_capacity_plan(peak_qps=500)
print(f'Pods needed: {cap["pods_needed_min"]} (min) / {cap["pods_recommended"]} (recommended)')
print(f'Monthly cost: ${cap["estimated_cost_monthly_usd"]}')

print()
print('=== DONE ===')
