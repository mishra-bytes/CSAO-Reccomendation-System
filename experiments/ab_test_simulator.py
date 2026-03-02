"""
A/B Test Offline Replay Simulator
===================================
Splits synthetic sessions into control (popularity baseline) and treatment
(full CSAO model), runs predictions from each, and computes statistical
significance of the lift.

Outputs:
  - artifacts/ab_test/ab_test_results.csv
  - artifacts/ab_test/ab_test_report.md

Usage:
    python -m experiments.ab_test_simulator [--n-bootstrap 1000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts._utils import load_project_config, load_unified_tables, load_feature_tables
from features.complementarity import build_complementarity_lookup
from ranking.training.dataset import build_training_dataset
from ranking.training.train import _temporal_train_valid_split, _group_from_query_ids
from evaluation.metrics.ranking_metrics import ndcg_at_k, precision_at_k
from evaluation.metrics.business_impact import compute_attach_rate


def _z_test_proportions(p1: float, n1: int, p2: float, n2: int) -> tuple[float, float]:
    """Two-proportion z-test. Returns (z_stat, p_value)."""
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    if p_pool == 0 or p_pool == 1:
        return 0.0, 1.0
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    z = (p1 - p2) / se
    # Two-sided p-value from standard normal
    from scipy import stats as sp_stats
    p_val = 2 * (1 - sp_stats.norm.cdf(abs(z)))
    return float(z), float(p_val)


def _bootstrap_ci(
    treatment_values: np.ndarray,
    control_values: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for difference in means.
    Returns (mean_diff, ci_lower, ci_upper)."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_bootstrap):
        t_sample = rng.choice(treatment_values, size=len(treatment_values), replace=True)
        c_sample = rng.choice(control_values, size=len(control_values), replace=True)
        diffs.append(t_sample.mean() - c_sample.mean())
    diffs = np.array(diffs)
    mean_diff = float(diffs.mean())
    ci_lo = float(np.percentile(diffs, 100 * alpha / 2))
    ci_hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return mean_diff, ci_lo, ci_hi


def run_ab_simulation(
    n_bootstrap: int = 1000,
    output_dir: str = "artifacts/ab_test",
) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    config = load_project_config()
    ranking_cfg = config.get("ranking", {})
    ranking_params = config.get("lightgbm", {})

    processed_dir = config.get("paths", {}).get("processed_dir", "data/processed")
    unified = load_unified_tables(processed_dir)
    features = load_feature_tables(processed_dir)
    comp_lookup = build_complementarity_lookup(features["complementarity"])

    print("[ab] Building training dataset …")
    data = build_training_dataset(
        unified=unified,
        user_features=features["user_features"],
        item_features=features["item_features"],
        comp_lookup=comp_lookup,
        config=config,
    )

    orders = unified.get("orders", pd.DataFrame())
    val_frac = float(config.get("train", {}).get("validation_fraction", 0.2))
    seed = int(ranking_cfg.get("random_state", 42))
    train_idx, val_idx = _temporal_train_valid_split(data, orders, val_frac, seed)

    q_train = _group_from_query_ids([data.query_ids[i] for i in train_idx])
    q_val = _group_from_query_ids([data.query_ids[i] for i in val_idx])

    X_train = data.X.iloc[train_idx]
    y_train = data.y[train_idx]
    X_val = data.X.iloc[val_idx]
    y_val = data.y[val_idx]

    # ── Train treatment model (full CSAO) ─────────────────────────────────
    print("[ab] Training treatment model (full CSAO) …")
    from lightgbm import LGBMRanker
    treatment_model = LGBMRanker(**ranking_params)
    treatment_model.fit(X_train, y_train, group=q_train,
                        eval_set=[(X_val, y_val)], eval_group=[q_val], eval_at=[10])
    treatment_scores = treatment_model.predict(X_val)

    # ── Control: popularity-only baseline ─────────────────────────────────
    print("[ab] Computing control (popularity baseline) …")
    pop_col = "item_order_count" if "item_order_count" in data.X.columns else None
    if pop_col:
        control_scores = data.X.iloc[val_idx][pop_col].values.astype(float)
    else:
        control_scores = np.random.default_rng(seed).random(len(val_idx))

    # ── Build prediction DataFrames ───────────────────────────────────────
    query_ids_val = [data.query_ids[i] for i in val_idx]
    item_ids_val = [data.candidate_items[i] for i in val_idx]

    treatment_preds = pd.DataFrame({
        "query_id": query_ids_val,
        "item_id": item_ids_val,
        "label": y_val.astype(int),
        "score": treatment_scores.astype(float),
    })
    control_preds = pd.DataFrame({
        "query_id": query_ids_val,
        "item_id": item_ids_val,
        "label": y_val.astype(int),
        "score": control_scores.astype(float),
    })

    # ── Split queries into control/treatment (50/50 random assignment) ────
    unique_queries = list(set(query_ids_val))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_queries)
    mid = len(unique_queries) // 2
    control_queries = set(unique_queries[:mid])
    treatment_queries = set(unique_queries[mid:])

    ctrl_df = control_preds[control_preds["query_id"].isin(control_queries)]
    treat_df = treatment_preds[treatment_preds["query_id"].isin(treatment_queries)]

    # ── Compute metrics per arm ───────────────────────────────────────────
    ctrl_ndcg = ndcg_at_k(ctrl_df, k=10)
    treat_ndcg = ndcg_at_k(treat_df, k=10)
    ctrl_prec = precision_at_k(ctrl_df, k=10)
    treat_prec = precision_at_k(treat_df, k=10)
    ctrl_attach = compute_attach_rate(ctrl_df, k=10)
    treat_attach = compute_attach_rate(treat_df, k=10)

    n_ctrl = len(control_queries)
    n_treat = len(treatment_queries)

    # ── Statistical significance ──────────────────────────────────────────
    # Z-test on attach rate (proportion test)
    z_stat, p_value = _z_test_proportions(treat_attach, n_treat, ctrl_attach, n_ctrl)

    # Bootstrap CI on NDCG lift
    # Compute per-query NDCG for bootstrap
    from evaluation.metrics.ranking_metrics import _top_k
    import math

    def _per_query_ndcg(df: pd.DataFrame, k: int = 10) -> np.ndarray:
        vals = []
        for _, group in df.groupby("query_id"):
            top = _top_k(group, k).reset_index(drop=True)
            dcg = sum((2.0**l - 1.0) / math.log2(i + 2) for i, l in enumerate(top["label"].astype(float)))
            ideal = sorted(group["label"].astype(float).tolist(), reverse=True)[:k]
            idcg = sum((2.0**l - 1.0) / math.log2(i + 2) for i, l in enumerate(ideal))
            vals.append(dcg / max(idcg, 1e-9))
        return np.array(vals)

    treat_ndcgs = _per_query_ndcg(treat_df)
    ctrl_ndcgs = _per_query_ndcg(ctrl_df)

    ndcg_diff, ndcg_ci_lo, ndcg_ci_hi = _bootstrap_ci(
        treat_ndcgs, ctrl_ndcgs, n_bootstrap=n_bootstrap, seed=seed,
    )

    # ── Results ───────────────────────────────────────────────────────────
    results = {
        "control_queries": n_ctrl,
        "treatment_queries": n_treat,
        "control_ndcg": ctrl_ndcg,
        "treatment_ndcg": treat_ndcg,
        "ndcg_lift": treat_ndcg - ctrl_ndcg,
        "ndcg_lift_ci_lower": ndcg_ci_lo,
        "ndcg_lift_ci_upper": ndcg_ci_hi,
        "control_precision": ctrl_prec,
        "treatment_precision": treat_prec,
        "control_attach_rate": ctrl_attach,
        "treatment_attach_rate": treat_attach,
        "attach_rate_lift_pp": (treat_attach - ctrl_attach) * 100,
        "z_statistic": z_stat,
        "p_value": p_value,
        "significant_at_005": p_value < 0.05,
    }

    results_df = pd.DataFrame([results])
    csv_path = out / "ab_test_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n[ab] Results saved → {csv_path}")

    # Print summary
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # ── Markdown report ───────────────────────────────────────────────────
    sig_label = "✅ YES" if results["significant_at_005"] else "❌ NO"
    md = [
        "# A/B Test Offline Replay Report",
        "",
        "## Setup",
        f"- **Control:** Popularity baseline ({n_ctrl} sessions)",
        f"- **Treatment:** Full CSAO model ({n_treat} sessions)",
        f"- **Assignment:** Random 50/50 query-level split",
        f"- **Bootstrap samples:** {n_bootstrap}",
        "",
        "## Results",
        "",
        "| Metric | Control | Treatment | Lift |",
        "|--------|---------|-----------|------|",
        f"| NDCG@10 | {ctrl_ndcg:.4f} | {treat_ndcg:.4f} | {treat_ndcg - ctrl_ndcg:+.4f} |",
        f"| Precision@10 | {ctrl_prec:.4f} | {treat_prec:.4f} | {treat_prec - ctrl_prec:+.4f} |",
        f"| Attach Rate | {ctrl_attach:.4f} | {treat_attach:.4f} | {(treat_attach - ctrl_attach)*100:+.1f} pp |",
        "",
        "## Statistical Significance",
        "",
        f"- **Z-statistic (attach rate):** {z_stat:.3f}",
        f"- **p-value:** {p_value:.4f}",
        f"- **Significant at α=0.05:** {sig_label}",
        f"- **NDCG lift 95% CI:** [{ndcg_ci_lo:+.4f}, {ndcg_ci_hi:+.4f}]",
        "",
        "## Interpretation",
        "",
    ]
    if results["significant_at_005"]:
        md.append("The treatment (CSAO model) shows a statistically significant improvement "
                   "over the popularity baseline, supporting deployment.")
    else:
        md.append("The difference did not reach statistical significance at α=0.05. "
                   "This may be due to insufficient session count or small effect size.")

    report_path = out / "ab_test_report.md"
    report_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[ab] Report saved → {report_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="A/B test offline replay simulator")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--output-dir", default="artifacts/ab_test")
    args = parser.parse_args()
    run_ab_simulation(n_bootstrap=args.n_bootstrap, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
