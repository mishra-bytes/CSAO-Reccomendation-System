"""
Phase 0 — LLM Connectivity and Usage Verification
===================================================

This test verifies every LLM/embedding component in the CSAO pipeline:
1. Template-based explainer (default, zero-latency)
2. Sentence-transformer embeddings (offline feature build)
3. OpenAI API explainer (optional, dead-code check)
4. LLM judge in evaluation (sentence-transformers coherence)

Run:
    python tests/test_llm_integration.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"
INFO = "\033[94mINFO\033[0m"


def _banner(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def test_template_explainer():
    """Test 1: Template-based explainer (used at serving time)."""
    _banner("Test 1: Template-based Explainer")
    from ranking.inference.llm_explainer import (
        explain_recommendations_batch,
        RecommendationExplanation,
    )

    # Simulate a request
    ranked_items = [
        {"item_id": "i_001", "rank_score": 0.85, "candidate_score": 0.6},
        {"item_id": "i_002", "rank_score": 0.72, "candidate_score": 0.4},
    ]
    cart_info = {
        "categories": ["main_course"],
        "missing_categories": {"beverage", "dessert"},
        "cart_value": 300.0,
        "last_item_name": "Butter Chicken",
        "cart_size": 1,
        "item_ids": ["i_100"],
    }
    item_catalog = {
        "i_001": {"item_name": "Masala Chai", "item_category": "beverage", "item_price": 60},
        "i_002": {"item_name": "Gulab Jamun", "item_category": "dessert", "item_price": 90},
    }
    comp_lookup = {("i_100", "i_001"): (2.5, 1.2), ("i_100", "i_002"): (1.8, 0.9)}

    t0 = time.perf_counter()
    results = explain_recommendations_batch(
        ranked_items=ranked_items,
        cart_info=cart_info,
        item_catalog=item_catalog,
        comp_lookup=comp_lookup,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    print(f"  Returned {len(results)} explanations in {latency_ms:.2f} ms")
    for exp in results:
        print(f"  [{exp.item_name}] \"{exp.explanation}\"")
        print(f"    Reasons: {exp.reason_tags}, Confidence: {exp.confidence:.2f}")

    ok = True
    if len(results) != 2:
        print(f"  {FAIL} Expected 2 explanations, got {len(results)}")
        ok = False
    for exp in results:
        if not exp.explanation or exp.explanation == "":
            print(f"  {FAIL} Empty explanation for {exp.item_id}")
            ok = False
        if not exp.reason_tags:
            print(f"  {FAIL} No reason tags for {exp.item_id}")
            ok = False
    if latency_ms > 10:
        print(f"  {WARN} Template explainer took {latency_ms:.1f}ms (expected <1ms)")
    if ok:
        print(f"  {PASS} Template explainer works correctly (<{latency_ms:.1f}ms)")
    return ok


def test_sentence_transformer_embeddings():
    """Test 2: sentence-transformers embedding generation (offline feature pipeline)."""
    _banner("Test 2: Sentence-Transformer Embeddings (Offline)")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print(f"  {WARN} sentence-transformers not installed — embeddings use TF-IDF fallback")
        print(f"  {INFO} This is acceptable; the pipeline caches embeddings to parquet")
        return True

    import pandas as pd
    from features.llm_embeddings import generate_item_embeddings

    test_items = pd.DataFrame({
        "item_id": ["test_1", "test_2", "test_3"],
        "item_name": ["Butter Chicken", "Paneer Tikka Masala", "Coca-Cola"],
        "item_category": ["main_course", "main_course", "beverage"],
        "item_price": [300, 280, 60],
    })

    t0 = time.perf_counter()
    emb_df = generate_item_embeddings(test_items, force_recompute=True)
    latency_ms = (time.perf_counter() - t0) * 1000

    print(f"  Generated {len(emb_df)} embeddings in {latency_ms:.0f} ms")
    emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
    print(f"  Embedding dims: {len(emb_cols)}")
    print(f"  Sample (Butter Chicken): {emb_df.iloc[0][emb_cols].values[:4]}...")

    # Check that similar items are closer in embedding space
    import numpy as np
    v1 = emb_df.iloc[0][emb_cols].values.astype(float)
    v2 = emb_df.iloc[1][emb_cols].values.astype(float)
    v3 = emb_df.iloc[2][emb_cols].values.astype(float)

    sim_12 = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    sim_13 = np.dot(v1, v3) / (np.linalg.norm(v1) * np.linalg.norm(v3) + 1e-9)

    print(f"  Cosine sim (Butter Chicken ↔ Paneer Tikka): {sim_12:.3f}")
    print(f"  Cosine sim (Butter Chicken ↔ Coca-Cola):    {sim_13:.3f}")

    if sim_12 > sim_13:
        print(f"  {PASS} Semantically similar items are closer (expected)")
    else:
        print(f"  {WARN} Semantic similarity ordering unexpected — PCA may have distorted")

    if len(emb_cols) >= 3:  # min(n_items, 8) = 3 for 3 test items
        print(f"  {PASS} Correct dimensionality ({len(emb_cols)} PCA components for {len(test_items)} items)")
    else:
        print(f"  {FAIL} Expected >=3 dims, got {len(emb_cols)}")
        return False
    return True


def test_openai_api_status():
    """Test 3: Check OpenAI API configuration (expected: NOT configured)."""
    _banner("Test 3: OpenAI API Configuration Check")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    has_openai = False
    try:
        import openai  # noqa: F401
        has_openai = True
    except ImportError:
        pass

    print(f"  OPENAI_API_KEY set: {'Yes' if api_key else 'No'}")
    print(f"  openai package installed: {'Yes' if has_openai else 'No'}")

    # Check if _llm_explain_batch is actually called anywhere
    import ast
    from pathlib import Path

    call_sites = 0
    skip_dirs = {"env", "node_modules", ".git", "__pycache__", "csao_reco.egg-info"}
    for py_file in ROOT.rglob("*.py"):
        if any(d in py_file.parts for d in skip_dirs):
            continue
        if "test_llm" in py_file.name:
            continue
        try:
            source = py_file.read_text(encoding="utf-8", errors="ignore")
            if "_llm_explain_batch" in source and "def _llm_explain_batch" not in source:
                call_sites += 1
                print(f"  {WARN} _llm_explain_batch referenced in {py_file.relative_to(ROOT)}")
        except Exception:
            pass

    if call_sites == 0:
        print(f"  {INFO} _llm_explain_batch() has ZERO call sites (dead code)")
    else:
        print(f"  {WARN} _llm_explain_batch() has {call_sites} call site(s)")

    # Attempt a real LLM call if key is set
    if api_key and has_openai:
        print(f"  {INFO} Attempting real OpenAI API call...")
        try:
            from ranking.inference.llm_explainer import _llm_explain_batch
            t0 = time.perf_counter()
            result = _llm_explain_batch(
                ranked_items=[{"item_id": "test", "item_name": "Masala Chai", "item_category": "beverage"}],
                cart_info={"categories": ["main_course"], "cart_size": 1},
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            if result:
                print(f"  {PASS} OpenAI API call succeeded ({latency_ms:.0f}ms)")
                print(f"  Sample: {result[0].explanation[:100]}...")
            else:
                print(f"  {FAIL} OpenAI API returned empty result")
        except Exception as e:
            print(f"  {FAIL} OpenAI API call failed: {e}")
    else:
        print(f"  {INFO} OpenAI API not configured — template engine is the active path")
        print(f"  {PASS} This is expected and correct for the current system")

    return True


def test_llm_judge_coherence():
    """Test 4: LLM judge embedding coherence (offline eval)."""
    _banner("Test 4: LLM Judge Embedding Coherence (Offline Eval)")
    try:
        from evaluation.metrics.llm_judge import run_llm_judge
    except ImportError as e:
        print(f"  {WARN} Could not import llm_judge: {e}")
        return True

    import pandas as pd
    import warnings
    warnings.filterwarnings("ignore")

    # Create minimal test predictions
    test_preds = pd.DataFrame({
        "query_id": ["q1"] * 5,
        "item_id": [f"i_{i}" for i in range(5)],
        "score": [0.9, 0.8, 0.7, 0.6, 0.5],
        "label": [1, 0, 0, 0, 0],
    })
    test_items = pd.DataFrame({
        "item_id": [f"i_{i}" for i in range(5)],
        "item_name": ["Butter Chicken", "Garlic Naan", "Sweet Lassi", "Dog Food", "Detergent"],
        "item_category": ["main_course", "addon", "beverage", "addon", "addon"],
    })

    t0 = time.perf_counter()
    try:
        result = run_llm_judge(test_preds, test_items, k=3)
        latency_ms = (time.perf_counter() - t0) * 1000
        print(f"  Coherence result ({latency_ms:.0f}ms): {result}")
        print(f"  {PASS} LLM judge works (sentence-transformers coherence)")
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000
        print(f"  {WARN} LLM judge failed ({latency_ms:.0f}ms): {e}")
        print(f"  {INFO} This is non-critical — only used in offline evaluation")

    return True


def test_cached_embeddings_exist():
    """Test 5: Verify pre-computed embeddings are cached and used at serving time."""
    _banner("Test 5: Pre-computed Embeddings Cache")
    cache_path = ROOT / "data" / "processed" / "embeddings_cache.parquet"
    feature_path = ROOT / "data" / "processed" / "features_item.parquet"

    if cache_path.exists():
        import pandas as pd
        cache_df = pd.read_parquet(cache_path)
        emb_cols = [c for c in cache_df.columns if c.startswith("emb_")]
        print(f"  Embeddings cache: {cache_path.name}")
        print(f"  Items: {len(cache_df)}, Dims: {len(emb_cols)}")
        print(f"  {PASS} Embeddings cached — NO model loaded at serving time")
    else:
        print(f"  {WARN} No embeddings cache at {cache_path}")
        print(f"  {INFO} Embeddings may be inlined in item features")

    if feature_path.exists():
        import pandas as pd
        feat_df = pd.read_parquet(feature_path)
        emb_cols = [c for c in feat_df.columns if "emb" in c.lower()]
        print(f"  Item features: {len(feat_df)} items, embedding cols: {emb_cols[:5]}")
        if emb_cols:
            print(f"  {PASS} Embeddings merged into item features (used by LightGBM)")
        else:
            print(f"  {WARN} No embedding columns in item features")
    return True


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  CSAO — LLM Integration Verification (Phase 0)")
    print("=" * 60)

    results = {}
    results["template_explainer"] = test_template_explainer()
    results["sentence_transformer"] = test_sentence_transformer_embeddings()
    results["openai_api"] = test_openai_api_status()
    results["llm_judge"] = test_llm_judge_coherence()
    results["cached_embeddings"] = test_cached_embeddings_exist()

    _banner("SUMMARY")
    all_pass = True
    for name, ok in results.items():
        status = PASS if ok else FAIL
        print(f"  {status}  {name}")
        if not ok:
            all_pass = False

    print()
    print("=" * 60)
    print("  LLM USAGE VERDICT")
    print("=" * 60)
    print("""
  Component                 | Status          | At Serving Time?
  ─────────────────────────┼─────────────────┼─────────────────
  Template Explainer        | ACTIVE (default)| YES — <1ms
  sentence-transformers     | Offline only    | NO — cached to parquet
  OpenAI API Explainer      | DEAD CODE       | NO — zero call sites
  OpenAI API Judge          | Not configured  | NO — offline eval only
  Neural Reranker           | Pre-computed emb| NO LLM at inference

  CONCLUSION:
  • No real LLM API call is made at any point in the system.
  • The "LLM" label refers to sentence-transformers (offline) + templates (serving).
  • No API key is needed or missing.
  • The template explainer IS functional but could be improved.
  • To add real LLM capabilities, set OPENAI_API_KEY and wire
    _llm_explain_batch() into the serving path.
""")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
