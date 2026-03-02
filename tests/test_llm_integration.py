from __future__ import annotations

import os
import time

from ranking.inference.llm_explainer import _llm_explain_batch, explain_recommendations_batch


def _sample_payload():
    ranked_items = [{"item_id": "tea", "rank_score": 0.8, "candidate_score": 0.4}]
    cart_info = {
        "categories": ["main_course"],
        "missing_categories": {"beverage"},
        "cart_value": 240.0,
        "last_item_name": "Egg Roll",
        "cart_size": 1,
        "item_ids": ["egg_roll"],
    }
    item_catalog = {
        "tea": {"item_name": "Masala Chai", "item_category": "beverage", "item_price": 40},
        "egg_roll": {"item_name": "Egg Roll", "item_category": "starter", "item_price": 120},
    }
    return ranked_items, cart_info, item_catalog


def test_llm_provider_configuration_and_real_call_or_safe_fallback():
    """Phase 0: verify OpenRouter wiring and behavior with/without key."""
    ranked_items, cart_info, item_catalog = _sample_payload()

    # Expected configuration for this repo.
    assert os.environ.get("OPENAI_API_KEY") in {None, ""}, (
        "Use OPENROUTER_API_KEY for this project; OPENAI_API_KEY should not be required."
    )

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    t0 = time.perf_counter()

    if api_key:
        lines = _llm_explain_batch(
            ranked_items=ranked_items,
            cart_info=cart_info,
            item_catalog=item_catalog,
            model="google/gemma-3-4b-it:free",
            max_items=1,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        print(f"OpenRouter latency_ms={latency_ms:.2f}")
        print(f"OpenRouter sample={lines[0] if lines else '<empty>'}")
        assert lines, "OPENROUTER_API_KEY is set but real OpenRouter call returned empty output."
    else:
        # Safe no-key mode: still return recommendation explanation via templates.
        out = explain_recommendations_batch(
            ranked_items=ranked_items,
            cart_info=cart_info,
            item_catalog=item_catalog,
            comp_lookup={("egg_roll", "tea"): (0.6, 0.2)},
            use_llm=True,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        print("OPENROUTER_API_KEY not set; validated template fallback path")
        print(f"Fallback latency_ms={latency_ms:.2f}")
        print(f"Fallback sample={out[0].explanation}")
        assert out and out[0].explanation, "No-key fallback must not crash and must return explanation."
