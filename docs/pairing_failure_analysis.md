# Pairing Failure Analysis (Egg Roll Case)

Deterministic reproduction is implemented in `tests/test_pairing_sanity.py`.

| Failure Mode | Root Cause | Component | Fix | Test Coverage |
|---|---|---|---|---|
| Egg Roll receives dessert-heavy add-ons | High raw co-occurrence lift from noisy sparse pairs dominates retrieval score | `CooccurrenceRetriever` + weighted merge in `CandidateGenerator` | Added **meal-semantic compatibility multiplier** (soft penalty for implausible category; boost for plausible category) before aggregation | `tests/test_pairing_sanity.py` |
| Missing cuisine/meal intent signal for short carts | Category affinity and popularity are context-light and can overfit popularity tails | `CandidateGenerator` score blending | Introduced policy inference from cart item names (Egg Roll/Biryani/Dosa/Pizza) with category-level soft priors | `tests/test_meal_semantics.py` |
| No explicit debug trace of retrieval-to-final path | Hard to explain why bad outputs occurred | Testing/observability gap | Added per-retriever printed trace + final ranked output in deterministic sanity test | `tests/test_pairing_sanity.py` |
