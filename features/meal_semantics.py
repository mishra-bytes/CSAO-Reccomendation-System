from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MealSemanticPolicy:
    name_tokens: tuple[str, ...]
    preferred_categories: tuple[str, ...]
    blocked_categories: tuple[str, ...] = ()


_POLICIES: tuple[MealSemanticPolicy, ...] = (
    MealSemanticPolicy(("biryani",), ("addon", "beverage", "dessert")),
    MealSemanticPolicy(("dosa",), ("addon", "beverage")),
    MealSemanticPolicy(("pizza",), ("addon", "beverage")),
    MealSemanticPolicy(("roll", "egg roll", "kathi"), ("addon", "beverage"), ("dessert",)),
)


def infer_policy(item_names: list[str]) -> MealSemanticPolicy | None:
    text = " ".join(n.lower() for n in item_names)
    for policy in _POLICIES:
        if any(tok in text for tok in policy.name_tokens):
            return policy
    return None


def category_compatibility_multiplier(
    cart_item_names: list[str],
    candidate_category: str,
) -> float:
    """Soft compatibility multiplier used by candidate generation/ranking.

    Values:
    - 1.15 for preferred categories
    - 0.60 for policy-blocked categories
    - 1.00 default
    """
    policy = infer_policy(cart_item_names)
    cat = (candidate_category or "").lower()
    if policy is None:
        return 1.0
    if cat in policy.preferred_categories:
        return 1.25
    if cat in policy.blocked_categories:
        return 0.30
    return 1.0
