from __future__ import annotations

from features.meal_semantics import category_compatibility_multiplier


def _best_category(cart_name: str, categories: list[str]) -> str:
    scored = sorted(
        categories,
        key=lambda c: category_compatibility_multiplier([cart_name], c),
        reverse=True,
    )
    return scored[0]


def test_egg_roll_cart_prefers_addon_or_beverage_over_dessert():
    cats = ["dessert", "addon", "beverage", "main_course"]
    best = _best_category("Egg Roll", cats)
    assert best in {"addon", "beverage"}
    assert category_compatibility_multiplier(["Egg Roll"], "dessert") < 1.0


def test_biryani_cart_prefers_meal_completion_categories():
    cats = ["addon", "beverage", "dessert", "main_course"]
    best = _best_category("Chicken Biryani", cats)
    assert best in {"addon", "beverage", "dessert"}


def test_dosa_cart_prefers_chutney_sambar_coffee_categories():
    cats = ["addon", "beverage", "dessert"]
    best = _best_category("Masala Dosa", cats)
    assert best in {"addon", "beverage"}


def test_pizza_cart_prefers_dip_bread_drink_categories():
    cats = ["addon", "beverage", "dessert"]
    best = _best_category("Farmhouse Pizza", cats)
    assert best in {"addon", "beverage"}
