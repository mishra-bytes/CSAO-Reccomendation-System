from __future__ import annotations

import pandas as pd

from candidate_generation.candidate_generator import CandidateGenerator


def _mini_data():
    items = pd.DataFrame(
        [
            {"item_id": "egg_roll", "item_name": "Egg Roll", "item_category": "starter", "item_price": 120},
            {"item_id": "mayo_dip", "item_name": "Mayo Dip", "item_category": "addon", "item_price": 25},
            {"item_id": "cola", "item_name": "Coke", "item_category": "beverage", "item_price": 50},
            {"item_id": "gulab", "item_name": "Gulab Jamun", "item_category": "dessert", "item_price": 90},
            {"item_id": "brownie", "item_name": "Chocolate Brownie", "item_category": "dessert", "item_price": 110},
        ]
    )
    orders = pd.DataFrame(
        [
            {"order_id": "o1", "restaurant_id": "r1"},
            {"order_id": "o2", "restaurant_id": "r1"},
            {"order_id": "o3", "restaurant_id": "r1"},
        ]
    )
    order_items = pd.DataFrame(
        [
            {"order_id": "o1", "item_id": "egg_roll", "position": 1},
            {"order_id": "o1", "item_id": "mayo_dip", "position": 2},
            {"order_id": "o1", "item_id": "cola", "position": 3},
            {"order_id": "o2", "item_id": "egg_roll", "position": 1},
            {"order_id": "o2", "item_id": "gulab", "position": 2},
            {"order_id": "o3", "item_id": "egg_roll", "position": 1},
            {"order_id": "o3", "item_id": "brownie", "position": 2},
        ]
    )
    complementarity = pd.DataFrame(
        [
            {"item_id": "egg_roll", "candidate_item_id": "gulab", "lift": 3.5, "pmi": 2.0},
            {"item_id": "egg_roll", "candidate_item_id": "brownie", "lift": 3.0, "pmi": 1.8},
            {"item_id": "egg_roll", "candidate_item_id": "cola", "lift": 1.2, "pmi": 0.5},
            {"item_id": "egg_roll", "candidate_item_id": "mayo_dip", "lift": 1.1, "pmi": 0.4},
        ]
    )
    category_affinity = pd.DataFrame(
        [
            {"from_category": "starter", "to_category": "dessert", "affinity": 0.9},
            {"from_category": "starter", "to_category": "beverage", "affinity": 0.6},
            {"from_category": "starter", "to_category": "addon", "affinity": 0.6},
        ]
    )
    return items, orders, order_items, complementarity, category_affinity


def test_egg_roll_pairing_trace_and_final_sanity():
    items, orders, order_items, comp, cat_aff = _mini_data()
    gen = CandidateGenerator(
        complementarity=comp,
        category_affinity=cat_aff,
        items=items,
        orders=orders,
        order_items=order_items,
        config={"candidate_generation": {"total_candidates": 5}},
    )
    cart = ["egg_roll"]
    exclude = set(cart)

    co = gen.co_retriever.retrieve(cart, k=5)
    session = gen.session_retriever.retrieve(cart, exclude=exclude, k=5)
    meal_gap = gen.meal_gap_retriever.retrieve(cart, restaurant_id="r1", exclude=exclude, k=5)
    category = gen.cat_retriever.retrieve(cart, restaurant_id="r1", exclude=exclude, k=5)
    pop = gen.pop_retriever.retrieve(restaurant_id="r1", exclude=exclude, k=5)

    print("cooccurrence", co)
    print("session", session)
    print("meal_gap", meal_gap)
    print("category", category)
    print("popularity", pop)

    final = gen.generate(cart, restaurant_id="r1", top_k=5)
    print("final", final)

    top_item = final[0][0]
    top_cat = items.set_index("item_id").loc[top_item, "item_category"]
    assert top_cat in {"addon", "beverage"}, f"Expected sensible top category, got {top_cat}"
