from __future__ import annotations

import pandas as pd

from serving.pipeline.cold_start import ColdStartContext, ColdStartHandler


def _cold_start_fixture():
    item_catalog = pd.DataFrame(
        [
            {"item_id": "i1", "item_category": "main_course"},
            {"item_id": "i2", "item_category": "beverage"},
            {"item_id": "i3", "item_category": "dessert"},
            {"item_id": "i4", "item_category": "addon"},
        ]
    )
    order_items = pd.DataFrame(
        [
            {"order_id": "o1", "restaurant_id": "r1", "item_id": "i1", "order_hour": 13},
            {"order_id": "o1", "restaurant_id": "r1", "item_id": "i2", "order_hour": 13},
            {"order_id": "o2", "restaurant_id": "r1", "item_id": "i4", "order_hour": 21},
            {"order_id": "o3", "restaurant_id": "r2", "item_id": "i3", "order_hour": 21},
        ]
    )
    user_features = pd.DataFrame([{"user_id": "warm_u", "order_frequency": 0.2}])
    return ColdStartHandler(item_catalog, order_items=order_items, user_features=user_features, config={"serving": {"default_top_n": 3}})


def test_new_user_known_restaurant_non_empty_cart_is_context_aware():
    h = _cold_start_fixture()
    d = h.handle(ColdStartContext(user_id="new_u", restaurant_id="r1", cart_item_ids=["i1"], hour_of_day=13))
    assert d.candidates
    assert d.strategy == "cart_aware_restaurant_popular"


def test_new_user_unknown_restaurant_no_crash():
    h = _cold_start_fixture()
    d = h.handle(ColdStartContext(user_id="new_u", restaurant_id="rX", cart_item_ids=[], hour_of_day=22))
    assert d.candidates
    assert d.strategy == "global_diverse_popular"
