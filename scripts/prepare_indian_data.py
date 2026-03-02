"""
prepare_indian_data.py
====================
Generates **Indian food-only** delivery data for the CSAO recommendation system.

Completely replaces the Instacart grocery-based data with:
  1. Synthetic Indian food delivery orders (authentic cuisines, realistic meal patterns)
  2. Real Indian Takeaway orders from Mendeley dataset (both restaurants)
  3. Food-similarity embeddings (cuisine + co-purchase SVD)

Outputs (same schema as prepare_real_data.py — drop-in replacement):
    data/raw/restaurant_orders.csv   ← Synthetic Indian food orders
    data/raw/mendeley_orders.csv     ← Real Indian Takeaway orders
    data/raw/recipe_embeddings.csv   ← 8-dim food embeddings

Usage:
    python scripts/prepare_indian_data.py

Then continue with the normal pipeline:
    python scripts/build_unified_data.py
    python scripts/build_features.py
    python scripts/train_ranker.py
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from demo_catalog import DISHES, RESTAURANT_NAMES  # noqa: E402

DOWNLOAD_DIR = ROOT / "data" / "raw" / "downloads"
OUTPUT_DIR = ROOT / "data" / "raw"

# ─── Configuration ────────────────────────────────────────────────────────────
NUM_RESTAURANTS = 500
NUM_USERS = 20_000
NUM_ORDERS = 200_000

CITIES = ["Delhi", "Mumbai", "Bangalore", "Hyderabad", "Pune", "Chennai", "Kolkata"]
CUISINES = ["North Indian", "South Indian", "Chinese", "Biryani",
            "Italian", "Street Food", "Bengali", "Mughlai"]

# How common each cuisine is (order volume share)
CUISINE_WEIGHTS = {
    "North Indian": 0.22, "South Indian": 0.16, "Biryani": 0.15,
    "Chinese": 0.14, "Mughlai": 0.10, "Street Food": 0.10,
    "Bengali": 0.07, "Italian": 0.06,
}

# Meal patterns: (probability, {category: count_or_(min,max)})
MEAL_PATTERNS = [
    (0.28, {"main_course": 1, "addon": (1, 2)}),                      # quick
    (0.25, {"main_course": 1, "addon": (1, 2), "beverage": 1}),       # standard
    (0.20, {"main_course": (1, 2), "starter": 1,
            "addon": (2, 3), "beverage": 1}),                         # full
    (0.10, {"main_course": (2, 3), "starter": (1, 2),
            "addon": (3, 4), "beverage": (1, 2), "dessert": (1, 2)}), # party
    (0.10, {"starter": (2, 3), "beverage": 1}),                       # snack
    (0.07, {"main_course": 1, "addon": 1, "dessert": 1}),             # dessert-meal
]

# When main_course is in cart, these items from the specified category get
# a 4× selection-probability boost, creating realistic co-occurrence signals.
PAIRING_BOOSTS: dict[str, dict[str, list[str]]] = {
    "North Indian": {
        "addon": ["Butter Naan", "Garlic Naan", "Tandoori Roti",
                  "Raita (Boondi)", "Laccha Paratha", "Plain Rice"],
        "beverage": ["Sweet Lassi", "Masala Chai", "Buttermilk (Chaas)"],
        "dessert": ["Gulab Jamun (2 pcs)", "Gajar Ka Halwa"],
    },
    "South Indian": {
        "addon": ["Sambar (extra)", "Coconut Chutney", "Tomato Chutney",
                  "Ghee (extra)", "Curd (extra)"],
        "beverage": ["Filter Coffee", "Buttermilk (Neer Mor)"],
        "dessert": ["Payasam", "Mysore Pak"],
    },
    "Biryani": {
        "addon": ["Raita", "Mirchi Ka Salan", "Salan (extra)",
                  "Onion Raita", "Boiled Egg (2 pcs)"],
        "beverage": ["Falooda", "Sweet Lassi"],
        "dessert": ["Gulab Jamun (2 pcs)", "Phirni"],
    },
    "Chinese": {
        "addon": ["Manchow Soup", "Hot & Sour Soup", "Sweet Corn Soup",
                  "Schezwan Sauce (extra)", "Steamed Rice"],
        "beverage": ["Iced Lemon Tea", "Fresh Lime Soda"],
        "dessert": ["Fried Ice Cream", "Chocolate Brownie"],
    },
    "Mughlai": {
        "addon": ["Butter Naan", "Sheermal", "Garlic Naan",
                  "Raita", "Mint Raita", "Tandoori Roti"],
        "beverage": ["Rooh Afza Sharbat", "Sweet Lassi", "Kahwa"],
        "dessert": ["Shahi Tukda", "Phirni"],
    },
    "Bengali": {
        "addon": ["Steamed Rice", "Luchi (2 pcs)", "Aloo Bhaja",
                  "Extra Rice", "Kasundi (Mustard Dip)"],
        "beverage": ["Masala Chai", "Mishti Doi Lassi"],
        "dessert": ["Mishti Doi", "Rasgulla (2 pcs)", "Sandesh"],
    },
    "Street Food": {
        "addon": ["Green Chutney", "Tamarind Chutney", "Extra Pav (2 pcs)",
                  "Sev (extra)", "Onion Topping", "Extra Cheese"],
        "beverage": ["Masala Chai (Cutting)", "Sugarcane Juice", "Nimbu Pani"],
        "dessert": ["Kulfi (Stick)", "Rabri Falooda"],
    },
    "Italian": {
        "addon": ["Extra Cheese Topping", "Garlic Mayo", "Oregano Seasoning",
                  "Olive Oil Dip", "Side Salad"],
        "starter": ["Garlic Bread (4 pcs)", "Bruschetta (3 pcs)"],
        "beverage": ["Cold Coffee", "Virgin Mojito", "Iced Tea (Peach)"],
        "dessert": ["Tiramisu", "Chocolate Lava Cake"],
    },
}

# Indian Food 101 category map (for Mendeley takeaway classification)
INDIAN_FOOD_CATEGORIES = {
    "main course": "main_course", "dessert": "dessert", "starter": "starter",
    "snack": "starter", "drink": "beverage", "side dish": "addon",
    "one dish meal": "main_course",
}

# Keyword fallback for Mendeley items (CORRECTED: naan/roti → addon, not main)
KEYWORD_CATEGORIES = {
    "rice": "main_course", "biryani": "main_course", "curry": "main_course",
    "dal": "main_course", "daal": "main_course", "paneer": "main_course",
    "chicken": "main_course", "lamb": "main_course", "mutton": "main_course",
    "fish": "main_course", "prawn": "main_course", "tikka": "main_course",
    "masala": "main_course", "korma": "main_course", "vindaloo": "main_course",
    "madras": "main_course", "jalfrezi": "main_course", "bhuna": "main_course",
    "dopiaza": "main_course", "dhansak": "main_course", "balti": "main_course",
    "pilau": "main_course", "pulao": "main_course", "tandoori": "main_course",
    # Breads → addon (NOT main_course)
    "naan": "addon", "roti": "addon", "paratha": "addon", "chapati": "addon",
    "kulcha": "addon", "puri": "addon",
    "kebab": "starter", "samosa": "starter", "pakora": "starter",
    "bhaji": "starter", "poppadom": "addon", "papadum": "addon",
    "chutney": "addon", "pickle": "addon", "raita": "addon",
    "salad": "addon", "sauce": "addon", "dip": "addon",
    "lassi": "beverage", "cola": "beverage", "coke": "beverage",
    "pepsi": "beverage", "water": "beverage", "juice": "beverage",
    "tea": "beverage", "coffee": "beverage", "lemonade": "beverage",
    "gulab": "dessert", "jamun": "dessert", "halwa": "dessert",
    "kheer": "dessert", "kulfi": "dessert", "ice cream": "dessert",
    "jalebi": "dessert", "barfi": "dessert", "sweet": "dessert",
    "rasmalai": "dessert", "payasam": "dessert",
}


# ══════════════════════════════════════════════════════════════════════════
# 1. Build Item Catalog from DISHES dict
# ══════════════════════════════════════════════════════════════════════════
def build_item_catalog() -> dict[str, dict]:
    """
    Build global catalog: dish_name → {item_id, item_name, item_type,
                                        min_price, max_price, cuisines}
    """
    catalog: dict[str, dict] = {}
    idx = 1
    for cuisine, categories in DISHES.items():
        if cuisine == "_default":
            continue
        for cat, dish_list in categories.items():
            for name, min_p, max_p in dish_list:
                if name not in catalog:
                    catalog[name] = {
                        "item_id": f"f_{idx:05d}",
                        "item_name": name,
                        "item_type": cat,
                        "min_price": float(min_p),
                        "max_price": float(max_p),
                        "cuisines": set(),
                    }
                    idx += 1
                catalog[name]["cuisines"].add(cuisine)
                # Widen price range across cuisines
                catalog[name]["min_price"] = min(catalog[name]["min_price"], float(min_p))
                catalog[name]["max_price"] = max(catalog[name]["max_price"], float(max_p))
    return catalog


# ══════════════════════════════════════════════════════════════════════════
# 2. Build Restaurants with Cuisine-Specific Menus
# ══════════════════════════════════════════════════════════════════════════
def build_restaurants(
    catalog: dict[str, dict], rng: np.random.Generator,
) -> tuple[list[dict], dict[str, list[dict]]]:
    """Create NUM_RESTAURANTS restaurants, each with a balanced cuisine menu."""

    restaurants: list[dict] = []
    menus: dict[str, list[dict]] = {}
    rest_idx = 1

    cuisine_weights = np.array([CUISINE_WEIGHTS[c] for c in CUISINES])
    cuisine_weights /= cuisine_weights.sum()
    rest_counts = np.round(cuisine_weights * NUM_RESTAURANTS).astype(int)
    # Adjust to hit exactly NUM_RESTAURANTS
    rest_counts[-1] = NUM_RESTAURANTS - rest_counts[:-1].sum()

    for ci, cuisine in enumerate(CUISINES):
        n_rest = max(10, int(rest_counts[ci]))
        name_pool = RESTAURANT_NAMES.get(cuisine, RESTAURANT_NAMES["North Indian"])

        # All items available for this cuisine
        cuisine_items = [v for v in catalog.values() if cuisine in v["cuisines"]]
        by_cat: dict[str, list[dict]] = {}
        for item in cuisine_items:
            by_cat.setdefault(item["item_type"], []).append(item)

        for j in range(n_rest):
            rest_id = f"r_{rest_idx:04d}"
            rest_idx += 1
            city = CITIES[int(rng.integers(0, len(CITIES)))]

            base_name = name_pool[j % len(name_pool)]
            name = f"{base_name} - {city}" if j >= len(name_pool) else base_name

            # Build balanced menu (25-45 items)
            menu_size = int(rng.integers(25, 46))
            cat_targets = {
                "main_course": 0.30, "addon": 0.28, "starter": 0.15,
                "beverage": 0.15, "dessert": 0.12,
            }
            menu: list[dict] = []

            for cat, frac in cat_targets.items():
                pool = by_cat.get(cat, [])
                if not pool:
                    continue
                n = min(len(pool), max(2, round(menu_size * frac)))
                chosen_idx = rng.choice(len(pool), size=n, replace=False)
                for cidx in chosen_idx:
                    item = pool[int(cidx)].copy()
                    # Stable per-restaurant price
                    item["rest_price"] = round(
                        rng.uniform(item["min_price"], item["max_price"]), 0
                    )
                    menu.append(item)

            restaurants.append({
                "restaurant_id": rest_id,
                "restaurant_name": name,
                "city": city,
                "cuisine": cuisine,
            })
            menus[rest_id] = menu

    return restaurants, menus


# ══════════════════════════════════════════════════════════════════════════
# 3. Generate Synthetic Orders
# ══════════════════════════════════════════════════════════════════════════
def generate_orders(
    restaurants: list[dict],
    menus: dict[str, list[dict]],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate NUM_ORDERS synthetic orders with cuisine-aware meal patterns."""

    n_rest = len(restaurants)
    rest_ids = [r["restaurant_id"] for r in restaurants]

    # Zipf-like restaurant popularity
    rest_pop = 1.0 / np.arange(1, n_rest + 1) ** 0.6
    rest_pop /= rest_pop.sum()

    # User IDs + per-user favorite restaurants (loyalty model)
    user_ids = [f"u_{i:05d}" for i in range(1, NUM_USERS + 1)]
    user_favs: dict[str, np.ndarray] = {}
    for uid in user_ids:
        n_favs = int(rng.integers(1, 4))
        user_favs[uid] = rng.choice(n_rest, size=n_favs, replace=False, p=rest_pop)

    # Pre-build per-restaurant structures
    menu_by_cat: dict[str, dict[str, list[dict]]] = {}
    boost_weights: dict[str, dict[str, np.ndarray]] = {}
    rest_cuisine_map = {r["restaurant_id"]: r["cuisine"] for r in restaurants}

    for rid in rest_ids:
        by_cat: dict[str, list[dict]] = {}
        for item in menus[rid]:
            by_cat.setdefault(item["item_type"], []).append(item)
        menu_by_cat[rid] = by_cat

        cuisine = rest_cuisine_map[rid]
        boosts = PAIRING_BOOSTS.get(cuisine, {})
        rw: dict[str, np.ndarray] = {}
        for cat, items in by_cat.items():
            w = np.ones(len(items))
            if cat in boosts:
                boosted_names = set(boosts[cat])
                for k, item in enumerate(items):
                    if item["item_name"] in boosted_names:
                        w[k] = 4.0
            rw[cat] = w / w.sum()
        boost_weights[rid] = rw

    # Meal pattern probabilities
    pattern_probs = np.array([p[0] for p in MEAL_PATTERNS])
    pattern_probs /= pattern_probs.sum()

    # Time-of-day distribution (lunch + dinner peaks)
    hour_w = np.zeros(24)
    for h in range(11, 15):
        hour_w[h] = 3.0  # lunch
    for h in range(18, 23):
        hour_w[h] = 4.0  # dinner
    hour_w[0] = 1.0       # midnight
    hour_w += 0.1
    hour_w /= hour_w.sum()

    base_date = pd.Timestamp("2024-01-01")
    rows: list[dict] = []

    for oid in range(1, NUM_ORDERS + 1):
        # Pick user
        uid = user_ids[int(rng.integers(0, NUM_USERS))]

        # 70% loyal (favorite restaurant), 30% exploration
        if rng.random() < 0.7:
            ridx = int(rng.choice(user_favs[uid]))
        else:
            ridx = int(rng.choice(n_rest, p=rest_pop))

        rest = restaurants[ridx]
        rid = rest["restaurant_id"]
        cuisine = rest["cuisine"]
        by_cat = menu_by_cat[rid]
        rw = boost_weights[rid]

        # Pick meal pattern
        pidx = int(rng.choice(len(MEAL_PATTERNS), p=pattern_probs))
        _, pattern = MEAL_PATTERNS[pidx]

        selected: list[dict] = []
        has_main = False

        for cat, count_spec in pattern.items():
            if isinstance(count_spec, tuple):
                n_items = int(rng.integers(count_spec[0], count_spec[1] + 1))
            else:
                n_items = int(count_spec)

            pool = by_cat.get(cat, [])
            if not pool:
                continue

            # Use boosted weights only when main_course already picked
            if has_main and cat in PAIRING_BOOSTS.get(cuisine, {}):
                weights = rw[cat]
            else:
                weights = np.ones(len(pool)) / len(pool)

            n_pick = min(n_items, len(pool))
            if n_pick <= 0:
                continue

            chosen = rng.choice(len(pool), size=n_pick, replace=False, p=weights)
            for ci in chosen:
                selected.append(pool[int(ci)])

            if cat == "main_course":
                has_main = True

        if not selected:
            continue

        # Timestamp
        day_offset = int(rng.integers(0, 365))
        hour = int(rng.choice(24, p=hour_w))
        minute = int(rng.integers(0, 60))
        order_time = base_date + pd.Timedelta(days=day_offset, hours=hour, minutes=minute)

        for pos, item in enumerate(selected, 1):
            price = item.get("rest_price", round(
                rng.uniform(item["min_price"], item["max_price"]), 0
            ))
            rows.append({
                "order_id": f"ord_{oid:06d}",
                "user_id": uid,
                "restaurant_id": rid,
                "order_time": order_time,
                "item_id": item["item_id"],
                "item_name": item["item_name"],
                "item_type": item["item_type"],
                "price": price,
                "quantity": 1,
                "line_total": price,
                "position": pos,
                "city": rest["city"],
                "cuisine": cuisine,
                "restaurant_name": rest["restaurant_name"],
            })

        if oid % 50_000 == 0:
            print(f"    Generated {oid:,}/{NUM_ORDERS:,} orders ({len(rows):,} rows)...")

    df = pd.DataFrame(rows)
    df = df.sort_values(["order_id", "position"]).reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════════════
# 4. Process Real Indian Takeaway (Mendeley Dataset)
# ══════════════════════════════════════════════════════════════════════════
def _find_file(base: Path, *possible_names: str) -> Path | None:
    if not base.exists():
        return None
    files = {f.name.lower(): f for f in base.iterdir() if f.is_file()}
    for name in possible_names:
        if name.lower() in files:
            return files[name.lower()]
    return None


def prepare_indian_takeaway(
    takeaway_dir: Path, indian_food_dir: Path | None,
) -> pd.DataFrame:
    """Transform Indian Takeaway data into mendeley_orders.csv format.

    Handles both restaurant-1 and restaurant-2.
    """
    print("\n[3/4] Processing Indian Takeaway data (Mendeley)...")

    csv_files = sorted(takeaway_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {takeaway_dir}")

    order_files = [f for f in csv_files
                   if "order" in f.name.lower() and "product" not in f.name.lower()]
    price_files = [f for f in csv_files
                   if "product" in f.name.lower() or "price" in f.name.lower()]

    # Load order files
    frames = []
    for f in (order_files or csv_files):
        try:
            df = pd.read_csv(f, encoding="latin-1")
            if len(df.columns) >= 3:
                tag = "r_takeaway_001" if "restaurant-1" in f.name.lower() else \
                      "r_takeaway_002" if "restaurant-2" in f.name.lower() else \
                      "r_takeaway_001"
                df["_restaurant_tag"] = tag
                frames.append(df)
                print(f"  Loaded {f.name}: {len(df):,} rows")
        except Exception as e:
            print(f"  Warning: Could not load {f.name}: {e}")

    # Price lookup
    price_lookup: dict[str, float] = {}
    for f in price_files:
        try:
            pdf = pd.read_csv(f, encoding="latin-1")
            pdf.columns = [c.strip().lower().replace(" ", "_") for c in pdf.columns]
            name_col = next((c for c in pdf.columns if "name" in c or "product" in c or "item" in c), None)
            pr_col = next((c for c in pdf.columns if "price" in c), None)
            if name_col and pr_col:
                for _, row in pdf.iterrows():
                    name = str(row[name_col]).strip().lower()
                    try:
                        pr = float(str(row[pr_col]).replace("£", "").replace(",", "").strip())
                        price_lookup[name] = pr
                    except ValueError:
                        pass
                print(f"  Price list {f.name}: {len(pdf):,} products")
        except Exception as e:
            print(f"  Warning: Could not load price file {f.name}: {e}")

    if not frames:
        raise FileNotFoundError(f"No valid order data in {takeaway_dir}")

    raw = pd.concat(frames, ignore_index=True)
    raw.columns = [c.strip().lower().replace(" ", "_") for c in raw.columns]
    print(f"  Combined: {len(raw):,} rows")

    # Column identification
    col_map: dict[str, str] = {}
    for target, candidates in {
        "order_id": ["order_number", "order_id", "ordernumber", "order_no"],
        "item_name": ["item_name", "item", "product_name", "product", "name"],
        "quantity": ["quantity", "qty"],
        "price": ["product_price", "price", "item_price", "unit_price", "amount"],
        "order_time": ["order_date", "date", "timestamp", "order_time", "datetime"],
    }.items():
        for c in candidates:
            if c in raw.columns:
                col_map[target] = c
                break

    if "order_id" not in col_map or "item_name" not in col_map:
        raise ValueError(f"Cannot find required columns in {list(raw.columns)}")

    # Indian Food 101 category lookup
    food_cat_map: dict[str, str] = {}
    if indian_food_dir and indian_food_dir.exists():
        food_file = _find_file(indian_food_dir, "indian_food.csv")
        if food_file:
            try:
                food101 = pd.read_csv(food_file)
                food101.columns = [c.strip().lower() for c in food101.columns]
                if "name" in food101.columns and "course" in food101.columns:
                    for _, row in food101.iterrows():
                        nm = str(row["name"]).strip().lower()
                        course = str(row["course"]).strip().lower()
                        food_cat_map[nm] = INDIAN_FOOD_CATEGORIES.get(course, "addon")
                    print(f"  Loaded {len(food_cat_map)} dish→category from Indian Food 101")
            except Exception as e:
                print(f"  Warning: Indian Food 101 load failed: {e}")

    def classify(item_name: str) -> str:
        nl = str(item_name).strip().lower()
        if nl in food_cat_map:
            return food_cat_map[nl]
        for fn, cat in food_cat_map.items():
            if fn in nl or nl in fn:
                return cat
        for kw, cat in KEYWORD_CATEGORIES.items():
            if kw in nl:
                return cat
        return "main_course"

    # Transform rows
    rng = np.random.default_rng(42)
    raw = raw.dropna(subset=[col_map["order_id"], col_map["item_name"]]).copy()

    order_ids = raw[col_map["order_id"]].unique()
    n_users = max(len(order_ids) // 8, 500)
    order_user_map = {oid: f"u_{abs(hash(str(oid))) % n_users:05d}" for oid in order_ids}

    result_rows = []
    order_col = col_map["order_id"]
    item_col = col_map["item_name"]
    qty_col = col_map.get("quantity")
    price_col = col_map.get("price")
    time_col = col_map.get("order_time")

    for _, row in raw.iterrows():
        oid = str(row[order_col])
        if oid.endswith(".0"):
            oid = oid[:-2]
        item_name = str(row[item_col])
        qty = max(1, int(row[qty_col]) if qty_col and pd.notna(row.get(qty_col)) else 1)

        # Price: lookup → inline → random, GBP → INR
        price = 0.0
        item_lower = item_name.strip().lower()
        if item_lower in price_lookup:
            price = round(price_lookup[item_lower] * 105, 2)
        elif price_col and pd.notna(row.get(price_col)):
            try:
                price = round(float(str(row[price_col]).replace("£", "").replace(",", "").strip()) * 105, 2)
            except ValueError:
                price = round(rng.uniform(80, 300), 2)
        else:
            price = round(rng.uniform(80, 300), 2)

        order_time = None
        if time_col and pd.notna(row.get(time_col)):
            try:
                order_time = pd.to_datetime(row[time_col], dayfirst=True)
            except Exception:
                pass
        if order_time is None:
            order_time = pd.Timestamp("2024-01-01") + pd.Timedelta(
                minutes=int(rng.integers(0, 525600))
            )

        item_id = f"t_{abs(hash(item_lower)) % 100000:05d}"
        rest_id = row.get("_restaurant_tag", "r_takeaway_001") \
            if "_restaurant_tag" in raw.columns else "r_takeaway_001"

        # Assign city based on restaurant
        city = "Delhi" if rest_id == "r_takeaway_001" else "Mumbai"
        cuisine = "North Indian" if rest_id == "r_takeaway_001" else "Mughlai"

        result_rows.append({
            "order_id": f"takeaway_{oid}",
            "user_id": order_user_map[row[order_col]],
            "restaurant_id": rest_id,
            "order_time": order_time,
            "item_id": item_id,
            "item_name": item_name.strip(),
            "item_type": classify(item_name),
            "price": price,
            "quantity": qty,
            "line_total": round(price * qty, 2),
            "position": 1,
            "city": city,
            "cuisine": cuisine,
            "restaurant_name": "Indian Takeaway" if rest_id == "r_takeaway_001"
                               else "Mughlai Kitchen",
        })

    result = pd.DataFrame(result_rows)
    result = result.sort_values(["order_id", "item_type"]).reset_index(drop=True)
    result["position"] = result.groupby("order_id").cumcount() + 1

    print(f"  Output: {result['order_id'].nunique():,} orders, "
          f"{result['user_id'].nunique():,} users, "
          f"{result['item_id'].nunique():,} items, {len(result):,} rows")
    print(f"  Categories:\n{result['item_type'].value_counts().to_string()}")
    return result


# ══════════════════════════════════════════════════════════════════════════
# 5. Generate Food Embeddings
# ══════════════════════════════════════════════════════════════════════════
def prepare_embeddings(
    primary_df: pd.DataFrame, takeaway_df: pd.DataFrame,
) -> pd.DataFrame:
    """8-dim embeddings: 4-dim cuisine/category + 4-dim co-purchase SVD."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    print("\n[4/4] Generating food-similarity embeddings...")

    combined = pd.concat([
        primary_df[["order_id", "item_id", "item_name", "item_type"]],
        takeaway_df[["order_id", "item_id", "item_name", "item_type"]],
    ], ignore_index=True) if not takeaway_df.empty else primary_df[
        ["order_id", "item_id", "item_name", "item_type"]
    ]

    item_ids = sorted(combined["item_id"].unique())
    print(f"  Total unique items: {len(item_ids):,}")

    # --- Part A: Category/cuisine features → SVD to 4 dims ---
    # Build a text descriptor per item: "category cuisine1 cuisine2 ..."
    item_meta: dict[str, str] = {}
    # From primary (synthetic) → items already have cuisine info
    item_cuisine_map: dict[str, set] = {}
    for _, row in primary_df.drop_duplicates("item_id")[
        ["item_id", "item_type", "item_name"]
    ].iterrows():
        iid = str(row["item_id"])
        cat = str(row["item_type"])
        # Find which cuisines this item belongs to from the DISHES dict
        cuisines = set()
        for cuisine, categories in DISHES.items():
            if cuisine == "_default":
                continue
            for c, dishes in categories.items():
                for d_name, _, _ in dishes:
                    if d_name == row["item_name"]:
                        cuisines.add(cuisine.lower().replace(" ", "_"))
        item_cuisine_map[iid] = cuisines
        item_meta[iid] = f"{cat} " + " ".join(cuisines) if cuisines else cat

    # Fill takeaway items
    for _, row in combined.drop_duplicates("item_id")[["item_id", "item_type"]].iterrows():
        iid = str(row["item_id"])
        if iid not in item_meta:
            item_meta[iid] = str(row["item_type"])

    meta_texts = [item_meta.get(iid, "addon") for iid in item_ids]
    tfidf_meta = TfidfVectorizer(max_features=50)
    meta_matrix = tfidf_meta.fit_transform(meta_texts)

    n_meta_dims = min(4, meta_matrix.shape[1] - 1) if meta_matrix.shape[1] > 1 else 1
    svd_meta = TruncatedSVD(n_components=n_meta_dims, random_state=42)
    meta_emb = svd_meta.fit_transform(meta_matrix)
    print(f"  Cuisine/category SVD ({n_meta_dims}d) variance: "
          f"{svd_meta.explained_variance_ratio_.sum():.3f}")

    # --- Part B: Co-purchase basket SVD → remaining dims ---
    co_dims = 8 - n_meta_dims
    baskets = combined.groupby("order_id")["item_id"].apply(
        lambda x: " ".join(x)
    ).reset_index()
    baskets.columns = ["order_id", "item_text"]

    tfidf_basket = TfidfVectorizer(max_features=10000, token_pattern=r"[a-z]_\d+")
    basket_matrix = tfidf_basket.fit_transform(baskets["item_text"])

    n_co_dims = min(co_dims, basket_matrix.shape[1] - 1) if basket_matrix.shape[1] > 1 else 1
    svd_basket = TruncatedSVD(n_components=n_co_dims, random_state=42)
    basket_vecs = svd_basket.fit_transform(basket_matrix)
    print(f"  Co-purchase SVD ({n_co_dims}d) variance: "
          f"{svd_basket.explained_variance_ratio_.sum():.3f}")

    # Average basket embeddings per item
    item_basket_map = combined.groupby("item_id")["order_id"].apply(set).to_dict()
    order_to_idx = {oid: i for i, oid in enumerate(baskets["order_id"])}

    co_emb = np.zeros((len(item_ids), n_co_dims))
    for idx, iid in enumerate(item_ids):
        basket_oids = item_basket_map.get(iid, set())
        idxs = [order_to_idx[oid] for oid in basket_oids if oid in order_to_idx]
        if idxs:
            co_emb[idx] = basket_vecs[idxs].mean(axis=0)[:n_co_dims]

    # Combine & pad to 8 dims
    full_emb = np.hstack([meta_emb, co_emb])
    if full_emb.shape[1] < 8:
        pad = np.zeros((full_emb.shape[0], 8 - full_emb.shape[1]))
        full_emb = np.hstack([full_emb, pad])
    full_emb = full_emb[:, :8]

    # L2 normalize
    norms = np.linalg.norm(full_emb, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    full_emb = full_emb / norms

    result = pd.DataFrame(full_emb, columns=[f"emb_{i}" for i in range(8)])
    result.insert(0, "item_id", item_ids)
    print(f"  Output: {len(result):,} item embeddings × 8 dims")
    return result


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("CSAO Indian Food Data Generator")
    print("  Replacing Instacart grocery data with authentic Indian food")
    print("=" * 70)

    rng = np.random.default_rng(42)

    # 1. Build catalog
    print("\n[1/4] Building Indian food catalog from DISHES dict...")
    catalog = build_item_catalog()
    cats = {}
    for item in catalog.values():
        cats[item["item_type"]] = cats.get(item["item_type"], 0) + 1
    print(f"  Catalog: {len(catalog)} unique dishes — {cats}")

    # 2. Build restaurants
    print(f"\n[2/4] Building {NUM_RESTAURANTS} restaurants with cuisine menus...")
    restaurants, menus = build_restaurants(catalog, rng)
    cuisines_created = {}
    for r in restaurants:
        c = r["cuisine"]
        cuisines_created[c] = cuisines_created.get(c, 0) + 1
    print(f"  Created {len(restaurants)} restaurants — {cuisines_created}")
    avg_menu = np.mean([len(m) for m in menus.values()])
    print(f"  Average menu size: {avg_menu:.1f} items")

    # 3. Generate synthetic orders
    print(f"\n[3/4] Generating {NUM_ORDERS:,} synthetic orders...")
    primary_df = generate_orders(restaurants, menus, rng)
    print(f"  Output: {primary_df['order_id'].nunique():,} orders, "
          f"{primary_df['user_id'].nunique():,} users, "
          f"{primary_df['item_id'].nunique():,} items, "
          f"{len(primary_df):,} rows")
    print(f"  Category distribution:\n"
          f"  {primary_df['item_type'].value_counts().to_dict()}")

    # 4. Process Indian Takeaway (if available)
    takeaway_dir = DOWNLOAD_DIR / "indian_takeaway"
    indian_food_dir = DOWNLOAD_DIR / "indian_food_101"
    if takeaway_dir.exists():
        takeaway_df = prepare_indian_takeaway(
            takeaway_dir,
            indian_food_dir if indian_food_dir.exists() else None,
        )
    else:
        print("\n  Indian Takeaway downloads not found — skipping Mendeley data")
        takeaway_df = pd.DataFrame(columns=[
            "order_id", "user_id", "restaurant_id", "order_time", "item_id",
            "item_name", "item_type", "price", "quantity", "line_total",
            "position", "city", "cuisine", "restaurant_name",
        ])

    # 5. Generate embeddings
    embeddings_df = prepare_embeddings(primary_df, takeaway_df)

    # 6. Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    out_primary = OUTPUT_DIR / "restaurant_orders.csv"
    out_mendeley = OUTPUT_DIR / "mendeley_orders.csv"
    out_embeddings = OUTPUT_DIR / "recipe_embeddings.csv"

    primary_df.to_csv(out_primary, index=False)
    if not takeaway_df.empty:
        takeaway_df.to_csv(out_mendeley, index=False)
    else:
        # Write empty CSV with headers so downstream doesn't break
        pd.DataFrame(columns=primary_df.columns).to_csv(out_mendeley, index=False)
    embeddings_df.to_csv(out_embeddings, index=False)

    print("\n" + "=" * 70)
    print("SUCCESS — Indian food data generated:")
    print(f"  {out_primary} ({len(primary_df):,} rows)")
    print(f"  {out_mendeley} ({len(takeaway_df):,} rows)")
    print(f"  {out_embeddings} ({len(embeddings_df):,} rows)")
    print("\nNext steps:")
    print("  python scripts/build_unified_data.py")
    print("  python scripts/build_features.py")
    print("  python scripts/train_ranker.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
