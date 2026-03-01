"""
prepare_real_data.py
====================
Transforms downloaded Kaggle datasets into the pipeline's expected format:

    data/raw/restaurant_orders.csv   ← Instacart Market Basket Analysis
    data/raw/mendeley_orders.csv     ← Indian Takeaway Orders
    data/raw/recipe_embeddings.csv   ← Food.com ingredient embeddings + co-purchase SVD

Prerequisite folder structure (after Kaggle downloads & extraction):

    data/raw/downloads/
    ├── instacart/
    │   ├── orders.csv
    │   ├── order_products__prior.csv
    │   ├── order_products__train.csv
    │   ├── products.csv
    │   ├── departments.csv
    │   └── aisles.csv
    ├── indian_takeaway/
    │   ├── restaurant-1-orders.csv
    │   ├── restaurant-1-products-price.csv
    │   ├── restaurant-2-orders.csv
    │   └── restaurant-2-products-price.csv
    ├── zomato_bangalore/
    │   └── zomato.csv
    ├── indian_food_101/
    │   └── indian_food.csv
    └── foodcom/
        ├── RAW_recipes.csv
        └── RAW_interactions.csv

Usage:
    python scripts/prepare_real_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DOWNLOAD_DIR = ROOT / "data" / "raw" / "downloads"
OUTPUT_DIR = ROOT / "data" / "raw"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INSTACART_MAX_ORDERS = 200_000       # cap for manageable size
INDIAN_FOOD_CATEGORIES = {
    # Indian Food 101 'course' → our item_type
    "main course": "main_course",
    "dessert": "dessert",
    "starter": "starter",
    "snack": "starter",
    "drink": "beverage",
    "side dish": "addon",
    "one dish meal": "main_course",
}

# Instacart department → food delivery item_type mapping
DEPT_TO_ITEM_TYPE = {
    "produce": "addon",
    "dairy eggs": "addon",
    "snacks": "starter",
    "beverages": "beverage",
    "frozen": "main_course",
    "bakery": "dessert",
    "canned goods": "addon",
    "deli": "main_course",
    "dry goods pasta": "main_course",
    "breakfast": "main_course",
    "meat seafood": "main_course",
    "pantry": "addon",
    "babies": "addon",
    "international": "main_course",
    "alcohol": "beverage",
    "pets": "addon",
    "household": "addon",
    "personal care": "addon",
    "other": "addon",
    "missing": "addon",
    "bulk": "addon",
}

# Cities and cuisines to assign to synthetic restaurants
CITIES = ["Delhi", "Mumbai", "Bangalore", "Hyderabad", "Pune", "Chennai", "Kolkata"]
CUISINES = ["North Indian", "South Indian", "Chinese", "Biryani", "Italian",
            "Street Food", "Bengali", "Mughlai"]


def _find_file(base: Path, *possible_names: str) -> Path | None:
    """Find a file case-insensitively in a directory."""
    if not base.exists():
        return None
    files = {f.name.lower(): f for f in base.iterdir() if f.is_file()}
    for name in possible_names:
        if name.lower() in files:
            return files[name.lower()]
    return None


def _find_csv_files(base: Path) -> list[Path]:
    """Find all CSV files in a directory."""
    if not base.exists():
        return []
    return sorted(base.glob("*.csv"))


# ===========================================================================
# 1. INSTACART → restaurant_orders.csv
# ===========================================================================

def prepare_instacart(instacart_dir: Path, zomato_dir: Path | None) -> pd.DataFrame:
    """Transform Instacart data into restaurant_orders.csv format."""
    print("\n[1/3] Processing Instacart Market Basket data...")

    # Load Instacart files
    orders_file = _find_file(instacart_dir, "orders.csv")
    prior_file = _find_file(instacart_dir, "order_products__prior.csv")
    train_file = _find_file(instacart_dir, "order_products__train.csv")
    products_file = _find_file(instacart_dir, "products.csv")
    departments_file = _find_file(instacart_dir, "departments.csv")
    aisles_file = _find_file(instacart_dir, "aisles.csv")

    if not all([orders_file, products_file, departments_file]):
        raise FileNotFoundError(
            f"Missing Instacart files in {instacart_dir}. "
            "Expected: orders.csv, products.csv, departments.csv, "
            "and order_products__prior.csv or order_products__train.csv"
        )

    orders = pd.read_csv(orders_file)
    products = pd.read_csv(products_file)
    departments = pd.read_csv(departments_file)

    # Merge products with departments
    products = products.merge(departments, on="department_id", how="left")
    if aisles_file:
        aisles = pd.read_csv(aisles_file)
        products = products.merge(aisles, on="aisle_id", how="left")

    # Load order-product mappings (combine prior + train)
    frames = []
    if prior_file:
        frames.append(pd.read_csv(prior_file))
    if train_file:
        frames.append(pd.read_csv(train_file))
    if not frames:
        raise FileNotFoundError("Need order_products__prior.csv or order_products__train.csv")
    order_products = pd.concat(frames, ignore_index=True)

    print(f"  Raw: {len(orders):,} orders, {len(order_products):,} order-product rows, "
          f"{len(products):,} products")

    # Cap orders for manageability
    if len(orders) > INSTACART_MAX_ORDERS:
        sampled_orders = orders.sample(n=INSTACART_MAX_ORDERS, random_state=42)
    else:
        sampled_orders = orders
    order_products = order_products[order_products["order_id"].isin(sampled_orders["order_id"])]

    # Merge to get full detail rows
    merged = order_products.merge(sampled_orders, on="order_id", how="inner")
    merged = merged.merge(products, on="product_id", how="left")

    print(f"  After sampling: {merged['order_id'].nunique():,} orders, {len(merged):,} rows")

    # --- Build timestamps from relative Instacart timing ---
    # Instacart has: order_dow (0-6), order_hour_of_day (0-23), days_since_prior_order
    rng = np.random.default_rng(42)
    base_date = pd.Timestamp("2025-01-01")

    # Create a per-user cumulative day offset from days_since_prior_order
    merged = merged.sort_values(["user_id", "order_number"])
    merged["days_since_prior_order"] = merged["days_since_prior_order"].fillna(0)

    # Build per-order timestamp (one per unique order)
    order_ts = merged.groupby("order_id").first()[
        ["user_id", "order_number", "order_dow", "order_hour_of_day", "days_since_prior_order"]
    ].reset_index()

    # Cumulative days per user
    order_ts = order_ts.sort_values(["user_id", "order_number"])
    order_ts["cum_days"] = order_ts.groupby("user_id")["days_since_prior_order"].cumsum()
    order_ts["order_time"] = base_date + pd.to_timedelta(order_ts["cum_days"], unit="D") + \
                             pd.to_timedelta(order_ts["order_hour_of_day"], unit="h") + \
                             pd.to_timedelta(rng.integers(0, 60, size=len(order_ts)), unit="m")

    merged = merged.drop(columns=["order_time"], errors="ignore")
    merged = merged.merge(order_ts[["order_id", "order_time"]], on="order_id", how="left")

    # --- Map departments to item_type ---
    merged["item_type"] = merged["department"].str.lower().map(DEPT_TO_ITEM_TYPE).fillna("addon")

    # --- Assign prices from realistic distributions ---
    # Use Zomato data for price calibration if available
    price_by_type = {
        "main_course": (180, 60),   # mean, std
        "starter": (120, 40),
        "beverage": (80, 30),
        "dessert": (130, 45),
        "addon": (60, 25),
    }
    if zomato_dir and zomato_dir.exists():
        zomato_file = _find_file(zomato_dir, "zomato.csv")
        if zomato_file:
            try:
                zomato = pd.read_csv(zomato_file, encoding="latin-1")
                if "approx_cost(for two people)" in zomato.columns:
                    costs = pd.to_numeric(
                        zomato["approx_cost(for two people)"].astype(str).str.replace(",", ""),
                        errors="coerce"
                    ).dropna()
                    median_cost_for_two = costs.median()
                    # Per-item ≈ cost_for_two / 3 items avg
                    scale = median_cost_for_two / 3.0
                    price_by_type = {
                        "main_course": (scale * 1.2, scale * 0.3),
                        "starter": (scale * 0.8, scale * 0.2),
                        "beverage": (scale * 0.5, scale * 0.15),
                        "dessert": (scale * 0.9, scale * 0.25),
                        "addon": (scale * 0.4, scale * 0.15),
                    }
                    print(f"  Price calibration from Zomato: median cost_for_two = ₹{median_cost_for_two:.0f}")
            except Exception as e:
                print(f"  Warning: Could not read Zomato data for price calibration: {e}")

    prices = np.zeros(len(merged))
    for itype, (mean, std) in price_by_type.items():
        mask = merged["item_type"] == itype
        n = mask.sum()
        if n > 0:
            prices[mask.values] = np.clip(rng.normal(mean, std, size=n), 20, 800).round(2)
    merged["price"] = prices

    # --- Assign restaurants by clustering products ---
    # Group products into ~500 synthetic restaurants based on their aisles/departments
    n_restaurants = 500
    product_rest_map = {}
    unique_products = merged[["product_id", "department", "item_type"]].drop_duplicates()

    # Cluster: products in same department go to same restaurants
    for dept, group in unique_products.groupby("department"):
        product_ids = group["product_id"].tolist()
        # Split into restaurant-sized chunks (20-80 items per restaurant)
        restaurant_size = rng.integers(20, 80)
        for i in range(0, len(product_ids), restaurant_size):
            rest_id = f"r_{len(product_rest_map) // restaurant_size + 1:04d}"
            for pid in product_ids[i:i + restaurant_size]:
                product_rest_map[pid] = rest_id

    # Cap restaurant IDs to n_restaurants
    unique_rests = list(set(product_rest_map.values()))
    if len(unique_rests) > n_restaurants:
        rest_remap = {r: unique_rests[i % n_restaurants] for i, r in enumerate(unique_rests)}
        product_rest_map = {pid: rest_remap[rid] for pid, rid in product_rest_map.items()}

    merged["restaurant_id"] = merged["product_id"].map(product_rest_map).fillna("r_0001")

    # Assign city and cuisine to each restaurant
    unique_rests = merged["restaurant_id"].unique()
    rest_city = {r: rng.choice(CITIES) for r in unique_rests}
    rest_cuisine = {r: rng.choice(CUISINES) for r in unique_rests}
    merged["city"] = merged["restaurant_id"].map(rest_city)
    merged["cuisine"] = merged["restaurant_id"].map(rest_cuisine)
    merged["restaurant_name"] = "Restaurant " + merged["restaurant_id"]

    # --- Build final output ---
    result = pd.DataFrame({
        "order_id": merged["order_id"].astype(str),
        "user_id": "u_" + merged["user_id"].astype(str).str.zfill(5),
        "restaurant_id": merged["restaurant_id"],
        "order_time": merged["order_time"],
        "item_id": "i_" + merged["product_id"].astype(str).str.zfill(5),
        "item_name": merged["product_name"],
        "item_type": merged["item_type"],
        "price": merged["price"],
        "quantity": 1,  # Instacart doesn't have quantity per product in this dataset
        "line_total": merged["price"],
        "position": merged["add_to_cart_order"],
        "city": merged["city"],
        "cuisine": merged["cuisine"],
        "restaurant_name": merged["restaurant_name"],
    })

    result = result.dropna(subset=["order_id", "user_id", "item_id", "order_time"])
    result = result.sort_values(["order_id", "position"]).reset_index(drop=True)

    print(f"  Output: {result['order_id'].nunique():,} orders, {result['user_id'].nunique():,} users, "
          f"{result['item_id'].nunique():,} items, {len(result):,} rows")

    return result


# ===========================================================================
# 2. INDIAN TAKEAWAY → mendeley_orders.csv
# ===========================================================================

def prepare_indian_takeaway(takeaway_dir: Path, indian_food_dir: Path | None) -> pd.DataFrame:
    """Transform Indian Takeaway data into mendeley_orders.csv format.

    Handles the known file structure:
      - restaurant-1-orders.csv, restaurant-2-orders.csv  (order rows)
      - restaurant-1-products-price.csv, restaurant-2-products-price.csv  (price lookups)
    """
    print("\n[2/4] Processing Indian Takeaway data...")

    csv_files = _find_csv_files(takeaway_dir)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {takeaway_dir}")

    # Separate order files from product-price files
    order_files = [f for f in csv_files if "order" in f.name.lower() and "product" not in f.name.lower()]
    price_files = [f for f in csv_files if "product" in f.name.lower() or "price" in f.name.lower()]

    # Load order files
    frames = []
    for f in (order_files if order_files else csv_files):
        try:
            df = pd.read_csv(f, encoding="latin-1")
            if len(df.columns) >= 3:
                # Tag which restaurant this came from
                if "restaurant-1" in f.name.lower():
                    df["_restaurant_tag"] = "r_takeaway_001"
                elif "restaurant-2" in f.name.lower():
                    df["_restaurant_tag"] = "r_takeaway_002"
                else:
                    df["_restaurant_tag"] = "r_takeaway_001"
                frames.append(df)
                print(f"  Loaded {f.name}: {len(df):,} rows")
        except Exception as e:
            print(f"  Warning: Could not load {f.name}: {e}")

    # Load product-price lookup files
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
                print(f"  Loaded price list {f.name}: {len(pdf):,} products")
        except Exception as e:
            print(f"  Warning: Could not load price file {f.name}: {e}")
    frames = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, encoding="latin-1")
            if len(df.columns) >= 3:  # skip tiny/metadata files
                frames.append(df)
                print(f"  Loaded {f.name}: {len(df):,} rows")
        except Exception as e:
            print(f"  Warning: Could not load {f.name}: {e}")

    if not frames:
        raise FileNotFoundError(f"No valid CSV data found in {takeaway_dir}")

    raw = pd.concat(frames, ignore_index=True)
    raw.columns = [c.strip().lower().replace(" ", "_") for c in raw.columns]

    print(f"  Combined: {len(raw):,} rows, columns: {list(raw.columns)}")

    # --- Identify columns (the dataset has varied column names across versions) ---
    col_map = {}

    # Order ID
    for candidate in ["order_number", "order_id", "ordernumber", "order_no"]:
        if candidate in raw.columns:
            col_map["order_id"] = candidate
            break

    # Item name
    for candidate in ["item_name", "item", "product_name", "product", "name"]:
        if candidate in raw.columns:
            col_map["item_name"] = candidate
            break

    # Quantity
    for candidate in ["quantity", "qty"]:
        if candidate in raw.columns:
            col_map["quantity"] = candidate
            break

    # Price
    for candidate in ["product_price", "price", "item_price", "unit_price", "amount"]:
        if candidate in raw.columns:
            col_map["price"] = candidate
            break

    # Order date/time
    for candidate in ["order_date", "date", "timestamp", "order_time", "datetime"]:
        if candidate in raw.columns:
            col_map["order_time"] = candidate
            break

    print(f"  Column mapping: {col_map}")

    if "order_id" not in col_map or "item_name" not in col_map:
        raise ValueError(
            f"Cannot find required columns. Available: {list(raw.columns)}. "
            "Need at least order_id and item_name equivalents."
        )

    # --- Build category lookup from Indian Food 101 ---
    food_category_map: dict[str, str] = {}
    if indian_food_dir and indian_food_dir.exists():
        food_file = _find_file(indian_food_dir, "indian_food.csv")
        if food_file:
            try:
                food101 = pd.read_csv(food_file)
                food101.columns = [c.strip().lower() for c in food101.columns]
                if "name" in food101.columns and "course" in food101.columns:
                    for _, row in food101.iterrows():
                        name = str(row["name"]).strip().lower()
                        course = str(row["course"]).strip().lower()
                        mapped = INDIAN_FOOD_CATEGORIES.get(course, "addon")
                        food_category_map[name] = mapped
                    print(f"  Loaded {len(food_category_map)} dish→category mappings from Indian Food 101")
            except Exception as e:
                print(f"  Warning: Could not load Indian Food 101: {e}")

    # --- Common Indian food keyword → category fallback ---
    KEYWORD_CATEGORIES = {
        "rice": "main_course", "biryani": "main_course", "naan": "main_course",
        "roti": "main_course", "paratha": "main_course", "curry": "main_course",
        "dal": "main_course", "daal": "main_course", "paneer": "main_course",
        "chicken": "main_course", "lamb": "main_course", "mutton": "main_course",
        "fish": "main_course", "prawn": "main_course", "tikka": "main_course",
        "masala": "main_course", "korma": "main_course", "vindaloo": "main_course",
        "madras": "main_course", "jalfrezi": "main_course", "bhuna": "main_course",
        "dopiaza": "main_course", "dhansak": "main_course", "balti": "main_course",
        "pilau": "main_course", "pulao": "main_course", "tandoori": "main_course",
        "kebab": "starter", "samosa": "starter", "pakora": "starter",
        "bhaji": "starter", "poppadom": "starter", "papadum": "starter",
        "chutney": "addon", "pickle": "addon", "raita": "addon",
        "salad": "addon", "sauce": "addon", "dip": "addon",
        "lassi": "beverage", "cola": "beverage", "coke": "beverage",
        "pepsi": "beverage", "water": "beverage", "juice": "beverage",
        "beer": "beverage", "drink": "beverage", "tea": "beverage",
        "coffee": "beverage", "mango": "beverage", "lemonade": "beverage",
        "gulab": "dessert", "jamun": "dessert", "halwa": "dessert",
        "kheer": "dessert", "kulfi": "dessert", "ice_cream": "dessert",
        "jalebi": "dessert", "barfi": "dessert", "sweet": "dessert",
        "rasmalai": "dessert", "gajar": "dessert",
    }

    def classify_item(item_name: str) -> str:
        name_lower = str(item_name).strip().lower()
        # Try exact match from Indian Food 101
        if name_lower in food_category_map:
            return food_category_map[name_lower]
        # Try partial match from Indian Food 101
        for food_name, cat in food_category_map.items():
            if food_name in name_lower or name_lower in food_name:
                return cat
        # Keyword fallback
        for keyword, cat in KEYWORD_CATEGORIES.items():
            if keyword in name_lower:
                return cat
        return "main_course"  # default for Indian takeaway items

    # --- Transform ---
    rng = np.random.default_rng(42)

    result_rows = []
    order_col = col_map["order_id"]
    item_col = col_map["item_name"]
    qty_col = col_map.get("quantity")
    price_col = col_map.get("price")
    time_col = col_map.get("order_time")

    # Assign user_ids: cluster orders by time proximity
    # (same-day orders within a few hours → likely same user)
    order_ids = raw[order_col].unique()
    n_users = max(len(order_ids) // 8, 500)  # ~8 orders per user on average
    order_user_map = {oid: f"u_{(hash(str(oid)) % n_users):05d}" for oid in order_ids}

    for idx, row in raw.iterrows():
        oid = str(row[order_col])
        item_name = str(row[item_col])

        qty = int(row[qty_col]) if qty_col and pd.notna(row.get(qty_col)) else 1
        qty = max(1, qty)

        # Price: use product-price lookup first, then inline price, convert GBP → INR
        price = 0.0
        item_name_lower = item_name.strip().lower()
        if item_name_lower in price_lookup:
            price = round(price_lookup[item_name_lower] * 105, 2)  # GBP → INR
        elif price_col and pd.notna(row.get(price_col)):
            price_str = str(row[price_col]).replace("£", "").replace(",", "").strip()
            try:
                price_gbp = float(price_str)
                price = round(price_gbp * 105, 2)  # GBP → INR
            except ValueError:
                price = round(rng.uniform(80, 300), 2)
        else:
            price = round(rng.uniform(80, 300), 2)

        # Timestamp
        order_time = None
        if time_col and pd.notna(row.get(time_col)):
            try:
                order_time = pd.to_datetime(row[time_col])
            except Exception:
                pass
        if order_time is None:
            order_time = pd.Timestamp("2025-01-01") + pd.Timedelta(minutes=int(rng.integers(0, 525600)))

        item_type = classify_item(item_name)

        # Build standardized item_id from item_name
        item_id = "t_" + str(abs(hash(item_name.strip().lower())) % 100000).zfill(5)

        # Use the restaurant tag from source file
        rest_id = row.get("_restaurant_tag", "r_takeaway_001") if "_restaurant_tag" in raw.columns else "r_takeaway_001"

        result_rows.append({
            "order_id": f"takeaway_{oid}",
            "user_id": order_user_map[row[order_col]],
            "restaurant_id": rest_id,
            "order_time": order_time,
            "item_id": item_id,
            "item_name": item_name.strip(),
            "item_type": item_type,
            "price": price,
            "quantity": qty,
            "line_total": round(price * qty, 2),
            "position": 1,  # will be recomputed per order below
            "city": "Delhi",  # London→Delhi for Indian food context
            "cuisine": "North Indian",
            "restaurant_name": "Indian Takeaway",
        })

    result = pd.DataFrame(result_rows)

    # Assign positions within each order
    result = result.sort_values(["order_id", "item_type"]).reset_index(drop=True)
    result["position"] = result.groupby("order_id").cumcount() + 1

    print(f"  Output: {result['order_id'].nunique():,} orders, {result['user_id'].nunique():,} users, "
          f"{result['item_id'].nunique():,} items, {len(result):,} rows")
    print(f"  Category distribution:\n{result['item_type'].value_counts().to_string()}")

    return result


# ===========================================================================
# 3. RECIPE EMBEDDINGS
# ===========================================================================

def prepare_embeddings(
    instacart_data: pd.DataFrame,
    takeaway_data: pd.DataFrame,
    foodcom_dir: Path | None,
) -> pd.DataFrame:
    """Generate food similarity embeddings.

    Strategy:
      1. If Food.com RAW_recipes.csv is available → TF-IDF on ingredient lists
         gives real food-similarity embeddings (items sharing ingredients are close).
      2. Always augment with co-purchase SVD from Instacart+Takeaway baskets.
      3. Final embedding = concat(ingredient_emb[4d], copurchase_emb[4d]) → 8d.
    """
    print("\n[3/4] Generating recipe embeddings...")

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    n_dims = 8
    half = n_dims // 2  # 4 dims ingredient, 4 dims co-purchase

    # Combine both order sources for item list
    combined = pd.concat([
        instacart_data[["order_id", "item_id", "item_name"]],
        takeaway_data[["order_id", "item_id", "item_name"]],
    ], ignore_index=True)
    item_ids = sorted(combined["item_id"].unique())
    item_name_map = combined.drop_duplicates("item_id").set_index("item_id")["item_name"].to_dict()

    print(f"  Total unique items: {len(item_ids):,}")

    # ----- Part A: Food.com ingredient-based embeddings -----
    ingredient_emb = np.zeros((len(item_ids), half))
    foodcom_used = False

    if foodcom_dir and foodcom_dir.exists():
        recipes_file = _find_file(foodcom_dir, "RAW_recipes.csv", "PP_recipes.csv")
        if recipes_file:
            try:
                print(f"  Loading Food.com recipes from {recipes_file.name}...")
                recipes = pd.read_csv(recipes_file)
                recipes.columns = [c.strip().lower() for c in recipes.columns]

                # RAW_recipes.csv has: name, id, ingredients (as string list), tags, nutrition
                if "ingredients" in recipes.columns and "name" in recipes.columns:
                    recipes["ingredients_clean"] = (
                        recipes["ingredients"].astype(str)
                        .str.replace(r"[\[\]'\"]", "", regex=True)
                        .str.lower()
                    )

                    # Build TF-IDF on ingredient lists
                    tfidf_ingr = TfidfVectorizer(max_features=5000, stop_words="english")
                    ingr_matrix = tfidf_ingr.fit_transform(recipes["ingredients_clean"])
                    svd_ingr = TruncatedSVD(n_components=half, random_state=42)
                    recipe_vecs = svd_ingr.fit_transform(ingr_matrix)
                    print(f"  Food.com SVD explained variance: {svd_ingr.explained_variance_ratio_.sum():.3f}")

                    # Build recipe name → embedding lookup
                    recipe_name_to_vec = {}
                    for i, name in enumerate(recipes["name"].astype(str).str.lower()):
                        recipe_name_to_vec[name] = recipe_vecs[i]

                    # Match our items to Food.com recipes by fuzzy name overlap
                    matched = 0
                    for idx, item_id in enumerate(item_ids):
                        item_name = str(item_name_map.get(item_id, "")).lower().strip()
                        if not item_name:
                            continue
                        # Try exact match first
                        if item_name in recipe_name_to_vec:
                            ingredient_emb[idx] = recipe_name_to_vec[item_name]
                            matched += 1
                            continue
                        # Try substring match (item name in recipe name or vice versa)
                        best_score = 0
                        best_vec = None
                        item_words = set(item_name.split())
                        for rname, rvec in recipe_name_to_vec.items():
                            rwords = set(rname.split())
                            overlap = len(item_words & rwords)
                            if overlap > best_score and overlap >= 1:
                                best_score = overlap
                                best_vec = rvec
                        if best_vec is not None and best_score >= 2:
                            ingredient_emb[idx] = best_vec
                            matched += 1

                    foodcom_used = True
                    print(f"  Matched {matched:,}/{len(item_ids):,} items to Food.com recipes")
                else:
                    print(f"  Food.com file missing 'ingredients'/'name' columns, skipping")
            except Exception as e:
                print(f"  Warning: Could not process Food.com data: {e}")

    if not foodcom_used:
        print("  Food.com not available — using co-purchase only for all 8 dims")
        half = 0  # give all 8 dims to co-purchase

    # ----- Part B: Co-purchase basket embeddings -----
    copurchase_dims = n_dims - half if foodcom_used else n_dims

    baskets = combined.groupby("order_id")["item_id"].apply(lambda x: " ".join(x)).reset_index()
    baskets.columns = ["order_id", "item_text"]

    print(f"  Building co-purchase embeddings from {len(baskets):,} baskets...")

    tfidf = TfidfVectorizer(max_features=10000, token_pattern=r"[a-z]_\d+")
    tfidf_matrix = tfidf.fit_transform(baskets["item_text"])

    svd = TruncatedSVD(
        n_components=min(copurchase_dims, tfidf_matrix.shape[1] - 1),
        random_state=42,
    )
    basket_vecs = svd.fit_transform(tfidf_matrix)
    print(f"  Co-purchase SVD explained variance: {svd.explained_variance_ratio_.sum():.3f}")

    # Average basket embeddings per item
    item_basket_map = combined.groupby("item_id")["order_id"].apply(set).to_dict()
    order_to_idx = {oid: i for i, oid in enumerate(baskets["order_id"])}

    copurchase_emb = np.zeros((len(item_ids), copurchase_dims))
    for idx, item_id in enumerate(item_ids):
        basket_oids = item_basket_map.get(item_id, set())
        idxs = [order_to_idx[oid] for oid in basket_oids if oid in order_to_idx]
        if idxs:
            emb = basket_vecs[idxs].mean(axis=0)
            copurchase_emb[idx] = emb[:copurchase_dims]

    # ----- Combine & normalize -----
    if foodcom_used:
        emb_matrix = np.hstack([ingredient_emb, copurchase_emb])
    else:
        emb_matrix = copurchase_emb

    # Pad to exactly n_dims if needed
    if emb_matrix.shape[1] < n_dims:
        pad = np.zeros((emb_matrix.shape[0], n_dims - emb_matrix.shape[1]))
        emb_matrix = np.hstack([emb_matrix, pad])
    emb_matrix = emb_matrix[:, :n_dims]

    # L2 normalize
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    emb_matrix = emb_matrix / norms

    result = pd.DataFrame(emb_matrix, columns=[f"emb_{i}" for i in range(n_dims)])
    result.insert(0, "item_id", item_ids)

    src = "Food.com ingredients + co-purchase" if foodcom_used else "co-purchase only"
    print(f"  Output: {len(result):,} item embeddings x {n_dims} dims ({src})")
    return result


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 70)
    print("CSAO Real Data Preparation Pipeline")
    print("=" * 70)

    # --- Check prerequisites ---
    instacart_dir = DOWNLOAD_DIR / "instacart"
    takeaway_dir = DOWNLOAD_DIR / "indian_takeaway"
    zomato_dir = DOWNLOAD_DIR / "zomato_bangalore"
    indian_food_dir = DOWNLOAD_DIR / "indian_food_101"
    foodcom_dir = DOWNLOAD_DIR / "foodcom"

    missing = []
    if not instacart_dir.exists():
        missing.append(f"  {instacart_dir}")
    if not takeaway_dir.exists():
        missing.append(f"  {takeaway_dir}")

    if missing:
        print("\nERROR: Required dataset directories not found:")
        for m in missing:
            print(m)
        print(f"\nPlease download datasets and extract into: {DOWNLOAD_DIR}")
        print("\nExpected structure:")
        print("  data/raw/downloads/instacart/       ← Instacart csvs")
        print("  data/raw/downloads/indian_takeaway/  ← Takeaway csvs")
        print("  data/raw/downloads/foodcom/           ← (optional) RAW_recipes.csv")
        print("  data/raw/downloads/zomato_bangalore/ ← (optional) zomato.csv")
        print("  data/raw/downloads/indian_food_101/  ← (optional) indian_food.csv")
        sys.exit(1)

    # Optional dirs
    if not zomato_dir.exists():
        print(f"\nNote: {zomato_dir} not found — will use default price distributions")
        zomato_dir = None
    if not indian_food_dir.exists():
        print(f"\nNote: {indian_food_dir} not found — will use keyword-based categories")
        indian_food_dir = None
    if not foodcom_dir.exists():
        print(f"\nNote: {foodcom_dir} not found — will use co-purchase embeddings only")
        foodcom_dir = None

    # --- Process ---
    instacart_df = prepare_instacart(instacart_dir, zomato_dir)
    takeaway_df = prepare_indian_takeaway(takeaway_dir, indian_food_dir)
    embeddings_df = prepare_embeddings(instacart_df, takeaway_df, foodcom_dir)

    # --- Save ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    out_primary = OUTPUT_DIR / "restaurant_orders.csv"
    out_mendeley = OUTPUT_DIR / "mendeley_orders.csv"
    out_embeddings = OUTPUT_DIR / "recipe_embeddings.csv"

    instacart_df.to_csv(out_primary, index=False)
    takeaway_df.to_csv(out_mendeley, index=False)
    embeddings_df.to_csv(out_embeddings, index=False)

    print("\n" + "=" * 70)
    print("SUCCESS — Output files:")
    print(f"  {out_primary}  ({len(instacart_df):,} rows)")
    print(f"  {out_mendeley} ({len(takeaway_df):,} rows)")
    print(f"  {out_embeddings} ({len(embeddings_df):,} rows)")
    print("\nNext steps:")
    print("  python scripts/build_unified_data.py")
    print("  python scripts/build_features.py")
    print("  python scripts/train_ranker.py")
    print("  python scripts/run_offline_eval.py")
    print("  python scripts/simulate_serving.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
