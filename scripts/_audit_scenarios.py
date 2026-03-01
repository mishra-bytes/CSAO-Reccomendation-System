"""Audit script: run 3 distinct cart scenarios through the full CSAO pipeline."""
import sys, yaml
sys.path.insert(0, ".")
import pandas as pd
from collections import Counter
from features.complementarity import build_complementarity_lookup
from candidate_generation.candidate_generator import CandidateGenerator
from ranking.inference.ranker import CSAORanker

with open("configs/base.yaml") as f: base_cfg = yaml.safe_load(f)
with open("configs/ranking.yaml") as f: rank_cfg = yaml.safe_load(f)
config = {**base_cfg, **rank_cfg}

items = pd.read_parquet("data/processed/items.parquet")
orders = pd.read_parquet("data/processed/orders.parquet")
oi = pd.read_parquet("data/processed/order_items.parquet")
uf = pd.read_parquet("data/processed/features_user.parquet")
itf = pd.read_parquet("data/processed/features_item.parquet")
comp = pd.read_parquet("data/processed/features_complementarity.parquet")
cat_aff = pd.read_parquet("data/processed/features_category_affinity.parquet")

comp_lookup = build_complementarity_lookup(comp)
cg = CandidateGenerator(comp, cat_aff, items, orders, oi, config)
ranker = CSAORanker(
    model_path="models/lgbm_ranker.joblib",
    feature_columns_path="models/feature_columns.json",
    user_features=uf, item_features=itf, items=items,
    complementarity_lookup=comp_lookup,
)

item_cat = items.set_index("item_id")["item_category"].to_dict()
item_name = items.set_index("item_id")["item_name"].to_dict()
item_price = items.set_index("item_id")["item_price"].to_dict()

SEP = "=" * 60

def run_scenario(name, user_id, restaurant_id, cart_items):
    print(f"\n{SEP}")
    print(f"SCENARIO: {name}")
    print(SEP)
    print(f"User: {user_id}")
    print(f"Restaurant: {restaurant_id}")
    print(f"Cart ({len(cart_items)} items):")
    for ci in cart_items:
        print(f"  {ci}: {item_name.get(ci,'?')} | {item_cat.get(ci,'?')} | price={item_price.get(ci,0):.0f}")

    cart_cats = set(item_cat.get(ci, "unknown") for ci in cart_items)
    print(f"Cart categories: {cart_cats}")

    candidates = cg.generate(cart_items, restaurant_id)
    print(f"\nCandidates generated: {len(candidates)}")
    print("Top 5 candidates (pre-ranking):")
    for cid, score in candidates[:5]:
        print(f"  {cid}: {item_name.get(cid,'?')} | {item_cat.get(cid,'?')} | score={score:.4f}")

    ranked = ranker.rank(user_id, restaurant_id, cart_items, candidates, top_n=10)
    print(f"\nFinal Top-10 Ranked Output:")
    for i, r in enumerate(ranked, 1):
        iid = r["item_id"]
        cat = item_cat.get(iid, "?")
        nm = item_name.get(iid, "?")
        pr = item_price.get(iid, 0)
        print(f"  #{i}: {iid} | {nm} | {cat} | price={pr:.0f} | rank={r['rank_score']:.4f} | cand={r['candidate_score']:.4f}")

    ranked_cats = [item_cat.get(r["item_id"], "?") for r in ranked]
    cat_dist = Counter(ranked_cats)
    print(f"\nRecommendation category distribution: {dict(cat_dist)}")
    fills_gap = set(ranked_cats) - cart_cats
    print(f"New categories introduced: {fills_gap if fills_gap else 'none'}")


# === SCENARIO 1: Cold Start ===
run_scenario(
    "Cold Start User - 1 item in cart, unknown user",
    user_id="u_99999",
    restaurant_id="r_0001",
    cart_items=["i_008"],
)

# === SCENARIO 2: Large Cart / Repeat User ===
run_scenario(
    "Large Cart - Repeat user, 4 items (all categories filled)",
    user_id="u_00001",
    restaurant_id="r_0010",
    cart_items=["i_008", "i_003", "i_009", "i_011"],
)

# === SCENARIO 3: Incomplete Meal ===
run_scenario(
    "Incomplete Meal - Main+Starter only, missing beverage & dessert",
    user_id="u_00042",
    restaurant_id="r_0050",
    cart_items=["i_008", "i_010"],
)
