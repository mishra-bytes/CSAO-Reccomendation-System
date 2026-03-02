"""Test with a synthetic restaurant (proper categories)."""
import json
import urllib.request

r = urllib.request.urlopen("http://localhost:8000/api/restaurants")
restaurants = json.loads(r.read())

# Find a synthetic restaurant (not takeaway)
for rest in restaurants:
    if "Takeaway" not in rest["name"] and rest["item_count"] >= 20:
        break

print(f"Restaurant: {rest['name']} ({rest['cuisine']}, {rest['city']})")
print(f"  Items: {rest['item_count']}")

r2 = urllib.request.urlopen(f"http://localhost:8000/api/restaurant/{rest['id']}/items")
items = json.loads(r2.read())
cats = {}
for i in items:
    cats[i["item_category"]] = cats.get(i["item_category"], 0) + 1
print(f"  Categories: {cats}")

# Pick a main_course for the cart
mains = [i for i in items if i["item_category"] == "main_course"]
if mains:
    cart = [mains[0]["item_id"]]
    print(f"\nCart: {mains[0]['item_name']} (main_course, Rs{mains[0]['item_price']})")

    req_data = json.dumps({
        "user_id": "u_00100",
        "restaurant_id": rest["id"],
        "cart_item_ids": cart,
        "top_n": 8,
    }).encode()
    req3 = urllib.request.Request(
        "http://localhost:8000/api/recommend",
        data=req_data,
        headers={"Content-Type": "application/json"},
    )
    r3 = urllib.request.urlopen(req3, timeout=30)
    resp = json.loads(r3.read())

    print(f"\nRecommendations ({len(resp.get('recommendations', []))}):")
    print("-" * 110)
    cats_seen = set()
    for rec in resp.get("recommendations", []):
        name = rec.get("item_name", "?")
        cat = rec.get("item_category", "?")
        price = rec.get("item_price", 0)
        score = rec.get("rank_score", 0)
        expl = rec.get("explanation", "")
        tags = rec.get("reason_tags", [])
        cats_seen.add(cat)
        print(f"  {name:30s} | {cat:12s} | Rs{price:>6.0f} | Score: {score:.4f} | {expl[:70]}")

    print("-" * 110)
    print(f"Category diversity: {len(cats_seen)} categories: {cats_seen}")
    print(f"Latency: {resp.get('total_ms', 0):.0f}ms")

# Also test with 2 items in cart (main + starter)
print("\n\n=== Test 2: Two items in cart ===")
starters = [i for i in items if i["item_category"] == "starter"]
if mains and starters:
    cart2 = [mains[0]["item_id"], starters[0]["item_id"]]
    print(f"Cart: {mains[0]['item_name']} + {starters[0]['item_name']}")
    req_data2 = json.dumps({
        "user_id": "u_00100",
        "restaurant_id": rest["id"],
        "cart_item_ids": cart2,
        "top_n": 8,
    }).encode()
    req4 = urllib.request.Request(
        "http://localhost:8000/api/recommend",
        data=req_data2,
        headers={"Content-Type": "application/json"},
    )
    r4 = urllib.request.urlopen(req4, timeout=30)
    resp2 = json.loads(r4.read())
    print(f"\nRecommendations ({len(resp2.get('recommendations', []))}):")
    cats_seen2 = set()
    for rec in resp2.get("recommendations", []):
        name = rec.get("item_name", "?")
        cat = rec.get("item_category", "?")
        score = rec.get("rank_score", 0)
        expl = rec.get("explanation", "")
        cats_seen2.add(cat)
        print(f"  {name:30s} | {cat:12s} | Score: {score:.4f} | {expl[:70]}")
    print(f"Category diversity: {len(cats_seen2)} categories: {cats_seen2}")
