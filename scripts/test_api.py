"""Quick test: verify recommendation API returns meaningful scores."""
import json
import urllib.request

# Get restaurants
r = urllib.request.urlopen("http://localhost:8000/api/restaurants")
restaurants = json.loads(r.read())
rest = restaurants[0]
print(f"Restaurant: {rest['name']} ({rest['cuisine']}, {rest['city']})")
print(f"  Items: {rest['item_count']}")

# Get restaurant items
r2 = urllib.request.urlopen(f"http://localhost:8000/api/restaurant/{rest['id']}/items")
items = json.loads(r2.read())
mains = [i for i in items if i["item_category"] == "main_course"]
addons = [i for i in items if i["item_category"] == "addon"]
beverages = [i for i in items if i["item_category"] == "beverage"]
print(f"  Menu: {len(mains)} mains, {len(addons)} addons, {len(beverages)} beverages")

if mains:
    cart = [mains[0]["item_id"]]
    print(f"\nCart: {mains[0]['item_name']} (main_course, Rs{mains[0]['item_price']})")

    # Get recommendations
    req_data = json.dumps({
        "user_id": "u_00001",
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
    print("-" * 100)
    cats_seen = set()
    for rec in resp.get("recommendations", []):
        name = rec.get("item_name", rec.get("item_id", "?"))
        cat = rec.get("item_category", "?")
        price = rec.get("item_price", 0)
        score = rec.get("rank_score", 0)
        expl = rec.get("explanation", "")
        tags = rec.get("reason_tags", [])
        cats_seen.add(cat)
        print(f"  {name:30s} | {cat:12s} | Rs{price:>6.0f} | Score: {score:.4f} | {expl}")
    
    print("-" * 100)
    print(f"Category diversity: {len(cats_seen)} categories: {cats_seen}")
    print(f"Latency: {resp.get('total_ms', 0):.0f}ms")
    print(f"Cart value: Rs{resp.get('cart_value', 0)}")
    print(f"Potential AOV increase: Rs{resp.get('potential_aov_increase', 0)}")
