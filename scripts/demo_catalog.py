"""
Realistic Indian restaurant food catalog for the CSAO demo.

Maps generic Instacart grocery names → authentic restaurant dish names,
grouped by cuisine type. Keeps the underlying ML model/IDs intact.
"""
from __future__ import annotations

import hashlib
import random
from typing import Any

# ── Cuisine-aware dish names ──────────────────────────────────────────────────

DISHES: dict[str, dict[str, list[tuple[str, float, float]]]] = {
    # cuisine → category → [(dish_name, min_price, max_price), ...]
    "North Indian": {
        "main_course": [
            ("Butter Chicken", 280, 350), ("Dal Makhani", 220, 280),
            ("Paneer Tikka Masala", 260, 320), ("Rajma Chawal", 180, 230),
            ("Chole Bhature", 180, 220), ("Kadai Paneer", 240, 300),
            ("Shahi Paneer", 250, 310), ("Palak Paneer", 230, 280),
            ("Malai Kofta", 260, 320), ("Aloo Gobi", 180, 220),
            ("Paneer Butter Masala", 260, 320), ("Chicken Curry", 240, 300),
            ("Mutton Rogan Josh", 350, 420), ("Egg Curry", 180, 230),
            ("Jeera Rice", 140, 180), ("Veg Biryani", 200, 260),
            ("Mix Veg Curry", 180, 230), ("Mushroom Masala", 220, 270),
            ("Matar Paneer", 230, 280), ("Dum Aloo", 200, 250),
        ],
        "starter": [
            ("Paneer Tikka", 220, 280), ("Chicken Tikka", 260, 320),
            ("Veg Seekh Kebab", 200, 260), ("Hara Bhara Kebab", 180, 230),
            ("Dahi Kebab", 200, 250), ("Tandoori Chicken", 280, 350),
            ("Amritsari Fish Tikka", 300, 370), ("Malai Chaap", 220, 280),
            ("Aloo Tikki", 120, 160), ("Papdi Chaat", 130, 170),
            ("Samosa (2 pcs)", 80, 120), ("Onion Bhaji", 120, 160),
        ],
        "beverage": [
            ("Masala Chai", 50, 80), ("Sweet Lassi", 80, 120),
            ("Mango Lassi", 100, 140), ("Salted Lassi", 70, 100),
            ("Rose Sharbat", 80, 110), ("Jaljeera", 60, 90),
            ("Nimbu Pani", 50, 70), ("Buttermilk (Chaas)", 50, 70),
            ("Thandai", 100, 140), ("Badam Milk", 90, 130),
        ],
        "dessert": [
            ("Gulab Jamun (2 pcs)", 80, 120), ("Rasmalai (2 pcs)", 100, 140),
            ("Gajar Ka Halwa", 120, 160), ("Kheer", 100, 140),
            ("Jalebi", 80, 120), ("Rabri", 100, 140),
            ("Moong Dal Halwa", 120, 160), ("Kulfi", 80, 110),
        ],
        "addon": [
            ("Butter Naan", 50, 70), ("Garlic Naan", 60, 80),
            ("Tandoori Roti", 30, 50), ("Laccha Paratha", 50, 70),
            ("Raita (Boondi)", 50, 70), ("Papad", 30, 40),
            ("Green Salad", 60, 80), ("Pickle (Achar)", 30, 50),
            ("Plain Rice", 100, 140), ("Onion Rings", 60, 80),
            ("Extra Gravy", 40, 60), ("Missi Roti", 40, 60),
            ("Rumali Roti", 40, 60), ("Pudina Raita", 50, 70),
            ("Masala Papad", 50, 60),
        ],
    },
    "South Indian": {
        "main_course": [
            ("Masala Dosa", 140, 180), ("Ghee Roast Dosa", 160, 200),
            ("Mysore Masala Dosa", 160, 210), ("Rava Dosa", 150, 190),
            ("Set Dosa (3 pcs)", 130, 170), ("Idli Sambar (4 pcs)", 100, 140),
            ("Vada Sambar (2 pcs)", 100, 140), ("Uttapam", 140, 180),
            ("Curd Rice", 120, 160), ("Lemon Rice", 130, 170),
            ("Bisi Bele Bath", 140, 180), ("Pongal", 120, 160),
            ("Appam with Stew", 160, 210), ("Chicken Chettinad", 280, 340),
            ("Fish Curry (Kerala)", 260, 320), ("Sambar Rice", 130, 170),
        ],
        "starter": [
            ("Medu Vada (2 pcs)", 80, 120), ("Onion Pakoda", 100, 140),
            ("Paniyaram (6 pcs)", 110, 150), ("Chicken 65", 220, 280),
            ("Gobi Manchurian Dry", 160, 210), ("Pepper Chicken", 240, 300),
            ("Banana Chips", 60, 90), ("Murukku", 70, 100),
        ],
        "beverage": [
            ("Filter Coffee", 50, 80), ("Masala Chai", 40, 60),
            ("Buttermilk (Neer Mor)", 40, 60), ("Tender Coconut Water", 60, 90),
            ("Jigarthanda", 80, 120), ("Rose Milk", 60, 90),
            ("Mango Juice", 80, 110), ("Sugarcane Juice", 50, 70),
        ],
        "dessert": [
            ("Payasam", 80, 120), ("Mysore Pak", 70, 100),
            ("Kesari Bath", 70, 100), ("Double Ka Meetha", 90, 130),
            ("Banana Sheera", 70, 100), ("Rava Ladoo (3 pcs)", 80, 110),
        ],
        "addon": [
            ("Sambar (extra)", 40, 60), ("Coconut Chutney", 30, 50),
            ("Tomato Chutney", 30, 50), ("Gun Powder", 20, 40),
            ("Ghee (extra)", 30, 40), ("Pickle (Avakaya)", 30, 50),
            ("Curd (extra)", 40, 60), ("Papad", 20, 30),
            ("Onion Uttapam", 120, 160), ("Plain Dosa", 100, 130),
            ("Mini Idli (8 pcs)", 90, 120), ("Podi Idli", 100, 140),
        ],
    },
    "Biryani": {
        "main_course": [
            ("Hyderabadi Chicken Biryani", 280, 350), ("Mutton Biryani", 350, 420),
            ("Veg Dum Biryani", 220, 280), ("Egg Biryani", 200, 260),
            ("Prawns Biryani", 320, 400), ("Paneer Biryani", 240, 300),
            ("Lucknowi Chicken Biryani", 280, 350), ("Keema Biryani", 300, 370),
            ("Chicken Fried Rice", 200, 260), ("Veg Pulao", 180, 230),
            ("Mushroom Biryani", 220, 280), ("Fish Biryani", 300, 370),
            ("Chicken 65 Biryani", 280, 340), ("Double Masala Biryani", 300, 370),
        ],
        "starter": [
            ("Chicken 65", 200, 260), ("Apollo Fish", 260, 320),
            ("Paneer 65", 200, 250), ("Kebab Platter", 300, 380),
            ("Tandoori Prawns", 320, 400), ("Mutton Seekh Kebab", 280, 350),
            ("Crispy Corn", 140, 180), ("Mushroom Pepper Fry", 160, 210),
        ],
        "beverage": [
            ("Falooda", 120, 160), ("Sweet Lassi", 70, 100),
            ("Masala Chai", 40, 60), ("Rose Sharbat", 60, 90),
            ("Pepsi / Coke", 40, 60), ("Fresh Lime Soda", 60, 90),
            ("Mango Shake", 100, 140), ("Cold Coffee", 90, 130),
        ],
        "dessert": [
            ("Phirni", 80, 120), ("Shahi Tukda", 100, 140),
            ("Double Ka Meetha", 90, 130), ("Qubani Ka Meetha", 100, 140),
            ("Gulab Jamun (2 pcs)", 70, 100), ("Ice Cream (Scoop)", 60, 90),
        ],
        "addon": [
            ("Raita", 50, 70), ("Mirchi Ka Salan", 60, 80),
            ("Gutti Vankaya Curry", 80, 110), ("Bagara Baingan", 80, 110),
            ("Boiled Egg (2 pcs)", 40, 60), ("Rumali Roti", 40, 60),
            ("Extra Chicken Piece", 80, 120), ("Green Salad", 50, 70),
            ("Papad", 20, 30), ("Onion Raita", 50, 70),
            ("Dahi Chutney", 40, 50), ("Salan (extra)", 50, 60),
        ],
    },
    "Chinese": {
        "main_course": [
            ("Veg Hakka Noodles", 180, 230), ("Chicken Fried Rice", 200, 260),
            ("Schezwan Noodles", 190, 240), ("Manchurian Rice", 190, 240),
            ("Chilli Chicken with Rice", 240, 300), ("Dragon Chicken", 260, 320),
            ("Singapore Noodles", 200, 260), ("Triple Schezwan Rice", 220, 280),
            ("Kung Pao Chicken", 260, 320), ("Veg Chow Mein", 170, 220),
            ("Chicken Manchurian with Rice", 240, 300), ("Paneer Chilli Rice", 210, 260),
        ],
        "starter": [
            ("Veg Manchurian Dry", 160, 210), ("Chilli Paneer Dry", 180, 240),
            ("Honey Chilli Potato", 150, 200), ("Crispy Corn", 130, 170),
            ("Spring Rolls (4 pcs)", 140, 180), ("Chicken Momos (8 pcs)", 160, 210),
            ("Veg Momos (8 pcs)", 130, 170), ("Dim Sum Platter", 200, 260),
            ("Chilli Fish", 240, 300), ("Golden Fried Prawns", 260, 320),
        ],
        "beverage": [
            ("Iced Lemon Tea", 70, 100), ("Fresh Lime Soda", 60, 90),
            ("Cold Coffee", 90, 130), ("Green Tea", 50, 70),
            ("Mango Shake", 100, 140), ("Blue Lagoon Mocktail", 120, 160),
            ("Virgin Mojito", 120, 160), ("Pepsi / Coke", 40, 60),
        ],
        "dessert": [
            ("Fried Ice Cream", 120, 160), ("Chocolate Brownie", 130, 170),
            ("Date Pancake", 100, 140), ("Toffee Banana", 100, 140),
            ("Lychee Pudding", 90, 130), ("Mango Sticky Rice", 110, 150),
        ],
        "addon": [
            ("Schezwan Sauce (extra)", 30, 50), ("Chilli Oil", 30, 40),
            ("Steamed Rice", 80, 110), ("Fried Rice (small)", 100, 140),
            ("Wonton Soup", 80, 120), ("Hot & Sour Soup", 90, 130),
            ("Manchow Soup", 90, 130), ("Extra Soy Sauce", 20, 30),
            ("Prawn Crackers", 60, 80), ("Chilli Garlic Sauce", 30, 40),
            ("Sweet Corn Soup", 80, 110), ("Fried Wonton", 100, 130),
        ],
    },
    "Mughlai": {
        "main_course": [
            ("Butter Chicken", 280, 350), ("Mutton Nihari", 340, 420),
            ("Chicken Korma", 260, 320), ("Shahi Paneer", 250, 310),
            ("Mughlai Paratha", 160, 210), ("Keema Matar", 220, 280),
            ("Chicken Changezi", 280, 340), ("Mutton Burrah", 350, 430),
            ("Navratan Korma", 240, 300), ("Paneer Lababdar", 250, 310),
            ("Chicken Biryani", 260, 320), ("Seekh Kebab Curry", 280, 340),
        ],
        "starter": [
            ("Seekh Kebab (4 pcs)", 240, 300), ("Galouti Kebab", 260, 330),
            ("Kakori Kebab", 260, 320), ("Tandoori Chicken", 280, 350),
            ("Reshmi Tikka", 250, 310), ("Shami Kebab (4 pcs)", 200, 260),
            ("Chicken Boti Kebab", 240, 300), ("Malai Tikka", 250, 310),
        ],
        "beverage": [
            ("Rooh Afza Sharbat", 60, 90), ("Sweet Lassi", 70, 100),
            ("Masala Chai", 40, 60), ("Kahwa", 80, 110),
            ("Thandai", 90, 130), ("Nimbu Pani", 40, 60),
            ("Kesar Badam Milk", 100, 140), ("Pepsi / Coke", 40, 60),
        ],
        "dessert": [
            ("Shahi Tukda", 100, 140), ("Phirni", 80, 120),
            ("Gulab Jamun (2 pcs)", 80, 110), ("Rabri", 90, 130),
            ("Firni", 80, 110), ("Kulfi Falooda", 120, 160),
        ],
        "addon": [
            ("Butter Naan", 50, 70), ("Sheermal", 60, 80),
            ("Tandoori Roti", 30, 50), ("Garlic Naan", 60, 80),
            ("Raita", 50, 70), ("Laccha Paratha", 60, 80),
            ("Green Chutney", 30, 40), ("Mint Raita", 50, 70),
            ("Papad", 20, 30), ("Onion Salad", 30, 50),
            ("Extra Gravy", 40, 60), ("Khameeri Roti", 50, 70),
        ],
    },
    "Bengali": {
        "main_course": [
            ("Maacher Jhol (Fish Curry)", 240, 300), ("Kosha Mangsho", 320, 400),
            ("Chicken Chaap", 260, 320), ("Shorshe Ilish", 360, 440),
            ("Chingri Malai Curry", 300, 380), ("Luchi with Aloor Dom", 160, 210),
            ("Begun Bhaja with Rice", 140, 180), ("Doi Maach", 260, 320),
            ("Aloo Posto", 160, 200), ("Mochar Ghonto", 180, 230),
            ("Dhokar Dalna", 160, 210), ("Chicken Kasha", 240, 300),
        ],
        "starter": [
            ("Fish Fry (Kolkata Style)", 180, 240), ("Chicken Cutlet", 160, 210),
            ("Beguni (2 pcs)", 60, 90), ("Phuchka (6 pcs)", 60, 90),
            ("Aloo Chop (2 pcs)", 60, 90), ("Prawn Cutlet", 200, 260),
            ("Dimer Devil (Egg)", 80, 120), ("Ghugni Chaat", 80, 110),
        ],
        "beverage": [
            ("Masala Chai", 30, 50), ("Mishti Doi Lassi", 70, 100),
            ("Aam Pora Shorbot", 70, 100), ("Rose Sherbet", 50, 80),
            ("Cold Coffee", 80, 110), ("Nimbu Pani", 40, 60),
        ],
        "dessert": [
            ("Rasgulla (2 pcs)", 60, 90), ("Mishti Doi", 60, 90),
            ("Sandesh", 70, 100), ("Pantua (2 pcs)", 70, 100),
            ("Chom Chom", 80, 110), ("Nalen Gur Payesh", 90, 130),
        ],
        "addon": [
            ("Steamed Rice", 60, 80), ("Luchi (2 pcs)", 40, 60),
            ("Papad", 20, 30), ("Kasundi (Mustard Dip)", 30, 50),
            ("Aloo Bhaja", 40, 60), ("Green Salad", 40, 60),
            ("Begun Bhaja (extra)", 50, 70), ("Pickle", 20, 40),
            ("Fried Brinjal", 50, 70), ("Extra Rice", 50, 70),
        ],
    },
    "Street Food": {
        "main_course": [
            ("Pav Bhaji", 140, 180), ("Chole Kulche", 130, 170),
            ("Vada Pav (2 pcs)", 60, 90), ("Dabeli (2 pcs)", 70, 100),
            ("Kathi Roll (Chicken)", 150, 200), ("Kathi Roll (Paneer)", 130, 180),
            ("Egg Roll", 100, 140), ("Bhel Puri", 80, 110),
            ("Sev Puri (6 pcs)", 80, 120), ("Misal Pav", 120, 160),
            ("Aloo Tikki Burger", 100, 140), ("Cheese Grilled Sandwich", 120, 160),
        ],
        "starter": [
            ("Pani Puri (6 pcs)", 60, 80), ("Dahi Puri (6 pcs)", 80, 110),
            ("Ragda Pattice", 80, 120), ("Samosa (2 pcs)", 50, 80),
            ("Kachori (2 pcs)", 60, 90), ("Aloo Tikki (2 pcs)", 60, 90),
            ("Papdi Chaat", 80, 110), ("Corn Cheese Balls", 120, 160),
        ],
        "beverage": [
            ("Masala Chai (Cutting)", 20, 40), ("Sugarcane Juice", 40, 60),
            ("Nimbu Pani", 30, 50), ("Kokum Sharbat", 40, 60),
            ("Buttermilk", 30, 50), ("Mango Panna", 50, 70),
            ("Lassi", 50, 80), ("Sol Kadi", 40, 60),
        ],
        "dessert": [
            ("Kulfi (Stick)", 40, 60), ("Rabri Falooda", 80, 120),
            ("Malpua (2 pcs)", 60, 90), ("Gola (Ice Candy)", 30, 50),
            ("Jalebi (100g)", 50, 70), ("Imarti", 50, 70),
        ],
        "addon": [
            ("Extra Pav (2 pcs)", 20, 30), ("Extra Cheese", 30, 50),
            ("Green Chutney", 10, 20), ("Tamarind Chutney", 10, 20),
            ("Sev (extra)", 10, 20), ("Onion Topping", 10, 20),
            ("Butter (extra)", 20, 30), ("Plain Puri (4 pcs)", 30, 50),
            ("Extra Gravy", 30, 40), ("Chilli Flakes", 10, 20),
        ],
    },
    "Italian": {
        "main_course": [
            ("Margherita Pizza", 250, 320), ("Penne Arrabiata", 220, 280),
            ("Spaghetti Aglio e Olio", 230, 290), ("Alfredo Pasta", 240, 300),
            ("Four Cheese Pizza", 300, 370), ("Chicken Parmigiana", 320, 400),
            ("Mushroom Risotto", 260, 320), ("Lasagna", 280, 350),
            ("Pepperoni Pizza", 280, 350), ("Veg Supreme Pizza", 260, 320),
            ("Pesto Pasta", 230, 290), ("Paneer Tikka Pizza", 260, 320),
        ],
        "starter": [
            ("Garlic Bread (4 pcs)", 120, 160), ("Bruschetta (3 pcs)", 160, 210),
            ("Stuffed Mushrooms", 180, 230), ("Soup of the Day", 120, 160),
            ("Caesar Salad", 180, 230), ("Cheesy Dip with Nachos", 160, 210),
            ("Fried Calamari", 220, 280), ("Caprese Salad", 180, 230),
        ],
        "beverage": [
            ("Virgin Mojito", 120, 160), ("Iced Tea (Peach)", 80, 110),
            ("Cold Coffee", 100, 140), ("Fresh Orange Juice", 100, 130),
            ("Blue Lagoon", 120, 160), ("Pepsi / Coke", 40, 60),
            ("Lemonade", 70, 100), ("Hot Chocolate", 120, 150),
        ],
        "dessert": [
            ("Tiramisu", 160, 210), ("Panna Cotta", 140, 180),
            ("Chocolate Lava Cake", 160, 210), ("Gelato (2 Scoops)", 120, 160),
            ("Cheesecake Slice", 160, 210), ("Brownie with Ice Cream", 150, 200),
        ],
        "addon": [
            ("Extra Cheese Topping", 50, 80), ("Olive Oil Dip", 30, 50),
            ("Garlic Mayo", 30, 50), ("Parmesan Shavings", 40, 60),
            ("Bread Basket", 80, 110), ("Side Salad", 80, 110),
            ("Oregano Seasoning", 10, 20), ("Chilli Flakes", 10, 20),
            ("Mushroom Topping", 40, 60), ("Jalapeno Topping", 30, 50),
        ],
    },
}

# Fallback for any cuisine not in the map above
DISHES["_default"] = DISHES["North Indian"]

# ── Restaurant name templates ─────────────────────────────────────────────────

RESTAURANT_NAMES: dict[str, list[str]] = {
    "North Indian": [
        "Punjab Grill", "Dhaba Express", "Moti Mahal Delux", "Pind Balluchi",
        "Kwality Restaurant", "Sagar Ratna", "Bikanervala", "Rajdhani Thali",
        "Gulati Restaurant", "Haveli", "Punjabi By Nature", "Frontier",
        "Bukhara Kitchen", "Dilli 32", "Paratha Junction",
    ],
    "South Indian": [
        "Saravana Bhavan", "Murugan Idli Shop", "A2B Adyar Ananda Bhavan",
        "Vasudev Adigas", "Dosa Plaza", "Madras Cafe", "Shri Krishna Bhavan",
        "Cafe Madras", "Udupi Palace", "MTR 1924", "Maiyas",
        "Nair's Restaurant", "Kailash Parbat", "Sangeetha Veg",
    ],
    "Biryani": [
        "Behrouz Biryani", "Paradise Biryani", "Bawarchi", "Meghana Foods",
        "Cafe Bahar", "Shah Ghouse", "Hyderabad House", "Pista House",
        "Lucky Biryani", "Al Baik",  "Biryani Blues", "Biryani Pot",
        "Royal Biryani", "Dum Pukht Biryani", "Shadab",
    ],
    "Chinese": [
        "Mainland China", "Wow! China", "Chili's", "Yo! China",
        "Hakka Noodle House", "Chopsticks", "Wok Express", "Dragon Bowl",
        "Chinese Wok", "Noodle Bar", "Hunan Express", "Szechuan Court",
        "Golden Dragon", "Wok & Roll", "Asia Kitchen",
    ],
    "Mughlai": [
        "Al Jawahar", "Karim's", "Moti Mahal", "Eden", "Dum Pukht",
        "Chor Bizarre", "Naivedhyam", "Sheesh Mahal", "Mughal Darbar",
        "Nawab Sahab", "Tunday Kababi", "Lucknowi Kitchen", "Imperial Spice",
    ],
    "Bengali": [
        "6 Ballygunge Place", "Oh! Calcutta", "Aaheli", "Bhojohori Manna",
        "Kewpie's Kitchen", "Kasturi", "Rokomari", "Ilish Mahal",
        "Kolkata Biryani House", "Balwant Singh Dhaba", "Saptapadi",
    ],
    "Street Food": [
        "Elco Pav Bhaji", "Sharma Ji Ki Chai", "Chaat Corner",
        "Bombay Chowpatty", "Delhi Chaat House", "Goli Vada Pav",
        "Jai Jawan", "Jumbo King", "Tibb's Frankie", "Bachelorr's",
        "Shree Thaker Bhojanalay", "Swati Snacks", "Sardarji Chaat Wala",
    ],
    "Italian": [
        "Pizza Hut", "Domino's", "La Pino'z", "Toscano", "Jamie's Pizzeria",
        "California Pizza Kitchen", "Olive Bar & Kitchen", "Smoke House Deli",
        "The Big Chill Cafe", "Pasta Street", "Di Ghent Cafe", "Mad Over Donuts",
    ],
}


def _hash_seed(item_id: str) -> int:
    """Deterministic seed from item_id so names are stable across restarts."""
    return int(hashlib.md5(item_id.encode()).hexdigest()[:8], 16)


def build_demo_catalog(
    items_df,
    restaurants_df,
    order_items_df,
    orders_df,
) -> tuple[dict[str, dict], list[dict], dict[str, list[str]]]:
    """
    Returns:
        item_catalog  – {item_id: {item_name, item_category, item_price}}
        restaurant_list – [{id, name, city, cuisine, item_count}]
        rest_items_map  – {restaurant_id: [item_id, ...]}
    """
    import pandas as pd

    # ── 1. Build restaurant → cuisine lookup ──────────────────────────────────
    rest_cuisine: dict[str, str] = {}
    rest_city: dict[str, str] = {}
    for _, row in restaurants_df.iterrows():
        rid = str(row["restaurant_id"])
        rest_cuisine[rid] = str(row.get("cuisine", "North Indian"))
        rest_city[rid] = str(row.get("city", ""))

    # ── 2. Build restaurant → item list ───────────────────────────────────────
    oi = order_items_df.merge(
        orders_df[["order_id", "restaurant_id"]], on="order_id", how="left"
    )
    raw_rest_items: dict[str, list[str]] = (
        oi.groupby("restaurant_id")["item_id"]
        .apply(lambda s: sorted(set(s.astype(str).tolist())))
        .to_dict()
    )

    # ── 3. Build per-restaurant menus with realistic names ────────────────────
    #    Strategy: for each restaurant pick up to 40 items, assign them to
    #    categories in realistic ratios, and give each a unique dish name
    #    from the cuisine's dish bank.  If an item already has a name from
    #    a previous restaurant we still use the first assignment (stable).

    item_catalog: dict[str, dict] = {}
    rest_items_map: dict[str, list[str]] = {}

    CATEGORY_WEIGHTS = [
        ("main_course", 0.30),
        ("addon", 0.28),
        ("starter", 0.15),
        ("beverage", 0.15),
        ("dessert", 0.12),
    ]

    for rid, item_ids in raw_rest_items.items():
        cuisine = rest_cuisine.get(rid, "North Indian")
        dish_bank = DISHES.get(cuisine, DISHES["_default"])

        menu_size = min(len(item_ids), 40)
        # Deterministically shuffle items for this restaurant
        rng = random.Random(_hash_seed(rid))
        pool = list(item_ids)
        rng.shuffle(pool)
        pool = pool[:menu_size]

        # Partition pool into categories
        offset = 0
        menu_items: list[str] = []
        for cat, weight in CATEGORY_WEIGHTS:
            n = max(2, round(menu_size * weight))
            cat_ids = pool[offset: offset + n]
            offset += n

            dishes_for_cat = dish_bank.get(cat, [])
            if not dishes_for_cat:
                dishes_for_cat = DISHES["_default"].get(cat, [])

            used_dish_names: set[str] = set()
            for i, iid in enumerate(cat_ids):
                # Only assign if item doesn't already have a name
                if iid in item_catalog:
                    menu_items.append(iid)
                    continue

                # Pick a dish name not yet used in this restaurant
                seed = _hash_seed(iid)
                dish_idx = (seed + i) % len(dishes_for_cat)
                attempts = 0
                while dishes_for_cat[dish_idx][0] in used_dish_names and attempts < len(dishes_for_cat):
                    dish_idx = (dish_idx + 1) % len(dishes_for_cat)
                    attempts += 1

                dish_name, min_p, max_p = dishes_for_cat[dish_idx]
                used_dish_names.add(dish_name)

                price = min_p + (seed % 100) / 100 * (max_p - min_p)
                price = round(price, 0)

                item_catalog[iid] = {
                    "item_name": dish_name,
                    "item_category": cat,
                    "item_price": price,
                }
                menu_items.append(iid)

        rest_items_map[rid] = menu_items

    # ── 4. Build restaurant list with realistic names ─────────────────────────
    restaurant_list = []
    used_names: dict[str, int] = {}
    for _, row in restaurants_df.iterrows():
        rid = str(row["restaurant_id"])
        cuisine = rest_cuisine.get(rid, "North Indian")
        city = rest_city.get(rid, "")

        name_pool = RESTAURANT_NAMES.get(
            cuisine, RESTAURANT_NAMES.get("North Indian", ["Restaurant"])
        )
        seed = _hash_seed(rid)
        name = name_pool[seed % len(name_pool)]
        # Deduplicate: append city or counter
        if name in used_names:
            used_names[name] += 1
            name = f"{name} - {city}" if city else f"{name} #{used_names[name]}"
        used_names.setdefault(name, 0)
        used_names[name] += 1

        restaurant_list.append({
            "id": rid,
            "name": name,
            "city": city,
            "cuisine": cuisine,
            "item_count": len(rest_items_map.get(rid, [])),
        })

    restaurant_list.sort(key=lambda x: x["item_count"], reverse=True)

    return item_catalog, restaurant_list, rest_items_map
