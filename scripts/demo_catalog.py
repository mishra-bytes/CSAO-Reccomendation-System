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

DISHES: dict[str, dict[str, list[tuple[str, float, float, bool]]]] = {
    # cuisine → category → [(dish_name, min_price, max_price, is_veg), ...]
    "North Indian": {
        "main_course": [
            ("Butter Chicken", 280, 350, False), ("Dal Makhani", 220, 280, True),
            ("Paneer Tikka Masala", 260, 320, True), ("Rajma Chawal", 180, 230, True),
            ("Chole Bhature", 180, 220, True), ("Kadai Paneer", 240, 300, True),
            ("Shahi Paneer", 250, 310, True), ("Palak Paneer", 230, 280, True),
            ("Malai Kofta", 260, 320, True), ("Aloo Gobi", 180, 220, True),
            ("Paneer Butter Masala", 260, 320, True), ("Chicken Curry", 240, 300, False),
            ("Mutton Rogan Josh", 350, 420, False), ("Egg Curry", 180, 230, False),
            ("Jeera Rice", 140, 180, True), ("Veg Biryani", 200, 260, True),
            ("Mix Veg Curry", 180, 230, True), ("Mushroom Masala", 220, 270, True),
            ("Matar Paneer", 230, 280, True), ("Dum Aloo", 200, 250, True),
        ],
        "starter": [
            ("Paneer Tikka", 220, 280, True), ("Chicken Tikka", 260, 320, False),
            ("Veg Seekh Kebab", 200, 260, True), ("Hara Bhara Kebab", 180, 230, True),
            ("Dahi Kebab", 200, 250, True), ("Tandoori Chicken", 280, 350, False),
            ("Amritsari Fish Tikka", 300, 370, False), ("Malai Chaap", 220, 280, True),
            ("Aloo Tikki", 120, 160, True), ("Papdi Chaat", 130, 170, True),
            ("Samosa (2 pcs)", 80, 120, True), ("Onion Bhaji", 120, 160, True),
        ],
        "beverage": [
            ("Masala Chai", 50, 80, True), ("Sweet Lassi", 80, 120, True),
            ("Mango Lassi", 100, 140, True), ("Salted Lassi", 70, 100, True),
            ("Rose Sharbat", 80, 110, True), ("Jaljeera", 60, 90, True),
            ("Nimbu Pani", 50, 70, True), ("Buttermilk (Chaas)", 50, 70, True),
            ("Thandai", 100, 140, True), ("Badam Milk", 90, 130, True),
        ],
        "dessert": [
            ("Gulab Jamun (2 pcs)", 80, 120, True), ("Rasmalai (2 pcs)", 100, 140, True),
            ("Gajar Ka Halwa", 120, 160, True), ("Kheer", 100, 140, True),
            ("Jalebi", 80, 120, True), ("Rabri", 100, 140, True),
            ("Moong Dal Halwa", 120, 160, True), ("Kulfi", 80, 110, True),
        ],
        "addon": [
            ("Butter Naan", 50, 70, True), ("Garlic Naan", 60, 80, True),
            ("Tandoori Roti", 30, 50, True), ("Laccha Paratha", 50, 70, True),
            ("Raita (Boondi)", 50, 70, True), ("Papad", 30, 40, True),
            ("Green Salad", 60, 80, True), ("Pickle (Achar)", 30, 50, True),
            ("Plain Rice", 100, 140, True), ("Onion Rings", 60, 80, True),
            ("Extra Gravy", 40, 60, True), ("Missi Roti", 40, 60, True),
            ("Rumali Roti", 40, 60, True), ("Pudina Raita", 50, 70, True),
            ("Masala Papad", 50, 60, True),
        ],
    },
    "South Indian": {
        "main_course": [
            ("Masala Dosa", 140, 180, True), ("Ghee Roast Dosa", 160, 200, True),
            ("Mysore Masala Dosa", 160, 210, True), ("Rava Dosa", 150, 190, True),
            ("Set Dosa (3 pcs)", 130, 170, True), ("Idli Sambar (4 pcs)", 100, 140, True),
            ("Vada Sambar (2 pcs)", 100, 140, True), ("Uttapam", 140, 180, True),
            ("Curd Rice", 120, 160, True), ("Lemon Rice", 130, 170, True),
            ("Bisi Bele Bath", 140, 180, True), ("Pongal", 120, 160, True),
            ("Appam with Stew", 160, 210, True), ("Chicken Chettinad", 280, 340, False),
            ("Fish Curry (Kerala)", 260, 320, False), ("Sambar Rice", 130, 170, True),
        ],
        "starter": [
            ("Medu Vada (2 pcs)", 80, 120, True), ("Onion Pakoda", 100, 140, True),
            ("Paniyaram (6 pcs)", 110, 150, True), ("Chicken 65", 220, 280, False),
            ("Gobi Manchurian Dry", 160, 210, True), ("Pepper Chicken", 240, 300, False),
            ("Banana Chips", 60, 90, True), ("Murukku", 70, 100, True),
        ],
        "beverage": [
            ("Filter Coffee", 50, 80, True), ("Masala Chai", 40, 60, True),
            ("Buttermilk (Neer Mor)", 40, 60, True), ("Tender Coconut Water", 60, 90, True),
            ("Jigarthanda", 80, 120, True), ("Rose Milk", 60, 90, True),
            ("Mango Juice", 80, 110, True), ("Sugarcane Juice", 50, 70, True),
        ],
        "dessert": [
            ("Payasam", 80, 120, True), ("Mysore Pak", 70, 100, True),
            ("Kesari Bath", 70, 100, True), ("Double Ka Meetha", 90, 130, True),
            ("Banana Sheera", 70, 100, True), ("Rava Ladoo (3 pcs)", 80, 110, True),
        ],
        "addon": [
            ("Sambar (extra)", 40, 60, True), ("Coconut Chutney", 30, 50, True),
            ("Tomato Chutney", 30, 50, True), ("Gun Powder", 20, 40, True),
            ("Ghee (extra)", 30, 40, True), ("Pickle (Avakaya)", 30, 50, True),
            ("Curd (extra)", 40, 60, True), ("Papad", 20, 30, True),
            ("Onion Uttapam", 120, 160, True), ("Plain Dosa", 100, 130, True),
            ("Mini Idli (8 pcs)", 90, 120, True), ("Podi Idli", 100, 140, True),
        ],
    },
    "Biryani": {
        "main_course": [
            ("Hyderabadi Chicken Biryani", 280, 350, False), ("Mutton Biryani", 350, 420, False),
            ("Veg Dum Biryani", 220, 280, True), ("Egg Biryani", 200, 260, False),
            ("Prawns Biryani", 320, 400, False), ("Paneer Biryani", 240, 300, True),
            ("Lucknowi Chicken Biryani", 280, 350, False), ("Keema Biryani", 300, 370, False),
            ("Chicken Fried Rice", 200, 260, False), ("Veg Pulao", 180, 230, True),
            ("Mushroom Biryani", 220, 280, True), ("Fish Biryani", 300, 370, False),
            ("Chicken 65 Biryani", 280, 340, False), ("Double Masala Biryani", 300, 370, False),
        ],
        "starter": [
            ("Chicken 65", 200, 260, False), ("Apollo Fish", 260, 320, False),
            ("Paneer 65", 200, 250, True), ("Kebab Platter", 300, 380, False),
            ("Tandoori Prawns", 320, 400, False), ("Mutton Seekh Kebab", 280, 350, False),
            ("Crispy Corn", 140, 180, True), ("Mushroom Pepper Fry", 160, 210, True),
        ],
        "beverage": [
            ("Falooda", 120, 160, True), ("Sweet Lassi", 70, 100, True),
            ("Masala Chai", 40, 60, True), ("Rose Sharbat", 60, 90, True),
            ("Pepsi / Coke", 40, 60, True), ("Fresh Lime Soda", 60, 90, True),
            ("Mango Shake", 100, 140, True), ("Cold Coffee", 90, 130, True),
        ],
        "dessert": [
            ("Phirni", 80, 120, True), ("Shahi Tukda", 100, 140, True),
            ("Double Ka Meetha", 90, 130, True), ("Qubani Ka Meetha", 100, 140, True),
            ("Gulab Jamun (2 pcs)", 70, 100, True), ("Ice Cream (Scoop)", 60, 90, True),
        ],
        "addon": [
            ("Raita", 50, 70, True), ("Mirchi Ka Salan", 60, 80, True),
            ("Gutti Vankaya Curry", 80, 110, True), ("Bagara Baingan", 80, 110, True),
            ("Boiled Egg (2 pcs)", 40, 60, False), ("Rumali Roti", 40, 60, True),
            ("Extra Chicken Piece", 80, 120, False), ("Green Salad", 50, 70, True),
            ("Papad", 20, 30, True), ("Onion Raita", 50, 70, True),
            ("Dahi Chutney", 40, 50, True), ("Salan (extra)", 50, 60, True),
        ],
    },
    "Chinese": {
        "main_course": [
            ("Veg Hakka Noodles", 180, 230, True), ("Chicken Fried Rice", 200, 260, False),
            ("Schezwan Noodles", 190, 240, True), ("Manchurian Rice", 190, 240, True),
            ("Chilli Chicken with Rice", 240, 300, False), ("Dragon Chicken", 260, 320, False),
            ("Singapore Noodles", 200, 260, True), ("Triple Schezwan Rice", 220, 280, True),
            ("Kung Pao Chicken", 260, 320, False), ("Veg Chow Mein", 170, 220, True),
            ("Chicken Manchurian with Rice", 240, 300, False), ("Paneer Chilli Rice", 210, 260, True),
        ],
        "starter": [
            ("Veg Manchurian Dry", 160, 210, True), ("Chilli Paneer Dry", 180, 240, True),
            ("Honey Chilli Potato", 150, 200, True), ("Crispy Corn", 130, 170, True),
            ("Spring Rolls (4 pcs)", 140, 180, True), ("Chicken Momos (8 pcs)", 160, 210, False),
            ("Veg Momos (8 pcs)", 130, 170, True), ("Dim Sum Platter", 200, 260, True),
            ("Chilli Fish", 240, 300, False), ("Golden Fried Prawns", 260, 320, False),
        ],
        "beverage": [
            ("Iced Lemon Tea", 70, 100, True), ("Fresh Lime Soda", 60, 90, True),
            ("Cold Coffee", 90, 130, True), ("Green Tea", 50, 70, True),
            ("Mango Shake", 100, 140, True), ("Blue Lagoon Mocktail", 120, 160, True),
            ("Virgin Mojito", 120, 160, True), ("Pepsi / Coke", 40, 60, True),
        ],
        "dessert": [
            ("Fried Ice Cream", 120, 160, True), ("Chocolate Brownie", 130, 170, True),
            ("Date Pancake", 100, 140, True), ("Toffee Banana", 100, 140, True),
            ("Lychee Pudding", 90, 130, True), ("Mango Sticky Rice", 110, 150, True),
        ],
        "addon": [
            ("Schezwan Sauce (extra)", 30, 50, True), ("Chilli Oil", 30, 40, True),
            ("Steamed Rice", 80, 110, True), ("Fried Rice (small)", 100, 140, True),
            ("Wonton Soup", 80, 120, True), ("Hot & Sour Soup", 90, 130, True),
            ("Manchow Soup", 90, 130, True), ("Extra Soy Sauce", 20, 30, True),
            ("Prawn Crackers", 60, 80, False), ("Chilli Garlic Sauce", 30, 40, True),
            ("Sweet Corn Soup", 80, 110, True), ("Fried Wonton", 100, 130, True),
        ],
    },
    "Mughlai": {
        "main_course": [
            ("Butter Chicken", 280, 350, False), ("Mutton Nihari", 340, 420, False),
            ("Chicken Korma", 260, 320, False), ("Shahi Paneer", 250, 310, True),
            ("Mughlai Paratha", 160, 210, False), ("Keema Matar", 220, 280, False),
            ("Chicken Changezi", 280, 340, False), ("Mutton Burrah", 350, 430, False),
            ("Navratan Korma", 240, 300, True), ("Paneer Lababdar", 250, 310, True),
            ("Chicken Biryani", 260, 320, False), ("Seekh Kebab Curry", 280, 340, False),
        ],
        "starter": [
            ("Seekh Kebab (4 pcs)", 240, 300, False), ("Galouti Kebab", 260, 330, False),
            ("Kakori Kebab", 260, 320, False), ("Tandoori Chicken", 280, 350, False),
            ("Reshmi Tikka", 250, 310, False), ("Shami Kebab (4 pcs)", 200, 260, False),
            ("Chicken Boti Kebab", 240, 300, False), ("Malai Tikka", 250, 310, False),
        ],
        "beverage": [
            ("Rooh Afza Sharbat", 60, 90, True), ("Sweet Lassi", 70, 100, True),
            ("Masala Chai", 40, 60, True), ("Kahwa", 80, 110, True),
            ("Thandai", 90, 130, True), ("Nimbu Pani", 40, 60, True),
            ("Kesar Badam Milk", 100, 140, True), ("Pepsi / Coke", 40, 60, True),
        ],
        "dessert": [
            ("Shahi Tukda", 100, 140, True), ("Phirni", 80, 120, True),
            ("Gulab Jamun (2 pcs)", 80, 110, True), ("Rabri", 90, 130, True),
            ("Firni", 80, 110, True), ("Kulfi Falooda", 120, 160, True),
        ],
        "addon": [
            ("Butter Naan", 50, 70, True), ("Sheermal", 60, 80, True),
            ("Tandoori Roti", 30, 50, True), ("Garlic Naan", 60, 80, True),
            ("Raita", 50, 70, True), ("Laccha Paratha", 60, 80, True),
            ("Green Chutney", 30, 40, True), ("Mint Raita", 50, 70, True),
            ("Papad", 20, 30, True), ("Onion Salad", 30, 50, True),
            ("Extra Gravy", 40, 60, True), ("Khameeri Roti", 50, 70, True),
        ],
    },
    "Bengali": {
        "main_course": [
            ("Maacher Jhol (Fish Curry)", 240, 300, False), ("Kosha Mangsho", 320, 400, False),
            ("Chicken Chaap", 260, 320, False), ("Shorshe Ilish", 360, 440, False),
            ("Chingri Malai Curry", 300, 380, False), ("Luchi with Aloor Dom", 160, 210, True),
            ("Begun Bhaja with Rice", 140, 180, True), ("Doi Maach", 260, 320, False),
            ("Aloo Posto", 160, 200, True), ("Mochar Ghonto", 180, 230, True),
            ("Dhokar Dalna", 160, 210, True), ("Chicken Kasha", 240, 300, False),
        ],
        "starter": [
            ("Fish Fry (Kolkata Style)", 180, 240, False), ("Chicken Cutlet", 160, 210, False),
            ("Beguni (2 pcs)", 60, 90, True), ("Phuchka (6 pcs)", 60, 90, True),
            ("Aloo Chop (2 pcs)", 60, 90, True), ("Prawn Cutlet", 200, 260, False),
            ("Dimer Devil (Egg)", 80, 120, False), ("Ghugni Chaat", 80, 110, True),
        ],
        "beverage": [
            ("Masala Chai", 30, 50, True), ("Mishti Doi Lassi", 70, 100, True),
            ("Aam Pora Shorbot", 70, 100, True), ("Rose Sherbet", 50, 80, True),
            ("Cold Coffee", 80, 110, True), ("Nimbu Pani", 40, 60, True),
        ],
        "dessert": [
            ("Rasgulla (2 pcs)", 60, 90, True), ("Mishti Doi", 60, 90, True),
            ("Sandesh", 70, 100, True), ("Pantua (2 pcs)", 70, 100, True),
            ("Chom Chom", 80, 110, True), ("Nalen Gur Payesh", 90, 130, True),
        ],
        "addon": [
            ("Steamed Rice", 60, 80, True), ("Luchi (2 pcs)", 40, 60, True),
            ("Papad", 20, 30, True), ("Kasundi (Mustard Dip)", 30, 50, True),
            ("Aloo Bhaja", 40, 60, True), ("Green Salad", 40, 60, True),
            ("Begun Bhaja (extra)", 50, 70, True), ("Pickle", 20, 40, True),
            ("Fried Brinjal", 50, 70, True), ("Extra Rice", 50, 70, True),
        ],
    },
    "Street Food": {
        "main_course": [
            ("Pav Bhaji", 140, 180, True), ("Chole Kulche", 130, 170, True),
            ("Vada Pav (2 pcs)", 60, 90, True), ("Dabeli (2 pcs)", 70, 100, True),
            ("Kathi Roll (Chicken)", 150, 200, False), ("Kathi Roll (Paneer)", 130, 180, True),
            ("Egg Roll", 100, 140, False), ("Bhel Puri", 80, 110, True),
            ("Sev Puri (6 pcs)", 80, 120, True), ("Misal Pav", 120, 160, True),
            ("Aloo Tikki Burger", 100, 140, True), ("Cheese Grilled Sandwich", 120, 160, True),
        ],
        "starter": [
            ("Pani Puri (6 pcs)", 60, 80, True), ("Dahi Puri (6 pcs)", 80, 110, True),
            ("Ragda Pattice", 80, 120, True), ("Samosa (2 pcs)", 50, 80, True),
            ("Kachori (2 pcs)", 60, 90, True), ("Aloo Tikki (2 pcs)", 60, 90, True),
            ("Papdi Chaat", 80, 110, True), ("Corn Cheese Balls", 120, 160, True),
        ],
        "beverage": [
            ("Masala Chai (Cutting)", 20, 40, True), ("Sugarcane Juice", 40, 60, True),
            ("Nimbu Pani", 30, 50, True), ("Kokum Sharbat", 40, 60, True),
            ("Buttermilk", 30, 50, True), ("Mango Panna", 50, 70, True),
            ("Lassi", 50, 80, True), ("Sol Kadi", 40, 60, True),
        ],
        "dessert": [
            ("Kulfi (Stick)", 40, 60, True), ("Rabri Falooda", 80, 120, True),
            ("Malpua (2 pcs)", 60, 90, True), ("Gola (Ice Candy)", 30, 50, True),
            ("Jalebi (100g)", 50, 70, True), ("Imarti", 50, 70, True),
        ],
        "addon": [
            ("Extra Pav (2 pcs)", 20, 30, True), ("Extra Cheese", 30, 50, True),
            ("Green Chutney", 10, 20, True), ("Tamarind Chutney", 10, 20, True),
            ("Sev (extra)", 10, 20, True), ("Onion Topping", 10, 20, True),
            ("Butter (extra)", 20, 30, True), ("Plain Puri (4 pcs)", 30, 50, True),
            ("Extra Gravy", 30, 40, True), ("Chilli Flakes", 10, 20, True),
        ],
    },
    "Italian": {
        "main_course": [
            ("Margherita Pizza", 250, 320, True), ("Penne Arrabiata", 220, 280, True),
            ("Spaghetti Aglio e Olio", 230, 290, True), ("Alfredo Pasta", 240, 300, True),
            ("Four Cheese Pizza", 300, 370, True), ("Chicken Parmigiana", 320, 400, False),
            ("Mushroom Risotto", 260, 320, True), ("Lasagna", 280, 350, True),
            ("Pepperoni Pizza", 280, 350, False), ("Veg Supreme Pizza", 260, 320, True),
            ("Pesto Pasta", 230, 290, True), ("Paneer Tikka Pizza", 260, 320, True),
        ],
        "starter": [
            ("Garlic Bread (4 pcs)", 120, 160, True), ("Bruschetta (3 pcs)", 160, 210, True),
            ("Stuffed Mushrooms", 180, 230, True), ("Soup of the Day", 120, 160, True),
            ("Caesar Salad", 180, 230, True), ("Cheesy Dip with Nachos", 160, 210, True),
            ("Fried Calamari", 220, 280, False), ("Caprese Salad", 180, 230, True),
        ],
        "beverage": [
            ("Virgin Mojito", 120, 160, True), ("Iced Tea (Peach)", 80, 110, True),
            ("Cold Coffee", 100, 140, True), ("Fresh Orange Juice", 100, 130, True),
            ("Blue Lagoon", 120, 160, True), ("Pepsi / Coke", 40, 60, True),
            ("Lemonade", 70, 100, True), ("Hot Chocolate", 120, 150, True),
        ],
        "dessert": [
            ("Tiramisu", 160, 210, True), ("Panna Cotta", 140, 180, True),
            ("Chocolate Lava Cake", 160, 210, True), ("Gelato (2 Scoops)", 120, 160, True),
            ("Cheesecake Slice", 160, 210, True), ("Brownie with Ice Cream", 150, 200, True),
        ],
        "addon": [
            ("Extra Cheese Topping", 50, 80, True), ("Olive Oil Dip", 30, 50, True),
            ("Garlic Mayo", 30, 50, True), ("Parmesan Shavings", 40, 60, True),
            ("Bread Basket", 80, 110, True), ("Side Salad", 80, 110, True),
            ("Oregano Seasoning", 10, 20, True), ("Chilli Flakes", 10, 20, True),
            ("Mushroom Topping", 40, 60, True), ("Jalapeno Topping", 30, 50, True),
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

                dish_entry = dishes_for_cat[dish_idx]
                dish_name = dish_entry[0]
                min_p, max_p = dish_entry[1], dish_entry[2]
                is_veg = bool(dish_entry[3]) if len(dish_entry) > 3 else True
                used_dish_names.add(dish_name)

                price = min_p + (seed % 100) / 100 * (max_p - min_p)
                price = round(price, 0)

                item_catalog[iid] = {
                    "item_name": dish_name,
                    "item_category": cat,
                    "item_price": price,
                    "is_veg": is_veg,
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
