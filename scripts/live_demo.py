"""
Live Interactive Demo — CSAO Recommendation Engine
===================================================

Launches a FastAPI server with a rich HTML dashboard at http://localhost:8000
where you can:
  • Browse restaurants and their menus
  • Build a cart by clicking items
  • Get real-time CSAO recommendations with LLM explanations
  • See latency breakdown, feature importance, and diversity metrics

Usage:
    python scripts/live_demo.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from candidate_generation.candidate_generator import CandidateGenerator
from features.complementarity import build_complementarity_lookup
from ranking.inference.ranker import CSAORanker
from scripts._utils import load_feature_tables, load_project_config, load_unified_tables
from serving.api.schemas import RecommendationRequest
from serving.pipeline.recommendation_service import RecommendationService, ServingArtifacts

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
print("[demo] Loading config and data artifacts...", flush=True)
config = load_project_config()
processed_dir = config.get("paths", {}).get("processed_dir", "data/processed")
ranking_cfg = config.get("ranking", {})

print("[demo]  .. loading tables", flush=True)
unified = load_unified_tables(processed_dir)
print("[demo]  .. loading features", flush=True)
feats = load_feature_tables(processed_dir)
print("[demo]  .. building comp lookup", flush=True)
comp_lookup = build_complementarity_lookup(feats["complementarity"])
model_path = Path(ranking_cfg.get("model_path", "models/lgbm_ranker.joblib"))
feature_cols_path = Path(ranking_cfg.get("feature_columns_path", "models/feature_columns.json"))

# Build service - sample orders for fast index building in candidate gen
print("[demo]  .. building candidate generator", flush=True)
_demo_orders = unified["orders"].tail(5000)
_demo_oi = unified["order_items"][
    unified["order_items"]["order_id"].isin(_demo_orders["order_id"])
]
candidate_generator = CandidateGenerator(
    complementarity=feats["complementarity"],
    category_affinity=feats["category_affinity"],
    items=unified["items"],
    orders=_demo_orders,
    order_items=_demo_oi,
    config=config,
)
print("[demo]  .. building ranker", flush=True)
ranker = CSAORanker(
    model_path=str(model_path),
    feature_columns_path=str(feature_cols_path),
    user_features=feats["user_features"],
    item_features=feats["item_features"],
    items=unified["items"],
    complementarity_lookup=comp_lookup,
)

# ── Build realistic demo catalog (cuisine-aware dish names & restaurant names) ─
print("[demo]  .. building demo catalog", flush=True)

# Check if items already have proper food names (Indian food data)
# by seeing if item_names match dishes in our known catalog
_sample_names = unified["items"]["item_name"].head(20).tolist()
from scripts.demo_catalog import DISHES as _DISHES_CHECK

_known_dish_names = set()
for _cuisine_dict in _DISHES_CHECK.values():
    if isinstance(_cuisine_dict, dict):
        for _cat_list in _cuisine_dict.values():
            if isinstance(_cat_list, list):
                for _entry in _cat_list:
                    if isinstance(_entry, tuple) and len(_entry) >= 1:
                        _known_dish_names.add(_entry[0])

_name_matches = sum(1 for n in _sample_names if n in _known_dish_names)
_use_passthrough = _name_matches > len(_sample_names) * 0.3  # >30% match = proper food data

if _use_passthrough:
    print("[demo]  .. items already have proper food names, using direct catalog", flush=True)
    # Build catalog directly from unified data (no remapping needed)
    item_catalog = {}
    for _, row in unified["items"].iterrows():
        iid = str(row["item_id"])
        item_catalog[iid] = {
            "item_name": str(row.get("item_name", iid)),
            "item_category": str(row.get("item_category", "unknown")),
            "item_price": float(row.get("item_price", 0)),
        }

    # Restaurant → item mapping from actual orders
    _oi_merged = unified["order_items"].merge(
        unified["orders"][["order_id", "restaurant_id"]], on="order_id", how="left"
    )
    rest_items_map = (
        _oi_merged.groupby("restaurant_id")["item_id"]
        .apply(lambda s: sorted(set(s.astype(str))))
        .to_dict()
    )

    # Restaurant list with actual names from data
    from scripts.demo_catalog import RESTAURANT_NAMES as _RNAMES
    restaurant_list = []
    for _, row in unified["restaurants"].iterrows():
        rid = str(row["restaurant_id"])
        restaurant_list.append({
            "id": rid,
            "name": str(row.get("restaurant_name", rid)),
            "city": str(row.get("city", "")),
            "cuisine": str(row.get("cuisine", "North Indian")),
            "item_count": len(rest_items_map.get(rid, [])),
        })
    restaurant_list.sort(key=lambda x: x["item_count"], reverse=True)
else:
    print("[demo]  .. using demo catalog remapping (legacy Instacart names)", flush=True)
    from scripts.demo_catalog import build_demo_catalog
    item_catalog, restaurant_list, rest_items_map = build_demo_catalog(
        items_df=unified["items"],
        restaurants_df=unified["restaurants"],
        order_items_df=unified["order_items"],
        orders_df=unified["orders"],
    )

service = RecommendationService(
    artifacts=ServingArtifacts(
        candidate_generator=candidate_generator,
        ranker=ranker,
        user_features=feats["user_features"],
        item_features=feats["item_features"],
        item_catalog=item_catalog,
    ),
    config=config,
)

# Sample user IDs
user_ids = sorted(unified["orders"]["user_id"].dropna().unique().tolist())[:200]

print(f"[demo] Ready! {len(item_catalog)} items, {len(restaurant_list)} restaurants", flush=True)

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(title="CSAO Live Demo", version="1.0.0")


class RecommendRequest(BaseModel):
    user_id: str
    restaurant_id: str
    cart_item_ids: list[str]
    top_n: int = 10


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD_HTML


@app.get("/api/restaurants")
def get_restaurants():
    return restaurant_list[:100]


@app.get("/api/restaurant/{restaurant_id}/items")
def get_restaurant_items(restaurant_id: str):
    item_ids = rest_items_map.get(restaurant_id, [])
    items = []
    for iid in item_ids:
        info = item_catalog.get(iid, {})
        items.append({
            "item_id": iid,
            "item_name": info.get("item_name", iid),
            "item_category": info.get("item_category", "unknown"),
            "item_price": info.get("item_price", 0),
        })
    return items


@app.get("/api/users")
def get_users():
    return user_ids[:50]


@app.post("/api/recommend")
def recommend(req: RecommendRequest):
    start = time.perf_counter()
    try:
        # Expand candidate pool — request more so we can filter to known menu items
        internal_req = RecommendationRequest(
            user_id=req.user_id,
            session_id=f"demo_{int(time.time())}",
            restaurant_id=req.restaurant_id,
            cart_item_ids=req.cart_item_ids,
            top_n=req.top_n * 20,  # over-fetch heavily to find same-restaurant items
        )
        resp = service.recommend(internal_req)

        # Filter to items on this restaurant's demo menu
        rest_menu = set(rest_items_map.get(req.restaurant_id, []))
        cart_set = set(req.cart_item_ids)

        # Pass 1: only items from this restaurant's menu
        recos = []
        seen = set()
        for rec in resp.recommendations:
            iid = rec.get("item_id", "")
            if iid in cart_set or iid in seen or iid not in rest_menu:
                continue
            info = item_catalog.get(iid, {})
            if not info:
                continue
            seen.add(iid)
            # Replace raw item-id references in explanation text with item names
            explanation = rec.get("explanation", "")
            for ref_iid in req.cart_item_ids:
                ref_info = item_catalog.get(ref_iid, {})
                if ref_iid in explanation and ref_info:
                    explanation = explanation.replace(ref_iid, ref_info["item_name"])
            recos.append({
                **rec,
                "item_name": info.get("item_name", iid),
                "item_category": info.get("item_category", "unknown"),
                "item_price": info.get("item_price", 0),
                "explanation": explanation,
            })
            if len(recos) >= req.top_n:
                break

        # Pass 2: if not enough, fill from the restaurant menu (category-diverse)
        if len(recos) < req.top_n:
            cart_cats = {item_catalog.get(c, {}).get("item_category") for c in req.cart_item_ids}
            remaining = [iid for iid in rest_menu if iid not in cart_set and iid not in seen]
            # Sort: prefer different categories than what's in cart, then by price
            remaining.sort(key=lambda iid: (
                item_catalog.get(iid, {}).get("item_category", "") in cart_cats,
                item_catalog.get(iid, {}).get("item_price", 0),
            ))
            for iid in remaining:
                info = item_catalog.get(iid, {})
                if not info:
                    continue
                recos.append({
                    "item_id": iid,
                    "score": 0.1,
                    "item_name": info["item_name"],
                    "item_category": info["item_category"],
                    "item_price": info["item_price"],
                    "explanation": f"Popular at this restaurant - try {info['item_name']}!",
                })
                if len(recos) >= req.top_n:
                    break

        total_ms = (time.perf_counter() - start) * 1000
        return {
            "recommendations": recos,
            "latency_ms": resp.latency_ms,
            "total_ms": round(total_ms, 2),
            "cart_value": sum(item_catalog.get(i, {}).get("item_price", 0) for i in req.cart_item_ids),
            "potential_aov_increase": sum(r.get("item_price", 0) for r in recos[:3]),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
def stats():
    return service.get_latency_stats()


# ---------------------------------------------------------------------------
# Dashboard HTML
# ---------------------------------------------------------------------------
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CSAO Recommendation Engine — Live Demo</title>
<style>
  :root {
    --zomato-red: #E23744;
    --zomato-dark: #1C1C1C;
    --zomato-light: #F8F8F8;
    --card-bg: #FFFFFF;
    --border: #E8E8E8;
    --success: #48BB78;
    --warning: #ECC94B;
    --info: #4299E1;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Segoe UI', -apple-system, sans-serif; background: var(--zomato-light); color: var(--zomato-dark); }

  .header {
    background: var(--zomato-red);
    color: white;
    padding: 16px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
  }
  .header h1 { font-size: 22px; font-weight: 700; }
  .header .badge { background: rgba(255,255,255,0.2); padding: 4px 12px; border-radius: 12px; font-size: 13px; }

  .container { max-width: 1400px; margin: 0 auto; padding: 24px; display: grid; grid-template-columns: 340px 1fr 400px; gap: 20px; }

  .panel {
    background: var(--card-bg);
    border-radius: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    overflow: hidden;
  }
  .panel-header {
    padding: 14px 18px;
    font-weight: 600;
    font-size: 15px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .panel-body { padding: 14px 18px; max-height: 70vh; overflow-y: auto; }

  /* Selectors */
  select, input {
    width: 100%;
    padding: 8px 12px;
    border: 1px solid var(--border);
    border-radius: 8px;
    font-size: 14px;
    margin-bottom: 10px;
    outline: none;
  }
  select:focus, input:focus { border-color: var(--zomato-red); }

  /* Menu items */
  .menu-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 12px;
    border-radius: 8px;
    cursor: pointer;
    transition: background 0.15s;
    margin-bottom: 4px;
  }
  .menu-item:hover { background: #FFF5F5; }
  .menu-item.in-cart { background: #FED7D7; }
  .menu-item .name { font-size: 14px; font-weight: 500; flex: 1; }
  .menu-item .cat-badge {
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 10px;
    background: #EDF2F7;
    color: #4A5568;
    margin: 0 8px;
    white-space: nowrap;
  }
  .menu-item .price { font-weight: 600; color: var(--zomato-dark); white-space: nowrap; }
  .add-btn {
    background: var(--zomato-red);
    color: white;
    border: none;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 12px;
    cursor: pointer;
    margin-left: 8px;
  }
  .remove-btn { background: #718096; }

  /* Cart */
  .cart-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid #F0F0F0;
    font-size: 14px;
  }
  .cart-total {
    font-weight: 700;
    font-size: 16px;
    padding: 12px 0;
    display: flex;
    justify-content: space-between;
    border-top: 2px solid var(--zomato-dark);
    margin-top: 8px;
  }

  /* Get Recommendations Button */
  .reco-btn {
    width: 100%;
    padding: 12px;
    background: var(--zomato-red);
    color: white;
    font-size: 15px;
    font-weight: 600;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    margin-top: 12px;
    transition: opacity 0.2s;
  }
  .reco-btn:hover { opacity: 0.9; }
  .reco-btn:disabled { background: #CBD5E0; cursor: not-allowed; }

  /* Recommendations */
  .reco-card {
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 10px;
    position: relative;
    transition: box-shadow 0.2s;
  }
  .reco-card:hover { box-shadow: 0 2px 8px rgba(226,55,68,0.15); }
  .reco-rank {
    position: absolute;
    top: -8px;
    left: 12px;
    background: var(--zomato-red);
    color: white;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    font-weight: 700;
  }
  .reco-name { font-weight: 600; font-size: 15px; margin-bottom: 4px; padding-left: 24px; }
  .reco-meta { display: flex; gap: 10px; font-size: 13px; color: #718096; margin-bottom: 8px; }
  .reco-explanation {
    background: #FFFBEB;
    border-left: 3px solid var(--warning);
    padding: 8px 12px;
    font-size: 13px;
    border-radius: 0 6px 6px 0;
    margin-bottom: 6px;
  }
  .reco-tags { display: flex; gap: 6px; flex-wrap: wrap; }
  .reco-tag {
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 10px;
    background: #EBF8FF;
    color: var(--info);
  }
  .reco-score-bar {
    height: 4px;
    background: #EDF2F7;
    border-radius: 2px;
    margin-top: 6px;
    overflow: hidden;
  }
  .reco-score-fill { height: 100%; background: var(--zomato-red); border-radius: 2px; transition: width 0.5s; }

  /* Latency panel */
  .latency-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 16px; }
  .latency-card {
    background: #F7FAFC;
    border-radius: 8px;
    padding: 10px;
    text-align: center;
  }
  .latency-value { font-size: 22px; font-weight: 700; color: var(--zomato-red); }
  .latency-label { font-size: 12px; color: #718096; }
  .sla-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 13px;
    font-weight: 600;
  }
  .sla-pass { background: #C6F6D5; color: #276749; }
  .sla-fail { background: #FED7D7; color: #9B2C2C; }

  /* AOV Impact */
  .impact-card {
    background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%);
    color: white;
    border-radius: 10px;
    padding: 16px;
    margin-top: 12px;
  }
  .impact-card h4 { font-size: 13px; opacity: 0.85; margin-bottom: 6px; }
  .impact-value { font-size: 28px; font-weight: 700; }

  .empty-state { text-align: center; padding: 40px 20px; color: #A0AEC0; }
  .empty-state .icon { font-size: 48px; margin-bottom: 12px; }
  .spinner { display: inline-block; width: 18px; height: 18px; border: 2px solid rgba(255,255,255,0.3); border-top-color: white; border-radius: 50%; animation: spin 0.6s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }

  @media (max-width: 1200px) { .container { grid-template-columns: 1fr; } }
</style>
</head>
<body>

<div class="header">
  <h1>&#127829; CSAO Recommendation Engine</h1>
  <div>
    <span class="badge">LightGBM Ranker</span>
    <span class="badge">59 Features</span>
    <span class="badge">MMR Diversity</span>
    <span class="badge">LLM Explanations</span>
  </div>
</div>

<div class="container">
  <!-- LEFT: Restaurant & Menu -->
  <div>
    <div class="panel">
      <div class="panel-header">&#127860; Restaurant & Menu</div>
      <div class="panel-body">
        <label style="font-size:13px;color:#718096;">Select Restaurant</label>
        <select id="restaurantSelect" onchange="loadMenu()">
          <option value="">-- Choose a restaurant --</option>
        </select>
        <label style="font-size:13px;color:#718096;">User ID</label>
        <select id="userSelect">
          <option value="">-- Choose a user --</option>
        </select>
        <div id="menuContainer">
          <div class="empty-state">
            <div class="icon">&#127869;</div>
            <p>Select a restaurant to see the menu</p>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- CENTER: Cart + Recommendations -->
  <div>
    <div class="panel" style="margin-bottom: 20px;">
      <div class="panel-header">&#128722; Your Cart <span id="cartCount" style="background:var(--zomato-red);color:white;padding:2px 8px;border-radius:10px;font-size:12px;margin-left:8px;">0</span></div>
      <div class="panel-body">
        <div id="cartContainer">
          <div class="empty-state">
            <div class="icon">&#128722;</div>
            <p>Add items from the menu to build your cart</p>
          </div>
        </div>
        <button class="reco-btn" id="recoBtn" onclick="getRecommendations()" disabled>
          &#10024; Get CSAO Recommendations
        </button>
      </div>
    </div>

    <div class="panel">
      <div class="panel-header">&#127775; Recommendations <span id="recoCount" style="background:var(--info);color:white;padding:2px 8px;border-radius:10px;font-size:12px;margin-left:8px;">0</span></div>
      <div class="panel-body" id="recoContainer">
        <div class="empty-state">
          <div class="icon">&#129300;</div>
          <p>Build a cart and click "Get Recommendations" to see AI-powered suggestions</p>
        </div>
      </div>
    </div>
  </div>

  <!-- RIGHT: Metrics -->
  <div>
    <div class="panel" style="margin-bottom: 20px;">
      <div class="panel-header">&#9889; Latency Breakdown</div>
      <div class="panel-body" id="latencyContainer">
        <div class="empty-state"><p>Make a request to see latency metrics</p></div>
      </div>
    </div>

    <div class="panel">
      <div class="panel-header">&#128200; Business Impact</div>
      <div class="panel-body" id="impactContainer">
        <div class="empty-state"><p>Make a request to see impact metrics</p></div>
      </div>
    </div>
  </div>
</div>

<script>
const cart = [];  // [{item_id, item_name, item_category, item_price}]
let currentMenu = [];
let selectedRestaurant = '';

// Init
async function init() {
  const [restaurants, users] = await Promise.all([
    fetch('/api/restaurants').then(r => r.json()),
    fetch('/api/users').then(r => r.json()),
  ]);
  const rSel = document.getElementById('restaurantSelect');
  restaurants.forEach(r => {
    const opt = document.createElement('option');
    opt.value = r.id;
    opt.textContent = `${r.name} — ${r.cuisine} (${r.city}) [${r.item_count} items]`;
    rSel.appendChild(opt);
  });
  const uSel = document.getElementById('userSelect');
  users.forEach(u => {
    const opt = document.createElement('option');
    opt.value = u;
    opt.textContent = u;
    uSel.appendChild(opt);
  });
  // Auto-select first
  if (restaurants.length) { rSel.value = restaurants[0].id; loadMenu(); }
  if (users.length) { uSel.value = users[0]; }
}

async function loadMenu() {
  selectedRestaurant = document.getElementById('restaurantSelect').value;
  if (!selectedRestaurant) return;
  const items = await fetch(`/api/restaurant/${selectedRestaurant}/items`).then(r => r.json());
  currentMenu = items;
  renderMenu();
}

function renderMenu() {
  const container = document.getElementById('menuContainer');
  if (!currentMenu.length) {
    container.innerHTML = '<div class="empty-state"><p>No items found</p></div>';
    return;
  }
  const cartIds = new Set(cart.map(c => c.item_id));
  const catColors = { main_course: '#FED7D7', beverage: '#C6F6D5', dessert: '#FEFCBF', starter: '#BEE3F8', addon: '#E9D8FD' };
  container.innerHTML = currentMenu.map(item => {
    const inCart = cartIds.has(item.item_id);
    return `<div class="menu-item ${inCart ? 'in-cart' : ''}" onclick="toggleCart('${item.item_id}')">
      <span class="name">${item.item_name}</span>
      <span class="cat-badge" style="background:${catColors[item.item_category] || '#EDF2F7'}">${item.item_category}</span>
      <span class="price">&#8377;${item.item_price.toFixed(0)}</span>
      <button class="add-btn ${inCart ? 'remove-btn' : ''}">${inCart ? '✕' : '+'}</button>
    </div>`;
  }).join('');
}

function toggleCart(itemId) {
  const idx = cart.findIndex(c => c.item_id === itemId);
  if (idx >= 0) {
    cart.splice(idx, 1);
  } else {
    const item = currentMenu.find(m => m.item_id === itemId);
    if (item) cart.push({...item});
  }
  renderCart();
  renderMenu();
}

function renderCart() {
  const container = document.getElementById('cartContainer');
  const count = document.getElementById('cartCount');
  const btn = document.getElementById('recoBtn');
  count.textContent = cart.length;
  btn.disabled = cart.length === 0;

  if (!cart.length) {
    container.innerHTML = '<div class="empty-state"><div class="icon">&#128722;</div><p>Add items from the menu</p></div>';
    return;
  }
  const total = cart.reduce((s, c) => s + c.item_price, 0);
  const catColors = { main_course: '#FED7D7', beverage: '#C6F6D5', dessert: '#FEFCBF', starter: '#BEE3F8', addon: '#E9D8FD' };
  container.innerHTML = cart.map(item => `
    <div class="cart-item">
      <span>${item.item_name} <span class="cat-badge" style="background:${catColors[item.item_category] || '#EDF2F7'};font-size:10px;">${item.item_category}</span></span>
      <span>&#8377;${item.item_price.toFixed(0)} <button class="add-btn remove-btn" onclick="toggleCart('${item.item_id}')" style="font-size:10px;">✕</button></span>
    </div>
  `).join('') + `<div class="cart-total"><span>Cart Total</span><span>&#8377;${total.toFixed(0)}</span></div>`;
}

async function getRecommendations() {
  if (!cart.length) return;
  const btn = document.getElementById('recoBtn');
  btn.innerHTML = '<span class="spinner"></span> Loading...';
  btn.disabled = true;

  const userId = document.getElementById('userSelect').value || 'u_00010';
  try {
    const resp = await fetch('/api/recommend', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        user_id: userId,
        restaurant_id: selectedRestaurant,
        cart_item_ids: cart.map(c => c.item_id),
        top_n: 10,
      }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || 'Request failed');
    renderRecommendations(data.recommendations);
    renderLatency(data.latency_ms, data.total_ms);
    renderImpact(data);
  } catch (e) {
    document.getElementById('recoContainer').innerHTML = `<div class="empty-state" style="color:red;">Error: ${e.message}</div>`;
  }
  btn.innerHTML = '&#10024; Get CSAO Recommendations';
  btn.disabled = false;
}

function renderRecommendations(recos) {
  const container = document.getElementById('recoContainer');
  document.getElementById('recoCount').textContent = recos.length;
  const maxScore = Math.max(...recos.map(r => r.rank_score || 0), 0.01);
  const catColors = { main_course: '#FED7D7', beverage: '#C6F6D5', dessert: '#FEFCBF', starter: '#BEE3F8', addon: '#E9D8FD' };

  container.innerHTML = recos.map((r, i) => `
    <div class="reco-card">
      <div class="reco-rank">${i + 1}</div>
      <div class="reco-name">${r.item_name || r.item_id}</div>
      <div class="reco-meta">
        <span class="cat-badge" style="background:${catColors[r.item_category] || '#EDF2F7'}">${r.item_category}</span>
        <span>&#8377;${(r.item_price || 0).toFixed(0)}</span>
        <span>Score: ${(r.rank_score || 0).toFixed(3)}</span>
      </div>
      ${r.explanation ? `<div class="reco-explanation">&#128161; ${r.explanation}</div>` : ''}
      <div class="reco-tags">${(r.reason_tags || []).map(t => `<span class="reco-tag">${t}</span>`).join('')}</div>
      <div class="reco-score-bar"><div class="reco-score-fill" style="width:${((r.rank_score || 0) / maxScore * 100).toFixed(1)}%"></div></div>
    </div>
  `).join('');
}

function renderLatency(latency, totalMs) {
  const container = document.getElementById('latencyContainer');
  const sla = 300;
  const pass = totalMs <= sla;
  const stages = Object.entries(latency).filter(([k]) => k !== 'total');

  container.innerHTML = `
    <div class="latency-grid">
      <div class="latency-card">
        <div class="latency-value">${totalMs.toFixed(1)}</div>
        <div class="latency-label">Total (ms)</div>
      </div>
      ${stages.map(([k, v]) => `
        <div class="latency-card">
          <div class="latency-value" style="font-size:18px;">${v.toFixed(1)}</div>
          <div class="latency-label">${k}</div>
        </div>
      `).join('')}
    </div>
    <div style="text-align:center;">
      <span class="sla-badge ${pass ? 'sla-pass' : 'sla-fail'}">
        SLA ${sla}ms: ${pass ? '&#10003; PASS' : '&#10007; FAIL'} (${totalMs.toFixed(1)}ms)
      </span>
    </div>
  `;
}

function renderImpact(data) {
  const container = document.getElementById('impactContainer');
  const potentialIncrease = data.potential_aov_increase || 0;
  const cartValue = data.cart_value || 0;
  const pctIncrease = cartValue > 0 ? (potentialIncrease / cartValue * 100) : 0;
  const uniqueCats = new Set((data.recommendations || []).map(r => r.item_category));

  container.innerHTML = `
    <div class="impact-card">
      <h4>Potential AOV Increase (Top 3)</h4>
      <div class="impact-value">+&#8377;${potentialIncrease.toFixed(0)}</div>
      <div style="opacity:0.8;font-size:13px;">+${pctIncrease.toFixed(1)}% cart value uplift</div>
    </div>
    <div style="margin-top:14px;">
      <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #eee;">
        <span style="color:#718096;font-size:13px;">Cart Value</span>
        <span style="font-weight:600;">&#8377;${cartValue.toFixed(0)}</span>
      </div>
      <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #eee;">
        <span style="color:#718096;font-size:13px;">Recos Shown</span>
        <span style="font-weight:600;">${(data.recommendations || []).length}</span>
      </div>
      <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #eee;">
        <span style="color:#718096;font-size:13px;">Category Diversity</span>
        <span style="font-weight:600;">${uniqueCats.size} categories</span>
      </div>
      <div style="display:flex;justify-content:space-between;padding:8px 0;">
        <span style="color:#718096;font-size:13px;">System</span>
        <span style="font-weight:600;">LightGBM + MMR + LLM</span>
      </div>
    </div>
  `;
}

init();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("  CSAO Live Demo — http://localhost:8000")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
