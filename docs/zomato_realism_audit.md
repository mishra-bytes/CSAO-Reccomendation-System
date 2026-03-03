# Zomato-Realism Audit — CSAO Recommendation System

> **Audit Date**: 2025-07-17  
> **Scope**: Data generation, domain logic, feature engineering, candidate retrieval  

---

## Dimension Scores

| # | Dimension | Score | Verdict |
|---|-----------|-------|---------|
| A | Cuisine Compatibility | **8/10** | Strong — restaurant-menu gate prevents cross-cuisine leakage |
| B | Meal Completion Logic | **8/10** | Strong — 6 archetypes, MealGapRetriever, course-type penalty |
| C | Price Realism | **9/10** | Excellent — 582 hand-calibrated prices matching live Zomato menus |
| D | Order Patterns | **7/10** | Good — lunch/dinner peaks, loyalty model, but no day-of-week/seasonal |
| E | Veg/Non-Veg Segregation | **2/10** | 🔴 Critical gap — no `is_veg` field anywhere in schemas or features |
| F | City/Region Diversity | **4/10** | 🔴 Major gap — city-cuisine assignment is random; no regional affinity |
| G | Restaurant Realism | **8/10** | Strong — menu gating, authentic names, realistic menu sizes |
| H | Cold-Start Handling | **9/10** | Excellent — 5-strategy cascade, 4 test scenarios, sub-100ms latency |

**Composite Realism Score: 6.9/10**

---

## A. Cuisine Compatibility (8/10)

**Strengths:**
- Restaurant-menu gate in `CandidateGenerator.generate()` ensures candidates MUST exist on the restaurant's actual menu — structurally prevents "Sushi + Biryani" disasters
- 8 cuisine-specific `PAIRING_BOOSTS` dicts with culturally accurate pairings (Butter Chicken → Butter Naan, Masala Dosa → Sambar, Biryani → Raita)
- 4× selection probability boost for within-cuisine pairings
- Each restaurant assigned a single cuisine; menu drawn only from that cuisine's dish pool

**Gaps:**
- Multi-cuisine restaurants (very common on Zomato — e.g., "North Indian, Chinese, Mughlai") are not modeled
- When `menu_items` is empty (new restaurant), the gate is bypassed, allowing cross-cuisine leakage

---

## B. Meal Completion Logic (8/10)

**Strengths:**
- Six meal archetypes: Quick (28%), Standard (25%), Full (20%), Party (10%), Snack (10%), Dessert-meal (7%)
- MealGapRetriever identifies missing categories from best-matching archetype
- Course-type penalty: duplicate mains get 0.3× score multiplier
- Category-weighted scoring: `main_course` 1.5×, `beverage`/`dessert` 1.2×

**Gaps:**
- Bread/rice not modeled as distinct meal components — naan/roti are `addon`, rice sometimes `main_course`
- No must-have item logic (curry without bread should trigger stronger signal than missing dessert)

---

## C. Price Realism (9/10)

**Strengths:**
- 582 dishes with individually calibrated min/max prices: Cutting Chai ₹20, Samosa ₹80, Shorshe Ilish ₹440
- Price sensitivity modeled: `price_sensitivity = avg_item_price_user / global_median_price`
- User segments: budget (<₹250 AOV), mid (₹250–500), premium (>₹500)
- Per-restaurant stable pricing

**Gaps:**
- No discount/offer modeling (Zomato heavily uses coupons)
- Fallback generator in `loaders.py` uses uniform ₹50–450 for ALL categories

---

## D. Order Patterns (7/10)

**Strengths:**
- Realistic cart sizes: Quick=2–3, Standard=3–4, Full=5–7, Party=7–12+ (mean 3.8)
- Temporal peaks: lunch (11:00–14:00, weight 3.0), dinner (18:00–22:00, weight 4.0)
- User loyalty: 70% favorite restaurants, 30% exploration
- Zipf-like restaurant popularity: `1/rank^0.6`

**Gaps:**
- No day-of-week variation (weekday vs. weekend)
- No seasonal/festival variation (Ramadan, Diwali)
- No heavy vs. light user modeling (all users ~10 orders avg)

---

## E. Veg/Non-Veg Segregation (2/10) 🔴

**This is the single most critical gap for Indian food realism.**

A judge familiar with Zomato would immediately notice the absence of the prominent 🟢/🔴 veg/non-veg indicator that appears on every single item in the Zomato app.

**What's missing:**
- No `is_veg` field in item schema or any data file
- No user dietary preference modeling
- No veg-only restaurant modeling (Saravana Bhavan, Bikanervala, Rajdhani)
- No filtering of non-veg recommendations for vegetarian users
- A strictly vegetarian user could be recommended Butter Chicken

**Mitigation:** The restaurant-menu gate and cuisine-specific pairings provide indirect veg clustering (South Indian restaurants are naturally veg-heavy), but this is incidental, not intentional.

---

## F. City/Region Diversity (4/10) 🔴

**What exists:**
- 7 metros: Delhi, Mumbai, Bangalore, Hyderabad, Pune, Chennai, Kolkata
- `user_cuisine_share__*` columns in features
- Defense doc claims tiered city behavior

**What's actually implemented:**
- City assignment to restaurants is `rng.integers(0, len(CITIES))` — **completely random**
- No city-cuisine affinity (Bengali restaurants equally likely in Delhi as Kolkata)
- No city-tier AOV differences (all cities use same price ranges)
- No city feature in ranking model
- Defense doc claims don't match code reality

---

## G. Restaurant Realism (8/10)

**Strengths:**
- Menus of 25–45 items with balanced categories (30% mains, 28% addons, 15% starters)
- Restaurant-menu gate ensures candidates come from actual menu
- Authentic names per cuisine (Karim's → Mughlai, Behrouz → Biryani)
- 500 restaurants across 8 cuisines with realistic distribution

**Gaps:**
- Single-cuisine-per-restaurant only (multi-cuisine is the norm on Zomato)
- No restaurant rating/quality modeling
- No operational constraints (hours, delivery radius, minimum order)

---

## H. Cold-Start Handling (9/10)

**Strengths:**
- 5-strategy cascade: Co-occurrence → Session co-visit → Meal-gap → Category → Popularity
- 4 tested scenarios: unseen user (49ms), unseen restaurant (62ms), unseen item (25ms), triple cold (34ms)
- Neural reranker has hash-based fallback for unseen items
- `fill_candidates` ensures target count always met

**Gaps:**
- Hash-based fallback embeddings carry no semantic meaning
- No cold-start for new cuisines/food categories

---

## Top 5 Most Unrealistic Aspects

| # | Issue | Severity | Fix Complexity |
|---|-------|----------|----------------|
| 1 | No veg/non-veg segregation — fundamental Indian food attribute absent | 🔴 Critical | Medium (2–3h) |
| 2 | City-cuisine assignment is random — defense doc claims don't match code | 🔴 High | Low (30min) |
| 3 | Single-cuisine restaurants only — multi-cuisine is the norm | 🟡 Medium | Medium (1h) |
| 4 | Bread/rice not modeled as distinct meal components | 🟡 Medium | Low (30min) |
| 5 | No day-of-week or seasonal order variation | 🟡 Low | Low (30min) |

## Top 5 Most Realistic/Impressive Aspects

| # | Aspect | Why It Matters |
|---|--------|----------------|
| 1 | Restaurant-menu gate | Production-grade cross-cuisine prevention |
| 2 | 8 cuisine-specific pairing boosts | Culturally accurate (Biryani→Raita, Dosa→Sambar) |
| 3 | MealGapRetriever | Genuinely CSAO-native intelligence |
| 4 | 582 hand-calibrated prices | Match actual Zomato menus |
| 5 | Cold-start cascade with 4 test scenarios | Graceful degradation, latency-tested |

---

## Remediation Priority

1. **Add `is_veg` to item catalog + veg user inference** — Highest realism lift per effort
2. **Add city-cuisine affinity weights** — Fix the claim-vs-code discrepancy
3. **Add `bread`/`rice` as meal component category** — Indian dining fundamental
4. **Add day-of-week temporal patterns** — Easy realism gain
5. **Consider multi-cuisine restaurants** — Nice-to-have, lower priority
