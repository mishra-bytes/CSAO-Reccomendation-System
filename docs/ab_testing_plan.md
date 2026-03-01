# A/B Testing Plan — CSAO Recommendation Engine

## 1. Executive Summary

This document defines the production A/B testing strategy for Zomato's Cart Super Add-On (CSAO) recommendation engine.  The test validates whether **cart-aware complementarity ranking with LLM-powered explanations** increases add-on attach rate and order value compared to the current popularity-based approach.

---

## 2. Hypothesis

**H₀ (null):** CSAO ranking does not increase the add-on attach rate compared to popularity-only add-ons.  
**H₁ (alternative):** CSAO ranking increases add-on attach rate by ≥ 2 percentage points (absolute lift).

| Dimension | Metric | Expected Lift |
|-----------|--------|--------------|
| Primary | Add-on attach rate | +2.0 pp (e.g., 8% → 10%) |
| Primary | Incremental GMV per order | +₹15–25 |
| Secondary | Add-on click-through rate | +1.5 pp |
| Secondary | Items per order (basket size) | +0.15 items |

---

## 3. Power Analysis & Sample Size

### Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Significance level (α) | 0.05 | Standard two-sided test |
| Statistical power (1−β) | 0.80 | Industry standard |
| Baseline attach rate (p₀) | 0.08 | Current production estimate |
| Minimum Detectable Effect (MDE) | 0.02 (absolute) | +2 pp lift (8% → 10%) |
| Test variant attach rate (p₁) | 0.10 | p₀ + MDE |
| Allocation ratio | 50/50 | Equal traffic split |

### Sample Size Calculation

Using the two-proportion z-test formula:

$$n = \frac{(Z_{\alpha/2} + Z_\beta)^2 \cdot [p_0(1-p_0) + p_1(1-p_1)]}{(p_1 - p_0)^2}$$

Where:
- $Z_{\alpha/2} = 1.96$ (two-sided, α = 0.05)
- $Z_\beta = 0.84$ (power = 0.80)
- $p_0 = 0.08$, $p_1 = 0.10$

$$n = \frac{(1.96 + 0.84)^2 \cdot [0.08 \times 0.92 + 0.10 \times 0.90]}{(0.02)^2}$$

$$n = \frac{7.84 \times [0.0736 + 0.09]}{0.0004} = \frac{7.84 \times 0.1636}{0.0004} = \frac{1.283}{0.0004} \approx 3{,}207$$

**Per arm: ~3,207 user-sessions.  Total: ~6,414 user-sessions.**

At Zomato's scale (~1 M daily orders in top metros), this requires approximately **1–2 days** of traffic at 1% allocation, or **6–12 hours** at 5% allocation.

### Sensitivity Table

| MDE (pp) | n per arm | Total sessions | Days @ 5% traffic |
|----------|-----------|----------------|-------------------|
| 1.0 | 12,522 | 25,044 | 1.3 |
| 1.5 | 5,605 | 11,210 | 0.6 |
| **2.0** | **3,207** | **6,414** | **0.3** |
| 3.0 | 1,440 | 2,880 | 0.2 |

---

## 4. Experiment Design

### Randomisation Unit
**User-session** — each unique (user_id, session_id) pair is assigned to exactly one variant for the duration of the session.  This prevents within-session contamination while allowing users to see different variants across sessions.

### Variants

| Group | Name | Description | Traffic |
|-------|------|-------------|---------|
| Control | `popularity_baseline` | Current popularity-only add-on suggestions | 50% |
| Treatment | `csao_ranker_v1` | CSAO candidate generation + LightGBM ranking + MMR diversity + LLM explanations | 50% |

### Stratification
Randomise **within strata** to ensure balanced covariates:
- **City tier**: Metro (Delhi, Mumbai, Bangalore) vs. Tier-2 (Pune, Hyderabad, Chennai)
- **Restaurant cuisine**: North Indian, South Indian, Chinese, Biryani, Other
- **Cart value band**: Low (<₹300), Mid (₹300–600), High (>₹600)
- **User tenure**: New (<30 days) vs. Returning (30+ days)

### Ramp-Up Schedule

| Phase | Traffic % | Duration | Purpose |
|-------|-----------|----------|---------|
| Burn-in | 1% | 2 days | Detect crashes, latency regressions |
| Ramp 1 | 5% | 3 days | Early signal, guardrail validation |  
| Ramp 2 | 25% | 5 days | Reach statistical significance |
| Full | 50% | 7+ days | Confirm stability across segments |

---

## 5. Sequential Testing Design

We use **group sequential testing** with **O'Brien-Fleming spending function** to allow valid early stopping while controlling Type I error.

### Why Sequential Testing?
- Fixed-horizon tests waste time if the effect is large and obvious early.
- Continuous monitoring without correction inflates false positive rate to 20–30%.
- O'Brien-Fleming boundaries are conservative early (hard to stop) but permissive late (easy to conclude).

### Interim Analyses

| Analysis | Information fraction | O'Brien-Fleming boundary (z) | Equivalent p-value |
|----------|---------------------|-------------------------------|-------------------|
| 1st look | 25% (day 3) | ±4.05 | 0.00005 |
| 2nd look | 50% (day 7) | ±2.86 | 0.0042 |
| 3rd look | 75% (day 10) | ±2.34 | 0.019 |
| Final | 100% (day 14) | ±2.02 | 0.043 |

**Overall α = 0.05 preserved across all looks.**

### Stopping Rules
- **Stop for efficacy**: If test statistic exceeds boundary at any interim analysis.
- **Stop for futility**: If conditional power < 10% at 50%+ information fraction.
- **Stop for harm**: If guardrail metric degrades > 1% with p < 0.01.

---

## 6. Metrics Framework

### Primary Metrics (Decision Metrics)

| Metric | Definition | Success Threshold |
|--------|-----------|-------------------|
| **Add-on attach rate** | `sessions_with_addon_added / sessions_with_reco_shown` | ≥ +2 pp lift |
| **Incremental GMV** | `mean(addon_price × quantity) per session` | ≥ +₹15/session |

### Secondary Metrics (Supportive)

| Metric | Definition | Expected Direction |
|--------|-----------|-------------------|
| Add-on CTR | Clicks on add-on panel / impressions | ↑ |
| Items per order | Mean basket size | ↑ |
| Recommendation diversity | Unique items shown / total slots | ↑ |
| Explanation engagement | Users who expand explanation tooltip | ↑ |

### Guardrail Metrics (Must Not Degrade)

| Metric | Threshold | Action if Violated |
|--------|-----------|-------------------|
| Checkout completion rate | Δ ≤ −0.5 pp | Pause experiment |
| Recommendation latency p95 | ≤ 200 ms | Pause, investigate |
| App crash rate | Δ ≤ +0.1 pp | Roll back immediately |
| Cart abandonment rate | Δ ≤ +1 pp | Pause experiment |
| Recommendation dismiss rate | Δ ≤ +2 pp | Reduce treatment traffic |

---

## 7. Segmented Analysis Plan

Post-experiment, compute treatment effects within:

| Segment | Why |
|---------|-----|
| New vs. returning users | Cold-start behavior differs |
| City tier | Food preferences vary by region |
| Cart value bands | High-value carts may have different add-on affinity |
| Time of day (meal period) | Lunch vs. dinner vs. snack |
| Cuisine type | Complementarity strength varies |
| Cart completeness (from model) | CSAO should shine when carts are "incomplete" |

Use **Bonferroni correction** (α/m) for multiple comparisons in segment-level tests.

---

## 8. Network Effect & Spillover Mitigation

- **No interference**: User-session randomisation ensures one user's treatment doesn't affect another user's control experience.
- **Restaurant-side spillover**: If CSAO drives more add-on orders for certain restaurants, their popularity scores change. Mitigated by using **pre-experiment popularity snapshots** in the control arm.
- **Novelty/primacy effects**: Monitor lift trajectory over time — if lift decays after day 7, it may be novelty.

---

## 9. Launch Decision Framework

| Outcome | Action |
|---------|--------|
| Primary metric significant + guardrails hold | **Ship 100%** |
| Primary metric significant + guardrail violated | Investigate root cause, re-test with fix |
| Primary metric not significant | Check secondary metrics, extend test or iterate |
| Primary metric significant negative | Roll back, analyse failure mode |

---

## 10. Implementation Checklist

- [ ] Feature flag: `csao_ranker_enabled` (boolean, per session)
- [ ] Logging: Emit `reco_shown`, `reco_clicked`, `addon_added` events with variant ID
- [ ] Metrics pipeline: Configure experiment dashboard with sequential boundaries
- [ ] Latency monitoring: Real-time p95 alert for treatment arm
- [ ] Rollback trigger: Automated if crash rate or latency exceeds threshold
- [ ] Data quality: Validate no logging discrepancies between arms (SRM check)

### Sample Ratio Mismatch (SRM) Check
Run a χ² test daily to verify the 50/50 split holds:

$$\chi^2 = \frac{(n_C - n_T)^2}{n_C + n_T}$$

Alert if p < 0.001 — indicates randomisation bug.

---

## 11. Timeline

| Week | Activity |
|------|----------|
| W1 | Instrument logging, deploy feature flag, burn-in at 1% |
| W2 | Ramp to 25%, first interim analysis |
| W3 | Full 50% traffic, second + third interim analyses |
| W4 | Final analysis, decision, documentation |

**Total experiment duration: 2–4 weeks** (depending on early stopping).
