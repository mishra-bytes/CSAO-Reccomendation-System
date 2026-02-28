# A/B Testing Plan

## Hypothesis

Cart-aware complementarity ranking increases add-on attach rate and order value.

## Experiment Design

- Control: popularity-only add-ons
- Treatment: CSAO candidate + ranking stack
- Unit: user-session
- Duration: 2-3 weeks

## Primary Metrics

- Add-on attach rate
- Incremental gross merchandise value
- Conversion uplift for add-on clicks

## Guardrails

- Checkout completion rate
- App latency (p95)
- Recommendation hide/dismiss rate

## Production TODOs

- Add sequential testing correction
- Add regional segmentation and restaurant-category stratification
