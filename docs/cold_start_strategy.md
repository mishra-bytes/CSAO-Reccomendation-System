# Cold Start Strategy

Flow-style hierarchy in `serving/pipeline/cold_start.py`:

1. **Warm user?**
   - Yes → main pipeline
   - No → continue
2. **New user + non-empty cart + known restaurant**
   - Use restaurant popularity + category diversity from cart
3. **New user + non-empty cart + unknown restaurant**
   - Use global popularity + category diversity
4. **New user + empty cart + known restaurant**
   - Use restaurant popularity + mealtime weighting
5. **Worst case (new user + empty cart + unknown restaurant)**
   - Use globally popular, category-diverse fallback

`tests/test_cold_start.py` validates no-crash and context-aware outputs.
