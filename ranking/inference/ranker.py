from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from common.feature_names import normalize_feature_name
from features.cart_features import build_cart_feature_vector


class CSAORanker:
    """Vectorized LightGBM ranker optimised for low-latency serving.

    Key optimisations vs. the original per-candidate-loop design:
    1. Cart features computed ONCE per request (shared across all candidates).
    2. User features looked up ONCE and broadcast.
    3. Item features pre-indexed as a NumPy matrix for O(1) row fetch.
    4. Complementarity aggregated via vectorised NumPy ops.
    5. Feature matrix assembled as a single NumPy array (no dict→DataFrame).
    6. model.predict called ONCE on the full batch.
    7. MMR re-ranking for diversity — reduces fatigue (same-item-everywhere problem).
    """

    # MMR diversity parameter: 0 = pure diversity, 1 = pure relevance
    DEFAULT_MMR_LAMBDA = 0.7

    def __init__(
        self,
        model_path: str,
        feature_columns_path: str,
        user_features: pd.DataFrame,
        item_features: pd.DataFrame,
        items: pd.DataFrame,
        complementarity_lookup: dict[tuple[str, str], tuple[float, float]],
    ) -> None:
        self.model = joblib.load(model_path)
        loaded_cols = json.loads(Path(feature_columns_path).read_text(encoding="utf-8"))
        self.feature_columns: list[str] = [normalize_feature_name(c) for c in loaded_cols]
        self._col_to_idx: dict[str, int] = {c: i for i, c in enumerate(self.feature_columns)}
        self._n_features = len(self.feature_columns)

        # --- Pre-index user features as dict[user_id -> np.array] ---------
        uf = user_features.drop_duplicates("user_id").copy()
        self._user_cols: list[str] = [
            normalize_feature_name(f"user__{c}") for c in uf.columns if c != "user_id"
        ]
        self._user_raw_cols = [c for c in uf.columns if c != "user_id"]
        uf_numeric = uf[self._user_raw_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        self._user_vectors: dict[str, np.ndarray] = {}
        for uid, row in zip(uf["user_id"].astype(str), uf_numeric.values):
            self._user_vectors[uid] = row.astype(np.float64)

        # --- Pre-index item features as dict[item_id -> np.array] ---------
        itf = item_features.drop_duplicates("item_id").copy()
        self._item_cols: list[str] = [
            normalize_feature_name(f"item__{c}") for c in itf.columns if c not in {"item_id", "item_category"}
        ]
        self._item_raw_cols = [c for c in itf.columns if c not in {"item_id", "item_category"}]
        itf_numeric = itf[self._item_raw_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        self._item_vectors: dict[str, np.ndarray] = {}
        for iid, row in zip(itf["item_id"].astype(str), itf_numeric.values):
            self._item_vectors[iid] = row.astype(np.float64)
        self._item_zero = np.zeros(len(self._item_raw_cols), dtype=np.float64)

        # Item lookup for cart features
        self.items = items[["item_id", "item_category", "item_price"]].drop_duplicates("item_id")
        self.comp_lookup = complementarity_lookup

        # Pre-compute column index mapping for fast assembly
        self._cart_col_indices: dict[str, int] | None = None  # lazy
        self._user_col_indices: list[int] | None = None
        self._item_col_indices: list[int] | None = None
        self._static_col_indices: dict[str, int] = {
            name: self._col_to_idx[name]
            for name in [
                "candidate_score", "comp_max_lift", "comp_mean_lift",
                "comp_max_pmi", "comp_mean_pmi",
                "ctx_restaurant_hash", "ctx_user_hash",
            ]
            if name in self._col_to_idx
        }
        # Resolve user/item column indices once
        self._user_col_indices = [self._col_to_idx.get(c, -1) for c in self._user_cols]
        self._item_col_indices = [self._col_to_idx.get(c, -1) for c in self._item_cols]

    # ------------------------------------------------------------------
    # Vectorised complementarity for ALL candidates at once
    # ------------------------------------------------------------------
    def _complement_agg_batch(
        self, cart_items: list[str], candidate_ids: list[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (max_lift, mean_lift, max_pmi, mean_pmi) arrays of shape (n_candidates,)."""
        n_cand = len(candidate_ids)
        n_cart = len(cart_items)
        if n_cart == 0:
            z = np.zeros(n_cand, dtype=np.float64)
            return z, z.copy(), z.copy(), z.copy()

        lifts = np.zeros((n_cand, n_cart), dtype=np.float64)
        pmis = np.zeros((n_cand, n_cart), dtype=np.float64)
        lookup = self.comp_lookup
        for ci, cand in enumerate(candidate_ids):
            for cj, cart_item in enumerate(cart_items):
                pair = lookup.get((cart_item, cand))
                if pair is not None:
                    lifts[ci, cj] = pair[0]
                    pmis[ci, cj] = pair[1]
        return (
            lifts.max(axis=1),
            lifts.mean(axis=1),
            pmis.max(axis=1),
            pmis.mean(axis=1),
        )

    # ------------------------------------------------------------------
    # Vectorised feature matrix construction
    # ------------------------------------------------------------------
    def _build_feature_matrix(
        self,
        user_id: str,
        restaurant_id: str,
        cart_items: list[str],
        candidate_ids: list[str],
        candidate_scores: np.ndarray,
    ) -> np.ndarray:
        """Build (n_candidates x n_features) float64 matrix without per-row dicts."""
        n = len(candidate_ids)
        X = np.zeros((n, self._n_features), dtype=np.float64)

        # 1) Cart features — computed once, broadcast to all rows
        cart_feats = build_cart_feature_vector(cart_items, self.items)
        if self._cart_col_indices is None:
            self._cart_col_indices = {}
            for k, v in cart_feats.items():
                if isinstance(v, (float, int, np.floating)):
                    norm_k = normalize_feature_name(k)
                    idx = self._col_to_idx.get(norm_k, -1)
                    if idx >= 0:
                        self._cart_col_indices[k] = idx
        for k, idx in self._cart_col_indices.items():
            val = cart_feats.get(k)
            if val is not None and isinstance(val, (float, int, np.floating)):
                X[:, idx] = float(val)

        # Also handle dynamic cart category columns not in cache
        for k, v in cart_feats.items():
            if isinstance(v, (float, int, np.floating)):
                norm_k = normalize_feature_name(k)
                idx = self._col_to_idx.get(norm_k, -1)
                if idx >= 0:
                    X[:, idx] = float(v)

        # 2) Complementarity — vectorised batch
        max_lift, mean_lift, max_pmi, mean_pmi = self._complement_agg_batch(cart_items, candidate_ids)
        for name, arr in [
            ("comp_max_lift", max_lift), ("comp_mean_lift", mean_lift),
            ("comp_max_pmi", max_pmi), ("comp_mean_pmi", mean_pmi),
        ]:
            idx = self._static_col_indices.get(name, -1)
            if idx >= 0:
                X[:, idx] = arr

        # 3) Candidate scores
        idx = self._static_col_indices.get("candidate_score", -1)
        if idx >= 0:
            X[:, idx] = candidate_scores

        # 4) Context hashes — scalar, broadcast
        rest_hash = float(abs(hash(str(restaurant_id))) % 1000)
        user_hash = float(abs(hash(str(user_id))) % 1000)
        idx_r = self._static_col_indices.get("ctx_restaurant_hash", -1)
        idx_u = self._static_col_indices.get("ctx_user_hash", -1)
        if idx_r >= 0:
            X[:, idx_r] = rest_hash
        if idx_u >= 0:
            X[:, idx_u] = user_hash

        # 5) User features — looked up once, broadcast
        user_vec = self._user_vectors.get(str(user_id))
        if user_vec is not None:
            for vi, col_idx in enumerate(self._user_col_indices):
                if col_idx >= 0 and vi < len(user_vec):
                    X[:, col_idx] = user_vec[vi]

        # 6) Item features — vectorised lookup
        for ri, iid in enumerate(candidate_ids):
            item_vec = self._item_vectors.get(iid, self._item_zero)
            for vi, col_idx in enumerate(self._item_col_indices):
                if col_idx >= 0 and vi < len(item_vec):
                    X[ri, col_idx] = item_vec[vi]

        # 7) CSAO intelligence features — cart completion & complement confidence
        from features.cart_features import _cart_completeness_score, _missing_categories

        # Build cart category set once
        item_lookup_idx = self.items.set_index("item_id", drop=False)
        cart_categories: set[str] = set()
        for ci in cart_items:
            if ci in item_lookup_idx.index:
                r = item_lookup_idx.loc[ci]
                cat_val = str(r["item_category"]) if isinstance(r, pd.Series) else str(r.iloc[0]["item_category"])
                if cat_val != "unknown":
                    cart_categories.add(cat_val)

        completeness_before = _cart_completeness_score(cart_categories)
        _, missing_ratio_before = _missing_categories(cart_categories)

        # Per-candidate: completeness_delta, fills_gap, complement_confidence, new_category
        completeness_delta = np.zeros(n, dtype=np.float64)
        fills_gap = np.zeros(n, dtype=np.float64)
        complement_conf = np.zeros(n, dtype=np.float64)
        new_category = np.zeros(n, dtype=np.float64)

        for ri, cand_id in enumerate(candidate_ids):
            cand_cat = "unknown"
            if cand_id in item_lookup_idx.index:
                cr = item_lookup_idx.loc[cand_id]
                cand_cat = str(cr["item_category"]) if isinstance(cr, pd.Series) else str(cr.iloc[0]["item_category"])

            after_set = cart_categories | {cand_cat}
            comp_after = _cart_completeness_score(after_set)
            completeness_delta[ri] = comp_after - completeness_before

            _, mr_after = _missing_categories(after_set)
            fills_gap[ri] = float(mr_after < missing_ratio_before)

            n_lift = sum(1 for ci in cart_items if self.comp_lookup.get((ci, cand_id), (0.0, 0.0))[0] > 0)
            complement_conf[ri] = n_lift / max(len(cart_items), 1)

            new_category[ri] = float(cand_cat not in cart_categories and cand_cat != "unknown")

        for name, arr in [
            ("completeness_delta", completeness_delta),
            ("fills_meal_gap", fills_gap),
            ("complement_confidence", complement_conf),
            ("candidate_new_category", new_category),
        ]:
            idx = self._col_to_idx.get(name, -1)
            if idx >= 0:
                X[:, idx] = arr

        return X

    def _mmr_rerank(
        self,
        ids: list[str],
        scores: np.ndarray,
        X: np.ndarray,
        top_n: int,
        mmr_lambda: float | None = None,
    ) -> list[int]:
        """Maximal Marginal Relevance re-ranking for diversity.

        MMR selects items that are both high-scoring AND dissimilar from
        already-selected items, breaking the fatigue pattern where the
        same popular items are recommended to everyone.

        score_mmr(i) = λ * relevance(i) - (1-λ) * max_sim(i, selected)

        Uses the feature vectors as item representations for similarity.
        """
        lam = mmr_lambda if mmr_lambda is not None else self.DEFAULT_MMR_LAMBDA
        n = len(ids)
        if n <= top_n:
            return list(np.argsort(scores)[::-1])

        # Normalize scores to [0, 1]
        s_min, s_max = scores.min(), scores.max()
        if s_max > s_min:
            norm_scores = (scores - s_min) / (s_max - s_min)
        else:
            norm_scores = np.ones(n, dtype=np.float64)

        # Build item representations from feature matrix (L2-normalised)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        X_normed = X / norms

        selected: list[int] = []
        remaining = set(range(n))

        for _ in range(top_n):
            best_idx = -1
            best_mmr = -np.inf

            for idx in remaining:
                relevance = norm_scores[idx]

                if not selected:
                    max_sim = 0.0
                else:
                    sims = X_normed[idx] @ X_normed[selected].T
                    max_sim = float(np.max(sims))

                mmr_score = lam * relevance - (1 - lam) * max_sim
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = idx

            if best_idx >= 0:
                selected.append(best_idx)
                remaining.discard(best_idx)
            else:
                break

        return selected

    def rank(
        self,
        user_id: str,
        restaurant_id: str,
        cart_items: list[str],
        candidates: list[tuple[str, float]],
        top_n: int = 10,
        use_mmr: bool = True,
        mmr_lambda: float | None = None,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []

        ids = [str(c[0]) for c in candidates]
        cand_scores = np.array([float(c[1]) for c in candidates], dtype=np.float64)

        # Vectorised feature matrix — no per-candidate Python loops for dicts
        X = self._build_feature_matrix(user_id, restaurant_id, cart_items, ids, cand_scores)

        # Single batched prediction call
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            scores = self.model.predict(X)

        # MMR diversity re-ranking (default ON) — fixes fatigue problem
        if use_mmr and len(ids) > top_n:
            top_indices = self._mmr_rerank(ids, scores, X, top_n, mmr_lambda)
        elif len(scores) > top_n * 2:
            # Fast top-N via argpartition instead of full sort
            top_indices = np.argpartition(scores, -top_n)[-top_n:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        else:
            top_indices = np.argsort(scores)[::-1][:top_n]

        return [
            {
                "item_id": ids[i],
                "rank_score": float(scores[i]),
                "candidate_score": float(cand_scores[i]),
            }
            for i in top_indices
        ]
