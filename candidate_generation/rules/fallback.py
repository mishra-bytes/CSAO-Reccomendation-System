from __future__ import annotations


def fill_candidates(
    ranked_candidates: list[tuple[str, float]],
    fallback_candidates: list[tuple[str, float]],
    exclude: set[str],
    target_k: int,
) -> list[tuple[str, float]]:
    seen = {item for item, _ in ranked_candidates}
    out = list(ranked_candidates)
    for item, score in fallback_candidates:
        if item in seen or item in exclude:
            continue
        out.append((item, score))
        seen.add(item)
        if len(out) >= target_k:
            break
    return out[:target_k]

