"""Transparent health-aware reranking rules shared by experiments and demo."""

from __future__ import annotations

import math

import numpy as np


FSA_MIN = 4.0
FSA_MAX = 12.0
WHO_MIN = 0.0
WHO_MAX = 7.0
DEFAULT_GAP_REFERENCE = 0.10


def health_utility(fsa_score, who_score):
    """Map FSA (lower is better) and WHO (higher is better) to [0, 1]."""
    fsa = np.asarray(fsa_score, dtype=float)
    who = np.asarray(who_score, dtype=float)
    fsa_component = np.clip((FSA_MAX - fsa) / (FSA_MAX - FSA_MIN), 0.0, 1.0)
    who_component = np.clip((who - WHO_MIN) / (WHO_MAX - WHO_MIN), 0.0, 1.0)
    return 0.5 * fsa_component + 0.5 * who_component


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    """Normalize one user's candidate scores without cross-user assumptions."""
    values = np.asarray(values, dtype=float)
    span = float(values.max() - values.min())
    if span == 0.0:
        return np.zeros_like(values)
    return (values - values.min()) / span


def stable_descending_order(scores: np.ndarray, meal_ids: np.ndarray) -> np.ndarray:
    """Sort by descending score, then ascending meal id for exact ties."""
    return np.lexsort((np.asarray(meal_ids, dtype=int), -np.asarray(scores, dtype=float)))


def preference_order(preference_scores: np.ndarray, meal_ids: np.ndarray) -> np.ndarray:
    """Keep the candidate generator's preference ordering."""
    return stable_descending_order(preference_scores, meal_ids)


def fixed_weight_order(
    preference_scores: np.ndarray,
    health_scores: np.ndarray,
    meal_ids: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Apply the same health weight to every user."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    combined = (1.0 - alpha) * minmax_normalize(preference_scores) + alpha * np.asarray(
        health_scores, dtype=float
    )
    return stable_descending_order(combined, meal_ids)


def hard_filter_order(
    preference_scores: np.ndarray,
    health_scores: np.ndarray,
    meal_ids: np.ndarray,
    target: float,
) -> np.ndarray:
    """Prioritize target-meeting meals, then backfill by preference if necessary."""
    base_order = preference_order(preference_scores, meal_ids)
    meets_target = np.asarray(health_scores, dtype=float) >= target
    return np.concatenate(
        (base_order[meets_target[base_order]], base_order[~meets_target[base_order]])
    )


def gap_rank_order(
    preference_scores: np.ndarray,
    health_scores: np.ndarray,
    meal_ids: np.ndarray,
    user_baseline: float,
    target: float,
    max_quota_share: float,
    gap_reference: float = DEFAULT_GAP_REFERENCE,
    final_k: int = 10,
) -> tuple[np.ndarray, dict[str, float]]:
    """Meet a gap-adaptive health quota using the fewest Top-K replacements."""
    if not 0.0 <= max_quota_share <= 1.0:
        raise ValueError("max_quota_share must be between 0 and 1")
    if gap_reference <= 0.0:
        raise ValueError("gap_reference must be positive")
    if final_k < 1 or final_k > len(preference_scores):
        raise ValueError("final_k must be within the candidate array")

    health_scores = np.asarray(health_scores, dtype=float)
    base_order = preference_order(preference_scores, meal_ids)
    candidate_ceiling = float(health_scores.max())
    target_gap = max(0.0, target - user_baseline)
    effective_target = min(target, candidate_ceiling)
    effective_gap = max(0.0, effective_target - user_baseline)
    feasibility_ratio = effective_gap / target_gap if target_gap > 0.0 else 0.0
    effective_intensity = (
        max_quota_share * min(target_gap / gap_reference, 1.0) * feasibility_ratio
    )

    if target_gap == 0.0 or effective_gap == 0.0 or effective_intensity == 0.0:
        return base_order, {
            "candidate_ceiling": candidate_ceiling,
            "effective_target": effective_target,
            "intervention_intensity": 0.0,
            "target_quota": 0.0,
        }

    meets_target = health_scores >= target
    target_quota = min(
        int(math.ceil(final_k * effective_intensity)),
        int(meets_target.sum()),
    )
    selected = [int(index) for index in base_order[:final_k]]
    current_target_count = sum(bool(meets_target[index]) for index in selected)
    needed = max(0, target_quota - current_target_count)
    if needed:
        promotable = [
            int(index) for index in base_order[final_k:] if meets_target[index]
        ][:needed]
        replace_positions = [
            position
            for position in range(final_k - 1, -1, -1)
            if not meets_target[selected[position]]
        ][: len(promotable)]
        for position, promoted_index in zip(replace_positions, promotable, strict=True):
            selected[position] = promoted_index

    selected_set = set(selected)
    selected_in_preference_order = [
        int(index) for index in base_order if int(index) in selected_set
    ]
    remaining = [int(index) for index in base_order if int(index) not in selected_set]
    order = np.asarray(selected_in_preference_order + remaining, dtype=int)
    return order, {
        "candidate_ceiling": candidate_ceiling,
        "effective_target": effective_target,
        "intervention_intensity": effective_intensity,
        "target_quota": float(target_quota),
    }
