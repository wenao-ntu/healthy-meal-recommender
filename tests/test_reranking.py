"""Unit tests for transparent health-aware ranking rules."""

from __future__ import annotations

import numpy as np
import pytest

from healthy_meal_recommender.reranking import (
    fixed_weight_order,
    gap_rank_order,
    hard_filter_order,
    health_utility,
)


def test_health_utility_respects_both_official_directions() -> None:
    """Best/worst canonical FSA and WHO endpoints must map to one/zero."""
    values = health_utility([4.0, 12.0, 8.0], [7.0, 0.0, 3.5])
    assert values.tolist() == pytest.approx([1.0, 0.0, 0.5])


def test_fixed_weight_and_filter_can_promote_healthier_candidates() -> None:
    """The two controls should visibly differ from pure preference."""
    preference = np.array([1.0, 0.8, 0.7])
    health = np.array([0.1, 0.2, 0.9])
    meal_ids = np.array([10, 20, 30])
    assert fixed_weight_order(preference, health, meal_ids, alpha=0.8)[0] == 2
    assert hard_filter_order(preference, health, meal_ids, target=0.5)[0] == 2


def test_gap_rank_intervenes_only_when_user_has_a_gap() -> None:
    """Users meeting the goal keep preference order; lower baselines adapt."""
    preference = np.array([1.0, 0.95, 0.8, 0.7])
    health = np.array([0.2, 0.3, 0.25, 0.8])
    meal_ids = np.array([10, 20, 30, 40])

    no_gap_order, no_gap = gap_rank_order(
        preference,
        health,
        meal_ids,
        user_baseline=0.6,
        target=0.5,
        max_quota_share=0.8,
        final_k=3,
    )
    gap_order, gap = gap_rank_order(
        preference,
        health,
        meal_ids,
        user_baseline=0.2,
        target=0.5,
        max_quota_share=0.8,
        final_k=3,
    )

    assert no_gap_order.tolist() == [0, 1, 2, 3]
    assert no_gap["intervention_intensity"] == 0.0
    assert gap_order[:3].tolist() == [0, 1, 3]
    assert gap["intervention_intensity"] == pytest.approx(0.8)
