"""Focused tests for the live demo recommendation path."""

from __future__ import annotations

from healthy_meal_recommender.demo import (
    DemoCandidate,
    DemoData,
    DemoMeal,
    build_demo_view,
)


def meal(meal_id: int, health: float) -> DemoMeal:
    """Build a minimal meal for ranking tests."""
    return DemoMeal(
        meal_id=meal_id,
        courses=(f"Starter {meal_id}", f"Main {meal_id}", f"Dessert {meal_id}"),
        fsa_score=8.0,
        who_score=3.5,
        health_utility=health,
        calories_kcal=500.0,
        protein_g=20.0,
        carbohydrates_g=60.0,
        sodium_mg=700.0,
    )


def demo_data(baseline: float) -> DemoData:
    """Create 100 ordered candidates with target meals below the first ten."""
    meals = {index: meal(index, 0.2 if index <= 10 else 0.5) for index in range(1, 101)}
    candidates = tuple(
        DemoCandidate(
            source_rank=index,
            preference_score=1.0 - index / 200,
            meal=meals[index],
        )
        for index in range(1, 101)
    )
    return DemoData(
        meals=meals,
        user_baselines={7: baseline},
        candidates={7: candidates},
        default_target=0.4,
        max_quota_share=0.6,
    )


def test_demo_preserves_preferences_when_user_has_no_gap() -> None:
    """A user already above the selected target should see no intervention."""
    view = build_demo_view(demo_data(baseline=0.5), user_id=7, target=0.4)
    assert view.gap == 0.0
    assert view.intervention_intensity == 0.0
    assert view.top10_overlap == 1.0
    assert not any(row.promoted for row in view.after)


def test_demo_uses_real_candidates_to_fill_a_gap_adaptive_quota() -> None:
    """A large gap should promote feasible target candidates without duplicates."""
    view = build_demo_view(demo_data(baseline=0.2), user_id=7, target=0.4)
    after_ids = [row.candidate.meal.meal_id for row in view.after]
    assert view.target_quota == 6
    assert view.after_goal_rate == 0.6
    assert len(set(after_ids)) == 10
    assert all(meal_id in range(1, 101) for meal_id in after_ids)
    assert sum(row.promoted for row in view.after) == 6
