"""Data loading and presentation models for the local Streamlit demo."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from healthy_meal_recommender.reranking import (
    gap_rank_order,
    health_utility,
    preference_order,
)


@dataclass(frozen=True)
class DemoMeal:
    """Meal fields displayed by the demo."""

    meal_id: int
    courses: tuple[str, str, str]
    fsa_score: float
    who_score: float
    health_utility: float
    calories_kcal: float | None
    protein_g: float | None
    carbohydrates_g: float | None
    sodium_mg: float | None


@dataclass(frozen=True)
class DemoCandidate:
    """One selected-model candidate."""

    source_rank: int
    preference_score: float
    meal: DemoMeal


@dataclass(frozen=True)
class DemoRecommendation:
    """A candidate annotated for one final recommendation list."""

    rank: int
    candidate: DemoCandidate
    promoted: bool
    meets_target: bool


@dataclass(frozen=True)
class DemoData:
    """All local artifacts required by the interface."""

    meals: dict[int, DemoMeal]
    user_baselines: dict[int, float]
    candidates: dict[int, tuple[DemoCandidate, ...]]
    default_target: float
    max_quota_share: float


@dataclass(frozen=True)
class DemoView:
    """Before/after recommendations and user-level explanation values."""

    user_id: int
    baseline: float
    target: float
    gap: float
    candidate_ceiling: float
    intervention_intensity: float
    target_quota: int
    before: tuple[DemoRecommendation, ...]
    after: tuple[DemoRecommendation, ...]
    before_mean_health: float
    after_mean_health: float
    before_goal_rate: float
    after_goal_rate: float
    top10_overlap: float


def optional_float(value: str) -> float | None:
    """Parse an optional numeric display field."""
    return None if value == "" else float(value)


def load_meals(path: Path) -> dict[int, DemoMeal]:
    """Load the compact processed meal catalog."""
    meals = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            meal_id = int(row["meal_id"])
            fsa_score = float(row["fsa_score"])
            who_score = float(row["who_score"])
            meals[meal_id] = DemoMeal(
                meal_id=meal_id,
                courses=(
                    row["appetizer_name"],
                    row["main_dish_name"],
                    row["dessert_name"],
                ),
                fsa_score=fsa_score,
                who_score=who_score,
                health_utility=float(health_utility(fsa_score, who_score)),
                calories_kcal=optional_float(row["calories_kcal"]),
                protein_g=optional_float(row["protein_g"]),
                carbohydrates_g=optional_float(row["carbohydrates_g"]),
                sodium_mg=optional_float(row["sodium_mg"]),
            )
    return meals


def load_baselines(path: Path) -> dict[int, float]:
    """Load train-only user baselines in the shared utility scale."""
    baselines = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            baselines[int(row["user_id"])] = float(
                health_utility(float(row["fsa_baseline"]), float(row["who_baseline"]))
            )
    return baselines


def load_candidates(
    path: Path, meals: dict[int, DemoMeal]
) -> dict[int, tuple[DemoCandidate, ...]]:
    """Load exported candidates and enforce the Top-100 contract."""
    candidates: dict[int, list[DemoCandidate]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            user_id = int(row["user_id"])
            candidates.setdefault(user_id, []).append(
                DemoCandidate(
                    source_rank=int(row["rank"]),
                    preference_score=float(row["preference_score"]),
                    meal=meals[int(row["meal_id"])],
                )
            )
    output = {}
    for user_id, rows in candidates.items():
        rows.sort(key=lambda row: row.source_rank)
        if [row.source_rank for row in rows] != list(range(1, 101)):
            raise ValueError(f"User {user_id} does not have candidate ranks 1..100")
        if len({row.meal.meal_id for row in rows}) != 100:
            raise ValueError(f"User {user_id} has duplicate meal candidates")
        output[user_id] = tuple(rows)
    return output


def load_demo_data(project_root: Path) -> DemoData:
    """Load the exact local artifacts produced by the reproducible pipeline."""
    processed_root = project_root / "data" / "processed" / "mealrecplus_h"
    result_path = project_root / "results" / "reranking" / "gap_rank.json"
    meals = load_meals(processed_root / "meal_catalog.csv")
    baselines = load_baselines(processed_root / "user_health_train.csv")
    candidates = load_candidates(
        processed_root
        / "candidates"
        / "neumf_standard_test_top100.csv",
        meals,
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    missing_baselines = set(candidates) - set(baselines)
    if missing_baselines:
        raise ValueError(f"Candidate users missing train baselines: {missing_baselines}")
    return DemoData(
        meals=meals,
        user_baselines=baselines,
        candidates=candidates,
        default_target=float(result["benchmark_goal"]["value"]),
        max_quota_share=float(
            result["selection"]["selected_gap_max_quota_share"]
        ),
    )


def annotate(
    ordered: list[DemoCandidate],
    target: float,
    base_top10_ids: set[int],
) -> tuple[DemoRecommendation, ...]:
    """Add UI explanation flags to an ordered Top-10."""
    return tuple(
        DemoRecommendation(
            rank=rank,
            candidate=candidate,
            promoted=candidate.meal.meal_id not in base_top10_ids,
            meets_target=candidate.meal.health_utility >= target,
        )
        for rank, candidate in enumerate(ordered, start=1)
    )


def build_demo_view(data: DemoData, user_id: int, target: float) -> DemoView:
    """Compute live GAP Rank recommendations for one anonymous user."""
    if user_id not in data.candidates:
        raise KeyError(f"Unknown demo user: {user_id}")
    rows = list(data.candidates[user_id])
    preference_scores = np.asarray([row.preference_score for row in rows])
    health_scores = np.asarray([row.meal.health_utility for row in rows])
    meal_ids = np.asarray([row.meal.meal_id for row in rows])
    baseline = data.user_baselines[user_id]

    base_order = preference_order(preference_scores, meal_ids)
    gap_order, metadata = gap_rank_order(
        preference_scores,
        health_scores,
        meal_ids,
        user_baseline=baseline,
        target=target,
        max_quota_share=data.max_quota_share,
        final_k=10,
    )
    before_rows = [rows[int(index)] for index in base_order[:10]]
    after_rows = [rows[int(index)] for index in gap_order[:10]]
    base_ids = {row.meal.meal_id for row in before_rows}
    before = annotate(before_rows, target, base_ids)
    after = annotate(after_rows, target, base_ids)
    before_health = np.asarray(
        [recommendation.candidate.meal.health_utility for recommendation in before]
    )
    after_health = np.asarray(
        [recommendation.candidate.meal.health_utility for recommendation in after]
    )
    after_ids = {recommendation.candidate.meal.meal_id for recommendation in after}
    return DemoView(
        user_id=user_id,
        baseline=baseline,
        target=target,
        gap=max(0.0, target - baseline),
        candidate_ceiling=float(metadata["candidate_ceiling"]),
        intervention_intensity=float(metadata["intervention_intensity"]),
        target_quota=int(metadata["target_quota"]),
        before=before,
        after=after,
        before_mean_health=float(before_health.mean()),
        after_mean_health=float(after_health.mean()),
        before_goal_rate=float((before_health >= target).mean()),
        after_goal_rate=float((after_health >= target).mean()),
        top10_overlap=len(base_ids & after_ids) / 10,
    )
