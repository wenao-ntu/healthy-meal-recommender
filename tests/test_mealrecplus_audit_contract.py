"""Contract tests for the committed MealRec+H audit evidence."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = PROJECT_ROOT / "results" / "data_audit" / "mealrecplus_h_audit.json"


def load_audit() -> dict[str, object]:
    """Load the small, committed audit result rather than raw upstream data."""
    return json.loads(AUDIT_PATH.read_text(encoding="utf-8"))


def test_source_and_license_are_explicit() -> None:
    """The exact source revision and unresolved license status must remain visible."""
    source = load_audit()["source"]
    assert source["commit"] == "8bd7d7d7effb505be4a7d2074b28abad802fe0f7"
    assert source["license_status"] == "no_explicit_license_file"
    assert source["readme_requires_citation"] is True


def test_interaction_splits_are_disjoint() -> None:
    """Official train/tune/test interactions must not overlap."""
    split = load_audit()["user_meal_split"]
    assert split["total_rows"] == split["total_unique_rows"] == 46_767
    assert split["train_tune_overlap"] == 0
    assert split["train_test_overlap"] == 0
    assert split["tune_test_overlap"] == 0
    assert split["cold_start_against_train"]["tune"]["cold_interactions"] == 9
    assert split["cold_start_against_train"]["test"]["cold_interactions"] == 4


def test_health_baselines_are_train_only() -> None:
    """Published user health vectors must match train-only recomputation exactly."""
    comparison = load_audit()["published_user_baseline_comparison"]
    for metric in ("fsa", "who"):
        train_only = comparison[f"{metric}_train_only"]
        assert train_only["users_compared"] == 1_575
        assert train_only["mean_absolute_error"] == 0
        assert train_only["max_absolute_error"] == 0


def test_bundle_and_nutrition_contracts() -> None:
    """Every meal has three categories and all required nutrient amounts exist."""
    audit = load_audit()
    bundle = audit["bundle_structure"]
    assert bundle["meals"] == 3_817
    assert bundle["courses_per_meal_counts"] == {"3": 3_817}
    assert bundle["meals_with_exact_categories_0_1_2"] == 3_817
    assert bundle["unique_courses_in_meals"] == 942

    course_metadata = audit["metadata"]["course_csv"]
    assert course_metadata["rows"] == 7_280
    assert course_metadata["nutrition_parse_errors"] == 0
    assert all(
        nutrient["rows"] == 7_280
        for nutrient in course_metadata["nutrient_amount_coverage"].values()
    )
