"""Run deterministic data-path and Streamlit render checks for the local demo."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

from streamlit.testing.v1 import AppTest

from healthy_meal_recommender.demo import build_demo_view, load_demo_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "app" / "streamlit_app.py"
BANNER_PATH = PROJECT_ROOT / "app" / "assets" / "healthy-meal-banner.png"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "demo" / "smoke_test.json"


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def representative_user(data) -> int:
    """Match the informative default-user policy in the Streamlit app."""
    records = sorted(
        (max(0.0, data.default_target - data.user_baselines[user_id]), user_id)
        for user_id in data.candidates
        if data.user_baselines[user_id] < data.default_target
    )
    return records[len(records) // 2][1]


def main() -> None:
    """Validate the live data route, UI structure, and no-intervention behavior."""
    data = load_demo_data(PROJECT_ROOT)
    user_id = representative_user(data)
    default_view = build_demo_view(data, user_id, round(data.default_target, 3))
    repeated_view = build_demo_view(data, user_id, round(data.default_target, 3))
    no_intervention_view = build_demo_view(data, user_id, 0.20)

    if default_view != repeated_view:
        raise AssertionError("Repeated demo recommendation was not deterministic")
    if len(default_view.after) != 10:
        raise AssertionError("Default view did not return exactly 10 meals")
    if len({row.candidate.meal.meal_id for row in default_view.after}) != 10:
        raise AssertionError("Default view contains duplicate meals")
    candidate_ids = {
        row.meal.meal_id for row in data.candidates[user_id]
    }
    if not {
        row.candidate.meal.meal_id for row in default_view.after
    }.issubset(candidate_ids):
        raise AssertionError("Demo returned a meal outside the real Top-100 candidates")
    if no_intervention_view.before != no_intervention_view.after:
        raise AssertionError("A user already meeting the target should not be reranked")

    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    if app.exception:
        raise AssertionError(f"Streamlit default render failed: {app.exception}")
    default_ui = {
        "title": [element.value for element in app.title],
        "selectboxes": len(app.selectbox),
        "sliders": len(app.slider),
        "metrics": len(app.metric),
        "dataframes": len(app.dataframe),
        "tabs": len(app.tabs),
    }
    if default_ui != {
        "title": ["Healthy Meal Navigator"],
        "selectboxes": 2,
        "sliders": 1,
        "metrics": 10,
        "dataframes": 4,
        "tabs": 3,
    }:
        raise AssertionError(f"Unexpected Streamlit UI structure: {default_ui}")

    app.slider[0].set_value(0.20).run()
    if app.exception:
        raise AssertionError(f"Streamlit low-target render failed: {app.exception}")
    if len(app.info) != 1 or "already meets" not in app.info[0].value:
        raise AssertionError("Low target did not show the no-intervention explanation")
    low_target_metrics = {element.label: element.value for element in app.metric}
    if low_target_metrics["Adaptive quota"] != "0 / 10":
        raise AssertionError("Low target did not reduce the adaptive quota to zero")
    if low_target_metrics["Preference overlap"] != "100%":
        raise AssertionError("Low target did not preserve the full preference Top-10")

    output = {
        "demo_smoke_schema_version": 1,
        "application": {
            "entrypoint": APP_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "framework": "Streamlit",
            "framework_version": version("streamlit"),
            "language": "English",
            "local_only": True,
        },
        "data_path": {
            "anonymous_users": len(data.candidates),
            "meal_catalog_rows": len(data.meals),
            "candidates_per_user": 100,
            "candidate_rows": sum(len(rows) for rows in data.candidates.values()),
            "candidate_generator": "NeuMF Standard",
            "uses_live_gap_rank_function": True,
            "contains_patient_pii": False,
        },
        "default_scenario": {
            "user_id": user_id,
            "train_only_baseline": default_view.baseline,
            "target": default_view.target,
            "adaptive_quota": default_view.target_quota,
            "promoted_meals": sum(row.promoted for row in default_view.after),
            "top10_overlap": default_view.top10_overlap,
            "before_mean_health": default_view.before_mean_health,
            "after_mean_health": default_view.after_mean_health,
            "before_goal_rate": default_view.before_goal_rate,
            "after_goal_rate": default_view.after_goal_rate,
        },
        "checks": {
            "ten_unique_real_candidates": True,
            "repeated_recommendation_identical": True,
            "no_gap_preserves_preference_order": True,
            "default_streamlit_render_has_no_exception": True,
            "low_target_streamlit_render_has_no_exception": True,
            "low_target_explanation_visible": True,
            "low_target_quota_is_zero": True,
            "low_target_overlap_is_100_percent": True,
        },
        "ui_structure": default_ui,
        "visual_asset": {
            "path": BANNER_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": file_sha256(BANNER_PATH),
            "contains_text": False,
            "generated_for_this_project": True,
        },
    }
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT.write_text(
        json.dumps(output, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
