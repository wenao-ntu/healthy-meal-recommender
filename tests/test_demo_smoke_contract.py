"""Contract tests for the committed local-demo smoke evidence."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = PROJECT_ROOT / "results" / "demo" / "smoke_test.json"


def load_result() -> dict[str, object]:
    """Load the committed demo evidence."""
    return json.loads(RESULT_PATH.read_text(encoding="utf-8"))


def test_demo_uses_real_anonymous_project_artifacts() -> None:
    """The UI must use the selected model's real candidates without patient PII."""
    result = load_result()
    data = result["data_path"]
    assert data["anonymous_users"] == 1_114
    assert data["meal_catalog_rows"] == 3_817
    assert data["candidates_per_user"] == 100
    assert data["candidate_rows"] == 111_400
    assert data["candidate_generator"] == "NeuMF Standard"
    assert data["uses_live_gap_rank_function"] is True
    assert data["contains_patient_pii"] is False


def test_demo_renders_and_handles_no_intervention() -> None:
    """The default and already-at-target paths must both be verified."""
    result = load_result()
    assert all(result["checks"].values())
    assert result["ui_structure"] == {
        "title": ["Healthy Meal Navigator"],
        "selectboxes": 2,
        "sliders": 1,
        "metrics": 10,
        "dataframes": 4,
        "tabs": 3,
    }
    assert result["default_scenario"]["after_goal_rate"] > result[
        "default_scenario"
    ]["before_goal_rate"]
    assert result["visual_asset"]["contains_text"] is False
    assert len(result["visual_asset"]["sha256"]) == 64
