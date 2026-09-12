"""Contracts for bounded NeuMF selection and its committed result."""

from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = PROJECT_ROOT / "results" / "baselines" / "neumf.json"
COMPARISON_PATH = (
    PROJECT_ROOT / "results" / "comparison" / "preference_models.csv"
)


def load_result() -> dict[str, object]:
    """Load the saved experiment evidence."""
    return json.loads(RESULT_PATH.read_text(encoding="utf-8"))


def test_neumf_selection_is_bounded_and_validation_only() -> None:
    """Exactly two declared capacities may compete before one test evaluation."""
    result = load_result()
    trials = result["selection"]["trials"]
    assert {trial["trial_id"] for trial in trials} == {"compact", "standard"}
    assert result["selection"]["test_evaluations_performed"] == 1
    winner = max(trials, key=lambda trial: trial["validation"]["ndcg@10"])
    assert result["selection"]["selected_trial"] == winner["trial_id"]
    assert all(trial["epochs_completed"] <= 20 for trial in trials)
    assert all(
        trial["hyperparameters"]["negative_samples_per_positive"] == 4
        for trial in trials
    )


def test_neumf_uses_shared_full_catalog_protocol() -> None:
    """NeuMF must be directly comparable with the committed MostPop result."""
    result = load_result()
    protocol = result["effective_protocol"]
    assert protocol["primary_metric"] == "NDCG@10"
    assert protocol["evaluation_mode"] == "full catalog"
    assert protocol["device"].startswith("cuda")
    assert result["dataset"]["train_interactions"] == 37_413
    assert result["dataset"]["validation_users"] == 1_130
    assert result["dataset"]["test_users"] == 1_114

    expected_metrics = {
        "recall@5",
        "recall@10",
        "recall@20",
        "ndcg@5",
        "ndcg@10",
        "ndcg@20",
    }
    for split in ("validation", "test"):
        assert set(result[split]) == expected_metrics
        assert all(0 <= value <= 1 for value in result[split].values())


def test_neumf_outputs_feed_comparison_and_demo() -> None:
    """The selected result must provide a report table and real recommendation samples."""
    result = load_result()
    assert len(result["recommendation_samples"]) == 3
    assert all(
        len(sample["recommendations"]) == 10
        for sample in result["recommendation_samples"]
    )
    assert result["selected_checkpoint"]["tracked_by_git"] is False

    with COMPARISON_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [(row["model"], row["split"]) for row in rows[:4]] == [
        ("MostPop", "validation"),
        ("MostPop", "test"),
        ("NeuMF", "validation"),
        ("NeuMF", "test"),
    ]
