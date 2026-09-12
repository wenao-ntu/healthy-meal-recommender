"""Contracts for the committed MostPop baseline result."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = PROJECT_ROOT / "results" / "baselines" / "mostpop.json"


def load_result() -> dict[str, object]:
    """Load the compact result rather than rerunning the full evaluator in unit tests."""
    return json.loads(RESULT_PATH.read_text(encoding="utf-8"))


def test_mostpop_uses_only_positive_training_counts() -> None:
    """The baseline must represent popularity, not sampled negative frequency."""
    result = load_result()
    assert result["model"]["report_name"] == "MostPop"
    assert result["effective_protocol"]["training_negative_sampling"] == "none"
    assert result["fit_checks"]["positive_interactions_counted"] == 37_413
    assert result["fit_checks"]["meals_with_positive_count"] == 3_810


def test_full_catalog_protocol_and_metrics_are_complete() -> None:
    """Validation and test must share the required full-sort ranking metrics."""
    result = load_result()
    assert result["effective_protocol"]["evaluation_mode"] == "full catalog"
    assert result["effective_protocol"]["primary_metric"] == "NDCG@10"
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
        assert result[split]["recall@5"] <= result[split]["recall@10"]
        assert result[split]["recall@10"] <= result[split]["recall@20"]


def test_recommendation_samples_are_real_and_compact() -> None:
    """Saved examples must use anonymous users and exactly ten catalog meals."""
    samples = load_result()["recommendation_samples"]
    assert [sample["history_band"] for sample in samples] == [
        "low_history",
        "median_history",
        "high_history",
    ]
    for sample in samples:
        assert isinstance(sample["user_id"], int)
        assert len(sample["recommendations"]) == 10
        assert all(len(meal["courses"]) == 3 for meal in sample["recommendations"])
