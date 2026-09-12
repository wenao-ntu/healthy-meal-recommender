"""Contracts for leakage-safe health reranking and GAP Rank evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = PROJECT_ROOT / "results" / "reranking" / "gap_rank.json"
COMPARISON_PATH = (
    PROJECT_ROOT / "results" / "comparison" / "health_reranking.csv"
)


def load_result() -> dict[str, object]:
    """Load the committed reranking evidence."""
    return json.loads(RESULT_PATH.read_text(encoding="utf-8"))


def test_health_goal_and_train_only_baseline_are_explicit() -> None:
    """The benchmark target must be non-clinical and free of evaluation labels."""
    result = load_result()
    assert result["health_utility"]["higher_is_healthier"] is True
    assert result["health_utility"]["weights"] == {"fsa": 0.5, "who": 0.5}
    assert result["benchmark_goal"]["quantile"] == 0.75
    assert result["benchmark_goal"]["clinical_threshold"] is False
    assert result["protocol"]["user_baselines"] == "training interactions only"
    assert result["protocol"]["labels_used_only_by_evaluator"] is True


def test_parameter_selection_is_validation_only_and_bounded() -> None:
    """Only four declared values per method may compete before test evaluation."""
    selection = load_result()["selection"]
    assert selection["test_metrics_used_for_parameter_selection"] is False
    assert {trial["alpha"] for trial in selection["fixed_weight_trials"]} == {
        0.1,
        0.2,
        0.3,
        0.4,
    }
    assert {
        trial["max_quota_share"] for trial in selection["gap_rank_trials"]
    } == {
        0.2,
        0.4,
        0.6,
        0.8,
    }


def test_all_required_controls_and_metrics_are_report_ready() -> None:
    """No rerank, fixed weight, hard filter, and GAP Rank must share metrics."""
    result = load_result()
    methods = {"preference_only", "fixed_weight", "hard_filter", "gap_rank"}
    for split in ("validation", "test"):
        assert set(result[split]) == methods
        for metrics in result[split].values():
            assert 0 <= metrics["recall@10"] <= 1
            assert 0 <= metrics["ndcg@10"] <= 1
            assert 0 <= metrics["goal_item_rate@10"] <= 1
            assert 0 <= metrics["preference_top10_overlap"] <= 1
        assert result[split]["preference_only"]["ndcg@10_retention"] == 1.0

    with COMPARISON_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8
    assert {(row["split"], row["method"]) for row in rows} == {
        (split, method) for split in ("validation", "test") for method in methods
    }


def test_gap_rank_keeps_the_declared_accuracy_constraint() -> None:
    """Selected GAP Rank must retain at least 90% validation NDCG."""
    result = load_result()
    assert result["selection"]["gap_retention_constraint_satisfied"] is True
    assert result["validation"]["gap_rank"]["ndcg@10_retention"] >= 0.90
    assert result["test_recommendation_artifact"]["rows"] == 11_140
    assert len(result["recommendation_samples"]) == 3
