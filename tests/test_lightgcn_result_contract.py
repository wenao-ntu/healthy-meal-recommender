"""Contracts for bounded LightGCN selection and final preference-model choice."""

from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = PROJECT_ROOT / "results" / "baselines" / "lightgcn.json"
COMPARISON_PATH = (
    PROJECT_ROOT / "results" / "comparison" / "preference_models.csv"
)
SELECTION_PATH = (
    PROJECT_ROOT / "results" / "comparison" / "preference_model_selection.json"
)


def load_json(path: Path) -> dict[str, object]:
    """Load a committed experiment artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def test_lightgcn_selection_is_bounded_and_validation_only() -> None:
    """Exactly two graph depths may compete before one test evaluation."""
    result = load_json(RESULT_PATH)
    trials = result["selection"]["trials"]
    assert {trial["trial_id"] for trial in trials} == {"1-layer", "2-layer"}
    assert result["selection"]["test_evaluations_performed"] == 1
    winner = max(trials, key=lambda trial: trial["validation"]["ndcg@10"])
    assert result["selection"]["selected_trial"] == winner["trial_id"]
    assert all(trial["epochs_completed"] <= 30 for trial in trials)


def test_lightgcn_protocol_and_assignment_boundary_are_explicit() -> None:
    """The graph model must share evaluation while distinguishing MF-BPR."""
    result = load_json(RESULT_PATH)
    assert result["model"]["optimization_objective"] == "BPR pairwise ranking loss"
    assert result["individual_assignment_boundary"]["same_algorithm"] is False
    assert "graph propagation" in result["individual_assignment_boundary"][
        "group_project_model"
    ]
    assert result["effective_protocol"]["primary_metric"] == "NDCG@10"
    assert result["dataset"]["train_interactions"] == 37_413
    for split in ("validation", "test"):
        assert all(0 <= value <= 1 for value in result[split].values())


def test_comparison_and_gap_rank_model_selection_use_validation() -> None:
    """The report table and candidate-generator choice must include all models."""
    with COMPARISON_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [(row["model"], row["split"]) for row in rows] == [
        ("MostPop", "validation"),
        ("MostPop", "test"),
        ("NeuMF", "validation"),
        ("NeuMF", "test"),
        ("LightGCN", "validation"),
        ("LightGCN", "test"),
    ]

    selection = load_json(SELECTION_PATH)
    assert selection["test_metrics_used_for_selection"] is False
    best = max(selection["candidates"], key=lambda row: row["validation_ndcg@10"])
    assert selection["selected_model"] == best["model"]
    assert selection["selected_config"] == best["selected_config"]
