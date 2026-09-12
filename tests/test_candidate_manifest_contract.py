"""Contracts for NeuMF candidate export evidence."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    PROJECT_ROOT / "results" / "candidates" / "neumf_standard_top100.json"
)


def load_manifest() -> dict[str, object]:
    """Load the committed candidate integrity evidence."""
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_candidates_come_from_validation_selected_neumf() -> None:
    """Candidate generation must preserve the validation-only model choice."""
    manifest = load_manifest()
    generator = manifest["candidate_generator"]
    assert generator["model"] == "NeuMF"
    assert generator["config"] == "standard"
    assert generator["test_metrics_used_for_selection"] is False
    assert len(generator["checkpoint_sha256"]) == 64


def test_candidate_files_have_fixed_size_and_no_history() -> None:
    """Every evaluation user must receive 100 unseen, unlabeled candidates."""
    manifest = load_manifest()
    protocol = manifest["protocol"]
    assert protocol["candidate_k"] == 100
    assert protocol["final_recommendation_k"] == 10
    assert protocol["candidate_files_contain_ground_truth_labels"] is False
    expected_users = {"validation": 1_130, "test": 1_114}
    for split_name, user_count in expected_users.items():
        split = manifest["splits"][split_name]
        assert split["users"] == user_count
        assert split["rows"] == user_count * 100
        assert split["history_overlap_rows"] == 0
        assert 0 <= split["candidate_recall@100"] <= 1
        assert len(split["sha256"]) == 64
