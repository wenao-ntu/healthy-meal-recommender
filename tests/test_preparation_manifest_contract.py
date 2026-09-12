"""Contracts for the committed MealRec+H preparation evidence."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    PROJECT_ROOT / "results" / "data_preparation" / "mealrecplus_h_manifest.json"
)
REPRODUCIBILITY_PATH = (
    PROJECT_ROOT / "results" / "data_preparation" / "reproducibility_check.json"
)


def load_json(path: Path) -> dict[str, object]:
    """Load a committed machine-readable evidence file."""
    return json.loads(path.read_text(encoding="utf-8"))


def test_prepared_counts_and_warm_start_policy() -> None:
    """The model split must preserve training data and explicitly filter cold meals."""
    manifest = load_json(MANIFEST_PATH)
    counts = manifest["counts"]
    assert counts == {
        "courses_used_by_meals": 942,
        "evaluation_exclusions": 13,
        "meals_catalog": 3_817,
        "meals_seen_in_train": 3_810,
        "test_interactions_original": 4_677,
        "test_interactions_warm": 4_673,
        "train_interactions": 37_413,
        "users": 1_575,
        "valid_interactions_original": 4_677,
        "valid_interactions_warm": 4_668,
    }
    assert manifest["validation"]["all_evaluation_meals_seen_in_train"] is True
    assert manifest["validation"]["split_overlap_rows"] == 0


def test_prepared_health_and_privacy_contracts() -> None:
    """Health baselines must be train-only and source usernames must remain unused."""
    manifest = load_json(MANIFEST_PATH)
    assert manifest["policies"]["source_user_mapping_used"] is False
    assert manifest["policies"]["user_health_baseline"].endswith(
        "training interactions only"
    )
    assert manifest["validation"]["max_user_fsa_baseline_error"] <= 1e-10
    assert manifest["validation"]["max_user_who_baseline_error"] <= 1e-10
    assert manifest["validation"]["zero_without_unit_treated_as_missing"] is True
    assert manifest["validation"]["courses_with_missing_nutrition"] == 5
    assert manifest["validation"]["meals_with_missing_nutrition"] > 0


def test_all_generated_artifacts_have_hashes_and_repeat_identically() -> None:
    """Every generated file must be traceable and reproducible byte-for-byte."""
    manifest = load_json(MANIFEST_PATH)
    assert len(manifest["outputs"]) == 7
    for artifact in manifest["outputs"].values():
        assert artifact["bytes"] > 0
        assert artifact["rows"] > 0
        assert len(artifact["sha256"]) == 64

    reproducibility = load_json(REPRODUCIBILITY_PATH)
    assert reproducibility["runs"] == 2
    assert reproducibility["manifests_byte_identical"] is True
    assert reproducibility["output_artifact_hashes_identical"] is True
    assert reproducibility["manifest_sha256_run_1"] == reproducibility[
        "manifest_sha256_run_2"
    ]
