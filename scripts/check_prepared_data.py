"""Validate prepared MealRec+H artifacts and load the formal RecBole benchmark."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from recbole.config import Config
from recbole.data import create_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "mealrecplus_h"
MANIFEST_PATH = PROCESSED_ROOT / "preparation_manifest.json"
CONFIG_PATH = PROJECT_ROOT / "configs" / "data" / "mealrecplus_h.yaml"


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """Check hashes and prove RecBole can build the fixed benchmark splits."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for relative, expected in manifest["outputs"].items():
        path = PROCESSED_ROOT / relative
        if path.stat().st_size != expected["bytes"] or file_sha256(path) != expected["sha256"]:
            raise ValueError(f"Prepared artifact differs from manifest: {relative}")

    config = Config(
        model="LightGCN",
        dataset="mealrecplus_h",
        config_file_list=[str(CONFIG_PATH)],
        config_dict={
            "data_path": str(PROCESSED_ROOT / "recbole"),
            "show_progress": False,
        },
    )
    dataset = create_dataset(config)
    train_data, valid_data, test_data = dataset.build()
    expected_counts = manifest["counts"]
    observed = {
        "train": len(train_data),
        "valid": len(valid_data),
        "test": len(test_data),
        "users_including_padding_token": dataset.user_num,
        "meals_including_padding_token": dataset.item_num,
    }
    expected = {
        "train": expected_counts["train_interactions"],
        "valid": expected_counts["valid_interactions_warm"],
        "test": expected_counts["test_interactions_warm"],
    }
    if {key: observed[key] for key in expected} != expected:
        raise ValueError(f"RecBole split counts disagree: {observed}")
    print(json.dumps(observed, indent=2))


if __name__ == "__main__":
    main()
