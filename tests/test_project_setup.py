"""Fast tests for repository and synthetic smoke-data contracts."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import healthy_meal_recommender


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_package_is_importable_from_src_layout() -> None:
    """The editable package should resolve to this repository's src directory."""
    package_file = Path(healthy_meal_recommender.__file__).resolve()
    assert PROJECT_ROOT / "src" in package_file.parents


def test_recbole_smoke_fixture_contract() -> None:
    """The fixture must remain large enough for per-user train/valid/test splits."""
    fixture = (
        PROJECT_ROOT
        / "tests"
        / "fixtures"
        / "recbole_smoke"
        / "recbole_smoke.inter"
    )
    with fixture.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    assert len(rows) == 30
    assert {row["user_id:token"] for row in rows} == {f"u{i}" for i in range(1, 7)}
    assert {row["item_id:token"] for row in rows} == {f"i{i}" for i in range(1, 9)}
    interactions_per_user = Counter(row["user_id:token"] for row in rows)
    assert set(interactions_per_user.values()) == {5}
