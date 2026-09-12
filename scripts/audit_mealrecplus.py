"""Audit the official MealRec+ H files without copying raw data into Git."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "raw" / "mealrecplus-source"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "data_audit" / "mealrecplus_h_audit.json"

COURSE_FIELDS = [
    "course_id",
    "course_name",
    "review_nums",
    "category",
    "aver_rate",
    "image_url",
    "ingredients",
    "cooking_directions",
    "nutritions",
    "reviews",
    "tags",
]
USER_COURSE_FIELDS = ["user_id", "course_id", "rating", "dateLastModified"]
WHO_NUTRIENTS = [
    "protein",
    "carbohydrates",
    "sugars",
    "sodium",
    "fat",
    "saturatedFat",
    "fiber",
]


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_pairs(path: Path) -> list[tuple[int, int]]:
    """Read a two-column, tab-separated integer relationship file."""
    pairs: list[tuple[int, int]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 2:
                raise ValueError(f"{path}:{line_number} expected 2 fields, got {len(fields)}")
            pairs.append((int(fields[0]), int(fields[1])))
    return pairs


def read_scores(path: Path) -> list[float]:
    """Read one numeric health score per line."""
    with path.open(encoding="utf-8") as handle:
        return [float(line.strip()) for line in handle if line.strip()]


def id_stats(values: Iterable[int]) -> dict[str, object]:
    """Summarize an integer identifier sequence."""
    value_set = set(values)
    minimum = min(value_set)
    maximum = max(value_set)
    return {
        "unique": len(value_set),
        "min": minimum,
        "max": maximum,
        "contiguous_zero_based": value_set == set(range(maximum + 1)),
    }


def score_stats(values: list[float]) -> dict[str, object]:
    """Summarize a health-score vector."""
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": round(fmean(values), 8),
        "non_finite": sum(not math.isfinite(value) for value in values),
    }


def mean_score_by_user(
    pairs: Iterable[tuple[int, int]], item_scores: list[float]
) -> dict[int, float]:
    """Calculate mean item healthiness from only the supplied interactions."""
    grouped: dict[int, list[float]] = defaultdict(list)
    for user_id, item_id in pairs:
        grouped[user_id].append(item_scores[item_id])
    return {user_id: fmean(scores) for user_id, scores in grouped.items()}


def compare_user_scores(
    calculated: dict[int, float], published: list[float]
) -> dict[str, object]:
    """Compare calculated user means with the published per-user vector."""
    absolute_errors = [abs(value - published[user_id]) for user_id, value in calculated.items()]
    return {
        "users_compared": len(absolute_errors),
        "mean_absolute_error": round(fmean(absolute_errors), 12),
        "max_absolute_error": round(max(absolute_errors), 12),
        "exact_within_1e-10": sum(error <= 1e-10 for error in absolute_errors),
    }


def audit_course_zip(path: Path) -> dict[str, object]:
    """Stream course metadata and audit the nutrition payload."""
    csv.field_size_limit(min(sys.maxsize, 2_147_483_647))
    category_counts: Counter[str] = Counter()
    nutrient_presence: Counter[str] = Counter()
    selected_nulls: Counter[str] = Counter()
    course_ids: set[str] = set()
    nutrition_parse_errors = 0
    row_count = 0

    with zipfile.ZipFile(path) as archive:
        members = [
            {"name": info.filename, "uncompressed_bytes": info.file_size}
            for info in archive.infolist()
        ]
        with archive.open("course.csv") as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
            reader = csv.DictReader(text)
            fields = reader.fieldnames or []
            for row in reader:
                row_count += 1
                course_ids.add(row["course_id"])
                category_counts[row["category"]] += 1
                for field in ("course_name", "category", "ingredients", "nutritions"):
                    selected_nulls[field] += not bool(row[field].strip())
                try:
                    nutritions = ast.literal_eval(row["nutritions"])
                except (SyntaxError, ValueError):
                    nutrition_parse_errors += 1
                    continue
                for nutrient in WHO_NUTRIENTS:
                    value = nutritions.get(nutrient)
                    if isinstance(value, dict) and isinstance(value.get("amount"), (int, float)):
                        nutrient_presence[nutrient] += 1

    return {
        "archive_members": members,
        "fields": fields,
        "fields_match_readme": fields == COURSE_FIELDS,
        "rows": row_count,
        "unique_course_ids": len(course_ids),
        "category_counts": dict(sorted(category_counts.items())),
        "selected_null_counts": dict(selected_nulls),
        "nutrition_parse_errors": nutrition_parse_errors,
        "nutrient_amount_coverage": {
            nutrient: {
                "rows": nutrient_presence[nutrient],
                "ratio": round(nutrient_presence[nutrient] / row_count, 8),
            }
            for nutrient in WHO_NUTRIENTS
        },
    }


def audit_user_course_zip(path: Path) -> dict[str, object]:
    """Stream user-course metadata and report only aggregate facts."""
    row_count = 0
    user_ids: set[str] = set()
    course_ids: set[str] = set()
    dates: list[str] = []
    with zipfile.ZipFile(path) as archive:
        members = [
            {"name": info.filename, "uncompressed_bytes": info.file_size}
            for info in archive.infolist()
        ]
        with archive.open("user_course.csv") as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
            reader = csv.DictReader(text)
            fields = reader.fieldnames or []
            for row in reader:
                row_count += 1
                user_ids.add(row["user_id"])
                course_ids.add(row["course_id"])
                date = row["dateLastModified"].strip()
                if date:
                    dates.append(date)
    return {
        "archive_members": members,
        "fields": fields,
        "fields_match_readme": fields == USER_COURSE_FIELDS,
        "rows": row_count,
        "unique_source_user_ids": len(user_ids),
        "unique_source_course_ids": len(course_ids),
        "date_min_lexical": min(dates),
        "date_max_lexical": max(dates),
    }


def get_source_commit(source: Path) -> str:
    """Read the exact source revision without changing the source clone."""
    process = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip()


def count_lines(path: Path) -> int:
    """Count non-empty text lines."""
    with path.open(encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def main() -> None:
    """Audit MealRec+ H and write a machine-readable evidence file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source = args.source.resolve()
    dataset = source / "MealRec+" / "MealRec+H"
    health = dataset / "healthiness"
    metadata = dataset / "meta_data"

    relation_paths = {
        "user_course": dataset / "user_course.txt",
        "course_category": dataset / "course_category.txt",
        "meal_course": dataset / "meal_course.txt",
        "user_meal_train": dataset / "user_meal_train.txt",
        "user_meal_tune": dataset / "user_meal_tune.txt",
        "user_meal_test": dataset / "user_meal_test.txt",
    }
    relations = {name: read_pairs(path) for name, path in relation_paths.items()}
    score_paths = {
        "course_fsa": health / "course_fsa.txt",
        "course_who": health / "course_who.txt",
        "meal_fsa": health / "meal_fsa.txt",
        "meal_who": health / "meal_who.txt",
        "user_fsa": health / "user_fsa.txt",
        "user_who": health / "user_who.txt",
    }
    scores = {name: read_scores(path) for name, path in score_paths.items()}

    split_names = ["user_meal_train", "user_meal_tune", "user_meal_test"]
    split_sets = {name: set(relations[name]) for name in split_names}
    all_user_meal = [pair for name in split_names for pair in relations[name]]
    all_user_meal_set = set(all_user_meal)

    meal_course_counts = Counter(meal_id for meal_id, _ in relations["meal_course"])
    category_values = Counter(category for _, category in relations["course_category"])
    course_categories: dict[int, set[int]] = defaultdict(set)
    for course_id, category_id in relations["course_category"]:
        course_categories[course_id].add(category_id)
    meal_categories: dict[int, set[int]] = defaultdict(set)
    for meal_id, course_id in relations["meal_course"]:
        meal_categories[meal_id].update(course_categories[course_id])
    unique_courses_by_category = {
        category_id: len(
            {
                course_id
                for course_id, category_ids in course_categories.items()
                if category_id in category_ids
            }
        )
        for category_id in sorted(category_values)
    }

    train_meals = {meal_id for _, meal_id in relations["user_meal_train"]}
    cold_start_evaluation = {}
    for split_name in ("user_meal_tune", "user_meal_test"):
        rows = relations[split_name]
        cold_meals = {meal_id for _, meal_id in rows} - train_meals
        warm_rows = [pair for pair in rows if pair[1] in train_meals]
        cold_start_evaluation[split_name.removeprefix("user_meal_")] = {
            "cold_meal_ids": sorted(cold_meals),
            "cold_interactions": sum(meal_id in cold_meals for _, meal_id in rows),
            "warm_interactions": len(warm_rows),
            "warm_users": len({user_id for user_id, _ in warm_rows}),
            "users_removed_by_warm_filter": len({user_id for user_id, _ in rows})
            - len({user_id for user_id, _ in warm_rows}),
        }

    published_user_fsa = scores["user_fsa"]
    published_user_who = scores["user_who"]
    train_fsa = mean_score_by_user(relations["user_meal_train"], scores["meal_fsa"])
    train_who = mean_score_by_user(relations["user_meal_train"], scores["meal_who"])
    full_fsa = mean_score_by_user(all_user_meal, scores["meal_fsa"])
    full_who = mean_score_by_user(all_user_meal, scores["meal_who"])

    artifact_paths = [
        *relation_paths.values(),
        *score_paths.values(),
        metadata / "course2index.txt",
        metadata / "user2index.txt",
        metadata / "course.zip",
        metadata / "user_course.zip",
    ]
    artifacts = {
        str(path.relative_to(source)).replace("\\", "/"): {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in artifact_paths
    }

    license_files = [
        path.name
        for path in source.iterdir()
        if path.is_file() and path.name.lower().startswith(("license", "copying"))
    ]

    report = {
        "audit_schema_version": 1,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repository": "https://github.com/WUT-IDEA/MealRecPlus",
            "commit": get_source_commit(source),
            "variant": "MealRec+H",
            "license_files": license_files,
            "license_status": (
                "explicit_license_present" if license_files else "no_explicit_license_file"
            ),
            "readme_requires_citation": True,
        },
        "artifacts": artifacts,
        "relationships": {
            name: {
                "rows": len(rows),
                "duplicate_rows": len(rows) - len(set(rows)),
                "left_ids": id_stats(left for left, _ in rows),
                "right_ids": id_stats(right for _, right in rows),
            }
            for name, rows in relations.items()
        },
        "user_meal_split": {
            "total_rows": len(all_user_meal),
            "total_unique_rows": len(all_user_meal_set),
            "train_tune_overlap": len(split_sets["user_meal_train"] & split_sets["user_meal_tune"]),
            "train_test_overlap": len(split_sets["user_meal_train"] & split_sets["user_meal_test"]),
            "tune_test_overlap": len(split_sets["user_meal_tune"] & split_sets["user_meal_test"]),
            "ratios": {
                name.removeprefix("user_meal_"): round(
                    len(relations[name]) / len(all_user_meal), 8
                )
                for name in split_names
            },
            "meal_interaction_timestamp_available": False,
            "cold_start_against_train": cold_start_evaluation,
        },
        "bundle_structure": {
            "meals": len(meal_course_counts),
            "courses_per_meal_counts": dict(sorted(Counter(meal_course_counts.values()).items())),
            "category_relationship_counts": dict(sorted(category_values.items())),
            "unique_courses_in_meals": len(course_categories),
            "unique_course_category_pairs": sum(len(values) for values in course_categories.values()),
            "courses_with_multiple_categories": sum(
                len(values) != 1 for values in course_categories.values()
            ),
            "unique_courses_by_category": unique_courses_by_category,
            "meals_with_exact_categories_0_1_2": sum(
                category_ids == {0, 1, 2} for category_ids in meal_categories.values()
            ),
        },
        "healthiness": {
            name: score_stats(values) for name, values in scores.items()
        },
        "published_user_baseline_comparison": {
            "fsa_train_only": compare_user_scores(train_fsa, published_user_fsa),
            "fsa_all_splits": compare_user_scores(full_fsa, published_user_fsa),
            "who_train_only": compare_user_scores(train_who, published_user_who),
            "who_all_splits": compare_user_scores(full_who, published_user_who),
        },
        "metadata": {
            "course_mapping_rows": count_lines(metadata / "course2index.txt"),
            "user_mapping_rows": count_lines(metadata / "user2index.txt"),
            "course_csv": audit_course_zip(metadata / "course.zip"),
            "user_course_csv": audit_user_course_zip(metadata / "user_course.zip"),
        },
        "privacy_scope": {
            "model_ids": "zero-based numeric indices",
            "source_user_mapping_present": True,
            "source_user_mapping_required_for_model": False,
            "medical_or_patient_fields_observed": False,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Audit written to {args.output}")
    print(
        json.dumps(
            {
                "source_commit": report["source"]["commit"],
                "license_status": report["source"]["license_status"],
                "user_meal_rows": report["user_meal_split"]["total_rows"],
                "course_rows": report["metadata"]["course_csv"]["rows"],
                "fsa_baseline_comparison": report["published_user_baseline_comparison"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
