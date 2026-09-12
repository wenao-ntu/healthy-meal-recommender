"""Create deterministic training and demo files from the audited MealRec+H source."""

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
from pathlib import Path
from statistics import fmean
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "raw" / "mealrecplus-source"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "mealrecplus_h"
DEFAULT_EVIDENCE = (
    PROJECT_ROOT / "results" / "data_preparation" / "mealrecplus_h_manifest.json"
)
AUDIT_PATH = PROJECT_ROOT / "results" / "data_audit" / "mealrecplus_h_audit.json"

EXPECTED_SOURCE_COMMIT = "8bd7d7d7effb505be4a7d2074b28abad802fe0f7"
CATEGORY_ID_TO_NAME = {0: "appetizer", 1: "main-dish", 2: "dessert"}
NUTRIENTS = {
    "calories": ("calories_kcal", "kcal"),
    "protein": ("protein_g", "g"),
    "carbohydrates": ("carbohydrates_g", "g"),
    "sugars": ("sugars_g", "g"),
    "sodium": ("sodium_mg", "mg"),
    "fat": ("fat_g", "g"),
    "saturatedFat": ("saturated_fat_g", "g"),
    "fiber": ("fiber_g", "g"),
}
USED_SOURCE_ARTIFACTS = (
    "MealRec+/MealRec+H/course_category.txt",
    "MealRec+/MealRec+H/healthiness/course_fsa.txt",
    "MealRec+/MealRec+H/healthiness/course_who.txt",
    "MealRec+/MealRec+H/healthiness/meal_fsa.txt",
    "MealRec+/MealRec+H/healthiness/meal_who.txt",
    "MealRec+/MealRec+H/healthiness/user_fsa.txt",
    "MealRec+/MealRec+H/healthiness/user_who.txt",
    "MealRec+/MealRec+H/meal_course.txt",
    "MealRec+/MealRec+H/meta_data/course.zip",
    "MealRec+/MealRec+H/meta_data/course2index.txt",
    "MealRec+/MealRec+H/user_meal_test.txt",
    "MealRec+/MealRec+H/user_meal_train.txt",
    "MealRec+/MealRec+H/user_meal_tune.txt",
)


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_pairs(path: Path) -> list[tuple[int, int]]:
    """Read a two-column tab-separated integer file."""
    pairs: list[tuple[int, int]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 2:
                raise ValueError(f"{path}:{line_number} expected 2 fields")
            pairs.append((int(fields[0]), int(fields[1])))
    return pairs


def read_scores(path: Path) -> list[float]:
    """Read one finite numeric score per non-empty line."""
    with path.open(encoding="utf-8") as handle:
        values = [float(line.strip()) for line in handle if line.strip()]
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"Non-finite score in {path}")
    return values


def format_number(value: float | int) -> str:
    """Serialize numeric values consistently across runs."""
    if isinstance(value, int):
        return str(value)
    return format(value, ".15g")


def clean_text(value: str | None) -> str:
    """Collapse embedded whitespace so output CSVs stay compact and one-row-per-line."""
    return " ".join((value or "").split())


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: Iterable[dict[str, object]],
    *,
    delimiter: str = ",",
) -> int:
    """Write a deterministic UTF-8 CSV and return its logical row count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter=delimiter,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def write_recbole_interactions(path: Path, rows: list[tuple[int, int]]) -> int:
    """Write RecBole atomic interactions using stable user/meal ordering."""
    return write_csv(
        path,
        ["user_id:token", "meal_id:token"],
        (
            {"user_id:token": user_id, "meal_id:token": meal_id}
            for user_id, meal_id in sorted(rows)
        ),
        delimiter="\t",
    )


def source_commit(source: Path) -> str:
    """Return the exact checked-out upstream commit."""
    result = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def verify_inputs(source: Path) -> dict[str, dict[str, object]]:
    """Match every used raw artifact against the committed audit evidence."""
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    if source_commit(source) != EXPECTED_SOURCE_COMMIT:
        raise ValueError("MealRecPlus source commit does not match the audited revision")

    verified: dict[str, dict[str, object]] = {}
    for relative in USED_SOURCE_ARTIFACTS:
        path = source / Path(relative)
        observed = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
        if observed != audit["artifacts"][relative]:
            raise ValueError(f"Raw artifact differs from audit evidence: {relative}")
        verified[relative] = observed
    return verified


def load_course_mapping(path: Path) -> dict[str, int]:
    """Map source recipe identifiers to zero-based course identifiers."""
    mapping: dict[str, int] = {}
    for source_id, internal_id in read_pairs(path):
        key = str(source_id)
        if key in mapping:
            raise ValueError(f"Duplicate source course id: {key}")
        mapping[key] = internal_id
    if set(mapping.values()) != set(range(len(mapping))):
        raise ValueError("Course indices are not contiguous and zero-based")
    return mapping


def nutrition_amounts(payload: str, course_id: int) -> dict[str, float | None]:
    """Parse nutrition amounts while preserving documented source placeholders."""
    parsed = ast.literal_eval(payload)
    values: dict[str, float | None] = {}
    for source_name, (output_name, expected_unit) in NUTRIENTS.items():
        nutrient = parsed.get(source_name)
        if not isinstance(nutrient, dict):
            raise ValueError(f"Course {course_id} lacks {source_name}")
        amount = nutrient.get("amount")
        if isinstance(amount, bool) or not isinstance(amount, (int, float)):
            raise ValueError(f"Course {course_id} has invalid {source_name} amount")
        unit = nutrient.get("unit")
        if unit is None and float(amount) == 0:
            values[output_name] = None
            continue
        if unit != expected_unit:
            raise ValueError(f"Course {course_id} has unexpected {source_name} unit")
        values[output_name] = float(amount)
    return values


def load_used_course_metadata(
    archive_path: Path,
    source_to_internal: dict[str, int],
    used_course_ids: set[int],
    course_categories: dict[int, int],
    course_fsa: list[float],
    course_who: list[float],
) -> dict[int, dict[str, object]]:
    """Stream the large archive and retain only courses that occur in a meal."""
    csv.field_size_limit(min(sys.maxsize, 2_147_483_647))
    selected: dict[int, dict[str, object]] = {}
    with zipfile.ZipFile(archive_path) as archive, archive.open("course.csv") as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
        for row in csv.DictReader(text):
            course_id = source_to_internal[row["course_id"]]
            if course_id not in used_course_ids:
                continue
            expected_category = CATEGORY_ID_TO_NAME[course_categories[course_id]]
            if row["category"] != expected_category:
                raise ValueError(f"Course {course_id} category disagrees with relationship file")
            nutrients = nutrition_amounts(row["nutritions"], course_id)
            selected[course_id] = {
                "course_id": course_id,
                "course_name": clean_text(row["course_name"]),
                "category": expected_category,
                "image_url": clean_text(row["image_url"]),
                "ingredients": clean_text(row["ingredients"]),
                "tags": clean_text(row["tags"]),
                **{
                    name: "" if value is None else format_number(value)
                    for name, value in nutrients.items()
                },
                "fsa_score": format_number(course_fsa[course_id]),
                "who_score": format_number(course_who[course_id]),
            }
    if set(selected) != used_course_ids:
        missing = sorted(used_course_ids - set(selected))
        raise ValueError(f"Missing metadata for used course ids: {missing[:10]}")
    return selected


def build_course_categories(rows: list[tuple[int, int]]) -> dict[int, int]:
    """Deduplicate repeated course-category pairs and require one category per course."""
    grouped: dict[int, set[int]] = defaultdict(set)
    for course_id, category_id in rows:
        grouped[course_id].add(category_id)
    if any(len(categories) != 1 for categories in grouped.values()):
        raise ValueError("A course belongs to multiple meal categories")
    result = {course_id: next(iter(categories)) for course_id, categories in grouped.items()}
    if set(result.values()) != set(CATEGORY_ID_TO_NAME):
        raise ValueError("Unexpected meal category ids")
    return result


def build_meal_courses(
    rows: list[tuple[int, int]], course_categories: dict[int, int]
) -> dict[int, dict[int, int]]:
    """Represent each meal as exactly one course from each required category."""
    grouped: dict[int, dict[int, int]] = defaultdict(dict)
    for meal_id, course_id in rows:
        category_id = course_categories[course_id]
        existing = grouped[meal_id].get(category_id)
        if existing is not None and existing != course_id:
            raise ValueError(f"Meal {meal_id} has multiple courses in category {category_id}")
        grouped[meal_id][category_id] = course_id
    if any(set(courses) != set(CATEGORY_ID_TO_NAME) for courses in grouped.values()):
        raise ValueError("Every meal must have appetizer, main dish and dessert")
    if set(grouped) != set(range(len(grouped))):
        raise ValueError("Meal ids are not contiguous and zero-based")
    return dict(grouped)


def build_user_health_rows(
    train: list[tuple[int, int]],
    meal_fsa: list[float],
    meal_who: list[float],
    published_fsa: list[float],
    published_who: list[float],
) -> tuple[list[dict[str, object]], float, float]:
    """Recompute user baselines from training interactions only and verify them."""
    grouped: dict[int, list[int]] = defaultdict(list)
    for user_id, meal_id in train:
        grouped[user_id].append(meal_id)
    if set(grouped) != set(range(len(published_fsa))) or len(published_fsa) != len(published_who):
        raise ValueError("Training split does not cover every published user")

    rows: list[dict[str, object]] = []
    fsa_errors: list[float] = []
    who_errors: list[float] = []
    for user_id in sorted(grouped):
        meal_ids = grouped[user_id]
        fsa_baseline = fmean(meal_fsa[meal_id] for meal_id in meal_ids)
        who_baseline = fmean(meal_who[meal_id] for meal_id in meal_ids)
        fsa_errors.append(abs(fsa_baseline - published_fsa[user_id]))
        who_errors.append(abs(who_baseline - published_who[user_id]))
        rows.append(
            {
                "user_id": user_id,
                "train_interactions": len(meal_ids),
                "fsa_baseline": format_number(fsa_baseline),
                "who_baseline": format_number(who_baseline),
            }
        )
    max_fsa_error = max(fsa_errors)
    max_who_error = max(who_errors)
    if max_fsa_error > 1e-10 or max_who_error > 1e-10:
        raise ValueError("Recomputed train-only user health baselines disagree with source")
    return rows, max_fsa_error, max_who_error


def build_meal_catalog_rows(
    meal_courses: dict[int, dict[int, int]],
    course_metadata: dict[int, dict[str, object]],
    meal_fsa: list[float],
    meal_who: list[float],
    train_counts: Counter[int],
) -> list[dict[str, object]]:
    """Create one compact, presentation-ready row per meal."""
    rows: list[dict[str, object]] = []
    nutrient_outputs = [output_name for output_name, _ in NUTRIENTS.values()]
    for meal_id in sorted(meal_courses):
        courses = meal_courses[meal_id]
        category_courses = {
            CATEGORY_ID_TO_NAME[category_id]: course_metadata[course_id]
            for category_id, course_id in courses.items()
        }
        calculated_fsa = fmean(float(course["fsa_score"]) for course in category_courses.values())
        calculated_who = fmean(float(course["who_score"]) for course in category_courses.values())
        if abs(calculated_fsa - meal_fsa[meal_id]) > 1e-10:
            raise ValueError(f"Meal {meal_id} FSA score disagrees with its courses")
        if abs(calculated_who - meal_who[meal_id]) > 1e-10:
            raise ValueError(f"Meal {meal_id} WHO score disagrees with its courses")

        row: dict[str, object] = {
            "meal_id": meal_id,
            "appetizer_course_id": category_courses["appetizer"]["course_id"],
            "main_dish_course_id": category_courses["main-dish"]["course_id"],
            "dessert_course_id": category_courses["dessert"]["course_id"],
            "appetizer_name": category_courses["appetizer"]["course_name"],
            "main_dish_name": category_courses["main-dish"]["course_name"],
            "dessert_name": category_courses["dessert"]["course_name"],
            "fsa_score": format_number(meal_fsa[meal_id]),
            "who_score": format_number(meal_who[meal_id]),
            "train_interactions": train_counts[meal_id],
        }
        for nutrient_name in nutrient_outputs:
            values = [course[nutrient_name] for course in category_courses.values()]
            if any(value == "" for value in values):
                row[nutrient_name] = ""
            else:
                row[nutrient_name] = format_number(sum(float(value) for value in values))
        rows.append(row)
    return rows


def relative_to_project(path: Path) -> str:
    """Use portable project-relative paths in evidence files."""
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def main() -> None:
    """Prepare RecBole inputs, compact demo tables and reproducibility evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--evidence-output", type=Path, default=DEFAULT_EVIDENCE)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    dataset = source / "MealRec+" / "MealRec+H"
    health = dataset / "healthiness"
    metadata = dataset / "meta_data"
    input_artifacts = verify_inputs(source)

    train = read_pairs(dataset / "user_meal_train.txt")
    valid_original = read_pairs(dataset / "user_meal_tune.txt")
    test_original = read_pairs(dataset / "user_meal_test.txt")
    if len(set(train)) != len(train) or len(set(valid_original)) != len(valid_original):
        raise ValueError("Duplicate interaction found in train or validation")
    if len(set(test_original)) != len(test_original):
        raise ValueError("Duplicate interaction found in test")
    if set(train) & set(valid_original) or set(train) & set(test_original):
        raise ValueError("Official train/evaluation splits overlap")
    if set(valid_original) & set(test_original):
        raise ValueError("Official validation/test splits overlap")

    train_meals = {meal_id for _, meal_id in train}
    valid = [pair for pair in valid_original if pair[1] in train_meals]
    test = [pair for pair in test_original if pair[1] in train_meals]
    exclusions = [
        {"split": split, "user_id": user_id, "meal_id": meal_id, "reason": "cold_meal"}
        for split, rows in (("valid", valid_original), ("test", test_original))
        for user_id, meal_id in rows
        if meal_id not in train_meals
    ]

    course_categories = build_course_categories(read_pairs(dataset / "course_category.txt"))
    meal_courses = build_meal_courses(
        read_pairs(dataset / "meal_course.txt"), course_categories
    )
    used_course_ids = {course_id for courses in meal_courses.values() for course_id in courses.values()}

    course_fsa = read_scores(health / "course_fsa.txt")
    course_who = read_scores(health / "course_who.txt")
    meal_fsa = read_scores(health / "meal_fsa.txt")
    meal_who = read_scores(health / "meal_who.txt")
    published_user_fsa = read_scores(health / "user_fsa.txt")
    published_user_who = read_scores(health / "user_who.txt")
    if len(course_fsa) != 7_280 or len(course_who) != 7_280:
        raise ValueError("Unexpected course health vector length")
    if len(meal_fsa) != len(meal_courses) or len(meal_who) != len(meal_courses):
        raise ValueError("Meal health vectors do not match meal ids")

    source_to_internal = load_course_mapping(metadata / "course2index.txt")
    course_metadata = load_used_course_metadata(
        metadata / "course.zip",
        source_to_internal,
        used_course_ids,
        course_categories,
        course_fsa,
        course_who,
    )
    user_health_rows, max_fsa_error, max_who_error = build_user_health_rows(
        train,
        meal_fsa,
        meal_who,
        published_user_fsa,
        published_user_who,
    )
    meal_catalog_rows = build_meal_catalog_rows(
        meal_courses, course_metadata, meal_fsa, meal_who, Counter(meal_id for _, meal_id in train)
    )
    nutrient_outputs = [name for name, _ in NUTRIENTS.values()]
    courses_with_missing_nutrition = sum(
        any(course[name] == "" for name in nutrient_outputs)
        for course in course_metadata.values()
    )
    meals_with_missing_nutrition = sum(
        any(row[name] == "" for name in nutrient_outputs) for row in meal_catalog_rows
    )

    recbole_dir = output / "recbole" / "mealrecplus_h"
    output_rows: dict[Path, int] = {}
    for split_name, rows in (("train", train), ("valid", valid), ("test", test)):
        path = recbole_dir / f"mealrecplus_h.{split_name}.inter"
        output_rows[path] = write_recbole_interactions(path, rows)

    course_fields = [
        "course_id",
        "course_name",
        "category",
        "image_url",
        "ingredients",
        "tags",
        *[name for name, _ in NUTRIENTS.values()],
        "fsa_score",
        "who_score",
    ]
    course_path = output / "course_catalog.csv"
    output_rows[course_path] = write_csv(
        course_path, course_fields, (course_metadata[key] for key in sorted(course_metadata))
    )

    meal_fields = [
        "meal_id",
        "appetizer_course_id",
        "main_dish_course_id",
        "dessert_course_id",
        "appetizer_name",
        "main_dish_name",
        "dessert_name",
        *[name for name, _ in NUTRIENTS.values()],
        "fsa_score",
        "who_score",
        "train_interactions",
    ]
    meal_path = output / "meal_catalog.csv"
    output_rows[meal_path] = write_csv(meal_path, meal_fields, meal_catalog_rows)

    user_path = output / "user_health_train.csv"
    output_rows[user_path] = write_csv(
        user_path,
        ["user_id", "train_interactions", "fsa_baseline", "who_baseline"],
        user_health_rows,
    )
    exclusions_path = output / "evaluation_exclusions.csv"
    output_rows[exclusions_path] = write_csv(
        exclusions_path,
        ["split", "user_id", "meal_id", "reason"],
        sorted(exclusions, key=lambda row: (str(row["split"]), int(row["user_id"]), int(row["meal_id"]))),
    )

    output_artifacts = {
        path.relative_to(output).as_posix(): {
            "bytes": path.stat().st_size,
            "rows": row_count,
            "sha256": file_sha256(path),
        }
        for path, row_count in sorted(output_rows.items(), key=lambda item: item[0].as_posix())
    }
    report = {
        "preparation_schema_version": 1,
        "source": {
            "repository": "https://github.com/WUT-IDEA/MealRecPlus",
            "commit": EXPECTED_SOURCE_COMMIT,
            "variant": "MealRec+H",
            "input_artifacts": input_artifacts,
        },
        "policies": {
            "official_split_preserved": True,
            "evaluation_policy": "warm-start meals only; exclusions listed explicitly",
            "user_health_baseline": "mean meal score from training interactions only",
            "metadata_scope": "only courses used by a meal; reviews and cooking text excluded",
            "source_user_mapping_used": False,
            "timestamps_available_for_meal_interactions": False,
        },
        "counts": {
            "users": len(user_health_rows),
            "meals_catalog": len(meal_catalog_rows),
            "meals_seen_in_train": len(train_meals),
            "courses_used_by_meals": len(course_metadata),
            "train_interactions": len(train),
            "valid_interactions_original": len(valid_original),
            "valid_interactions_warm": len(valid),
            "test_interactions_original": len(test_original),
            "test_interactions_warm": len(test),
            "evaluation_exclusions": len(exclusions),
        },
        "validation": {
            "split_overlap_rows": 0,
            "all_evaluation_meals_seen_in_train": True,
            "courses_per_meal": 3,
            "required_categories_per_meal": ["appetizer", "main-dish", "dessert"],
            "zero_without_unit_treated_as_missing": True,
            "courses_with_missing_nutrition": courses_with_missing_nutrition,
            "meals_with_missing_nutrition": meals_with_missing_nutrition,
            "max_user_fsa_baseline_error": max_fsa_error,
            "max_user_who_baseline_error": max_who_error,
        },
        "outputs": output_artifacts,
        "reproducibility": {
            "stable_sort": "user_id then meal_id; entity tables by numeric id",
            "encoding": "UTF-8",
            "line_ending": "LF",
            "generation_timestamp_included": False,
        },
    }
    manifest_text = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    processed_manifest = output / "preparation_manifest.json"
    processed_manifest.write_text(manifest_text, encoding="utf-8", newline="\n")
    args.evidence_output.parent.mkdir(parents=True, exist_ok=True)
    args.evidence_output.write_text(manifest_text, encoding="utf-8", newline="\n")

    print(f"Prepared data: {relative_to_project(output)}")
    print(f"Evidence: {relative_to_project(args.evidence_output)}")
    print(
        json.dumps(
            {
                "train": len(train),
                "valid_warm": len(valid),
                "test_warm": len(test),
                "excluded": len(exclusions),
                "meals": len(meal_catalog_rows),
                "courses_retained": len(course_metadata),
                "users": len(user_health_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
