"""Tune health rerankers on validation, then evaluate selected settings on test."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from healthy_meal_recommender.reranking import (
    DEFAULT_GAP_REFERENCE,
    fixed_weight_order,
    gap_rank_order,
    hard_filter_order,
    health_utility,
    preference_order,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "mealrecplus_h"
CANDIDATE_DIR = PROCESSED_ROOT / "candidates"
CANDIDATE_MANIFEST = (
    PROJECT_ROOT / "results" / "candidates" / "neumf_standard_top100.json"
)
MEAL_CATALOG = PROCESSED_ROOT / "meal_catalog.csv"
USER_BASELINES = PROCESSED_ROOT / "user_health_train.csv"
INTERACTION_DIR = PROCESSED_ROOT / "recbole" / "mealrecplus_h"
NEUMF_RESULT = PROJECT_ROOT / "results" / "baselines" / "neumf.json"
DEFAULT_RESULT = PROJECT_ROOT / "results" / "reranking" / "gap_rank.json"
DEFAULT_COMPARISON = (
    PROJECT_ROOT / "results" / "comparison" / "health_reranking.csv"
)
DEFAULT_TEST_RECOMMENDATIONS = (
    PROCESSED_ROOT / "reranked" / "gap_rank_test_top10.csv"
)

FINAL_K = 10
TARGET_QUANTILE = 0.75
PREFERENCE_RETENTION_FLOOR = 0.90
FIXED_ALPHA_GRID = (0.1, 0.2, 0.3, 0.4)
GAP_QUOTA_SHARE_GRID = (0.2, 0.4, 0.6, 0.8)


@dataclass(frozen=True)
class Meal:
    """Health metadata needed by reranking and examples."""

    meal_id: int
    health: float
    fsa: float
    who: float
    courses: tuple[str, str, str]
    seen_in_train: bool


@dataclass(frozen=True)
class Candidate:
    """One preference candidate enriched with health utility."""

    meal: Meal
    preference_score: float


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_meals() -> dict[int, Meal]:
    """Load meal metadata and calculate the shared health utility once."""
    meals = {}
    with MEAL_CATALOG.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            meal_id = int(row["meal_id"])
            fsa = float(row["fsa_score"])
            who = float(row["who_score"])
            meals[meal_id] = Meal(
                meal_id=meal_id,
                health=float(health_utility(fsa, who)),
                fsa=fsa,
                who=who,
                courses=(
                    row["appetizer_name"],
                    row["main_dish_name"],
                    row["dessert_name"],
                ),
                seen_in_train=int(row["train_interactions"]) > 0,
            )
    return meals


def load_user_baselines() -> dict[int, float]:
    """Load train-only user health baselines and convert them to utility."""
    baselines = {}
    with USER_BASELINES.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            baselines[int(row["user_id"])] = float(
                health_utility(float(row["fsa_baseline"]), float(row["who_baseline"]))
            )
    return baselines


def load_candidates(split: str, meals: dict[int, Meal]) -> dict[int, list[Candidate]]:
    """Load an unlabeled Top-100 file and enforce its per-user contract."""
    path = CANDIDATE_DIR / f"neumf_standard_{split}_top100.csv"
    candidates: dict[int, list[Candidate]] = {}
    ranks: dict[int, list[int]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            user_id = int(row["user_id"])
            candidates.setdefault(user_id, []).append(
                Candidate(
                    meal=meals[int(row["meal_id"])],
                    preference_score=float(row["preference_score"]),
                )
            )
            ranks.setdefault(user_id, []).append(int(row["rank"]))
    for user_id in candidates:
        if ranks[user_id] != list(range(1, 101)):
            raise ValueError(f"User {user_id} does not have exact candidate ranks 1..100")
        if len({row.meal.meal_id for row in candidates[user_id]}) != 100:
            raise ValueError(f"User {user_id} has duplicate candidates")
    return candidates


def load_ground_truth(split: str) -> dict[int, set[int]]:
    """Load labels only inside the evaluator, never into candidate files."""
    suffix = "valid" if split == "validation" else "test"
    path = INTERACTION_DIR / f"mealrecplus_h.{suffix}.inter"
    positives: dict[int, set[int]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader)
        for user_id, meal_id in reader:
            positives.setdefault(int(user_id), set()).add(int(meal_id))
    return positives


def rank_users(
    candidates: dict[int, list[Candidate]],
    baselines: dict[int, float],
    method: str,
    target: float,
    parameter: float | None = None,
) -> dict[int, dict[str, object]]:
    """Apply one method to every user and retain only the final Top-10."""
    rankings = {}
    for user_id in sorted(candidates):
        rows = candidates[user_id]
        preference = np.asarray([row.preference_score for row in rows])
        health = np.asarray([row.meal.health for row in rows])
        meal_ids = np.asarray([row.meal.meal_id for row in rows])
        metadata = {
            "candidate_ceiling": float(health.max()),
            "effective_target": min(target, float(health.max())),
            "intervention_intensity": 0.0,
        }
        if method == "preference_only":
            order = preference_order(preference, meal_ids)
        elif method == "fixed_weight":
            if parameter is None:
                raise ValueError("fixed_weight requires alpha")
            order = fixed_weight_order(preference, health, meal_ids, alpha=parameter)
            metadata["intervention_intensity"] = parameter
        elif method == "hard_filter":
            order = hard_filter_order(preference, health, meal_ids, target=target)
        elif method == "gap_rank":
            if parameter is None:
                raise ValueError("gap_rank requires max_quota_share")
            order, metadata = gap_rank_order(
                preference,
                health,
                meal_ids,
                user_baseline=baselines[user_id],
                target=target,
                max_quota_share=parameter,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        target_count = int((health >= target).sum())
        rankings[user_id] = {
            "items": [rows[int(index)] for index in order[:FINAL_K]],
            "intervention_intensity": float(metadata["intervention_intensity"]),
            "candidate_ceiling": float(metadata["candidate_ceiling"]),
            "target_candidates": target_count,
            "backfill_items": max(0, FINAL_K - target_count)
            if method == "hard_filter"
            else 0,
        }
    return rankings


def evaluate_rankings(
    rankings: dict[int, dict[str, object]],
    ground_truth: dict[int, set[int]],
    baselines: dict[int, float],
    target: float,
    preference_rankings: dict[int, dict[str, object]],
) -> dict[str, float]:
    """Evaluate relevance, health benefit, goal coverage, and fidelity."""
    recalls = []
    ndcgs = []
    health_means = []
    goal_item_rates = []
    any_goal = []
    full_goal = []
    overlaps = []
    intervention_intensities = []
    feasible_any = []
    feasible_full = []
    backfills = []
    discounts = 1.0 / np.log2(np.arange(2, FINAL_K + 2))

    for user_id in sorted(rankings):
        items = rankings[user_id]["items"]
        recommended_ids = [row.meal.meal_id for row in items]
        relevant = ground_truth[user_id]
        relevance = np.asarray([meal_id in relevant for meal_id in recommended_ids])
        recalls.append(float(relevance.sum()) / len(relevant))
        dcg = float((relevance * discounts).sum())
        ideal_length = min(FINAL_K, len(relevant))
        idcg = float(discounts[:ideal_length].sum())
        ndcgs.append(dcg / idcg)

        health = np.asarray([row.meal.health for row in items])
        health_means.append(float(health.mean()))
        meets_goal = health >= target
        goal_item_rates.append(float(meets_goal.mean()))
        any_goal.append(float(meets_goal.any()))
        full_goal.append(float(meets_goal.all()))
        base_ids = {
            row.meal.meal_id for row in preference_rankings[user_id]["items"]
        }
        overlaps.append(len(base_ids & set(recommended_ids)) / FINAL_K)
        intervention_intensities.append(
            float(rankings[user_id]["intervention_intensity"])
        )
        target_candidates = int(rankings[user_id]["target_candidates"])
        feasible_any.append(float(target_candidates >= 1))
        feasible_full.append(float(target_candidates >= FINAL_K))
        backfills.append(float(rankings[user_id]["backfill_items"]))

    mean_health = float(np.mean(health_means))
    mean_baseline = float(np.mean([baselines[user_id] for user_id in rankings]))
    return {
        "recall@10": float(np.mean(recalls)),
        "ndcg@10": float(np.mean(ndcgs)),
        "mean_health_utility@10": mean_health,
        "mean_health_lift_vs_user_baseline@10": mean_health - mean_baseline,
        "goal_item_rate@10": float(np.mean(goal_item_rates)),
        "users_with_any_goal_item@10": float(np.mean(any_goal)),
        "users_with_all_goal_items@10": float(np.mean(full_goal)),
        "preference_top10_overlap": float(np.mean(overlaps)),
        "mean_intervention_intensity": float(np.mean(intervention_intensities)),
        "candidate_goal_feasible_user_rate@100": float(np.mean(feasible_any)),
        "candidate_goal_fully_feasible_user_rate@100": float(np.mean(feasible_full)),
        "hard_filter_backfill_items_per_user": float(np.mean(backfills)),
    }


def with_retention(
    metrics: dict[str, float], preference_metrics: dict[str, float]
) -> dict[str, float]:
    """Attach relevance retention relative to no reranking."""
    return {
        **metrics,
        "ndcg@10_retention": metrics["ndcg@10"] / preference_metrics["ndcg@10"],
        "recall@10_retention": metrics["recall@10"]
        / preference_metrics["recall@10"],
    }


def evaluate_method(
    method: str,
    parameter: float | None,
    candidates: dict[int, list[Candidate]],
    ground_truth: dict[int, set[int]],
    baselines: dict[int, float],
    target: float,
    preference_rankings: dict[int, dict[str, object]],
    preference_metrics: dict[str, float],
) -> tuple[dict[int, dict[str, object]], dict[str, float]]:
    """Rank and evaluate one method under the shared protocol."""
    rankings = rank_users(candidates, baselines, method, target, parameter)
    metrics = evaluate_rankings(
        rankings, ground_truth, baselines, target, preference_rankings
    )
    return rankings, with_retention(metrics, preference_metrics)


def select_constrained_trial(
    trials: list[dict[str, object]], parameter_name: str
) -> dict[str, object]:
    """Maximize goal coverage subject to validation NDCG retention."""
    eligible = [
        trial
        for trial in trials
        if trial["validation"]["ndcg@10_retention"] >= PREFERENCE_RETENTION_FLOOR
    ]
    pool = eligible or trials
    selected = sorted(
        pool,
        key=lambda trial: (
            -trial["validation"]["goal_item_rate@10"],
            -trial["validation"]["mean_health_utility@10"],
            -trial["validation"]["ndcg@10"],
            trial[parameter_name],
        ),
    )[0]
    return {
        **selected,
        "retention_constraint_satisfied": bool(eligible),
    }


def validate_preference_reproduction(
    split: str, metrics: dict[str, float], neumf_result: dict[str, object]
) -> None:
    """Ensure exported candidate order reproduces the original Top-10 metrics."""
    for metric in ("recall@10", "ndcg@10"):
        if not math.isclose(
            metrics[metric],
            float(neumf_result[split][metric]),
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise AssertionError(
                f"{split} {metric} differs from committed NeuMF result: "
                f"{metrics[metric]} vs {neumf_result[split][metric]}"
            )


def write_comparison(
    path: Path,
    validation: dict[str, dict[str, float]],
    test: dict[str, dict[str, float]],
    selected_alpha: float,
    selected_quota_share: float,
) -> None:
    """Write the compact main table used by the report."""
    fields = [
        "split",
        "method",
        "parameter",
        "recall@10",
        "ndcg@10",
        "ndcg@10_retention",
        "mean_health_utility@10",
        "mean_health_lift_vs_user_baseline@10",
        "goal_item_rate@10",
        "preference_top10_overlap",
    ]
    rows = []
    parameters = {
        "preference_only": "none",
        "fixed_weight": f"alpha={selected_alpha}",
        "hard_filter": "target priority + fallback",
        "gap_rank": f"max_quota_share={selected_quota_share}",
    }
    for split_name, split_result in (("validation", validation), ("test", test)):
        for method in ("preference_only", "fixed_weight", "hard_filter", "gap_rank"):
            rows.append(
                {
                    "split": split_name,
                    "method": method,
                    "parameter": parameters[method],
                    **{field: split_result[method][field] for field in fields[3:]},
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def choose_sample_users(
    baselines: dict[int, float], target: float, available_users: set[int]
) -> list[tuple[str, int]]:
    """Choose no-, medium-, and high-gap anonymous examples."""
    records = sorted(
        (max(0.0, target - baselines[user_id]), user_id) for user_id in available_users
    )
    no_gap = min(records, key=lambda row: (row[0] != 0.0, row[1]))
    positive = [row for row in records if row[0] > 0.0]
    return [
        ("no_gap", no_gap[1]),
        ("median_positive_gap", positive[len(positive) // 2][1]),
        ("largest_gap", positive[-1][1]),
    ]


def recommendation_samples(
    preference: dict[int, dict[str, object]],
    gap_rank: dict[int, dict[str, object]],
    baselines: dict[int, float],
    target: float,
) -> list[dict[str, object]]:
    """Create three compact before/after examples for report and demo checks."""
    samples = []
    for label, user_id in choose_sample_users(baselines, target, set(gap_rank)):
        before = preference[user_id]["items"]
        after = gap_rank[user_id]["items"]

        def serialize(rows: list[Candidate]) -> list[dict[str, object]]:
            return [
                {
                    "meal_id": row.meal.meal_id,
                    "preference_score": row.preference_score,
                    "health_utility": row.meal.health,
                    "fsa_score": row.meal.fsa,
                    "who_score": row.meal.who,
                    "courses": list(row.meal.courses),
                }
                for row in rows
            ]

        before_ids = {row.meal.meal_id for row in before}
        after_ids = {row.meal.meal_id for row in after}
        samples.append(
            {
                "label": label,
                "user_id": user_id,
                "train_only_health_baseline": baselines[user_id],
                "target": target,
                "gap": max(0.0, target - baselines[user_id]),
                "intervention_intensity": gap_rank[user_id]["intervention_intensity"],
                "top10_overlap": len(before_ids & after_ids) / FINAL_K,
                "preference_only": serialize(before),
                "gap_rank": serialize(after),
            }
        )
    return samples


def write_test_recommendations(
    path: Path, rankings: dict[int, dict[str, object]], baselines: dict[int, float], target: float
) -> None:
    """Save final test Top-10 recommendations as a reproducible demo artifact."""
    fields = [
        "user_id",
        "rank",
        "meal_id",
        "preference_score",
        "health_utility",
        "user_health_baseline",
        "health_target",
        "intervention_intensity",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for user_id in sorted(rankings):
            for rank, candidate in enumerate(rankings[user_id]["items"], start=1):
                writer.writerow(
                    {
                        "user_id": user_id,
                        "rank": rank,
                        "meal_id": candidate.meal.meal_id,
                        "preference_score": format(candidate.preference_score, ".12g"),
                        "health_utility": format(candidate.meal.health, ".12g"),
                        "user_health_baseline": format(baselines[user_id], ".12g"),
                        "health_target": format(target, ".12g"),
                        "intervention_intensity": format(
                            rankings[user_id]["intervention_intensity"], ".12g"
                        ),
                    }
                )


def main() -> None:
    """Run leakage-safe validation selection and one final test comparison."""
    required = [
        CANDIDATE_MANIFEST,
        MEAL_CATALOG,
        USER_BASELINES,
        NEUMF_RESULT,
        CANDIDATE_DIR / "neumf_standard_validation_top100.csv",
        CANDIDATE_DIR / "neumf_standard_test_top100.csv",
        INTERACTION_DIR / "mealrecplus_h.valid.inter",
        INTERACTION_DIR / "mealrecplus_h.test.inter",
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")

    meals = load_meals()
    baselines = load_user_baselines()
    train_health = np.asarray([meal.health for meal in meals.values() if meal.seen_in_train])
    target = float(np.quantile(train_health, TARGET_QUANTILE))
    neumf_result = json.loads(NEUMF_RESULT.read_text(encoding="utf-8"))

    # All tunable choices are made here using validation only.
    validation_candidates = load_candidates("validation", meals)
    validation_truth = load_ground_truth("validation")
    validation_preference = rank_users(
        validation_candidates, baselines, "preference_only", target
    )
    validation_preference_metrics = evaluate_rankings(
        validation_preference,
        validation_truth,
        baselines,
        target,
        validation_preference,
    )
    validation_preference_metrics = with_retention(
        validation_preference_metrics, validation_preference_metrics
    )
    validate_preference_reproduction(
        "validation", validation_preference_metrics, neumf_result
    )

    fixed_trials = []
    for alpha in FIXED_ALPHA_GRID:
        _, metrics = evaluate_method(
            "fixed_weight",
            alpha,
            validation_candidates,
            validation_truth,
            baselines,
            target,
            validation_preference,
            validation_preference_metrics,
        )
        fixed_trials.append({"alpha": alpha, "validation": metrics})
    selected_fixed = select_constrained_trial(fixed_trials, "alpha")

    gap_trials = []
    for max_quota_share in GAP_QUOTA_SHARE_GRID:
        _, metrics = evaluate_method(
            "gap_rank",
            max_quota_share,
            validation_candidates,
            validation_truth,
            baselines,
            target,
            validation_preference,
            validation_preference_metrics,
        )
        gap_trials.append(
            {"max_quota_share": max_quota_share, "validation": metrics}
        )
    selected_gap = select_constrained_trial(gap_trials, "max_quota_share")

    validation_results = {"preference_only": validation_preference_metrics}
    _, validation_results["fixed_weight"] = evaluate_method(
        "fixed_weight",
        float(selected_fixed["alpha"]),
        validation_candidates,
        validation_truth,
        baselines,
        target,
        validation_preference,
        validation_preference_metrics,
    )
    _, validation_results["hard_filter"] = evaluate_method(
        "hard_filter",
        None,
        validation_candidates,
        validation_truth,
        baselines,
        target,
        validation_preference,
        validation_preference_metrics,
    )
    _, validation_results["gap_rank"] = evaluate_method(
        "gap_rank",
        float(selected_gap["max_quota_share"]),
        validation_candidates,
        validation_truth,
        baselines,
        target,
        validation_preference,
        validation_preference_metrics,
    )

    # Test labels are loaded only after both parameters have been frozen.
    test_candidates = load_candidates("test", meals)
    test_truth = load_ground_truth("test")
    test_preference = rank_users(test_candidates, baselines, "preference_only", target)
    test_preference_metrics = evaluate_rankings(
        test_preference, test_truth, baselines, target, test_preference
    )
    test_preference_metrics = with_retention(
        test_preference_metrics, test_preference_metrics
    )
    validate_preference_reproduction("test", test_preference_metrics, neumf_result)
    test_results = {"preference_only": test_preference_metrics}
    test_rankings = {}
    for method, parameter in (
        ("fixed_weight", float(selected_fixed["alpha"])),
        ("hard_filter", None),
        ("gap_rank", float(selected_gap["max_quota_share"])),
    ):
        rankings, metrics = evaluate_method(
            method,
            parameter,
            test_candidates,
            test_truth,
            baselines,
            target,
            test_preference,
            test_preference_metrics,
        )
        test_rankings[method] = rankings
        test_results[method] = metrics

    write_comparison(
        DEFAULT_COMPARISON,
        validation_results,
        test_results,
        float(selected_fixed["alpha"]),
        float(selected_gap["max_quota_share"]),
    )
    write_test_recommendations(
        DEFAULT_TEST_RECOMMENDATIONS,
        test_rankings["gap_rank"],
        baselines,
        target,
    )
    result = {
        "reranking_result_schema_version": 1,
        "method": {
            "name": "GAP Rank",
            "purpose": "user health-goal-gap adaptive reranking",
            "candidate_generator": "NeuMF Standard",
            "final_k": FINAL_K,
            "candidate_k": 100,
        },
        "health_utility": {
            "formula": "0.5 * clip((12 - FSA) / 8) + 0.5 * clip(WHO / 7)",
            "range": [0.0, 1.0],
            "higher_is_healthier": True,
            "fsa_direction": "lower is healthier; canonical range 4..12",
            "who_direction": "higher is healthier; canonical range 0..7",
            "weights": {"fsa": 0.5, "who": 0.5},
        },
        "benchmark_goal": {
            "definition": "75th percentile of health utility among train-seen meals",
            "quantile": TARGET_QUANTILE,
            "value": target,
            "source_scope": "meal_catalog rows with train_interactions > 0",
            "clinical_threshold": False,
            "interpretation": "offline wellness ranking target, not medical advice",
        },
        "gap_rank_formula": {
            "user_gap": "max(0, target - train-only user baseline)",
            "effective_target": "min(target, maximum candidate health utility)",
            "feasibility_ratio": "max(0, effective target - baseline) / user gap",
            "intervention_intensity": "max_quota_share * min(user_gap / 0.10, 1) * feasibility ratio",
            "target_quota": "ceil(10 * intervention intensity), capped by feasible target candidates",
            "minimal_change_rule": "if current Top-10 misses the quota, promote the highest-preference target meals and replace the lowest-ranked non-target meals; preserve preference order otherwise",
            "gap_reference": DEFAULT_GAP_REFERENCE,
        },
        "selection": {
            "rule": "maximize validation goal_item_rate@10 subject to at least 90% validation NDCG@10 retention; health utility, NDCG, then smaller parameter break ties",
            "preference_retention_floor": PREFERENCE_RETENTION_FLOOR,
            "test_metrics_used_for_parameter_selection": False,
            "fixed_weight_trials": fixed_trials,
            "selected_fixed_alpha": selected_fixed["alpha"],
            "fixed_retention_constraint_satisfied": selected_fixed[
                "retention_constraint_satisfied"
            ],
            "gap_rank_trials": gap_trials,
            "selected_gap_max_quota_share": selected_gap["max_quota_share"],
            "gap_retention_constraint_satisfied": selected_gap[
                "retention_constraint_satisfied"
            ],
        },
        "protocol": {
            "labels_used_only_by_evaluator": True,
            "user_baselines": "training interactions only",
            "validation_purpose": "parameter selection and method checking",
            "test_purpose": "one final evaluation after parameters are frozen",
            "hard_filter_fallback": "if fewer than 10 candidates meet target, fill remaining slots by preference",
            "preference_metrics_reproduced_with_absolute_tolerance": 1e-6,
        },
        "provenance": {
            "candidate_manifest": CANDIDATE_MANIFEST.relative_to(PROJECT_ROOT).as_posix(),
            "candidate_manifest_sha256": file_sha256(CANDIDATE_MANIFEST),
            "meal_catalog_sha256": file_sha256(MEAL_CATALOG),
            "user_baselines_sha256": file_sha256(USER_BASELINES),
            "neumf_result_sha256": file_sha256(NEUMF_RESULT),
        },
        "validation": validation_results,
        "test": test_results,
        "test_recommendation_artifact": {
            "path": DEFAULT_TEST_RECOMMENDATIONS.relative_to(PROJECT_ROOT).as_posix(),
            "rows": len(test_rankings["gap_rank"]) * FINAL_K,
            "sha256": file_sha256(DEFAULT_TEST_RECOMMENDATIONS),
            "tracked_by_git": False,
        },
        "recommendation_samples": recommendation_samples(
            test_preference,
            test_rankings["gap_rank"],
            baselines,
            target,
        ),
        "limitations": [
            "The composite health utility is an interpretable project proxy, not a clinical score.",
            "Offline goal satisfaction does not establish real-world dietary or health outcomes.",
            "Reranking can only choose from the preference model's Top-100 candidates.",
        ],
    }
    DEFAULT_RESULT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_RESULT.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "target": target,
                "selected_fixed_alpha": selected_fixed["alpha"],
                "selected_gap_max_quota_share": selected_gap["max_quota_share"],
                "validation": validation_results,
                "test": test_results,
                "result": DEFAULT_RESULT.relative_to(PROJECT_ROOT).as_posix(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
