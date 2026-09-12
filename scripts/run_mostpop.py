"""Train and evaluate the deterministic MostPop preference baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from importlib.metadata import version
from logging import getLogger
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_trainer, init_logger, init_seed
from recbole.utils.case_study import full_sort_scores

from healthy_meal_recommender.models import MostPop


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG = PROJECT_ROOT / "configs" / "data" / "mealrecplus_h.yaml"
EVALUATION_CONFIG = PROJECT_ROOT / "configs" / "evaluation" / "topn.yaml"
MODEL_CONFIG = PROJECT_ROOT / "configs" / "models" / "mostpop.yaml"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "mealrecplus_h"
DATA_MANIFEST = (
    PROJECT_ROOT / "results" / "data_preparation" / "mealrecplus_h_manifest.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "baselines" / "mostpop.json"
SAMPLE_SIZE = 3
SAMPLE_TOPK = 10


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_metrics(metrics: dict[str, object]) -> dict[str, float]:
    """Convert framework numeric scalars into stable JSON floats."""
    return {str(key): float(value) for key, value in sorted(metrics.items())}


def load_meal_catalog(path: Path) -> dict[int, dict[str, str]]:
    """Load compact meal metadata for human-readable recommendation examples."""
    with path.open(encoding="utf-8", newline="") as handle:
        return {int(row["meal_id"]): row for row in csv.DictReader(handle)}


def choose_sample_users(test_data) -> list[tuple[str, int]]:
    """Select low, median and high history users deterministically."""
    records: list[tuple[int, int, int]] = []
    dataset = test_data.dataset
    user_field = dataset.uid_field
    for internal_user_id in test_data.uid_list.tolist():
        raw_user_id = int(dataset.id2token(user_field, internal_user_id))
        history_count = len(test_data.uid2history_item[internal_user_id])
        records.append((history_count, raw_user_id, internal_user_id))
    records.sort()
    positions = [0, len(records) // 2, len(records) - 1]
    labels = ["low_history", "median_history", "high_history"]
    return [(label, records[position][2]) for label, position in zip(labels, positions)]


def finite_ranked_items(
    scores: torch.Tensor,
    dataset,
) -> list[tuple[float, int, int]]:
    """Rank finite item scores with raw meal id as a deterministic tie-break."""
    item_field = dataset.iid_field
    internal_ids = np.arange(1, dataset.item_num)
    raw_ids = dataset.id2token(item_field, internal_ids)
    candidates = [
        (float(scores[internal_id]), int(raw_id), int(internal_id))
        for internal_id, raw_id in zip(internal_ids, raw_ids)
        if torch.isfinite(scores[internal_id])
    ]
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates


def recommendation_samples(model, test_data) -> list[dict[str, object]]:
    """Produce three traceable examples from the real test candidate space."""
    dataset = test_data.dataset
    user_field = dataset.uid_field
    item_field = dataset.iid_field
    meal_catalog = load_meal_catalog(PROCESSED_ROOT / "meal_catalog.csv")
    samples: list[dict[str, object]] = []
    for history_band, internal_user_id in choose_sample_users(test_data):
        scores = full_sort_scores(
            [internal_user_id], model, test_data, device=model.device
        ).squeeze(0)
        ranked = finite_ranked_items(scores.cpu(), dataset)[:SAMPLE_TOPK]
        ground_truth = sorted(
            int(value)
            for value in dataset.id2token(
                item_field, test_data.uid2positive_item[internal_user_id].numpy()
            )
        )
        recommendations: list[dict[str, object]] = []
        for score, meal_id, model_item_id in ranked:
            meal = meal_catalog[meal_id]
            recommendations.append(
                {
                    "meal_id": meal_id,
                    "preference_score": score,
                    "training_popularity": int(model.item_cnt[model_item_id].item()),
                    "courses": [
                        meal["appetizer_name"],
                        meal["main_dish_name"],
                        meal["dessert_name"],
                    ],
                    "fsa_score": float(meal["fsa_score"]),
                    "who_score": float(meal["who_score"]),
                }
            )
        recommended_ids = {row["meal_id"] for row in recommendations}
        samples.append(
            {
                "history_band": history_band,
                "user_id": int(dataset.id2token(user_field, internal_user_id)),
                "history_items_excluded": len(
                    test_data.uid2history_item[internal_user_id]
                ),
                "test_ground_truth_meal_ids": ground_truth,
                "hits_at_10": len(recommended_ids & set(ground_truth)),
                "recommendations": recommendations,
            }
        )
    return samples


def validate_config_files(paths: Iterable[Path]) -> None:
    """Fail clearly when preparation or an experiment configuration is missing."""
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")


def main() -> None:
    """Run MostPop once and write a compact, deterministic result artifact."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    required = [DATA_CONFIG, EVALUATION_CONFIG, MODEL_CONFIG, DATA_MANIFEST]
    validate_config_files(required)

    config = Config(
        model=MostPop,
        dataset="mealrecplus_h",
        config_file_list=[str(path) for path in required[:3]],
        config_dict={
            "data_path": str(PROCESSED_ROOT / "recbole"),
        },
    )
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    init_seed(config["seed"], config["reproducibility"])
    model = MostPop(config, train_data._dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    valid_result = trainer.evaluate(
        valid_data,
        load_best_model=False,
        show_progress=False,
    )
    test_result = trainer.evaluate(
        test_data,
        load_best_model=False,
        show_progress=False,
    )
    best_valid_score = valid_result[config["valid_metric"].lower()]

    positive_interactions_counted = int(model.item_cnt.sum().item())
    if positive_interactions_counted != len(train_data._dataset):
        raise ValueError("MostPop counts do not equal the positive training interactions")
    if config["train_neg_sample_args"]["distribution"] != "none":
        raise ValueError("MostPop must not count sampled negative interactions")

    result = {
        "result_schema_version": 1,
        "model": {
            "report_name": "MostPop",
            "model_class": "healthy_meal_recommender.models.MostPop",
            "implementation": (
                "exact positive training-item frequency via torch.bincount; "
                "stable internal-id tie-break"
            ),
            "recbole_version": version("recbole"),
        },
        "provenance": {
            "data_manifest": "results/data_preparation/mealrecplus_h_manifest.json",
            "data_manifest_sha256": file_sha256(DATA_MANIFEST),
            "config_files": [path.relative_to(PROJECT_ROOT).as_posix() for path in required[:3]],
        },
        "effective_protocol": {
            "seed": config["seed"],
            "device": str(config["device"]),
            "metrics": list(config["metrics"]),
            "topk": list(config["topk"]),
            "primary_metric": config["valid_metric"],
            "evaluation_mode": "full catalog",
            "validation_history_excluded": "train interactions",
            "test_history_excluded": "train and validation interactions",
            "training_negative_sampling": config["train_neg_sample_args"]["distribution"],
            "metric_decimal_place": config["metric_decimal_place"],
        },
        "dataset": {
            "users": dataset.user_num - 1,
            "train_meals": dataset.item_num - 1,
            "train_interactions": len(train_data._dataset),
            "validation_interactions": len(valid_data.dataset),
            "validation_users": len(valid_data.uid_list),
            "test_interactions": len(test_data.dataset),
            "test_users": len(test_data.uid_list),
        },
        "fit_checks": {
            "positive_interactions_counted": positive_interactions_counted,
            "meals_with_positive_count": int((model.item_cnt > 0).sum().item()),
            "maximum_training_popularity": int(model.item_cnt.max().item()),
        },
        "best_validation_score": float(best_valid_score),
        "validation": json_metrics(valid_result),
        "test": json_metrics(test_result),
        "recommendation_samples": recommendation_samples(model, test_data),
        "limitations": [
            "MostPop is non-personalized except for excluding each user's history.",
            "It is a sanity-check baseline, not the proposed health-aware method.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    logger.info("MostPop result written to %s", args.output)
    print(
        json.dumps(
            {
                "validation": result["validation"],
                "test": result["test"],
                "output": args.output.relative_to(PROJECT_ROOT).as_posix(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
