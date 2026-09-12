"""Select between two NeuMF capacities, then evaluate the winner once on test."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_model, get_trainer, init_logger, init_seed
from recbole.utils.case_study import full_sort_scores


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG = PROJECT_ROOT / "configs" / "data" / "mealrecplus_h.yaml"
EVALUATION_CONFIG = PROJECT_ROOT / "configs" / "evaluation" / "topn.yaml"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "mealrecplus_h"
DATA_MANIFEST = (
    PROJECT_ROOT / "results" / "data_preparation" / "mealrecplus_h_manifest.json"
)
MOSTPOP_RESULT = PROJECT_ROOT / "results" / "baselines" / "mostpop.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "baselines" / "neumf.json"
DEFAULT_COMPARISON = (
    PROJECT_ROOT / "results" / "comparison" / "preference_models.csv"
)
TRIAL_CONFIGS = {
    "compact": PROJECT_ROOT / "configs" / "models" / "neumf-compact.yaml",
    "standard": PROJECT_ROOT / "configs" / "models" / "neumf-standard.yaml",
}
SAMPLE_TOPK = 10


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_metrics(metrics: dict[str, object]) -> dict[str, float]:
    """Convert framework scalars to stable JSON floats."""
    return {str(key): float(value) for key, value in sorted(metrics.items())}


def build_config(model_config: Path) -> Config:
    """Construct the effective config from shared data/evaluation and one trial."""
    return Config(
        model="NeuMF",
        dataset="mealrecplus_h",
        config_file_list=[
            str(DATA_CONFIG),
            str(EVALUATION_CONFIG),
            str(model_config),
        ],
        config_dict={"data_path": str(PROCESSED_ROOT / "recbole")},
    )


def build_pipeline(config: Config):
    """Build deterministic data loaders, model and trainer for one config."""
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    init_seed(config["seed"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    return dataset, train_data, valid_data, test_data, model, trainer


def checkpoint_best_epoch(path: Path) -> int:
    """Read the one-based best epoch stored by RecBole."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return int(checkpoint["epoch"]) + 1


def train_trial(trial_id: str, model_config: Path) -> dict[str, object]:
    """Train one candidate and return validation-only selection evidence."""
    config = build_config(model_config)
    dataset, train_data, valid_data, _, model, trainer = build_pipeline(config)
    checkpoint_path = PROJECT_ROOT / "checkpoints" / "neumf" / f"{trial_id}.pth"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.saved_model_file = str(checkpoint_path)
    best_score, best_result = trainer.fit(
        train_data,
        valid_data,
        verbose=False,
        saved=True,
        show_progress=False,
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Best checkpoint was not saved: {checkpoint_path}")
    outcome = {
        "trial_id": trial_id,
        "config_file": model_config.relative_to(PROJECT_ROOT).as_posix(),
        "hyperparameters": {
            "mf_embedding_size": config["mf_embedding_size"],
            "mlp_embedding_size": config["mlp_embedding_size"],
            "mlp_hidden_size": list(config["mlp_hidden_size"]),
            "dropout_prob": config["dropout_prob"],
            "learning_rate": config["learning_rate"],
            "negative_samples_per_positive": config["train_neg_sample_args"][
                "sample_num"
            ],
            "maximum_epochs": config["epochs"],
            "early_stopping_checks": config["stopping_step"],
        },
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "epochs_completed": len(trainer.train_loss_dict),
        "best_epoch": checkpoint_best_epoch(checkpoint_path),
        "best_validation_score": float(best_score),
        "validation": json_metrics(best_result),
        "checkpoint": checkpoint_path.relative_to(PROJECT_ROOT).as_posix(),
        "dataset_shape_check": {
            "users": dataset.user_num - 1,
            "train_meals": dataset.item_num - 1,
            "train_interactions": len(train_data._dataset),
        },
    }
    del dataset, train_data, valid_data, model, trainer
    torch.cuda.empty_cache()
    return outcome


def select_trial(trials: list[dict[str, object]]) -> dict[str, object]:
    """Select only by validation NDCG@10, with trial id as a stable tie-break."""
    return sorted(
        trials,
        key=lambda trial: (-float(trial["validation"]["ndcg@10"]), str(trial["trial_id"])),
    )[0]


def load_meal_catalog() -> dict[int, dict[str, str]]:
    """Load compact meal metadata for readable examples."""
    with (PROCESSED_ROOT / "meal_catalog.csv").open(encoding="utf-8", newline="") as handle:
        return {int(row["meal_id"]): row for row in csv.DictReader(handle)}


def choose_sample_users(test_data) -> list[tuple[str, int]]:
    """Select the same low, median and high history bands used by MostPop."""
    dataset = test_data.dataset
    records: list[tuple[int, int, int]] = []
    for internal_user_id in test_data.uid_list.tolist():
        raw_user_id = int(dataset.id2token(dataset.uid_field, internal_user_id))
        history_count = len(test_data.uid2history_item[internal_user_id])
        records.append((history_count, raw_user_id, internal_user_id))
    records.sort()
    positions = [0, len(records) // 2, len(records) - 1]
    labels = ["low_history", "median_history", "high_history"]
    return [(label, records[position][2]) for label, position in zip(labels, positions)]


def ranked_candidates(scores: torch.Tensor, dataset) -> list[tuple[float, int]]:
    """Apply a stable raw-meal-id tie-break to finite model scores."""
    internal_ids = np.arange(1, dataset.item_num)
    raw_ids = dataset.id2token(dataset.iid_field, internal_ids)
    candidates = [
        (float(scores[internal_id]), int(raw_id))
        for internal_id, raw_id in zip(internal_ids, raw_ids)
        if torch.isfinite(scores[internal_id])
    ]
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates


def recommendation_samples(model, test_data) -> list[dict[str, object]]:
    """Save three anonymous, presentation-ready predictions from the selected model."""
    dataset = test_data.dataset
    meal_catalog = load_meal_catalog()
    samples: list[dict[str, object]] = []
    for history_band, internal_user_id in choose_sample_users(test_data):
        scores = full_sort_scores(
            [internal_user_id], model, test_data, device=model.device
        ).squeeze(0)
        ranked = ranked_candidates(scores.cpu(), dataset)[:SAMPLE_TOPK]
        ground_truth = sorted(
            int(value)
            for value in dataset.id2token(
                dataset.iid_field, test_data.uid2positive_item[internal_user_id].numpy()
            )
        )
        recommendations = []
        for score, meal_id in ranked:
            meal = meal_catalog[meal_id]
            recommendations.append(
                {
                    "meal_id": meal_id,
                    "preference_score": score,
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
                "user_id": int(dataset.id2token(dataset.uid_field, internal_user_id)),
                "history_items_excluded": len(
                    test_data.uid2history_item[internal_user_id]
                ),
                "test_ground_truth_meal_ids": ground_truth,
                "hits_at_10": len(recommended_ids & set(ground_truth)),
                "recommendations": recommendations,
            }
        )
    return samples


def evaluate_selected_trial(selected: dict[str, object]):
    """Load the validation-selected checkpoint and perform the sole test evaluation."""
    model_config = PROJECT_ROOT / str(selected["config_file"])
    config = build_config(model_config)
    dataset, train_data, valid_data, test_data, model, trainer = build_pipeline(config)
    checkpoint_path = PROJECT_ROOT / str(selected["checkpoint"])
    checkpoint = torch.load(
        checkpoint_path,
        map_location=config["device"],
        weights_only=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))
    test_result = trainer.evaluate(
        test_data,
        load_best_model=False,
        show_progress=False,
    )
    samples = recommendation_samples(model, test_data)
    dataset_summary = {
        "users": dataset.user_num - 1,
        "train_meals": dataset.item_num - 1,
        "train_interactions": len(train_data._dataset),
        "validation_interactions": len(valid_data.dataset),
        "validation_users": len(valid_data.uid_list),
        "test_interactions": len(test_data.dataset),
        "test_users": len(test_data.uid_list),
    }
    return config, json_metrics(test_result), samples, dataset_summary


def write_comparison(path: Path, mostpop: dict[str, object], neumf: dict[str, object]) -> None:
    """Write the report-ready preference baseline table."""
    fields = [
        "model",
        "selected_config",
        "split",
        "recall@5",
        "recall@10",
        "recall@20",
        "ndcg@5",
        "ndcg@10",
        "ndcg@20",
    ]
    rows = []
    for model_name, selected_config, result in (
        ("MostPop", "none", mostpop),
        ("NeuMF", str(neumf["selection"]["selected_trial"]), neumf),
    ):
        for split in ("validation", "test"):
            rows.append(
                {
                    "model": model_name,
                    "selected_config": selected_config,
                    "split": split,
                    **result[split],
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Train two bounded trials, select on validation, then evaluate test once."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--comparison-output", type=Path, default=DEFAULT_COMPARISON)
    args = parser.parse_args()
    required = [DATA_CONFIG, EVALUATION_CONFIG, DATA_MANIFEST, MOSTPOP_RESULT]
    required.extend(TRIAL_CONFIGS.values())
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")

    trials = [train_trial(trial_id, path) for trial_id, path in TRIAL_CONFIGS.items()]
    selected = select_trial(trials)
    config, test_result, samples, dataset_summary = evaluate_selected_trial(selected)
    mostpop = json.loads(MOSTPOP_RESULT.read_text(encoding="utf-8"))
    validation = selected["validation"]
    result = {
        "result_schema_version": 1,
        "model": {
            "report_name": "NeuMF",
            "recbole_name": "NeuMF",
            "implementation": "RecBole pointwise neural collaborative filtering",
            "recbole_version": version("recbole"),
        },
        "provenance": {
            "data_manifest": "results/data_preparation/mealrecplus_h_manifest.json",
            "data_manifest_sha256": file_sha256(DATA_MANIFEST),
            "shared_config_files": [
                DATA_CONFIG.relative_to(PROJECT_ROOT).as_posix(),
                EVALUATION_CONFIG.relative_to(PROJECT_ROOT).as_posix(),
            ],
        },
        "selection": {
            "rule": "highest validation NDCG@10; trial id breaks exact ties",
            "trials": trials,
            "selected_trial": selected["trial_id"],
            "test_evaluations_performed": 1,
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
            "training_negative_sampling": "uniform",
            "negative_samples_per_positive": config["train_neg_sample_args"]["sample_num"],
        },
        "dataset": dataset_summary,
        "validation": validation,
        "test": test_result,
        "comparison_to_mostpop": {
            "validation_ndcg@10_delta": validation["ndcg@10"]
            - mostpop["validation"]["ndcg@10"],
            "test_ndcg@10_delta": test_result["ndcg@10"]
            - mostpop["test"]["ndcg@10"],
            "test_recall@10_delta": test_result["recall@10"]
            - mostpop["test"]["recall@10"],
        },
        "selected_checkpoint": {
            "path": selected["checkpoint"],
            "tracked_by_git": False,
            "recreated_by": "scripts/run_neumf.py",
        },
        "recommendation_samples": samples,
        "limitations": [
            "Only two model capacities are compared to control project scope.",
            "Selection uses validation NDCG@10; the test split is not used for tuning.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_comparison(args.comparison_output, mostpop, result)
    print(
        json.dumps(
            {
                "trials": [
                    {
                        "trial_id": trial["trial_id"],
                        "epochs_completed": trial["epochs_completed"],
                        "best_epoch": trial["best_epoch"],
                        "validation_ndcg@10": trial["validation"]["ndcg@10"],
                    }
                    for trial in trials
                ],
                "selected_trial": selected["trial_id"],
                "validation": validation,
                "test": test_result,
                "output": args.output.relative_to(PROJECT_ROOT).as_posix(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
