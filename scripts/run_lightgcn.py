"""Select a LightGCN depth and finalize the preference-model comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_model, get_trainer, init_logger, init_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG = PROJECT_ROOT / "configs" / "data" / "mealrecplus_h.yaml"
EVALUATION_CONFIG = PROJECT_ROOT / "configs" / "evaluation" / "topn.yaml"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "mealrecplus_h"
DATA_MANIFEST = (
    PROJECT_ROOT / "results" / "data_preparation" / "mealrecplus_h_manifest.json"
)
MOSTPOP_RESULT = PROJECT_ROOT / "results" / "baselines" / "mostpop.json"
NEUMF_RESULT = PROJECT_ROOT / "results" / "baselines" / "neumf.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "baselines" / "lightgcn.json"
DEFAULT_COMPARISON = (
    PROJECT_ROOT / "results" / "comparison" / "preference_models.csv"
)
DEFAULT_SELECTION = (
    PROJECT_ROOT / "results" / "comparison" / "preference_model_selection.json"
)
TRIAL_CONFIGS = {
    "1-layer": PROJECT_ROOT / "configs" / "models" / "lightgcn-1layer.yaml",
    "2-layer": PROJECT_ROOT / "configs" / "models" / "lightgcn-2layer.yaml",
}


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_metrics(metrics: dict[str, object]) -> dict[str, float]:
    """Convert framework scalars into stable JSON floats."""
    return {str(key): float(value) for key, value in sorted(metrics.items())}

def build_config(model_config: Path) -> Config:
    """Combine the shared protocol with one LightGCN depth."""
    return Config(
        model="LightGCN",
        dataset="mealrecplus_h",
        config_file_list=[
            str(DATA_CONFIG),
            str(EVALUATION_CONFIG),
            str(model_config),
        ],
        config_dict={"data_path": str(PROCESSED_ROOT / "recbole")},
    )


def build_pipeline(config: Config):
    """Build deterministic loaders, model and trainer."""
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    init_seed(config["seed"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    return dataset, train_data, valid_data, test_data, model, trainer


def load_local_checkpoint(path: Path, model, device) -> dict[str, object]:
    """Load only a checkpoint generated locally by this project script."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))
    return checkpoint


def train_trial(trial_id: str, model_config: Path) -> dict[str, object]:
    """Train one depth and retain validation-only selection evidence."""
    config = build_config(model_config)
    dataset, train_data, valid_data, _, model, trainer = build_pipeline(config)
    checkpoint_path = PROJECT_ROOT / "checkpoints" / "lightgcn" / f"{trial_id}.pth"
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
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    outcome = {
        "trial_id": trial_id,
        "config_file": model_config.relative_to(PROJECT_ROOT).as_posix(),
        "hyperparameters": {
            "embedding_size": config["embedding_size"],
            "n_layers": config["n_layers"],
            "reg_weight": config["reg_weight"],
            "learning_rate": config["learning_rate"],
            "negative_samples_per_positive": config["train_neg_sample_args"][
                "sample_num"
            ],
            "maximum_epochs": config["epochs"],
            "early_stopping_checks": config["stopping_step"],
        },
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "epochs_completed": len(trainer.train_loss_dict),
        "best_epoch": int(checkpoint["epoch"]) + 1,
        "best_validation_score": float(best_score),
        "validation": json_metrics(best_result),
        "checkpoint": checkpoint_path.relative_to(PROJECT_ROOT).as_posix(),
        "dataset_shape_check": {
            "users": dataset.user_num - 1,
            "train_meals": dataset.item_num - 1,
            "train_interactions": len(train_data._dataset),
        },
    }
    del checkpoint, dataset, train_data, valid_data, model, trainer
    torch.cuda.empty_cache()
    return outcome


def select_trial(trials: list[dict[str, object]]) -> dict[str, object]:
    """Choose the highest validation NDCG@10 with a stable tie-break."""
    return sorted(
        trials,
        key=lambda trial: (-float(trial["validation"]["ndcg@10"]), str(trial["trial_id"])),
    )[0]


def evaluate_selected_trial(selected: dict[str, object]):
    """Perform the only test evaluation after depth selection."""
    config = build_config(PROJECT_ROOT / str(selected["config_file"]))
    dataset, train_data, valid_data, test_data, model, trainer = build_pipeline(config)
    load_local_checkpoint(PROJECT_ROOT / str(selected["checkpoint"]), model, config["device"])
    test_result = trainer.evaluate(
        test_data,
        load_best_model=False,
        show_progress=False,
    )
    dataset_summary = {
        "users": dataset.user_num - 1,
        "train_meals": dataset.item_num - 1,
        "train_interactions": len(train_data._dataset),
        "validation_interactions": len(valid_data.dataset),
        "validation_users": len(valid_data.uid_list),
        "test_interactions": len(test_data.dataset),
        "test_users": len(test_data.uid_list),
    }
    return config, json_metrics(test_result), dataset_summary


def write_comparison(
    path: Path,
    mostpop: dict[str, object],
    neumf: dict[str, object],
    lightgcn: dict[str, object],
) -> None:
    """Write all selected preference baselines in report-ready form."""
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
    selected_configs = {
        "MostPop": "none",
        "NeuMF": neumf["selection"]["selected_trial"],
        "LightGCN": lightgcn["selection"]["selected_trial"],
    }
    for model_name, result in (
        ("MostPop", mostpop),
        ("NeuMF", neumf),
        ("LightGCN", lightgcn),
    ):
        for split in ("validation", "test"):
            rows.append(
                {
                    "model": model_name,
                    "selected_config": selected_configs[model_name],
                    "split": split,
                    **result[split],
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_preference_selection(
    path: Path, neumf: dict[str, object], lightgcn: dict[str, object]
) -> dict[str, object]:
    """Choose the GAP Rank candidate generator using validation accuracy only."""
    candidates = [
        {
            "model": "NeuMF",
            "selected_config": neumf["selection"]["selected_trial"],
            "validation_ndcg@10": neumf["validation"]["ndcg@10"],
            "result_file": "results/baselines/neumf.json",
            "checkpoint": neumf["selected_checkpoint"]["path"],
        },
        {
            "model": "LightGCN",
            "selected_config": lightgcn["selection"]["selected_trial"],
            "validation_ndcg@10": lightgcn["validation"]["ndcg@10"],
            "result_file": "results/baselines/lightgcn.json",
            "checkpoint": lightgcn["selected_checkpoint"]["path"],
        },
    ]
    selected = sorted(
        candidates,
        key=lambda row: (-float(row["validation_ndcg@10"]), str(row["model"])),
    )[0]
    selection = {
        "selection_schema_version": 1,
        "purpose": "preference candidate generator for GAP Rank",
        "rule": "highest validation NDCG@10 among personalized models",
        "test_metrics_used_for_selection": False,
        "candidates": candidates,
        "selected_model": selected["model"],
        "selected_config": selected["selected_config"],
        "selected_checkpoint": selected["checkpoint"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return selection


def main() -> None:
    """Train two depths, test the winner once and finalize model selection."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--comparison-output", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--selection-output", type=Path, default=DEFAULT_SELECTION)
    args = parser.parse_args()
    required = [DATA_CONFIG, EVALUATION_CONFIG, DATA_MANIFEST, MOSTPOP_RESULT, NEUMF_RESULT]
    required.extend(TRIAL_CONFIGS.values())
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")

    trials = [train_trial(trial_id, path) for trial_id, path in TRIAL_CONFIGS.items()]
    selected = select_trial(trials)
    config, test_result, dataset_summary = evaluate_selected_trial(selected)
    mostpop = json.loads(MOSTPOP_RESULT.read_text(encoding="utf-8"))
    neumf = json.loads(NEUMF_RESULT.read_text(encoding="utf-8"))
    validation = selected["validation"]
    result = {
        "result_schema_version": 1,
        "model": {
            "report_name": "LightGCN",
            "recbole_name": "LightGCN",
            "implementation": "RecBole graph neighborhood propagation",
            "optimization_objective": "BPR pairwise ranking loss",
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
            "training_negative_sampling": "uniform pairwise",
            "negative_samples_per_positive": config["train_neg_sample_args"]["sample_num"],
        },
        "dataset": dataset_summary,
        "validation": validation,
        "test": test_result,
        "comparison": {
            "validation_ndcg@10_delta_vs_neumf": validation["ndcg@10"]
            - neumf["validation"]["ndcg@10"],
            "test_ndcg@10_delta_vs_neumf": test_result["ndcg@10"]
            - neumf["test"]["ndcg@10"],
            "test_ndcg@10_delta_vs_mostpop": test_result["ndcg@10"]
            - mostpop["test"]["ndcg@10"],
        },
        "selected_checkpoint": {
            "path": selected["checkpoint"],
            "tracked_by_git": False,
            "recreated_by": "scripts/run_lightgcn.py",
        },
        "individual_assignment_boundary": {
            "shared_component": "pairwise BPR loss only",
            "group_project_model": "LightGCN graph propagation over user-meal interactions",
            "individual_model": "MF-BPR direct matrix-factorization embeddings",
            "same_algorithm": False,
        },
        "limitations": [
            "Only one and two propagation layers are compared to control project scope.",
            "Selection uses validation NDCG@10; test is not used for tuning.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_comparison(args.comparison_output, mostpop, neumf, result)
    preference_selection = write_preference_selection(args.selection_output, neumf, result)
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
                "gap_rank_candidate_model": preference_selection["selected_model"],
                "output": args.output.relative_to(PROJECT_ROOT).as_posix(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
