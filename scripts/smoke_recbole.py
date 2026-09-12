"""Run one-epoch NeuMF and LightGCN tests on a synthetic interaction dataset."""

from __future__ import annotations

import json
from pathlib import Path

from recbole.quick_start import run_recbole


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_model(model: str) -> dict[str, object]:
    """Train and evaluate one RecBole model with a deterministic tiny dataset."""
    config = {
        "data_path": str(PROJECT_ROOT / "tests" / "fixtures"),
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "TIME_FIELD": "timestamp",
        "load_col": {"inter": ["user_id", "item_id", "timestamp"]},
        "field_separator": "\t",
        "epochs": 1,
        "train_batch_size": 16,
        "eval_batch_size": 64,
        "embedding_size": 8,
        "mf_embedding_size": 8,
        "mlp_embedding_size": 8,
        "mlp_hidden_size": [16, 8],
        "dropout_prob": 0.0,
        "n_layers": 1,
        "learning_rate": 0.001,
        "eval_step": 1,
        "stopping_step": 1,
        "metrics": ["Recall", "NDCG"],
        "topk": [3],
        "valid_metric": "NDCG@3",
        "eval_args": {
            "split": {"RS": [0.6, 0.2, 0.2]},
            "order": "RO",
            "group_by": "user",
            "mode": "full",
        },
        "seed": 2022,
        "reproducibility": True,
        "use_gpu": True,
        "gpu_id": 0,
        "show_progress": False,
        "save_dataset": False,
        "save_dataloaders": False,
        "checkpoint_dir": str(PROJECT_ROOT / "checkpoints" / "smoke"),
    }
    result = run_recbole(
        model=model,
        dataset="recbole_smoke",
        config_dict=config,
        saved=False,
    )
    return {
        "model": model,
        "best_valid_score": float(result["best_valid_score"]),
        "valid": dict(result["best_valid_result"]),
        "test": dict(result["test_result"]),
    }


def main() -> None:
    """Run both non-overlapping project models and print machine-readable results."""
    results = [run_model(model) for model in ("NeuMF", "LightGCN")]
    print("SMOKE_RESULTS=" + json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
