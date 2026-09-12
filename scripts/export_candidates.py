"""Export deterministic Top-100 candidates from the selected NeuMF checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from recbole.utils.case_study import full_sort_scores

from run_neumf import (
    DATA_MANIFEST,
    PROJECT_ROOT,
    PROCESSED_ROOT,
    TRIAL_CONFIGS,
    build_config,
    build_pipeline,
    file_sha256,
)


SELECTION_PATH = (
    PROJECT_ROOT / "results" / "comparison" / "preference_model_selection.json"
)
DEFAULT_OUTPUT_DIR = PROCESSED_ROOT / "candidates"
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "results" / "candidates" / "neumf_standard_top100.json"
)
DEFAULT_CANDIDATE_K = 100
DEFAULT_USER_BATCH_SIZE = 64


def load_selected_neumf() -> tuple[dict[str, object], Path, Path]:
    """Resolve the validation-selected model, config, and local checkpoint."""
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    if selection["selected_model"] != "NeuMF":
        raise ValueError("Candidate exporter currently expects selected_model=NeuMF")
    selected_config = str(selection["selected_config"])
    if selected_config not in TRIAL_CONFIGS:
        raise ValueError(f"Unknown selected NeuMF config: {selected_config}")
    checkpoint_path = PROJECT_ROOT / str(selection["selected_checkpoint"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Missing selected checkpoint: {checkpoint_path}. Run scripts/run_neumf.py first."
        )
    return selection, TRIAL_CONFIGS[selected_config], checkpoint_path


def stable_top_candidates(
    scores: torch.Tensor,
    internal_item_ids: np.ndarray,
    raw_item_ids: np.ndarray,
    candidate_k: int,
) -> list[tuple[float, int, int]]:
    """Rank finite scores with numeric raw meal id as an exact tie-break."""
    finite = torch.isfinite(scores[internal_item_ids]).numpy()
    candidates = [
        (float(scores[internal_id]), int(raw_id), int(internal_id))
        for internal_id, raw_id in zip(
            internal_item_ids[finite], raw_item_ids[finite], strict=True
        )
    ]
    candidates.sort(key=lambda row: (-row[0], row[1]))
    if len(candidates) < candidate_k:
        raise ValueError(
            f"Only {len(candidates)} eligible meals remain; cannot export Top-{candidate_k}"
        )
    return candidates[:candidate_k]


def export_split(
    split_name: str,
    model,
    data_loader,
    output_path: Path,
    candidate_k: int,
    user_batch_size: int,
) -> dict[str, object]:
    """Export one evaluation split and return integrity statistics."""
    dataset = data_loader.dataset
    internal_item_ids = np.arange(1, dataset.item_num, dtype=np.int64)
    raw_item_ids = np.asarray(
        dataset.id2token(dataset.iid_field, internal_item_ids), dtype=np.int64
    )
    internal_user_ids = sorted(int(value) for value in data_loader.uid_list.tolist())
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row_count = 0
    history_overlap_count = 0
    recalled_fractions: list[float] = []
    minimum_score = float("inf")
    maximum_score = float("-inf")
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["user_id", "rank", "meal_id", "preference_score"],
            lineterminator="\n",
        )
        writer.writeheader()
        for start in range(0, len(internal_user_ids), user_batch_size):
            batch_user_ids = internal_user_ids[start : start + user_batch_size]
            batch_scores = full_sort_scores(
                batch_user_ids, model, data_loader, device=model.device
            ).cpu()
            for row_index, internal_user_id in enumerate(batch_user_ids):
                ranked = stable_top_candidates(
                    batch_scores[row_index],
                    internal_item_ids,
                    raw_item_ids,
                    candidate_k,
                )
                raw_user_id = int(
                    dataset.id2token(dataset.uid_field, internal_user_id)
                )
                history = set(
                    int(value)
                    for value in data_loader.uid2history_item[internal_user_id].tolist()
                )
                ranked_internal_ids = {row[2] for row in ranked}
                history_overlap_count += len(ranked_internal_ids & history)
                positive_items = set(
                    int(value)
                    for value in data_loader.uid2positive_item[internal_user_id].tolist()
                )
                recalled_fractions.append(
                    len(ranked_internal_ids & positive_items) / len(positive_items)
                )
                for rank, (score, raw_item_id, _) in enumerate(ranked, start=1):
                    writer.writerow(
                        {
                            "user_id": raw_user_id,
                            "rank": rank,
                            "meal_id": raw_item_id,
                            "preference_score": format(score, ".12g"),
                        }
                    )
                    row_count += 1
                    minimum_score = min(minimum_score, score)
                    maximum_score = max(maximum_score, score)

    if history_overlap_count:
        raise AssertionError(
            f"{split_name} candidates contain {history_overlap_count} history items"
        )
    expected_rows = len(internal_user_ids) * candidate_k
    if row_count != expected_rows:
        raise AssertionError(f"Expected {expected_rows} rows, wrote {row_count}")
    return {
        "users": len(internal_user_ids),
        "rows": row_count,
        "candidate_recall@100": float(np.mean(recalled_fractions)),
        "history_overlap_rows": history_overlap_count,
        "score_min": minimum_score,
        "score_max": maximum_score,
        "output": output_path.relative_to(PROJECT_ROOT).as_posix(),
        "bytes": output_path.stat().st_size,
        "sha256": file_sha256(output_path),
    }


def main() -> None:
    """Load the selected model once and export validation/test candidates."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K)
    parser.add_argument("--user-batch-size", type=int, default=DEFAULT_USER_BATCH_SIZE)
    args = parser.parse_args()
    if args.candidate_k < 10:
        raise ValueError("candidate-k must be at least the final recommendation size 10")
    if args.user_batch_size < 1:
        raise ValueError("user-batch-size must be positive")
    required = [SELECTION_PATH, DATA_MANIFEST]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")

    selection, model_config, checkpoint_path = load_selected_neumf()
    config = build_config(model_config)
    dataset, train_data, valid_data, test_data, model, _ = build_pipeline(config)
    # RecBole 1.2.1 stores config metadata alongside weights. This checkpoint is
    # generated locally by scripts/run_neumf.py; never use full loading for an
    # external or untrusted checkpoint.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=config["device"],
        weights_only=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))
    model.eval()

    split_results = {}
    with torch.no_grad():
        for split_name, data_loader in (("validation", valid_data), ("test", test_data)):
            output_path = args.output_dir / f"neumf_standard_{split_name}_top100.csv"
            split_results[split_name] = export_split(
                split_name,
                model,
                data_loader,
                output_path,
                args.candidate_k,
                args.user_batch_size,
            )

    manifest = {
        "candidate_manifest_schema_version": 1,
        "candidate_generator": {
            "model": str(selection["selected_model"]),
            "config": str(selection["selected_config"]),
            "checkpoint": checkpoint_path.relative_to(PROJECT_ROOT).as_posix(),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "selected_by": "highest validation NDCG@10 among personalized models",
            "test_metrics_used_for_selection": False,
        },
        "protocol": {
            "candidate_k": args.candidate_k,
            "final_recommendation_k": 10,
            "catalog": "3,810 meals seen in training",
            "validation_history_excluded": "train interactions",
            "test_history_excluded": "train and validation interactions",
            "tie_break": "ascending numeric raw meal_id",
            "candidate_files_contain_ground_truth_labels": False,
        },
        "provenance": {
            "data_manifest": DATA_MANIFEST.relative_to(PROJECT_ROOT).as_posix(),
            "data_manifest_sha256": file_sha256(DATA_MANIFEST),
            "model_selection": SELECTION_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "model_selection_sha256": file_sha256(SELECTION_PATH),
        },
        "dataset_check": {
            "users": dataset.user_num - 1,
            "train_meals": dataset.item_num - 1,
            "train_interactions": len(train_data._dataset),
        },
        "splits": split_results,
        "reproducibility": {
            "device": str(config["device"]),
            "seed": int(config["seed"]),
            "user_batch_size": args.user_batch_size,
            "generation_timestamp_included": False,
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
