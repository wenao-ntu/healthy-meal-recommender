# Preference model training

## Bounded experimental scope

The preference stage compares one non-personalized baseline and two established personalized architectures. Hyperparameter exploration is deliberately bounded so the project emphasizes the end-to-end system rather than an exhaustive search.

NeuMF compares two predefined capacities:

| Trial | MF and MLP embeddings | MLP hidden layers | Other settings |
|---|---:|---|---|
| Compact | 32 / 32 | 64 -> 32 | Identical to Standard |
| Standard | 64 / 64 | 128 -> 64 | Identical to Compact |

Both use Adam with learning rate 0.001, four uniformly sampled training negatives per positive interaction, at most 20 epochs, and early stopping after four validations without improvement. Pretraining and AMP are disabled, and seed 2026 is fixed.

## Selection protocol

1. Fit both trials on training interactions and evaluate full-catalog validation NDCG@10.
2. Select the higher validation NDCG@10; resolve an exact tie deterministically by trial name.
3. Evaluate only the selected trial on the test split.
4. Do not change capacity, learning rate, or training duration in response to test results.

This protocol provides a traceable selection process without unnecessary parameter proliferation. LightGCN follows the same train-validation-test boundary.

## Checkpoints and repository policy

Best checkpoints are stored locally under `checkpoints/` for candidate generation, but binary weights are excluded from Git. The repository retains configurations, executable scripts, compact result JSON, and comparison CSV files so the models can be retrained.

## NeuMF results

```powershell
.\.venv\Scripts\python.exe .\scripts\run_neumf.py
```

| Trial | Parameters | Completed epochs | Best epoch | Validation NDCG@10 |
|---|---:|---:|---:|---:|
| Compact | 351,073 | 20 | 18 | 0.051674 |
| Standard | 714,433 | 20 | 18 | **0.063192** |

NeuMF Standard is selected. It achieves test Recall@10 = 0.087280 and NDCG@10 = 0.064976, an absolute NDCG@10 gain of 0.050525 over the final MostPop baseline.

PyTorch 2.6 and later default to weights-only loading, while RecBole 1.2.1 checkpoints also contain configuration objects. The script permits full deserialization only for a local checkpoint created during the same trusted run, then copies its `state_dict` into the model. Downloaded or otherwise untrusted checkpoints must not use this path.

## LightGCN results

LightGCN fixes a 64-dimensional embedding, learning rate 0.001, regularization `1e-5`, and one training negative per positive. It compares one and two graph-propagation layers:

| Trial | Parameters | Completed epochs | Best epoch | Validation NDCG@10 |
|---|---:|---:|---:|---:|
| 1-layer | 344,768 | 14 | 8 | **0.045095** |
| 2-layer | 344,768 | 11 | 5 | 0.034124 |

The 1-layer trial is selected and achieves test Recall@10 = 0.059513 and NDCG@10 = 0.041006. It outperforms MostPop but does not match NeuMF Standard on validation NDCG@10. NeuMF Standard is therefore selected as the GAP Rank candidate generator without using test metrics for that decision.

## Outputs

- `results/baselines/neumf.json`: validation trials, selected configuration, one test result, and anonymous examples.
- `results/baselines/lightgcn.json`: bounded LightGCN trials and selected result.
- `results/comparison/preference_models.csv`: aligned metrics for the selected model variants.
- `results/comparison/preference_model_selection.json`: the validation-only candidate-generator decision.
