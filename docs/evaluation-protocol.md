# Evaluation protocol

## Preference evaluation

MostPop, NeuMF, and LightGCN differ only in their scoring model. They share the same data, candidate catalog, history exclusions, and RecBole full-sort evaluator. The report emphasizes Recall@10 and NDCG@10; committed result files also retain @5 and @20 as compact robustness checks.

## Full-catalog Top-N protocol

- The candidate catalog contains the 3,810 meals observed in training; negatives are not sampled for evaluation.
- Validation includes 1,130 users. Each user's training interactions are excluded before ranking.
- Test includes 1,114 users. Each user's training and validation interactions are excluded before ranking.
- Relevant items are meals in the corresponding validation or test split.
- Preprocessing separately records the 13 interactions involving meals unseen in training.

This setting requires each model to retrieve relevant meals from the complete known catalog and is closer to a deployed recommender than evaluation against a small sampled negative set.

## Metrics and selection

- **Recall@K** measures the proportion of a user's relevant meals retrieved in the first K positions.
- **NDCG@K** additionally rewards relevant meals ranked near the top.
- **Validation NDCG@10** is the primary preference-model selection metric.

The test split is evaluated only after a trial has been selected from validation results. Health-aware outcomes are evaluated separately so relevance and wellness effects remain interpretable.

## Health-reranking evaluation

All rerankers use NeuMF Standard Top-100 candidates and produce Top-10 lists. In addition to Recall@10 and NDCG@10, the experiment reports:

- NDCG@10 retention relative to the preference-only list;
- mean health utility@10;
- change relative to each user's train-only baseline;
- goal-meeting item rate@10;
- overlap with the original preference Top-10;
- candidate feasibility and hard-filter fallback counts.

Parameters are selected on validation data and frozen before a single test comparison. Definitions and results are in `docs/health-reranking.md`.

## MostPop baseline

MostPop scores each meal by its positive interaction count in the training split. It reads neither validation nor test interactions for scoring. `torch.bincount` computes exact counts; an extremely small deterministic internal-ID offset resolves complete ties without changing any real count difference.

Previously seen meals are excluded for each user, so displayed recommendation lists can differ across users even though the underlying scores are global. MostPop is therefore a minimum-complexity sanity check, not a learned personalized model.

## Reproduction

```powershell
.\.venv\Scripts\python.exe .\scripts\run_mostpop.py
```

`results/baselines/mostpop.json` records configuration provenance, data-manifest hashes, all requested metrics, count assertions, and three anonymous recommendation examples.
