# Health-aware reranking with GAP Rank

## Objective

Preference models estimate what a user is likely to prefer, but they do not ensure that the resulting list improves on the user's historical meal profile. This stage fixes NeuMF Standard as the candidate generator and selects a final Top-10 from each anonymous user's Top-100 unseen meals.

The objective is to improve alignment with a train-defined wellness target while retaining most of the personalized ranking quality. Health scores are ranking proxies and must not be interpreted as diagnosis, treatment advice, or guaranteed health outcomes.

## Health utility and target

MealRec+H provides FSA and WHO scores. Lower FSA and higher WHO indicate healthier meals. The project applies transparent equal-weight normalization:

```text
FSA component  = clip((12 - FSA) / 8, 0, 1)
WHO component  = clip(WHO / 7, 0, 1)
health utility = 0.5 * FSA component + 0.5 * WHO component
```

Health utility ranges from 0 to 1, with higher values representing better alignment under these two source scores. A user's baseline is derived from the mean FSA and WHO values of training interactions only.

The default offline target is the 75th percentile of health utility among meals observed in training: `0.360119`. This target means entering the relatively healthiest quarter of the training meal catalog. It is reproducible and interpretable, but it is not a clinical threshold.

## Compared methods

All methods use the same NeuMF Top-100 candidates, target, and final list length:

| Method | Definition |
|---|---|
| Preference only | Retain the original NeuMF order |
| Fixed weight | Apply one preference-health mixture weight to every user |
| Hard filter | Rank target-meeting meals first and backfill by preference if necessary |
| GAP Rank | Adapt a target-meeting quota to the user's gap and candidate feasibility, then make minimum replacements |

Fixed-weight preference scores are min-max normalized within each user's Top-100 before combination with health utility. Hard filtering represents an intentionally health-dominant extreme rather than the final design.

## GAP Rank procedure

For user `u`:

1. Compute `user gap = max(0, target - train-only baseline)`.
2. If even the healthiest candidate cannot reach the original target, cap the effective target at the candidate maximum for feasibility calculations. Evaluation continues to use the original target.
3. Compute `intervention intensity = max quota share * min(user gap / 0.10, 1) * feasibility ratio`.
4. Set the target-meeting Top-10 quota to `ceil(10 * intervention intensity)`, limited by the number of feasible candidates.
5. Leave the list unchanged if its original Top-10 already satisfies the quota. Otherwise, promote the highest-preference target-meeting candidates and replace only the lowest-ranked non-meeting items. Preserve relative preference order among items that remain.

The method therefore varies intervention across users and avoids replacing items once the necessary quota is satisfied.

## Validation-only parameter selection

- Fixed-weight candidates: `0.1`, `0.2`, `0.3`, and `0.4`.
- GAP Rank maximum-quota-share candidates: `0.2`, `0.4`, `0.6`, and `0.8`.
- Selection rule: among configurations retaining at least 90% of validation NDCG@10, maximize the goal-meeting item rate; then resolve ties by health utility, NDCG, and the smaller parameter.
- Selected fixed weight: `0.4`.
- Selected GAP Rank maximum quota share: `0.6`.
- Test evaluation occurs only after the parameters are frozen.

Candidate CSV files do not contain relevance labels. Reranking reads only user ID, meal ID, preference score, train-only user baseline, and meal health utility. Validation and test labels are used exclusively inside metric computation.

## Test results

| Method | Recall@10 | NDCG@10 | NDCG retention | Health utility@10 | Change from user baseline | Goal-meeting rate@10 | Preference Top-10 overlap |
|---|---:|---:|---:|---:|---:|---:|---:|
| Preference only | 0.087280 | 0.064976 | 100.00% | 0.306749 | -0.005177 | 17.96% | 100.00% |
| Fixed weight | 0.086577 | 0.062958 | 96.89% | 0.323349 | +0.011423 | 25.17% | 90.09% |
| Hard filter | 0.034101 | 0.026742 | 41.16% | 0.420505 | +0.108579 | 98.83% | 19.13% |
| **GAP Rank** | **0.077011** | **0.060721** | **93.45%** | **0.336569** | **+0.024644** | **38.80%** | **79.17%** |

GAP Rank does not dominate the fixed-weight baseline on every measure. Instead, it selects a stronger health improvement that still satisfies the validation retention constraint. Compared with fixed weighting, GAP Rank increases the test goal-meeting rate by 13.63 percentage points while retaining 93.45% of preference-only NDCG@10. Hard filtering demonstrates the relevance loss produced by optimizing almost exclusively for target attainment.

## Why these metrics are sufficient

- Recall@10 and NDCG@10 preserve comparability with the preference experiment.
- NDCG@10 retention directly measures ranking-quality preservation.
- Mean health utility@10 and change from the train-only baseline measure overall list movement.
- Goal-meeting item rate@10 measures target attainment in the final list.
- Preference Top-10 overlap describes how many items were replaced.
- Candidate feasibility and hard-filter fallback counts distinguish method limitations from missing feasible candidates.

Together these metrics report relevance, wellness improvement, target attainment, and the trade-off between them without adding redundant measures.

## Reproduction and outputs

```powershell
.\.venv\Scripts\python.exe .\scripts\export_candidates.py
.\.venv\Scripts\python.exe .\scripts\run_reranking.py
```

- `results/candidates/neumf_standard_top100.json`: candidate provenance, row counts, history exclusion, hashes, and Recall@100 checks.
- `results/reranking/gap_rank.json`: target definition, validation selection, test results, anonymous cases, and limitations.
- `results/comparison/health_reranking.csv`: report-ready validation and test comparison rows.
- `data/processed/mealrecplus_h/reranked/gap_rank_test_top10.csv`: rebuildable local Top-10 output used by the demonstration; not committed.

## Limitations

- Equal-weight FSA/WHO utility is a transparent project proxy, not a clinically validated composite health index.
- Offline target attainment does not prove adoption or improvement in real health outcomes.
- Reranking can select only within the preference model's Top-100 candidates.
- One public dataset and one official split do not establish generalization across populations or food cultures.
