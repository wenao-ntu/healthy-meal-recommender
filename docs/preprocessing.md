# MealRec+H preprocessing specification

## Purpose

The preprocessing pipeline produces the minimum artifacts required for an end-to-end personalized wellness recommender: trainable interactions, fixed evaluation splits, interpretable health attributes, and meal metadata for the demonstration. Review text and other fields unrelated to these tasks are excluded.

## Input and output

The source is pinned to official MealRec+H commit `8bd7d7d7effb505be4a7d2074b28abad802fe0f7`. Before transformation, the script verifies the sizes and SHA-256 hashes of audited source files.

Generated artifacts are written under the Git-ignored directory `data/processed/mealrecplus_h/`:

| Artifact | Purpose |
|---|---|
| `recbole/mealrecplus_h/mealrecplus_h.train.inter` | Shared training input for MostPop, NeuMF, and LightGCN |
| `recbole/mealrecplus_h/mealrecplus_h.valid.inter` | Model selection and parameter selection |
| `recbole/mealrecplus_h/mealrecplus_h.test.inter` | One frozen final evaluation |
| `course_catalog.csv` | The 942 courses used by meals, with compact nutrition and display fields |
| `meal_catalog.csv` | 3,817 three-course meals, health scores, and available nutrition totals |
| `user_health_train.csv` | Train-only health baselines for 1,575 anonymous users |
| `evaluation_exclusions.csv` | The 13 cold-meal interactions excluded from warm-start evaluation |

Compact manifests, hashes, and repeatability evidence are committed under `results/data_preparation/` so the transformation can be reviewed without redistributing the source data.

## Transformation rules

1. Preserve the official train/tune/test assignment; the source `tune` split is named `valid` in project code. Meal interactions have no timestamps, so the split is not described as temporal.
2. Retain all 37,413 training interactions. Remove only validation and test interactions involving meals unseen in training, producing 4,668 validation and 4,673 test interactions.
3. Record the 13 excluded interactions and their reasons instead of silently dropping them.
4. Recompute each user's mean FSA and WHO baseline exclusively from training interactions. Official user vectors are used only as equality checks.
5. Retain only the 942 courses referenced by the 3,817 meals. Reviews, complete cooking directions, and source username mappings are not copied to processed data.
6. Treat five courses whose eight nutrition amounts are all `0` with missing units as source missing-value placeholders. Their course-level FSA and WHO scores remain usable, but meal nutrition totals involving them are left unavailable rather than interpreted as zero.
7. Sort tables by numeric identifiers, write UTF-8 with LF line endings and fixed numeric formatting, and omit generation timestamps. Repeated runs must be byte-identical.

## Execution and validation

```powershell
.\.venv\Scripts\python.exe .\scripts\prepare_mealrecplus.py
.\.venv\Scripts\python.exe .\scripts\check_prepared_data.py
.\.venv\Scripts\python.exe .\scripts\verify_preparation.py
```

`check_prepared_data.py` validates schemas and asks RecBole to load all three `.inter` files. This catches naming, header, and configuration incompatibilities before model training. `verify_preparation.py` checks the committed manifest and repeatability evidence.

## Downstream use

- All preference models share the same interactions and evaluation exclusions.
- GAP Rank reads candidate scores, `meal_catalog.csv`, and `user_health_train.csv`; it does not read validation or test labels.
- The Streamlit application uses the same meal table and model candidates as the offline pipeline rather than a hand-authored example list.
