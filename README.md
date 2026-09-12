# Personalized Healthy Meal Recommender

This repository contains the implementation and reproducibility evidence for the CA6124 group project. The system first learns each anonymous user's meal preferences, then applies goal-adaptive health reranking to improve the nutritional profile of the final list while preserving most of the preference-ranking quality.

The project is a wellness recommender, not a diagnostic or clinical decision system.

## Research question

Can a personalized recommender adapt the strength of health-aware reranking to each user's gap from a wellness target, while retaining most of the relevance achieved by the preference model?

The proposed method, **GAP Rank**, uses NeuMF Standard to generate Top-100 candidates. It then estimates the intervention required from the user's train-only health baseline, the selected target, and the availability of target-meeting candidates. Only the minimum number of low-ranked items required to satisfy the resulting quota are replaced.

## System architecture

```text
MealRec+H source data
        |
        v
Source audit and deterministic preprocessing
        |
        v
MostPop / NeuMF / LightGCN preference models
        |
        v
NeuMF Standard Top-100 unseen candidates
        |
        v
GAP Rank goal-adaptive health reranking
        |
        v
Offline evaluation and Streamlit demonstration
```

## Main contributions

- A reproducible data pipeline pinned to MealRec+H commit `8bd7d7d7effb505be4a7d2074b28abad802fe0f7`.
- A fair full-catalog comparison of MostPop, NeuMF, and LightGCN on the same official splits.
- A transparent health utility based on normalized FSA and WHO meal scores.
- GAP Rank, which adapts a target-meeting quota to each user's wellness gap and candidate feasibility.
- A working Streamlit application driven by real model candidates and the same reranking function used in the offline experiment.

## Repository layout

```text
app/                          Streamlit application and visual asset
configs/                      Data, model, and evaluation configurations
data/raw/                     Local source data; ignored by Git
data/processed/               Rebuilt model and demo artifacts; ignored by Git
docs/                         Technical and reproducibility documentation
requirements/                 Direct dependencies and the tested environment lock
results/                      Compact machine-readable experiment evidence
scripts/                      Reproducible pipeline entry points
src/healthy_meal_recommender/ Core Python package
tests/                        Unit, contract, and demo smoke tests
```

Raw data, processed data, virtual environments, model checkpoints, caches, and training logs are intentionally excluded from Git.

## Tested environment

- Windows 11 and Python 3.10.21
- PyTorch 2.7.1 with CUDA 12.8
- RecBole 1.2.1
- Streamlit 1.63.0 and Plotly 7.0.0
- NumPy 1.26.4, pandas 2.2.3, and SciPy 1.10.1

See [docs/environment.md](docs/environment.md) for installation details and compatibility notes.

## Installation

Create and activate a Python 3.10 virtual environment named `.venv`, then run:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip==25.3 setuptools==80.9.0 wheel==0.45.1
.\.venv\Scripts\python.exe -m pip install -r .\requirements\torch-cu128.txt
.\.venv\Scripts\python.exe -m pip install -r .\requirements\dev.txt
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation --editable .
```

For a CPU-only environment, install an appropriate PyTorch build first and then install the remaining direct dependencies from `requirements/dev.txt`.

## Data preparation

Download and audit the pinned MealRec+H source:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\download_mealrecplus.ps1
.\.venv\Scripts\python.exe .\scripts\audit_mealrecplus.py
```

Build and verify the deterministic model inputs:

```powershell
.\.venv\Scripts\python.exe .\scripts\prepare_mealrecplus.py
.\.venv\Scripts\python.exe .\scripts\check_prepared_data.py
.\.venv\Scripts\python.exe .\scripts\verify_preparation.py
```

The pipeline retains all 37,413 training interactions and removes only 13 validation/test interactions involving meals unseen in training. The warm-start validation and test sets contain 4,668 and 4,673 interactions respectively. User health baselines are recomputed from training interactions only.

See the [dataset card](docs/data-card-mealrecplus-h.md) and [preprocessing specification](docs/preprocessing.md) for provenance, integrity checks, privacy boundaries, and output schemas.

## Preference models

Run the three preference-model stages:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_mostpop.py
.\.venv\Scripts\python.exe .\scripts\run_neumf.py
.\.venv\Scripts\python.exe .\scripts\run_lightgcn.py
```

All models use the same full-catalog protocol. Validation NDCG@10 is the primary model-selection metric; the test split is evaluated only after a configuration has been selected.

| Model | Validation Recall@10 | Validation NDCG@10 | Test Recall@10 | Test NDCG@10 |
|---|---:|---:|---:|---:|
| MostPop | 0.025886 | 0.018718 | 0.023558 | 0.014451 |
| NeuMF Standard | **0.087420** | **0.063192** | **0.087280** | **0.064976** |
| LightGCN 1-layer | 0.061233 | 0.045095 | 0.059513 | 0.041006 |

NeuMF Standard is selected as the candidate generator because it achieves the highest validation NDCG@10 among the personalized models. Details are in [docs/model-training.md](docs/model-training.md) and [docs/evaluation-protocol.md](docs/evaluation-protocol.md).

## Health-aware reranking

Export NeuMF candidates and run the reranking experiment:

```powershell
.\.venv\Scripts\python.exe .\scripts\export_candidates.py
.\.venv\Scripts\python.exe .\scripts\run_reranking.py
```

The health utility is:

```text
FSA component  = clip((12 - FSA) / 8, 0, 1)
WHO component  = clip(WHO / 7, 0, 1)
health utility = 0.5 * FSA component + 0.5 * WHO component
```

The default target, `0.360119`, is the 75th percentile of health utility among meals observed in training. It is a reproducible ranking target, not a clinical threshold.

| Method | Recall@10 | NDCG@10 | NDCG retention | Health utility@10 | Goal-meeting rate@10 |
|---|---:|---:|---:|---:|---:|
| Preference only | 0.087280 | 0.064976 | 100.00% | 0.306749 | 17.96% |
| Fixed weight | 0.086577 | 0.062958 | 96.89% | 0.323349 | 25.17% |
| Hard filter | 0.034101 | 0.026742 | 41.16% | 0.420505 | 98.83% |
| **GAP Rank** | **0.077011** | **0.060721** | **93.45%** | **0.336569** | **38.80%** |

GAP Rank increases the goal-meeting rate by 20.84 percentage points over the preference-only list while retaining 93.45% of its test NDCG@10. The hard-filter comparison shows why health improvement must be balanced against preference relevance. Full definitions and validation-only parameter selection are documented in [docs/health-reranking.md](docs/health-reranking.md).

## Interactive demonstration

After generating the local artifacts, validate and start the application:

```powershell
.\.venv\Scripts\python.exe .\scripts\check_demo.py
.\.venv\Scripts\python.exe -m streamlit run .\app\streamlit_app.py
```

The application supports 1,114 anonymous test users, a configurable wellness target, live GAP Rank recomputation, before-and-after metrics, meal composition, available nutrition totals, and the offline comparison table. It stops with rebuild instructions if a required artifact is missing; it never substitutes fabricated recommendations. See [docs/demo.md](docs/demo.md).

## Verification

```powershell
.\.venv\Scripts\python.exe .\scripts\check_environment.py
.\.venv\Scripts\python.exe .\scripts\smoke_recbole.py
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
```

The committed version passes 31 automated tests. Compact evidence files under `results/` record data hashes, configuration provenance, selection rules, metrics, and demo contract checks without redistributing the source dataset or binary checkpoints.

## Responsible-use boundaries

- All model-facing users are anonymous numeric identifiers.
- Source username mappings, reviews, raw images, and other unnecessary fields are not included in processed data.
- FSA and WHO scores are general food-wellness indicators, not individualized medical advice.
- The dataset includes simulated meal construction and mined interactions, so results must not be interpreted as observed clinical outcomes or complete real-world diets.
- The upstream repository does not declare a formal license. This repository therefore provides download and verification scripts but does not redistribute the raw dataset.

## Dataset reference

MealRec+H is distributed through the [MealRecPlus repository](https://github.com/WUT-IDEA/MealRecPlus) and accompanies *MealRec+: A Meal Recommendation Dataset with Meal-Course Affiliation for Personalization and Healthiness*, SIGIR 2024, [https://doi.org/10.1145/3626772.3657857](https://doi.org/10.1145/3626772.3657857).
