# Data directories

Raw and processed datasets are not committed to this repository.

Expected layout:

```text
data/
|-- raw/          Downloaded source data before project transformations
`-- processed/    Audited, encoded, and split model and demo artifacts
```

Reproduce the pinned MealRec+H source and audit it with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\download_mealrecplus.ps1
.\.venv\Scripts\python.exe .\scripts\audit_mealrecplus.py
```

The source is pinned to official commit `8bd7d7d7effb505be4a7d2074b28abad802fe0f7`. Provenance and usage constraints are documented in `docs/data-card-mealrecplus-h.md`; the machine-readable audit is stored in `results/data_audit/mealrecplus_h_audit.json`.

The upstream repository is public and requests citation of the accompanying paper, but it does not include a formal license file. This project therefore does not redistribute the raw dataset.
