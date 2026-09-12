# Reproducible environment

## Tested configuration

| Component | Tested value |
|---|---|
| Operating system | Windows 11 |
| Python | 3.10.21 |
| Virtual environment | Repository-local `.venv/` |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU, approximately 12 GB VRAM |
| Packaging tools | pip 25.3, setuptools 80.9.0, wheel 0.45.1 |
| PyTorch | 2.7.1+cu128 |
| RecBole | 1.2.1 |
| Streamlit | 1.63.0 |
| Plotly | 7.0.0 |

## Python version choice

Python 3.10 is compatible with both the tested PyTorch Windows build and RecBole 1.2.1. The numerical stack is intentionally conservative because RecBole 1.2.x predates several breaking changes in newer NumPy and SciPy releases. In particular, `numpy<2` avoids removed aliases still referenced by this RecBole generation.

Relevant upstream references:

- PyTorch installation guide: https://pytorch.org/get-started/locally/
- RecBole repository: https://github.com/RUCAIBox/RecBole
- RecBole dependency compatibility discussion: https://github.com/RUCAIBox/RecBole/issues/2105

## Environment creation

Create an isolated Python 3.10 environment from any suitable interpreter:

```powershell
<PYTHON310_PATH> -m venv --copies .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
python --version
```

If PowerShell policy prevents activation, call the environment interpreter directly:

```powershell
.\.venv\Scripts\python.exe --version
```

Install dependencies in layers so compatibility failures remain easy to diagnose:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip==25.3 setuptools==80.9.0 wheel==0.45.1
.\.venv\Scripts\python.exe -m pip install -r .\requirements\torch-cu128.txt
.\.venv\Scripts\python.exe -m pip install -r .\requirements\dev.txt
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation --editable .
```

The editable installation points the environment at `src/` without copying source files. The `.venv/` directory is ignored by Git.

## Dependency layers

1. Packaging: pip 25.3, setuptools 80.9.0, and wheel 0.45.1.
2. Deep learning: PyTorch 2.7.1+cu128.
3. Numerical and data: NumPy 1.26.4, pandas 2.2.3, SciPy 1.10.1, and scikit-learn 1.5.2.
4. Recommendation framework: RecBole 1.2.1.
5. Demonstration: Streamlit 1.63.0 and Plotly 7.0.0.
6. Engineering: pytest 8.3.5 and ruff 0.11.13.

`requirements/lock-win-py310-cu128.txt` records the complete resolved environment used for the reported runs. The smaller requirement files document direct dependencies and installation order.

## Verification

```powershell
.\.venv\Scripts\python.exe .\scripts\check_environment.py
.\.venv\Scripts\python.exe .\scripts\smoke_recbole.py
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
```

Observed verification summary:

```text
pip check: No broken requirements found
PyTorch: 2.7.1+cu128
CUDA runtime: 12.8
CUDA available: true
Compute capability: 12.0
NeuMF one-epoch smoke test: passed
LightGCN one-epoch smoke test: passed
Streamlit default and no-intervention render: passed
ruff: passed
pytest: 31 passed
```

The one-epoch model smoke tests use the small synthetic fixture under `tests/fixtures/`; their metrics are software checks and are not reported as experimental results.
