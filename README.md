# Paradigm_Shift_Act_Inf

Minimal structural production branch for graph-based Bayesian paradigm dynamics.

## Kept production surface

Top-level files:

- `.gitignore`
- `.gitattributes`
- `README.md`
- `PRODUCTION_SCOPE.md`
- `requirements-production.txt`

Directories:

- `scripts/`
  - `module_map.py`
  - `production_smoke.py`
  - `production_check.py`
- `src/`
  - `src/structural/**`
  - `src/config.py`
  - `src/network.py`
  - `src/__init__.py`
- `tests/`
  - `tests/__init__.py`
  - `tests/test_bayesnet.py`
  - `tests/test_graphs.py`
  - `tests/test_linalg.py`
  - `tests/test_structural_kernel.py`
- `notebooks/`
  - `notebooks/README.md`
  - `notebooks/29_bayesnet_cpd_over_time.ipynb`
  - `notebooks/_build_nb29.py`
- `results/` (with `.gitkeep`)

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-production.txt
python scripts/production_check.py
```

## Notebook reference

- `notebooks/29_bayesnet_cpd_over_time.ipynb`
