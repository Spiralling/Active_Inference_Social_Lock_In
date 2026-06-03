# Production scope

This branch provides a minimal structural production surface.

## Supported in production

- `src/structural/**`
- `src/config.py` (minimal `NetworkConfig` only)
- `src/network.py`
- `src/__init__.py`
- Scripts:
  - `scripts/module_map.py`
  - `scripts/production_smoke.py`
  - `scripts/production_check.py`
- Tests:
  - `tests/test_bayesnet.py`
  - `tests/test_graphs.py`
  - `tests/test_linalg.py`
  - `tests/test_structural_kernel.py::test_run_final_matches_step[derived]`
- Notebook:
  - `notebooks/29_bayesnet_cpd_over_time.ipynb`

## Explicitly removed from production scope

- Unified layer and adapters
- POMDP layer
- Scalar v2 root modules unrelated to structural graph builders
- Legacy notebooks, paper sources, notes, experiments, and archive directories

These removed layers are not supported on `production/structural-minimal`.
