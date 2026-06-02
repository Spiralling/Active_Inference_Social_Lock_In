# Paradigm_Shift_Act_Inf

Multi-agent active-inference simulation accompanying the IWAI 2026 paper *Variational Paradigm Dynamics* (working title).

## What

Extension of Hyland & Albarracin (2025), *On the Variational Costs of Changing Our Minds* (arXiv:2509.17957), from single-agent belief revision to a coupled multi-agent setting on a small-world network. Demonstrates Kuhnian paradigm-shift dynamics: communities lag behind environmental drift, with hysteretic phase transitions in their adaptation, dissociating from Bayesian rule-induction accounts (Oldenburg & Zhi-Xuan 2024).

## Codebase map

The model has gone through three generations. They coexist in `src/`; this is which is which:

| Generation | Location | Status | Notebooks | Tests |
|---|---|---|---|---|
| **Structural** | `src/structural/` | **Current paper object** — a paradigm is a Gaussian Bayes net of commitments; learning adds Fisher information; restructuring is Schur-complement carry-over + Bayesian model reduction; conservatism λ_i is *derived* from structural position. | `notebooks/18`, `19` | `tests/test_structural.py` |
| Scalar | `src/*.py` (top level: `world`, `inference`, `policy`, `trust`, `utility`, `resource`, `population`, …) | Baseline the structural model departs from (single categorical θ, trust matrix, resource flow). | archived | `tests/test_tier1_invariants.py`, `tests/test_tier2_sanity.py` |
| POMDP | `src/pomdp/` | Exploratory categorical-POMDP scaffold (Phase-0/1 de-risking). | `notebooks/14`–`17` | `tests/test_pomdp_scaffold.py` |

Other top-level directories:

```
paper/         LNCS submission (main.tex + sections/ + notation.tex + bib_additions/)
notes/         Research notes; current background draft in notes/background_drafts/background.tex
archive/       Frozen prior versions
scripts/       One-off dev/inspection tools
tests/         pytest suite (46 tests; run from repo root)
results/       Run output (gitignored)
```

Superseded writing and notebooks are archived in place (see **Archived material** below), each with a dated `_archive/` README.

## Running

```bash
python -m venv .venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt

# Smoke test
python -c "import pymdp; import jax; print('pymdp + jax OK')"

# Run the test suite (from repo root)
python -m pytest tests/ -q

# The simulation is driven from notebooks, not a CLI entrypoint.
# Start with the current structural model:
jupyter lab notebooks/18_structural_phlogiston.ipynb
```

## Building the paper

Uses Springer LNCS class. On MiKTeX (Windows), `llncs` is auto-installed if missing; otherwise: `mpm --install=llncs`. On TeXLive: `tlmgr install llncs`.

```bash
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

The structural background walkthrough builds separately:

```bash
cd notes/background_drafts/_build
python _prepare.py        # regenerates background_compile.tex + merged refs.bib
pdflatex main.tex
```

Strip identifying metadata before submission: `exiftool -all= main.pdf`.

## Anonymization

Submission is **double-blind**. The paper compiles with `\author{Anonymous}` by default. Real authors live in `paper/authors-deanon.tex` (gitignored). Do not link this repo from the paper; do not commit anything that identifies the authors.

## Deadlines

- 2026-05-24: 250-word abstract registration
- 2026-06-07: 12-page LNCS full paper

## Archived material

Superseded work is kept (not deleted) for provenance:

- `notebooks/_archive/` — scalar/vector notebooks 04–13.
- `notebooks/_v1_archive/` — the original v1 scalar notebooks 00–05.
- `notes/archive/pre_2026-05-22/` — research notes predating the structural pivot (PDFs date-prefixed).
- `notes/background_drafts/_archive/` — predecessors of the current `background.tex` (the S1–S7 fragments, the scalar walkthrough, the lit-review draft).
- `paper/_archive/` — standalone pre-pivot writeups (pomdp_paradigm_model, structured_belief_revision) and figure scratch.
- `archive/v02-norms-as-shared-precision-priors/` — frozen v02 draft.
