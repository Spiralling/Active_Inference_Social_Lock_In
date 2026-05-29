# Paradigm_Shift_Act_Inf

Multi-agent active-inference simulation accompanying the IWAI 2026 paper *Variational Paradigm Dynamics* (working title).

## What

Extension of Hyland & Albarracin (2025), *On the Variational Costs of Changing Our Minds* (arXiv:2509.17957), from single-agent belief revision to a coupled multi-agent setting on a small-world network. Demonstrates Kuhnian paradigm-shift dynamics: communities lag behind environmental drift, with hysteretic phase transitions in their adaptation.

## Model substrates

The codebase contains three model substrates:

1. **Continuous Gaussian substrate** (`src/*.py`) — the original v2 implementation. Agents hold Gaussian beliefs over a continuous parameter; trust is Gamma-conjugate edge learning. Proven monostable via Jacobian analysis (NB14). Retained as a regression baseline.

2. **Categorical POMDP scaffold** (`src/pomdp/step.py`, `agent_pop.py`, `gen_model.py`) — fixed-state epistemic POMDP with per-paradigm likelihoods, belief-utility in the EFE policy, and softmax-precision social coupling. Phase-1 complete (28 tests pass). Discovered the martingale wall: belief-utility in the policy cannot flip the inference basin.

3. **Simplified theory-laden model** (`src/pomdp/simple_step.py`) — drops the POMDP action loop. Adds a latent context variable c in {normal_science, crisis} to the joint state (theta, c). Context carried forward through a proper transition model B_c. Used for E1-E3.

4. **Continuous-lambda model** (`src/pomdp/cont_step.py`) — direct multi-agent lift of Hyland & Albarracin (2025) Eq. 13: `q*(theta) ∝ prior(theta) · p(o|theta)^(1/lambda)`. Per-agent lambda tempers the likelihood; lambda evolves from KL cost of previous update (large changes → more conservative). Optional resource coupling gates exploration via trust-derived patronage. This is the active experimental substrate for E4-E6.

## Layout

```
src/              Core simulation modules
  pomdp/          Categorical POMDP scaffold + simplified model
experiments/      YAML configs (E1-E3) + sweep runner
tests/            Tiered test suite (60 tests)
notebooks/        Exploratory + figure generation (NB15-17 are active)
  _v1_archive/    Old continuous-substrate notebooks
notes/            Design docs, theory brief, architecture review
paper/            LNCS submission (anonymized; do NOT commit authors-deanon.tex)
lit-review/       Topical lit reviews (markdown)
archive/          Frozen v02 norms-as-shared-precision-priors draft
```

## Running

```bash
python -m venv .venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt

# Smoke test
python -c "import pymdp; import jax; print('pymdp + jax OK')"

# Tests
pytest tests/                              # full suite (46 tests)
pytest tests/test_simple_step.py -v        # simplified model only

# Experiments (simplified model)
python -m experiments.run_experiment experiments/configs/E1_stationary.yaml --dry-run
python -m experiments.run_experiment experiments/configs/E1_stationary.yaml
```

## Experiments

| ID | Model | Environment | Sweep | What it tests |
|----|-------|-------------|-------|---------------|
| E1 | simple | Stationary, no social | eps_resolve x eps_crisis | Individual-level theory-ladenness (baseline) |
| E2 | simple | Discrete shift at t=100 | q_reliability x eps_resolve | Binary-context phase diagram |
| E3 | simple | Slow ramp over 200 steps | q_reliability x eps_resolve | Hysteresis under drift |
| E4 | cont | Discrete shift at t=100 | q_reliability x lambda_init | **Continuous-lambda phase diagram (Fig 3)** |
| E5 | cont | Discrete shift at t=100 | q_reliability x lambda_init | Continuous-lambda + resources |
| E6 | cont | Discrete shift at t=100 | lambda_init x resource ON/OFF | Resource ablation (heuristic) |
| E7 | simple (C=5) | Discrete shift at t=100 | q_reliability x eps_resolve | **Principled phase diagram (Fig 3)** |
| E8 | simple (C=5) | Discrete shift at t=100 | eps_resolve x resource ON/OFF | **Principled resource ablation (Fig 4)** |

Configs in `experiments/configs/`. Results in `experiments/results/`.
Detailed results: `notes/experiment_results_2026_05_27.md` (E1-E3),
`notes/experiment_results_2026_05_28.md` (E4, E6),
`notes/experiment_results_2026_05_28_principled.md` (E7, E8).

## Building the paper

Uses Springer LNCS class. On MiKTeX (Windows), `llncs` is auto-installed if missing; otherwise: `mpm --install=llncs`. On TeXLive: `tlmgr install llncs`.

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Strip identifying metadata before submission:

```bash
exiftool -all= main.pdf
```

## Anonymization

Submission is **double-blind**. The paper compiles with `\author{Anonymous}` by default. Real authors live in `paper/authors-deanon.tex` (gitignored). Do not link this repo from the paper; do not commit anything that identifies the authors.

## Deadlines

- 2026-05-24: 250-word abstract registration (done)
- 2026-06-07: 12-page LNCS full paper
