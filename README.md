# Paradigm_Shift_Act_Inf

Multi-agent active-inference simulation accompanying the IWAI 2026 paper *Variational Paradigm Dynamics* (working title).

## What

Extension of Hyland & Albarracin (2025), *On the Variational Costs of Changing Our Minds* (arXiv:2509.17957), from single-agent belief revision to a coupled multi-agent setting on a small-world network. Demonstrates Kuhnian paradigm-shift dynamics: communities lag behind environmental drift, with hysteretic phase transitions in their adaptation, dissociating from Bayesian rule-induction accounts (Oldenburg & Zhi-Xuan 2024).

Three intended results:

1. Multi-agent extension of the variational cost of mind-change.
2. (λ, environment-drift-rate) phase diagram + hysteresis under slow drift.
3. Clean dissociation from a Bayesian rule-induction baseline.

## Layout

```
paper/         LNCS submission target (anonymized; do NOT commit authors-deanon.tex)
lit-review/    Topical lit reviews (markdown, organising prior reading)
archive/       Frozen v02 norms-as-shared-precision-priors draft
src/           Multi-agent coupling layer on top of pymdp
experiments/   Configs + run_experiment entrypoint
results/       Output (gitignored if large)
notebooks/     Exploratory + figure generation
```

## Running

```bash
python -m venv .venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt

# Smoke test
python -c "import pymdp; import jax; print('pymdp + jax OK')"

# Run an experiment (after src/ is implemented)
python experiments/run_experiment.py --config experiments/configs/E1_baseline.yaml
```

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

- 2026-05-24: 250-word abstract registration
- 2026-06-07: 12-page LNCS full paper

## Plan

Detailed implementation plan: `C:\Users\Jonas\.claude\plans\wobbly-squishing-widget.md`
