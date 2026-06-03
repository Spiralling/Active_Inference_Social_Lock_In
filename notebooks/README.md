# Notebooks

Numbered roughly chronologically; the higher the number, the more current. The
**structural** notebooks (18+) exercise the current paper object (paradigm = a
Gaussian Bayes net); 14–17 are the exploratory POMDP scaffold; everything earlier
is archived.

## Live

| Notebook | Generation | Purpose |
|---|---|---|
| `29_bayesnet_cpd_over_time` | structural | CPD/edge interpretability over time: re-read posterior precision states as explicit Bayes-net conditionals. |
| `28_unified_staircase_lockin` | unified | Lock-in, flip, and the staircase test on the unified substrate (`src/unified`); committed-minority tipping surface. |
| `27_conditional_bayes_net` | structural | Open the per-agent net: conditional (hub-marginalized) structure; rigid-wiring finding. |
| `26_method_diffusion_flip` | structural | A new *method* social-contagions to the entrenched old guard, who then flip from their own measurements. |
| `25_initial_conditions` | structural | Transient suppression vs enablement; timed bridge (`run_bridge`); settling time. |
| `22`–`24` | structural | Phil-sci severe tests: degenerating programme, committed-minority tipping. |
| `20`–`21` | structural | Derived evidential precision (`precision.py`); shells. |
| `18_structural_phlogiston` | structural | Structural paradigm shift: phlogiston as a Gaussian Bayes net (the founding experiment). |
| `19_biased_experiment_lockin` | structural | Biased experiment selection: from horizon lock-in to *permanent* lock-in. |
| `14`–`17` | pomdp | Categorical-POMDP scaffold (Jacobian scan, social-fold go/no-go, environment-coupled fold, smoke). |

Each live notebook `NN_name.ipynb` is **generated** from `_build_nbNN.py` (see
below); the `.ipynb` is a build artifact.

## Archived

- `_archive/` — scalar/vector notebooks **04–13** (pre-2026-05-22, predating the
  structural pivot) plus their `_build_nb12.py` / `_build_nb13.py` generators.
- `_v1_archive/` — the original **v1** scalar notebooks 00–05 (frozen).

See the repo `README.md` codebase map for how each generation relates to `src/`.

---

# Authoring a new notebook

Notebooks are **not hand-edited**. Each one is emitted by a Python *build script*
`_build_nbNN.py`, so the source lives in one reviewable, diff-able place and the
notebook can be regenerated and re-executed deterministically. To add notebook
`NN`:

1. Copy the most recent `_build_nbNN.py` (currently `_build_nb29.py`) to
   `_build_nb<NN>.py` and edit it.
2. Generate, then execute in place:
   ```
   python notebooks/_build_nb<NN>.py
   jupyter nbconvert --to notebook --execute --inplace notebooks/<NN>_name.ipynb
   ```
3. Add a row to the **Live** table above.

## The build-script skeleton

```python
"""Builder for notebooks/NN_name.ipynb (run once, then nbconvert --execute).

<one-line scientific purpose>.
SCALE knob: 'iterate' (fast) -> 'publication' -> 'heavy'.
"""
import pathlib
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def code(s): C.append(nbf.v4.new_code_cell(s))

md(r"""# NN - Title

What this notebook does, and the one honest question it answers.""")

code(r"""import sys, pathlib
ROOT = pathlib.Path.cwd()
sys.path.insert(0, str(ROOT if (ROOT / 'src').exists() else ROOT.parent))
import numpy as np, jax, jax.numpy as jnp, dataclasses
import matplotlib.pyplot as plt
from src.structural.phlogiston import StructuralConfig
from src.structural import graphs, observables as obs, phlogiston as ph
from src.structural.kernel import Network
plt.rcParams.update({'figure.dpi': 110, 'axes.grid': True, 'grid.alpha': 0.3})
FIGDIR = ROOT / 'figures_nbNN'; FIGDIR.mkdir(exist_ok=True)

# SCALE knob: validate fast, then crank.
SCALE = 'iterate'  # 'iterate' | 'publication' | 'heavy'
N, T, SEEDS = {'iterate': (60, 200, 6), 'publication': (250, 800, 20),
               'heavy': (500, 1500, 50)}[SCALE]""")

# ... §-sections: each a md() reading-guide cell + a code() experiment+plot cell ...

nb['cells'] = C
out = pathlib.Path(__file__).resolve().parent / 'NN_name.ipynb'
nbformat.write(nb, out)
print('wrote', out)
```

## Conventions (follow the recent builders)

- **`md()` / `code()` cell-builders.** Append markdown and code cells in order;
  use raw strings (`r"""..."""`) so LaTeX/regex survive. Pair every code cell with
  a short markdown reading-guide.
- **`src` import path.** The first code cell inserts the repo root on `sys.path`
  (the `ROOT if (ROOT/'src').exists() else ROOT.parent` line) so the notebook runs
  whether the kernel CWD is the repo root or `notebooks/`.
- **`SCALE` knob.** Expose `'iterate' | 'publication' | 'heavy'` and key `N, T,
  SEEDS, grid` off it. Validate at `'iterate'`, crank for the figure.
- **Figures.** One directory per notebook: `FIGDIR = ROOT / 'figures_nbNN'`;
  `fig.savefig(FIGDIR / 'name.png', ...)`.
- **Honest verdict.** End with a `## Verdict` markdown cell stating plainly what
  the model *did* and *did not* do, including null/negative results. A structural
  wall is a finding — surface it, don't force the target.

## Setting up the population (the current interface)

Two **independent** axes — keep them separate:

- **Beliefs** = the `groups` spec (who believes what): a list of
  `{"count", "paradigm": "phlogiston"|"oxygen", "stance", "prec_scale",
  "stance_sd", "prec_sd"}`. `groups=None` is the homogeneous baseline.
- **Topology** = a `graphs.*` graph (who talks to whom), passed to
  `Network.init(cfg, key, groups=..., graph=...)`.

```python
from src.structural import graphs

# Topology — pick any family, independent of beliefs:
g = graphs.erdos_renyi(N, mean_degree=6, seed=0)   # G(n,p) random
g = graphs.scale_free(N, mean_degree=6, seed=0)    # Barabasi-Albert (hubs)
g = graphs.watts_strogatz(N, mean_degree=6, seed=0)# small-world
g = graphs.community([N//2, N//2], intra=0.5, inter=0.05, seed=0)  # SBM blocks
g = graphs.ring(N); graphs.lattice(side, side); graphs.complete(N)  # deterministic

net = Network.init(cfg, jax.random.PRNGKey(0), groups=groups, graph=g)
m_t, grav_t = net.run_trace()          # population order parameter + self-censorship
idx_t       = net.run_trace_index()    # per-agent index (mean over a block = its curve)
net.isolated().run_trace()             # no-fusion baseline (graph op)
```

For a **timed bridge** (Model B), build a community graph with `inter=0` and let
`run_bridge` open it — the membership lives on the graph, so no `groups` argument
is threaded through:

```python
net = Network.init(cfg, key, groups=groups,
                   graph=graphs.community([n0, n1], intra=0.5, inter=0.0, seed=0))
m_t, _ = net.run_bridge(t_incubate=80, inter_prob=0.3)
```

The **default path** (omit `graph=`) builds the topology from `cfg.network`
(`NetworkConfig(kind=..., mean_degree=..., intra_prob=..., inter_prob=...)`) and
is byte-identical to the old behaviour — existing notebooks keep working. New work
should prefer the explicit `graph=` form: it makes "what topology?" a visible,
swappable line rather than a config side-effect.
