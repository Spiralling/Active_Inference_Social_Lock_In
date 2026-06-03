"""Builder for notebooks/29_bayesnet_cpd_over_time.ipynb (run once, then nbconvert --execute).

The overhaul demo: the phlogiston scenario rebuilt on a WELL-DEFINED Bayes net
(``src/structural/bayesnet.py`` -- explicit per-node CPDs) instead of the legacy
joint-precision blob. Shows the three things the joint-only form could not:

  1. every node has a named conditional distribution p(node | parents);
  2. the per-node belief (marginal) is well defined and we can watch it over time;
  3. with *relational* observations the CPD edge weights actually MOVE -- the Bayes
     net learns its structure -- whereas node-wise reads freeze the coupling (the bug).
"""
import pathlib
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def code(s): C.append(nbf.v4.new_code_cell(s))

md(r"""# 29 - The paradigm as a *well-defined* Bayes net: CPDs that change over time

**Why this notebook exists.** The structural model used to carry a paradigm as one
joint precision matrix `Pi` ("off-diagonal `Pi[i,j]` = edge"). That is the
*undirected Gaussian-MRF* reading -- it has **no conditional distributions**, no
per-node belief you can name, and because the old observation operator read one node
per row, the Fisher deposit `HᵀH` was **diagonal**: data only ever touched `Pi`'s
diagonal, so the off-diagonal *structure was frozen in the prior forever*. The "net"
was decorative.

**The fix** (`src/structural/bayesnet.py`). A real Bayes net: a DAG whose joint
factorises into explicit per-node CPDs

$$x_v \mid x_{\mathrm{pa}(v)} \sim \mathcal N\!\big(b_v + \textstyle\sum_{u}B_{vu}x_u,\; s_v\big).$$

We store those CPD parameters `(B, b, s)`; the joint `(Π, h)` is *derived*
(`to_info`), and a posterior is *re-read* as CPDs (`from_info`) -- an exact bijection.
Crucially, a **relational** observation (an `H` row reading a *combination* of nodes,
e.g. a mass balance `calx − mass − gas`) deposits genuine **off-diagonal** Fisher
information, so the edges *learn*.

**What we show:** §1 the explicit CPDs & DAG · §2 the exact CPD↔precision bijection ·
§3 per-node beliefs **over time** · §4 structure (edges) **over time**: relational vs
node-wise (the bug, pinned visually) · §5 edge-level Bayesian Model Reduction.""")

code(r"""import sys, pathlib
ROOT = pathlib.Path.cwd()
sys.path.insert(0, str(ROOT if (ROOT / 'src').exists() else ROOT.parent))
import numpy as np, jax, jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.dpi': 110, 'axes.grid': True, 'grid.alpha': 0.3})

from src.structural.phlogiston import (
    StructuralConfig, phlogiston_bn, gravimetric_H, gravimetric_rows,
    H_observable, phi_true_at, NODE_NAMES, DISAGREEMENT_NODES)
from src.structural.bayesnet import LinearGaussianBN, relational_operator
from src.structural import linalg

cfg = StructuralConfig(t_shift=40, n_steps=120)
bn = phlogiston_bn(cfg, conviction=2.0)
print('nodes (d):', bn.dim)
print('node order:', bn.names)""")

md(r"""## §1 — The explicit CPDs and the DAG

Each node now carries a *named conditional distribution*. The hidden hub
`phlogiston` is the root common cause (a parent of every combustion/mass
commitment); the anomaly `calx_heavier_than_metal` is a child of the mass-law nodes
(the belt mass-balance). This is the structure the old `Pi`-blob only implied.""")

code(r"""for n in ['phlogiston', 'combustion_releases', 'mass_change_sign', 'calx_heavier_than_metal']:
    c = bn.cpd(n)
    pa = ', '.join(f'{p}·{w:+.2f}' for p, w in c['weights'].items()) or '(root)'
    print(f"p({n} | {pa})   intercept={c['intercept']:+.2f}  var={c['variance']:.2f}")

# the DAG as a directed adjacency (B[child, parent] != 0)
B = np.asarray(bn.B)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(B != 0, cmap='Greys', vmin=0, vmax=1)
ax.set_xticks(range(bn.dim)); ax.set_yticks(range(bn.dim))
ax.set_xticklabels(bn.names, rotation=90, fontsize=7)
ax.set_yticklabels(bn.names, fontsize=7)
ax.set_xlabel('parent'); ax.set_ylabel('child'); ax.set_title('DAG: B[child, parent] ≠ 0')
plt.tight_layout(); plt.savefig('figures_nb29_dag.png'); plt.show()
print('saved figures_nb29_dag.png')""")

md(r"""## §2 — The CPD ↔ precision bijection is exact

The CPDs compile to the joint precision form the dynamics consume, and a posterior
re-reads back to CPDs to machine precision. Two coordinates, one object.""")

code(r"""gbn = bn.to_info()                         # CPDs -> joint (Pi, h)
bn_back = LinearGaussianBN.from_info(gbn)   # joint -> CPDs
print('round-trip max |ΔB| =', float(np.abs(np.asarray(bn.B) - np.asarray(bn_back.B)).max()))
print('round-trip max |Δb| =', float(np.abs(np.asarray(bn.b) - np.asarray(bn_back.b)).max()))
print('round-trip max |Δs| =', float(np.abs(np.asarray(bn.s) - np.asarray(bn_back.s)).max()))
mu, cov = bn.joint()
print('prior marginal means:')
for n in ['combustion_releases', 'mass_change_sign', 'calx_heavier_than_metal']:
    i = bn.index(n); print(f'  {n:28s} {float(mu[i]):+.3f}  (var {float(cov[i,i]):.3f})')""")

md(r"""## §3 — Per-node beliefs **over time**

We feed the regime-switching world (`phi_true_at`: calx reads *lighter* in the
phlogiston regime, *heavier* after `t_shift`) through the **relational gravimetric
operator** and re-read the CPDs every step. Now we can plot the belief at *each
named node* as it evolves -- the thing the joint-only form collapsed to a scalar.""")

code(r"""def rollout(bn0, H, cfg, seed=0, noise=0.1):
    '''Return per-step (means (T,d), variances (T,d), nets [LinearGaussianBN]).'''
    key = jax.random.PRNGKey(seed)
    cur = bn0; means, varis, nets = [], [], []
    for t in range(cfg.n_steps):
        phi = phi_true_at(cfg, t)
        key, k = jax.random.split(key)
        o = H @ phi + noise * jax.random.normal(k, (H.shape[0],))
        cur = cur.observe(H, o, sigma_o=cfg.sigma_o)
        mu, cov = cur.joint()
        means.append(np.asarray(mu)); varis.append(np.diag(np.asarray(cov))); nets.append(cur)
    return np.array(means), np.array(varis), nets

Hg = gravimetric_H(cfg)
means, varis, nets = rollout(bn, Hg, cfg)
T = cfg.n_steps

fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 4.5))
key_nodes = ['phlogiston', 'combustion_releases', 'mass_change_sign', 'calx_heavier_than_metal']
for n in key_nodes:
    a0.plot(means[:, bn.index(n)], label=n)
a0.axvline(cfg.t_shift, color='k', ls='--', lw=1, label='regime shift')
a0.set_xlabel('step'); a0.set_ylabel('marginal mean'); a0.set_title('per-node belief mean over time')
a0.legend(fontsize=7)

im = a1.imshow(means.T, aspect='auto', cmap='coolwarm', vmin=-1.5, vmax=1.5)
a1.set_yticks(range(bn.dim)); a1.set_yticklabels(bn.names, fontsize=7)
a1.axvline(cfg.t_shift, color='k', ls='--', lw=1)
a1.set_xlabel('step'); a1.set_title('all node means  (T × d)')
fig.colorbar(im, ax=a1, shrink=0.8)
plt.tight_layout(); plt.savefig('figures_nb29_beliefs.png'); plt.show()

ci = bn.index('calx_heavier_than_metal')
print(f"calx_heavier belief: prior {means[0,ci]:+.2f} -> final {means[-1,ci]:+.2f} (world flips it heavier)")""")

md(r"""## §4 — Structure **over time**: relational learns, node-wise is frozen

This is the heart of the bug and the fix. We track the inferential coupling
`Π[calx, mass_change_sign]` (the belt edge) and a hub edge over time, under two
observation operators:

* **relational** `gravimetric_H` (rows that read *combinations*) — the edges move;
* **node-wise** `H_observable` (one node per row) — the off-diagonal coupling is
  **exactly frozen** (diagonal Fisher), so the structure can never be learned.""")

code(r"""def coupling_trace(nets, u, v):
    iu, iv = bn.index(u), bn.index(v)
    return np.array([np.asarray(n.to_info().Pi)[iu, iv] for n in nets])

def cpd_weight_trace(nets, child, parent):
    iv, iu = bn.index(child), bn.index(parent)
    return np.array([float(np.asarray(n.B)[iv, iu]) for n in nets])

# node-wise rollout (the legacy frozen-structure regime)
Hn = H_observable(cfg)
_, _, nets_node = rollout(bn, Hn, cfg)

belt_rel = coupling_trace(nets,      'calx_heavier_than_metal', 'mass_change_sign')
belt_nod = coupling_trace(nets_node, 'calx_heavier_than_metal', 'mass_change_sign')
hub_rel  = coupling_trace(nets,      'combustion_releases',     'phlogiston')
hub_nod  = coupling_trace(nets_node, 'combustion_releases',     'phlogiston')

fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 4.5))
a0.plot(belt_rel, label='relational obs', lw=2)
a0.plot(belt_nod, label='node-wise obs (frozen)', lw=2, ls='--')
a0.axvline(cfg.t_shift, color='k', ls=':', lw=1)
a0.set_title('belt coupling  Π[calx, mass_change]'); a0.set_xlabel('step'); a0.legend(fontsize=8)

a1.plot(hub_rel, label='relational obs', lw=2)
a1.plot(hub_nod, label='node-wise obs (frozen)', lw=2, ls='--')
a1.axvline(cfg.t_shift, color='k', ls=':', lw=1)
a1.set_title('hub coupling  Π[combustion, phlogiston]'); a1.set_xlabel('step'); a1.legend(fontsize=8)
plt.tight_layout(); plt.savefig('figures_nb29_structure.png'); plt.show()

print(f"belt coupling drift: relational {belt_rel[-1]-belt_rel[0]:+.3f}  vs  node-wise {belt_nod[-1]-belt_nod[0]:+.3f}")
print(f"=> node-wise obs leaves the off-diagonal coupling ~frozen; relational obs moves it (structure learned).")

# the directed CPD weight moving under relational evidence:
w = cpd_weight_trace(nets, 'calx_heavier_than_metal', 'mass_change_sign')
print(f"CPD weight (mass_change -> calx): prior {w[0]:+.3f} -> final {w[-1]:+.3f}")""")

md(r"""## §5 — Bayesian Model Reduction at the edge level

With explicit CPDs, BMR is the canonical Bayes-net move: *should the directed edge
`parent → child` be pruned?* We score the closed-form log Bayes factor `ΔF` for
pruning the belt edge against the data seen so far (Friston/Penny post-hoc identity,
`linalg.savage_dickey`). `ΔF > 0` ⇒ the data are content without the edge.""")

code(r"""# summarise the relational data as one Fisher deposit, then score the edge.
key = jax.random.PRNGKey(1); J = jnp.zeros((bn.dim, bn.dim)); j = jnp.zeros((bn.dim,))
for t in range(cfg.n_steps):
    phi = phi_true_at(cfg, t); key, k = jax.random.split(key)
    o = Hg @ phi + 0.1 * jax.random.normal(k, (Hg.shape[0],))
    Jd, jd = linalg.fisher_deposit(Hg, o, cfg.sigma_o); J = J + Jd; j = j + jd

for (child, parent) in [('calx_heavier_than_metal', 'mass_change_sign'),
                        ('combustion_releases', 'phlogiston')]:
    out = bn.bmr_prune_edge(child, parent, likelihood=(J, j))
    verdict = 'PRUNE (data content without it)' if out['favour_prune'] else 'KEEP (data hold the edge)'
    print(f"edge {parent:18s} -> {child:24s}  ΔF = {float(out['delta_F']):+8.2f}   {verdict}")""")

md(r"""## Verdict

The paradigm is now a **well-defined Bayes net**: explicit per-node CPDs, a derived
joint, an exact bijection, per-node beliefs we can watch evolve, and -- with
relational observations -- **edges that actually learn from data** instead of a
structure frozen in the prior. The node-wise vs relational panel in §4 is the bug
and its fix on one axis: the legacy operator could never have moved the coupling.

Next: migrate the multi-agent dynamics (`step.py` / `kernel.py`) onto the
relational operator so the population-level rollouts inherit structure learning, and
re-run the lock-in / staircase experiments (nb24–28) on the corrected substrate.""")

nb["cells"] = C
out = pathlib.Path(__file__).resolve().parent / "29_bayesnet_cpd_over_time.ipynb"
nbf.write(nb, str(out))
print("wrote", out)
