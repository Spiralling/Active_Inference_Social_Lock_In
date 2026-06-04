"""Builder for notebooks/43_multiagent_topology.ipynb (run once, then nbconvert --execute).

nb43: the GENUINE multi-agent lift of nb42. nb42's "population" was a conviction *sweep*
(one shared evidence stream thresholded by lambda_i) plus a one-shot 50/50 Fisher average --
agents never actually learned from each other. Here N agents sit on a trust graph and pool
precision (step.fuse: Pi_i <- sum_j W_ij Pi_j); the conviction-gated belt prune is read out
per agent; and the independent variable is the network TOPOLOGY. A small low-conviction
VANGUARD is the only source that runs the refuting gravimetric experiment; everyone else
self-censors completely and is a PURE RECEIVER, so the vanguard's disconfirming precision
must FLOW through the graph to reach them. The Fiedler connectivity lambda_2 and the
vanguard's placement decide whether it arrives -> structural revolution vs evidential lock-in.

Thin notebook over scripts/run_multiagent_topology.py: it imports run_world + the sweep
helpers and builds the F1-F5 figure panel. The simulation logic lives in the script (so it is
re-runnable and asserts its own null controls there); the notebook is the figure layer.
"""
import pathlib
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def code(s): C.append(nbf.v4.new_code_cell(s))

md(r"""# 43 — Genuine multi-agent structure learning: topology governs revolution vs lock-in

**What this is.** nb38 showed a single agent's structural prune is a **conviction** decision —
prune edge `e` iff its evidence for redundancy clears its conviction protection,
`ΔF_e > λ·v_e` (Eqs. 7–8). nb42 *thresholded* one shared evidence stream by a spread of `λ_i`
and did a one-shot 50/50 Fisher average for the social part — a conviction sweep, **not a
simulation**: agents never learned from each other.

**Here they do.** `N` agents sit on a trust graph and **learn from each other by pooling
precision** — `step.fuse`: `Pi_i ← Σ_j W_ij Pi_j`, the codebase's literal representation of
inter-agent learning. Every agent holds the *same* over-wired phlogiston prior; the only
heterogeneity is:

* a small **vanguard** (low conviction, `γ = 0`) — the *only* source that runs the refuting
  gravimetric experiment;
* the **rest** self-censor completely (`γ_rest = 1`, **pure receivers**) and hold moderate
  conviction poised just above the revolution boundary.

So the vanguard's disconfirming precision is **scarce** and must **diffuse over the graph** to
reach the rest. The independent variable is the **topology**; the question is whether it
governs a structural **revolution** (the community prunes the contested phlogiston mass-law
belt) versus an evidential **lock-in** (it stalls). The prune is a *read-out* of where each
fused agent lands — the only inter-agent influence is the genuine precision fusion.

**The contested belt** is the two native phlogiston mass-law edges
`mass_change_sign → calx_heavier_than_metal` and `gas_consumed → calx_heavier_than_metal`
(highest conviction protection `v_e`). An agent has **revolted** when it has pruned *both*.

> **Honest-findings note (the no-pool degeneracy).** This repo has repeatedly found naive
> posterior fusion *washes out* heterogeneity. It does here too — *unless* self-censorship is
> near-total. At `γ_rest = 0.95` (nb42's bubble value) the residual 5% channel already
> saturates the belt evidence over the horizon, and topology washes out. Only when the rest are
> **pure receivers** does the vanguard's diffusion become the sole evidence source and topology
> become the load-bearing variable. That requirement is the finding, not a knob hidden away.""")

code(r"""%matplotlib inline
import sys, pathlib
ROOT = pathlib.Path.cwd()
ROOT = ROOT if (ROOT / 'scripts').exists() else ROOT.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
plt.rcParams.update({'figure.dpi': 120, 'axes.grid': True, 'grid.alpha': 0.3})

from src.structural import graphs, shells
from src.structural.phlogiston import StructuralConfig
from scripts.run_multiagent_topology import (
    run_world, build_substrate, topology_panel, bridge_sweep, placement_contrast,
    CALIB, BELT)

RESULTS = ROOT / 'results' / 'structural_multiagent_topology'
RESULTS.mkdir(parents=True, exist_ok=True)

# the calibrated operating point (built once; shared by every run -> the prior is common,
# the heterogeneity is evidence-access γ and conviction-threshold λ).
cfg = StructuralConfig(t_shift=CALIB['t_shift'], n_steps=CALIB['n_steps'], sigma_o=CALIB['sigma_o'])
sub = build_substrate(cfg)
N = 60
kw = dict(CALIB)
print('operating point:', {k: CALIB[k] for k in ('conviction_rest','lambda_vanguard','n_vanguard','gamma_rest','sigma_o')})
print(f'contested belt edges (highest protection): {list(BELT)}')
print(f'v_e(belt) = {[round(float(sub.v_e[i]),2) for i in sub.belt_ix]}   '
      f'balanced λ* = {sub.lstar:.2f}   ({sub.n_edges if hasattr(sub,"n_edges") else len(sub.edges)} edges total)')""")

# ----------------------------------------------------------------------
md(r"""## F1 — The headline: connectivity governs revolution vs lock-in

One representative graph per family, ordered by **algebraic connectivity** `λ₂` (the Fiedler
value — 0 for a disconnected graph, larger for a better-mixed one). For each we run the full
simulation and read the **final revolted fraction** (community pruned the contested belt) and
the **mean edges kept** (how much over-wired structure survives).

A disconnected community (`isolated`, `λ₂ = 0`) lets only the vanguard revolt; a well-mixed one
lets the vanguard's evidence reach everyone. The transition between is the model's claim that
**network structure decides whether a paradigm's contested core ever moves.**""")

code(r"""panel = topology_panel(N, sub=sub, cfg=cfg, vanguard_placement='central', **kw)
panel = sorted(panel, key=lambda r: r['lambda2'])     # order by connectivity for a clean curve
lam2 = np.array([r['lambda2'] for r in panel])
rev  = np.array([r['final_revolted_fraction'] for r in panel])
kept = np.array([r['kept_t'][-1].mean() for r in panel])
deg  = np.array([r['mean_degree'] for r in panel])
labels = [r['panel_label'] for r in panel]

fig, (a0, a1) = plt.subplots(1, 2, figsize=(13.5, 5.0))
sc = a0.scatter(lam2, rev, c=lam2, cmap='viridis', s=120, zorder=3, edgecolor='k')
a0.plot(lam2, rev, color='0.6', lw=1.2, zorder=1)
for x, y, l, k in zip(lam2, rev, labels, deg):
    a0.annotate(f'{l}\n⟨k⟩={k:.1f}', (x, y), fontsize=7.5, textcoords='offset points',
                xytext=(6, 6))
a0.set_xscale('symlog', linthresh=0.05)
a0.set_xlabel(r'algebraic connectivity  $\lambda_2$  (symlog)')
a0.set_ylabel('final revolted fraction'); a0.set_ylim(-0.05, 1.08)
a0.set_title('F1 — topology governs revolution vs lock-in')

a1.scatter(lam2, kept, c=lam2, cmap='viridis', s=120, zorder=3, edgecolor='k')
a1.plot(lam2, kept, color='0.6', lw=1.2, zorder=1)
a1.axhline(sub.n_edges if hasattr(sub,'n_edges') else len(sub.edges), color='crimson', ls=':',
           lw=1, label='over-wired (all edges)')
a1.set_xscale('symlog', linthresh=0.05)
a1.set_xlabel(r'$\lambda_2$  (symlog)'); a1.set_ylabel('final mean edges kept')
a1.set_title('structure retained falls as the community mixes'); a1.legend(fontsize=8)
plt.tight_layout(); plt.savefig(RESULTS / 'F1_topology.png')
print('revolted fraction by λ₂-ordered topology:')
for l, x, y in zip(labels, lam2, rev):
    print(f'  {l:>16}: λ₂={x:7.3f}  revolted={y:.3f}')""")

# ----------------------------------------------------------------------
md(r"""## F2 — The clean knob: bridge density (de-confounding `λ₂` from degree)

The F1 panel mixes families with different *degrees*, so connectivity and density are
confounded. Here we hold the **within-block structure fixed** (three dense communities,
`intra` constant) and vary only the **cross-block bridge density** `inter`. That sweeps `λ₂`
while keeping the local structure constant — the controlled connectivity experiment. The
vanguard sits in **block 0**: at `inter = 0` only its own block can revolt; opening bridges
lets the disconfirming evidence cross into the other two blocks.""")

code(r"""sweep = bridge_sweep(N, sub=sub, cfg=cfg, **kw)
inter = np.array([r['inter'] for r in sweep])
l2    = np.array([r['lambda2'] for r in sweep])
rv    = np.array([r['final_revolted_fraction'] for r in sweep])

fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 4.8))
a0.plot(inter, rv, 'o-', color='seagreen', lw=2.2)
a0.set_xlabel('cross-block bridge density  inter  (intra fixed)')
a0.set_ylabel('final revolted fraction'); a0.set_ylim(-0.03, 1.05)
a0.set_title('F2 — bridging the blocks spreads the revolution')
a1.plot(l2, rv, 'o-', color='teal', lw=2.2)
for x, y, it in zip(l2, rv, inter):
    a1.annotate(f'{it:.3f}', (x, y), fontsize=7, textcoords='offset points', xytext=(4, -9))
a1.set_xlabel(r'algebraic connectivity $\lambda_2$ (set by the bridge)')
a1.set_ylabel('final revolted fraction'); a1.set_ylim(-0.03, 1.05)
a1.set_title('same data vs λ₂  (≤6 points — read the trend, not a law)')
plt.tight_layout(); plt.savefig(RESULTS / 'F2_bridge.png')
print('inter →', list(np.round(inter,3)), '\nrevolted →', list(np.round(rv,3)))""")

# ----------------------------------------------------------------------
md(r"""## F3 — Vanguard placement: *where* the scarce source sits

If evidence is scarce and must diffuse, *where* the vanguard sits should matter (the nb10
question). On a fixed **scale-free** graph (degree-heterogeneous: a few hubs, many leaves) we
put the vanguard on the **central** (highest-degree) nodes versus the **peripheral**
(lowest-degree) nodes. A hub vanguard floods the network; a peripheral one is quarantined by
its own low connectivity.""")

code(r"""place = placement_contrast(N, sub=sub, cfg=cfg, **kw)
fig, (a0, a1) = plt.subplots(1, 2, figsize=(12.5, 4.6))
groups = ['central', 'peripheral']
vals = [place[g]['final_revolted_fraction'] for g in groups]
a0.bar(groups, vals, color=['steelblue', '0.6'], width=0.55)
for i, v in enumerate(vals):
    a0.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=11)
a0.set_ylabel('final revolted fraction'); a0.set_ylim(0, 1.1)
a0.set_title('F3 — central vs peripheral vanguard (scale-free graph)')
for g, col in [('central', 'steelblue'), ('peripheral', 'crimson')]:
    rt = place[g]['revolted_t'].mean(axis=1)
    a1.plot(place[g]['snap_t'], rt, lw=2.4, color=col, label=g)
a1.axvline(cfg.t_shift, color='k', ls=':', lw=1); a1.text(cfg.t_shift+1, 0.02, 'regime shift', fontsize=8)
a1.set_xlabel('step'); a1.set_ylabel('fraction revolted'); a1.set_ylim(-0.03, 1.05)
a1.set_title('a hub source floods; a peripheral one is quarantined'); a1.legend(fontsize=9)
plt.tight_layout(); plt.savefig(RESULTS / 'F3_placement.png')
print('central revolted =', round(vals[0],3), ' peripheral revolted =', round(vals[1],3))""")

# ----------------------------------------------------------------------
md(r"""## F4 — The dynamics: structure(t) and the fragmented field

The same panel, resolved in time. **Left:** the population-mean edges kept over the run, one
curve per topology (coloured by `λ₂`) — well-mixed graphs shed the contested structure, the
disconnected ones hold it. **Right:** the **residual structural disagreement** (mean Frobenius
spread of the agents' precision matrices, `shells.residual_disagreement`). It *stays high* when
an entrenched bloc never receives the evidence — the **fragmented field**, the lock-in
signature: a community split not by what the world shows but by who is connected to whom.""")

code(r"""fig, (a0, a1) = plt.subplots(1, 2, figsize=(13.5, 5.0))
order = np.argsort(lam2)
norm = plt.Normalize(0, 1)
for j in order:
    r = panel[j]
    c = cm.viridis(norm(min(r['lambda2'] / 1.3, 1.0)))
    a0.plot(r['snap_t'], r['kept_t'].mean(axis=1), lw=2.2, color=c,
            label=f"{r['panel_label']} (λ₂={r['lambda2']:.2f})")
    a1.plot(r['snap_t'], r['disagreement_t'], lw=2.2, color=c)
a0.axvline(cfg.t_shift, color='k', ls=':', lw=1)
a0.set_xlabel('step'); a0.set_ylabel('population-mean edges kept')
a0.set_title('F4 — structure(t): well-mixed graphs prune the belt'); a0.legend(fontsize=7.5, loc='lower left')
a1.axvline(cfg.t_shift, color='k', ls=':', lw=1)
a1.set_xlabel('step'); a1.set_ylabel('residual structural disagreement')
a1.set_title('persistent disagreement = fragmented field (lock-in)')
plt.tight_layout(); plt.savefig(RESULTS / 'F4_dynamics.png')""")

# ----------------------------------------------------------------------
md(r"""## F5 — Fusion-mode contrast (the honesty panel)

Is the effect an artefact of *posterior pooling* (sharing whole belief nets, which can wash
out)? We re-run one fixed graph under three sharing rules:

* **posterior** — fuse whole nets `Pi_i ← Σ_j W_ij Pi_j` (multi-hop; identical priors fuse
  idempotently, so only the deposits diffuse);
* **deposit_pool** — pool only each step's *Fisher deposits* (1-hop, prior never averaged in);
* **deposit_keep** — each agent keeps its own deposit whole + a leak of its neighbours'.

If the topology effect survived only under posterior pooling it would be suspect. It does not —
all three give a topology-dependent revolt; posterior gives the strongest spread because it
diffuses the *accumulated* evidence, not just the current step's.""")

code(r"""gmid = graphs.community([N // 3] * 3, intra=0.45, inter=0.0, seed=0).with_bridge(0.03)
modes = ['posterior', 'deposit_pool', 'deposit_keep']
mvals, miso, mcomp = [], [], []
iso, comp = graphs.complete(N).isolated(), graphs.complete(N)
for mode in modes:
    mvals.append(run_world(gmid, sub=sub, cfg=cfg, fuse_mode=mode, vanguard_placement='block0', **kw)['final_revolted_fraction'])
    miso.append(run_world(iso, sub=sub, cfg=cfg, fuse_mode=mode, **kw)['final_revolted_fraction'])
    mcomp.append(run_world(comp, sub=sub, cfg=cfg, fuse_mode=mode, **kw)['final_revolted_fraction'])

x = np.arange(len(modes)); w = 0.26
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.bar(x - w, miso, w, label='isolated (λ₂=0)', color='0.7')
ax.bar(x,      mvals, w, label='community+bridge', color='mediumseagreen')
ax.bar(x + w,  mcomp, w, label='complete (λ₂=N)', color='steelblue')
ax.set_xticks(x); ax.set_xticklabels(modes)
ax.set_ylabel('final revolted fraction'); ax.set_ylim(0, 1.1)
ax.set_title('F5 — the topology effect survives every fusion mode'); ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig(RESULTS / 'F5_fusion_modes.png')
for m, vi, vc in zip(modes, miso, mcomp):
    print(f'  {m:>13}: isolated={vi:.2f}  complete={vc:.2f}  spread={vc-vi:+.2f}')""")

# ----------------------------------------------------------------------
md(r"""## Verdict

A *genuine* simulation — `N` agents pooling precision over a trust graph, the conviction-gated
belt prune read out per agent — turns the network **topology** into the order parameter for a
scientific **revolution vs lock-in**:

* **F1 / F2** — final revolted fraction rises **monotonically with `λ₂`** and saturates once the
  community is well-mixed; the controlled bridge sweep (intra fixed) reproduces the rise as a
  clean knob. A disconnected community lets only the vanguard revolt; a connected one lets the
  scarce disconfirming evidence diffuse to everyone.
* **F3** — *where* the vanguard sits matters: a hub source floods the network, a peripheral one
  is quarantined (the nb10 question, here a clean central-≫-peripheral gap).
* **F4** — the lock-in signature is a **persistent residual structural disagreement**: a
  fragmented field, a community split by connectivity rather than by evidence.
* **F5** — the effect is **not** an artefact of one fusion rule; it survives posterior pooling
  *and* both deposit-pooling variants.

**The honest boundary condition.** Topology is load-bearing only because the disconfirming
evidence is *scarce*: the rest must be near-total self-censors (`γ_rest → 1`, pure receivers).
At `γ_rest = 0.95` the residual channel self-saturates the belt evidence and topology washes out
(the no-pool degeneracy); at `γ_rest = 0` everyone gathers the evidence directly and topology
stops mattering (a control the script asserts). The model says **a network governs a paradigm
shift precisely when the refuting evidence is scarce and must travel** — and it stalls into a
fragmented lock-in when the graph cannot carry it.""")

nb["cells"] = C
out = pathlib.Path(__file__).resolve().parent / "43_multiagent_topology.ipynb"
nbf.write(nb, str(out))
print("wrote", out)
