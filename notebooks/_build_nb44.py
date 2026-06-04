"""Builder for notebooks/44_cosmology_tracking_levers.ipynb (run once, then nbconvert --execute).

nb44: a CHANGING cosmology world on the reusable simulation engine. nb43 asked whether network
topology governs a structural revolution vs lock-in under ONE shift. Here the true world is three
genuinely different cosmology theories and it changes ~3 times; we ask which population setups keep
re-tracking each successive theory vs collapse and get stuck, comparing two diversity levers --
network DISCONNECTION and CONVICTION -- head to head.

The simulation logic lives in two re-runnable scripts that assert their own controls:
  * scripts/run_cosmology_tracking.py  -- Lens B (population means-tracking + the lever 2x2);
  * scripts/run_cosmology_regrowth.py  -- Lens A (genuine node-wake structure re-growth).
This notebook is the figure + explanation layer over them (mirrors the nb43 pattern, with more
prose and explicit disclaimers, per the brief).
"""
import pathlib
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def code(s): C.append(nbf.v4.new_code_cell(s))

md(r"""# 44 — A changing cosmology on a reusable engine: do communities keep re-tracking, or get stuck?

**The question.** nb43 showed that under **one** paradigm shift, a network's connectivity `λ₂`
governs whether a community stages a structural *revolution* or an evidential *lock-in*. This
notebook makes the world **keep changing**: the truth is three genuinely different cosmology
theories, and it cycles through them in three epochs. Agents must **re-grow their picture of the
world** to track each successive theory. We ask: **which population setups keep enough diversity
to keep re-tracking, and which collapse and get stuck on an old theory?** — comparing two levers
that could preserve diversity, **network disconnection** and **conviction**, head-to-head.

**One engine, environments as presets.** The run-loop is now a single reusable engine,
`src/structural/simulation.run_simulation`, that takes a `Scenario` (the environment). nb43's
phlogiston experiment is `run_simulation(phlogiston_scenario(...), …)`; this cosmology experiment
is the *same call* with `cosmology_scenario(...)` plugged in. (The phlogiston scenario reproduces
nb43 **byte-for-byte** — the regression gate that let us build cosmology on top with confidence.)

**The world (3 epochs).** `n_steps = 180`, transitions at `T1 = 60` and `T2 = 120`:

| epoch | steps | true theory | what couples the anomalies |
|------:|:-----:|:------------|:---------------------------|
| 0 | `[0, 60)`   | **dark matter**        | anomalies ↘ *dark-matter* commitment |
| 1 | `[60, 120)` | **modified gravity**   | anomalies ↘ *modified-gravity* commitment |
| 2 | `[120, 180)`| **scale-variant laws** | anomalies ↘ *scale-variant* commitment |

Because each theory couples the rotation-curve and large-scale-structure anomalies to a
**different** commitment, "which structure should be kept" changes every epoch — a population that
over-committed in epoch 0 has lost the structure epoch 1 needs.

**Two complementary lenses (neither is "the good one").**
* **Lens B — population tracking + levers.** `N≈150` agents in three heterogeneous communities
  observe the changing world and pool precision over a trust graph. We read out *which theory each
  community holds over time* and sweep the two diversity levers as a 2×2.
* **Lens A — genuine node re-growth.** A host-loop agent on the same world where epoch 2 carries a
  genuinely **unconceived** `dark_energy` node its menu lacks. The witness that structure can be
  re-grown as a *node*, not just re-weighted.

> ### ⚠️ Disclaimers — read before interpreting anything below
> 1. **This is a stylized Gaussian model, not real cosmology inference.** "dark matter / modified
>    gravity / scale-variant laws" are three *prior topologies* on a 6-node toy basis. Nothing here
>    is evidence about actual cosmology; it is a model of *how committed communities track a moving
>    target*.
> 2. **No forgetting (by design).** Agents accumulate evidence as in nb43 — there is no forgetting
>    mechanism of any kind. With pure accumulation, early-epoch evidence is heavy by later epochs,
>    so absolute tracking **lags by ~one epoch** and late-epoch tracking is harder *by
>    construction*. Read the lever comparison as **relative** diversity-preservation, not as
>    absolute correctness.
> 3. **Tracking ≠ being right.** A disconnected community that ends on the eventually-true theory is
>    *committed* to it, not *correct* — it would hold that theory regardless of the evidence.
> 4. **Edge structure is operator-set here.** In this fixed-operator linear-Gaussian model the
>    learned *couplings* accumulate uniformly (the Fisher deposit `HᵀH` is set by the experiment,
>    not the data), so theory identity rides the **means**. Genuine data-driven structure re-growth
>    is what **Lens A** isolates.""")

code(r"""%matplotlib inline
import sys, pathlib
ROOT = pathlib.Path.cwd()
ROOT = ROOT if (ROOT / 'scripts').exists() else ROOT.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.dpi': 120, 'axes.grid': True, 'grid.alpha': 0.3})

from src.structural import scenarios as sc
# Lens B helpers (the simulation + readouts live in the re-runnable script)
from scripts.run_cosmology_tracking import (
    run_cell, held_by_community, CELLS, COMM_LABELS, HOMES, EPOCH_NAMES,
    T1, T2, N_STEPS, CONVICTION_LEVEL, INTER_CONNECTED)
# Lens A helper
from scripts import run_cosmology_regrowth as la

RESULTS = ROOT / 'results' / 'cosmology_tracking'
RESULTS.mkdir(parents=True, exist_ok=True)
(ROOT / 'results' / 'cosmology_regrowth').mkdir(parents=True, exist_ok=True)

scn = sc.cosmology_scenario(n_steps=N_STEPS, t1=T1, t2=T2, sigma_o=0.5)
theory_mu = sc.cosmology_theory_means()       # (3, 6) the three candidate theories' means
print('cosmology epochs:', EPOCH_NAMES, ' transitions at T1=%d, T2=%d, n_steps=%d' % (T1, T2, N_STEPS))
print('communities:', COMM_LABELS, ' conviction homes:', HOMES)""")

# ----------------------------------------------------------------------
md(r"""## The world: three theories the truth cycles through

Each epoch's "theory" is a full Gaussian Bayes net (an existing `landscape_presets` preset). The
agents only ever *see* the theory's **mean** vector (the world's commitments, read through a noisy
observation operator). Below are the three candidate-theory means the populations are tracking
toward — note they differ sharply on the three rival commitments (dark-matter / scale-variant /
modified-gravity), which is what makes "which theory do I hold" a clean read-out.""")

code(r"""names = scn.names
fig, ax = plt.subplots(figsize=(10, 3.2))
im = ax.imshow(theory_mu, aspect='auto', cmap='RdBu_r', vmin=-1.6, vmax=1.6)
ax.set_yticks(range(3)); ax.set_yticklabels([n.replace('_',' ') for n in EPOCH_NAMES])
ax.set_xticks(range(len(names))); ax.set_xticklabels([n.replace('_',' ') for n in names], rotation=40, ha='right', fontsize=8)
for e in range(3):
    for k in range(len(names)):
        ax.text(k, e, f'{theory_mu[e,k]:.1f}', ha='center', va='center', fontsize=7)
ax.set_title('the three candidate cosmology theories (mean commitment vector per epoch)')
fig.colorbar(im, ax=ax, shrink=0.8, label='prior mean')
plt.tight_layout(); plt.show()""")

# ----------------------------------------------------------------------
md(r"""## Lens B — the population and the two levers

`N≈150` agents in **three communities** that differ in:
* **conservatism** — the prior precision (stiffness) of their belief, sampled per community
  (`conservative` ≫ `moderate` ≫ `frontier`); a stiffer prior moves later.
* **conviction** — *when the conviction lever is on*, a per-community value tilt
  `h ← h + λ·U` toward that community's **home theory**. The `conservative` bloc is committed to the
  old incumbent (**dark matter**), the `frontier` bloc to the eventual truth (**scale-variant**).

Each step every agent **observes** the current world and **fuses** precision over the trust graph
(`step.fuse` — the codebase's literal representation of learning from peers). The two diversity
levers:

* **(a) disconnection** — drop the cross-community bridge density to 0 (`λ₂ → 0`). Preserves
  diversity by *blocking transmission*.
* **(b) conviction** — the per-community tilt above. Preserves diversity by *blocking revision*.

We run all four cells — **neither / disconnection / conviction / both** — and read out **which
theory each community holds at each step** (project its mean belief onto the three candidate
theories). The grey staircase is the true theory; a curve that follows it is *tracking*, one that
freezes on an earlier level is *stuck*.""")

code(r"""# run the four lever cells at one seed (the scripts run multi-seed + controls; here seed 0 for the figures)
rep = {label: run_cell(scn, disconnect=disc, conviction=conv, seed=0)[0] for (label, disc, conv) in CELLS}
cid = rep['both']['pop']['cluster_id']

fig, axs = plt.subplots(2, 2, figsize=(13.5, 9), sharex=True, sharey=True)
cols = ['crimson', 'darkorange', 'seagreen']
for ax, (label, disc, conv) in zip(axs.ravel(), CELLS):
    r = rep[label]['r']; snap_t = r['snap_t']
    true_per_snap = np.asarray(scn.epoch_t)[snap_t]
    held = held_by_community(r, cid, theory_mu)
    ax.step(snap_t, true_per_snap, where='post', color='0.35', lw=5, alpha=0.30, label='TRUE theory')
    for k in range(held.shape[0]):
        ax.plot(snap_t, held[k] + 0.05*(k-1), 'o-', color=cols[k], lw=1.8, ms=4,
                label=f'{COMM_LABELS[k]} (home {HOMES[k][:4]})')
    for tt in (T1, T2): ax.axvline(tt, color='k', ls=':', lw=0.8)
    ax.set_title(f"{label}   (λ₂={r['lambda2']:.2f})", fontsize=11)
    ax.set_yticks([0,1,2]); ax.set_yticklabels([n.replace('_',' ') for n in EPOCH_NAMES], fontsize=8)
axs[0,0].set_ylabel('held theory'); axs[1,0].set_ylabel('held theory')
axs[1,0].set_xlabel('step'); axs[1,1].set_xlabel('step')
axs[0,0].legend(fontsize=7.5, loc='center left')
fig.suptitle('Lens B — which theory each community holds over time (tracking vs stuck)', fontsize=13)
plt.tight_layout(); plt.savefig(RESULTS / 'F1_tracking_2x2.png'); plt.show()

for label, _, _ in CELLS:
    held = held_by_community(rep[label]['r'], cid, theory_mu)
    print(f"{label:>14}: end-theories = {list(int(x) for x in held[:,-1])}  (0=dark,1=modi,2=scale)")""")

# ----------------------------------------------------------------------
md(r"""### Reading the 2×2 — the levers fail *differently*

* **neither** — all three communities **re-track** the world (with the built-in ~one-epoch lag),
  ending together on the latest theory. Re-tracking *works* by default; the cost is that the
  communities homogenize (no diversity left).
* **disconnection only** — they still all re-track to the same theory (everyone observes the same
  world, so the shared signal dominates), but the *structural* diversity in their precision is
  preserved (next figure). Disconnection alone does **not** produce theory-divergence.
* **conviction only** — competing convictions get **washed out by fusion**: connected agents
  average their tilts away and the population drifts to a single compromise theory. Conviction
  *alone* does not preserve diversity — the **no-pool degeneracy** this repo keeps finding (naive
  posterior fusion erases heterogeneity) bites the conviction lever too.
* **both** — *persistent lock-in*: the `conservative` bloc, committed to dark matter **and** shielded
  by disconnection, **stays stuck on the old theory forever** (it stays *wrong*), while the other
  blocs settle on their own homes → the communities end on **three different theories** and never
  reconcile.

**The "do we need both?" answer is yes.** Conviction supplies the *pull* that resists revision;
disconnection supplies the *shield* that stops fusion from dissolving it. Either alone re-tracks;
only together do you get a durable, fragmented lock-in. The two failure modes the model exhibits —
*"communities track different theories and never converge"* (disconnection's signature) and *"a
committed bloc stays wrong"* (conviction's signature) — only co-occur in the **both** cell.""")

# ----------------------------------------------------------------------
md(r"""## Structural diversity over time

Beyond *which theory* each community holds, we track **residual structural disagreement**
(`shells.residual_disagreement`: the mean Frobenius spread of the agents' precision matrices). It
falls to ~0 when the population reaches consensus and stays high when communities never reconcile —
the *fragmented field* signature of lock-in.""")

code(r"""fig, ax = plt.subplots(figsize=(8.5, 4.8))
colmap = {'neither':'0.5', 'disconnection':'steelblue', 'conviction':'darkorange', 'both':'crimson'}
for label, _, _ in CELLS:
    r = rep[label]['r']
    ax.plot(r['snap_t'], r['disagreement_t'], 'o-', lw=2, color=colmap[label], label=label)
for tt in (T1, T2): ax.axvline(tt, color='k', ls=':', lw=0.8)
ax.set_xlabel('step'); ax.set_ylabel('residual structural disagreement')
ax.set_title('structural diversity (0 = consensus/diversity lost; high = fragmented field)')
ax.legend(fontsize=9); plt.tight_layout(); plt.savefig(RESULTS / 'F2_diversity.png'); plt.show()
print('final disagreement:', {l: round(float(rep[l]['r']['disagreement_t'][-1]),2) for l,_,_ in CELLS})""")

md(r"""Note the split: **disconnection** keeps disagreement high (≈3) whether or not conviction is on
— it is the lever that physically preserves diversity — while the two **connected** cells collapse
to ≈0. This is the honest refinement of the plan's hypothesis: *disconnection preserves structural
diversity; conviction converts that diversity into a persistent theory-divergence only when
disconnection protects it from fusion.*""")

# ----------------------------------------------------------------------
md(r"""## ⚠️ Honest structural caveat — the contested *edges* don't discriminate the theory

We *designed* the observation operator to be **relational** (an "anomaly-balance" row per rival
commitment) so the contested anomaly↔commitment couplings could be *learned* from data. But in a
fixed-operator linear-Gaussian model the precision deposit is `J = HᵀH/σ²` — set by the
*experiment*, not by the *data*. So every contested edge accumulates **the same** off-diagonal
Fisher regardless of which theory is true, swamping the tiny (±0.4) prior coupling differences. The
plot below shows the six contested couplings all pile up together.

**Consequence:** theory identity is carried by the **means**, not by the learned couplings — which
is exactly why Lens B is honestly *means-tracking*, and why genuine data-driven structure re-growth
needs the different mechanism of Lens A (below).""")

code(r"""r_both = rep['both']['r']; idx = {n:i for i,n in enumerate(scn.names)}
learned = np.array([r_both['snap_Pi'][-1][:, idx[a], idx[c]].mean() for (a,c) in scn.edges])
true_off = sc.cosmology_true_couplings()   # (3, 6) true coupling per epoch
fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 4.2))
elabels = [f'{a[:4]}→{c.split("_")[0][:4]}' for (a,c) in scn.edges]
a0.bar(range(len(learned)), np.abs(learned), color='slategray')
a0.set_xticks(range(len(learned))); a0.set_xticklabels(elabels, rotation=40, ha='right', fontsize=7)
a0.set_ylabel('|learned Pi[a,c]| (final)'); a0.set_title('learned couplings accumulate ~UNIFORMLY (operator-set)')
im = a1.imshow(true_off, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
a1.set_yticks(range(3)); a1.set_yticklabels([n[:9] for n in EPOCH_NAMES])
a1.set_xticks(range(len(elabels))); a1.set_xticklabels(elabels, rotation=40, ha='right', fontsize=7)
a1.set_title('the TRUE couplings differ per epoch (±0.4) — but are swamped'); fig.colorbar(im, ax=a1, shrink=0.8)
plt.tight_layout(); plt.savefig(RESULTS / 'F3_structural_caveat.png'); plt.show()
print('learned |coupling| (should be ~equal across all 6):', np.round(np.abs(learned),1))""")

# ----------------------------------------------------------------------
md(r"""## Lens A — genuine node re-growth: an *unconceived* `dark_energy` node

Lens B's edges can't discriminate the theory, so where *does* structure get genuinely re-grown? In
the **node** that the menu lacks. Here the same cosmology world carries, in **epoch 2** (`t ≥ T2`),
a hidden `dark_energy` node that couples three commitments the agent's 6-node menu holds
independent. The agent can't represent it as one of its nodes — so its footprint shows up as a
**coherent pattern in the prediction errors** (errors taken against the *current epoch's* theory
mean, so the theory flips at T1/T2 are subtracted out and only the unconceived shift remains).

The agent reads the residual floor `λ_max(R)` from its recent errors; when it clears a trigger and
*holds*, the agent **wakes a new node** (`action.wake_hub`) — the paradigm grows a dimension. The
witness: the wake fires **after** `T2`, never before, and not at all when there is no hidden cause.""")

code(r"""rep_a = la.single_run(coupling=1.6, seed=0)
f, dF = rep_a['floor_t'], rep_a['delta_F_t']
na, nw = rep_a['n_active_t'], rep_a['n_woken_t']; ws = rep_a['wake_step']
fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
axs[0].plot(f, lw=2); axs[0].axhline(la.TRIGGER, color='grey', ls=':', label='trigger'); axs[0].legend(fontsize=8)
axs[0].set_title('residual floor (the TRIGGER)'); axs[0].set_ylabel(r'$\lambda_{max}(R)$')
axs[1].plot(dF, lw=2); axs[1].axhline(0, color='grey', lw=1); axs[1].set_title('expansion model Bayes factor'); axs[1].set_ylabel(r'$\Delta F$')
axs[2].step(range(len(na)), na, where='post', lw=2, label='true active causes')
axs[2].step(range(len(nw)), nw, where='post', lw=2, ls='--', label='nodes woken (agent)')
axs[2].set_yticks([0,1]); axs[2].set_title('agent grows a node when the world does'); axs[2].legend(fontsize=8)
for ax in axs:
    ax.axvline(la.T2, color='purple', lw=1, alpha=0.6); ax.set_xlabel('step')
    if ws>=0: ax.axvline(ws, color='seagreen', ls='--', lw=1.5)
plt.tight_layout(); plt.savefig(RESULTS.parent / 'cosmology_regrowth' / 'F4_regrowth.png'); plt.show()
print(f"wake step = {ws} (T2={la.T2}); woke AFTER T2 = {rep_a['woke_after_T2']}; "
      f"drive-edge F1: {rep_a['f1_unwoken']:.2f} (no node) -> {rep_a['f1_woken']:.2f} (woken)")""")

code(r"""# discovery threshold: how strong must the unconceived coupling be to be discovered?
couplings = [0.0, 0.4, 0.8, 1.2, 1.6]
wf = []
for cpl in couplings:
    woke = [la.single_run(coupling=cpl, seed=s)['wake_step'] >= 0 for s in (0,1,2)]
    wf.append(np.mean(woke))
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(couplings, wf, 'o-', lw=2, color='seagreen'); ax.set_ylim(-0.05, 1.08)
ax.set_xlabel('unconceived coupling strength'); ax.set_ylabel('fraction of seeds that discover it')
ax.set_title('Lens A — discovery threshold for the unconceived dark_energy node')
plt.tight_layout(); plt.show()
print('wake fraction by coupling:', dict(zip(couplings, [round(float(x),2) for x in wf])))""")

# ----------------------------------------------------------------------
md(r"""## Verdict

A changing-cosmology world on the **same reusable engine** as nb43 (phlogiston is just a different
`Scenario`) lets us ask whether communities keep re-tracking a moving truth, and which diversity
lever keeps them able to:

* **Lens B / tracking.** By default communities **re-track** each successive theory (lagged by the
  no-forgetting ratchet). The two diversity levers fail *differently*: **disconnection** preserves
  structural diversity but communities still track the shared world; **conviction** alone washes out
  under fusion; only **both together** produce a durable lock-in where a committed bloc stays on the
  old theory and the field fragments onto different theories.
* **Lens A / node re-growth.** Genuine data-driven structure learning shows up not in the contested
  edges (operator-set in this model) but in **waking a node** for an *unconceived* cause — which the
  agent does **after** that cause appears, never before, with a clean discovery threshold.

**What we can and cannot conclude.** We can say: *in this stylized model, re-tracking is the default,
and durable lock-in requires both a conviction that resists revision and a disconnection that shields
it from fusion.* We **cannot** say anything about real cosmology, nor that a "stuck" community is
wrong because of conviction alone, nor that a community tracking the eventually-true theory is right
(it may be committed by luck). And because there is **no forgetting**, absolute late-epoch tracking
is hard by construction — the lever comparison is **relative**. Those caveats are the honest frame
for everything above.""")

nb["cells"] = C
out = pathlib.Path(__file__).resolve().parent / "44_cosmology_tracking_levers.ipynb"
nbf.write(nb, str(out))
print("wrote", out)
