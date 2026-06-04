"""Lens A -- genuine node-wake structure re-growth on the changing cosmology world.

Lens B (``run_cosmology_tracking.py``) showed that in a fixed-operator linear-Gaussian model
the contested *edges* accumulate uniformly (Fisher ``HᵀH`` is operator-set, not data-set), so
theory identity rides the MEANS and Lens B is honestly means-tracking. This companion shows the
ONE place genuine data-driven structure re-growth happens: a hidden node the agent's menu does
not contain.

The world cycles through the three cosmology theories (dark matter -> modified gravity ->
scale-variant laws). In the **third** epoch (``t >= T2``) a genuinely *unconceived*
``dark_energy`` node switches on, coupling a set of commitments the 6-node cosmology menu holds
independent. A host-loop agent (the wake GROWS the basis, a variable-dimension move that cannot
live in ``jax.lax.scan``) observes node-wise, reads the residual floor from its recent
prediction errors (``action.residual_from_errors`` -- the footprint of a hidden cause lives in
the errors when ``Π`` stays diagonal), and scores ``{sample, reduce, expand}``. The witness:
the agent **wakes a new node AFTER T2, not before**, and the null (``coupling = 0``) never wakes.

This reuses ``world_net.scheduled_world`` / ``agnostic_prior`` / ``sample_world`` (now with the
optional ``phi_fn`` / ``measured_nodes_fn`` hooks so they run on the cosmology basis) and the
generic ``action.*`` ledger + ``world_net.recovery_scores``. Agents are independent
(cross-basis structural contagion is out of scope).

Run it::

    python scripts/run_cosmology_regrowth.py

Prints the mechanisms-in-a-row, the coupling sweep (discovery threshold + null), asserts the
null/positive controls, and writes ``simulation_arrays.npz`` + ``summary.json`` + a figure to
``results/cosmology_regrowth/``.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib                       # backend set to "Agg" inside main() (not at import,
import matplotlib.pyplot as plt         # so importing this module in a notebook keeps %inline)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.structural import world_net, action
from src.structural.belief import add_fisher
from src.structural.world import fisher_deposit
from src.structural.phlogiston import StructuralConfig
from src.structural.world_net import LatentWorldConfig, CauseSpec
from src.structural import scenarios as sc
from src.structural.landscape_presets import cosmology_basis


# ----------------------------------------------------------------------
# Configuration.
# ----------------------------------------------------------------------

T1, T2, N_STEPS = 60, 120, 180
SIGMA_O = 0.5
WINDOW = 30           # trailing error window (a changing world -> recent errors matter)
TRIGGER = 2.0         # residual-floor height to consider the wake
SUSTAIN = 5           # the floor must hold for this many steps (kills noise spikes)
WARMUP = 5            # ignore the small-sample transient at the very start
HUB_SELF_PREC = 2.0
HUB_NAME = "dark_energy"
# the unconceived node couples these (held independent on the agent's menu)
DRIVES = ("rotation_curve_anomaly", "large_scale_structure_anomaly", "law_coherence")

_BASIS = cosmology_basis()
_NAMES = _BASIS.names
_D = len(_NAMES)
_THEORY_MU = sc.cosmology_theory_means(_BASIS)


def _epoch_of(t: int) -> int:
    return 0 if t < T1 else (1 if t < T2 else 2)


def _phi_fn(cfg, t):
    """The base 6-node world truth at step t: the current epoch's theory mean."""
    return jnp.asarray(_THEORY_MU[_epoch_of(t)])


def _measured_nodes(cfg):
    """Every cosmology node is measured (no hidden hub on this basis)."""
    return tuple(cfg.node_names)


_CFG6 = StructuralConfig(node_names=_NAMES, n_steps=N_STEPS, sigma_o=SIGMA_O)
_H_NODE = jnp.eye(_D)                                     # node-wise observation operator
_DRIVE_IDX = [_NAMES.index(n) for n in DRIVES]


@dataclass(frozen=True)
class StepRecord:
    floor: float          # residual-floor height (the TRIGGER), windowed
    delta_F: float        # expansion model Bayes factor
    score: float
    accepted: bool
    n_active: int         # # true hidden causes active by now (0 before T2, 1 after)
    n_woken: int          # # nodes the agent has woken


# ----------------------------------------------------------------------
# One host-loop agent traversing the changing cosmology world.
# ----------------------------------------------------------------------

def single_run(coupling: float, *, seed: int = 0, latent_mean: float = 1.6,
               proposal_rate: float | None = None) -> dict:
    """Traverse the 3-epoch cosmology world on the fixed 6-node menu, scoring the expansion of
    an unconceived ``dark_energy`` node each step. The cause activates at ``T2`` (epoch 2) and
    drives ``DRIVES`` with a coherent shift; the agent reads the residual from its *recent*
    prediction errors (vs the current epoch's theory mean -- the menu's best prediction, so the
    theory flips at T1/T2 are subtracted out and only the unconceived shift remains).

    ``proposal_rate`` selects WHEN the agent attempts a structural edit:
      * ``None`` (default): the deterministic debounce -- wake when the floor clears the trigger
        for ``SUSTAIN`` consecutive steps (byte-identical to the original).
      * a float ``lambda``: structural-edit attempts arrive as a POISSON PROCESS (the paper's
        node-arrival model) -- each step an attempt arrives with prob ``1 - e^{-lambda}`` and the
        wake fires on the first arrival for which the floor already clears the trigger, so the
        discovery TIME is a random waiting time of rate ``lambda``. A SEPARATE arrival RNG leaves
        the world-sample stream (hence the floor trajectory) identical to the deterministic run --
        only the wake *timing* is Poisson-gated."""
    causes = (CauseSpec(activate=T2, drives=DRIVES, coupling=coupling,
                        mean=latent_mean, name=HUB_NAME),)
    prior = world_net.agnostic_prior(_CFG6, LatentWorldConfig(), t=0, phi_fn=_phi_fn)
    agent = prior
    errors: list[np.ndarray] = []
    key = jax.random.PRNGKey(seed)
    streak = 0
    wake_step = -1
    recs: list[StepRecord] = []
    f1_unwoken = f1_woken = float("nan")
    arr_rng = np.random.default_rng(seed + 10_007) if proposal_rate is not None else None
    p_arrival = (1.0 - np.exp(-proposal_rate)) if proposal_rate is not None else None

    for t in range(N_STEPS):
        wnet = world_net.scheduled_world(_CFG6, t, causes, world_var=1.0, phi_fn=_phi_fn)
        key, k1 = jax.random.split(key)
        o = np.asarray(world_net.sample_world(wnet, _CFG6, k1, measured_nodes_fn=_measured_nodes))
        # prediction error vs the current epoch's theory mean (the menu's best prediction)
        errors.append(o[_DRIVE_IDX] - np.asarray(_THEORY_MU[_epoch_of(t)])[_DRIVE_IDX])
        win = np.array(errors[-WINDOW:])
        R = action.residual_from_errors(jnp.asarray(win))
        prop = action.propose_hub(agent, DRIVES, residual=R)
        J, j = fisher_deposit(_H_NODE, jnp.asarray(o), SIGMA_O)
        agent = add_fisher(agent, J, j)
        scx = action.expansion_score(agent, prior, prop, HUB_NAME,
                                     hub_self_prec=HUB_SELF_PREC)
        eligible = (t >= WARMUP) and scx.accept and (prop.strength >= TRIGGER)
        if proposal_rate is None:                  # deterministic debounce (existing behaviour)
            streak = streak + 1 if eligible else 0
            fire = streak >= SUSTAIN
        else:                                      # Poisson arrival of a structural-edit attempt
            fire = bool(arr_rng.random() < p_arrival) and eligible
        n_active = int(t >= T2 and coupling > 0.0)
        if fire and wake_step < 0:
            wake_step = t
            woken = action.wake_hub(agent, prop, HUB_NAME, hub_self_prec=HUB_SELF_PREC)
            f1_unwoken = world_net.recovery_scores(agent, wnet, DRIVES)["edge_f1"]
            f1_woken = world_net.recovery_scores(woken, wnet, DRIVES)["edge_f1"]
        recs.append(StepRecord(floor=prop.strength, delta_F=scx.delta_F, score=scx.score,
                               accepted=eligible, n_active=n_active,
                               n_woken=int(wake_step >= 0 and t >= wake_step)))

    return {
        "floor_t": np.array([r.floor for r in recs]),
        "delta_F_t": np.array([r.delta_F for r in recs]),
        "score_t": np.array([r.score for r in recs]),
        "n_active_t": np.array([r.n_active for r in recs]),
        "n_woken_t": np.array([r.n_woken for r in recs]),
        "wake_step": wake_step,
        "f1_unwoken": f1_unwoken,
        "f1_woken": f1_woken,
        "woke_after_T2": bool(wake_step >= T2),
        "structure_recovered": bool(wake_step >= 0 and f1_woken > f1_unwoken),
    }


# ----------------------------------------------------------------------
# Reporting + figures.
# ----------------------------------------------------------------------

def mechanisms_in_a_row(tele: dict, coupling: float):
    f = tele["floor_t"]
    ws = tele["wake_step"]
    print("\n=== mechanisms in a row (representative: a strong unconceived dark_energy node) ===")
    print(f"  [world]  3 cosmology epochs; in epoch 2 (t>={T2}) an UNCONCEIVED '{HUB_NAME}' node "
          f"switches on, coupling {DRIVES} at {coupling}.")
    print(f"  [init]   the agent's 6-node menu holds those commitments INDEPENDENT -- no slot for "
          f"the cause.")
    print(f"  [t<T1]   epoch 0; residual floor ~ {f[WARMUP:T1].max():.2f} (< trigger {TRIGGER}: "
          f"errors vs the menu's prediction are just noise)")
    print(f"  [T1..T2] epoch 1 (theory flips, but errors are vs the CURRENT epoch mean -> still "
          f"just noise); floor max {f[T1:T2].max():.2f}")
    if ws >= 0:
        print(f"  [t>=T2]  the unconceived shift accumulates; the floor climbs to {f[T2:].max():.2f} "
              f"and the wake fires at t={ws} (= {ws - T2} steps after the node appears)")
        print(f"  [wake]   '{HUB_NAME}' is wired in -- the paradigm grows a NODE (not just an edge).")
        print(f"  [recover] drive-marginal edge-F1: {tele['f1_unwoken']:.2f} (no node) -> "
              f"{tele['f1_woken']:.2f} (woken)  structure_recovered={tele['structure_recovered']}")
    else:
        print("  [no wake] the floor never cleared the trigger (the null / no-cause control).")


def make_figure(tele: dict, path: Path):
    f, dF = tele["floor_t"], tele["delta_F_t"]
    na, nw = tele["n_active_t"], tele["n_woken_t"]
    ws = tele["wake_step"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    axs[0].plot(f, lw=2); axs[0].axhline(TRIGGER, color="grey", ls=":", lw=1, label="trigger")
    axs[0].set_title("residual floor (the TRIGGER)"); axs[0].set_ylabel(r"$\lambda_{max}(R)$")
    axs[0].legend(fontsize=8)
    axs[1].plot(dF, lw=2); axs[1].axhline(0, color="grey", lw=1)
    axs[1].set_title("expansion model Bayes factor"); axs[1].set_ylabel(r"$\Delta F$")
    axs[2].step(range(len(na)), na, where="post", lw=2, label="true active causes")
    axs[2].step(range(len(nw)), nw, where="post", lw=2, ls="--", label="nodes woken (agent)")
    axs[2].set_title("does the agent grow a node when the world does?")
    axs[2].set_yticks([0, 1]); axs[2].legend(fontsize=8)
    for ax in axs:
        ax.axvline(T2, color="purple", ls="-", lw=1, alpha=0.5)
        ax.text(T2 + 1, ax.get_ylim()[1] * 0.05, "T2 (dark_energy on)", color="purple", fontsize=7)
        if ws >= 0:
            ax.axvline(ws, color="seagreen", ls="--", lw=1.5)
        ax.set_xlabel("step")
    if ws >= 0:
        axs[0].text(ws, axs[0].get_ylim()[1] * 0.9, f" wake t={ws}", color="seagreen", fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close(fig)


# ----------------------------------------------------------------------
# main.
# ----------------------------------------------------------------------

def main() -> int:
    matplotlib.use("Agg")                  # batch (file-only) figures when run as a script
    out_dir = ROOT / "results" / "cosmology_regrowth"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- representative run + mechanisms in a row ----
    rep = single_run(coupling=1.6, seed=0)
    mechanisms_in_a_row(rep, coupling=1.6)
    make_figure(rep, out_dir / "cosmology_regrowth.png")

    # ---- coupling sweep: discovery threshold + the null ----
    couplings = [0.0, 0.4, 0.8, 1.2, 1.6]
    seeds = [0, 1, 2]
    print("\n=== coupling sweep (does dark_energy get discovered, and WHEN?) ===")
    wake_frac, mean_wake, after_T2_frac = [], [], []
    for cpl in couplings:
        woke, steps, after = [], [], []
        for s in seeds:
            te = single_run(coupling=cpl, seed=s)
            woke.append(te["wake_step"] >= 0)
            if te["wake_step"] >= 0:
                steps.append(te["wake_step"])
                after.append(te["woke_after_T2"])
        wf = float(np.mean(woke))
        wake_frac.append(wf)
        mean_wake.append(float(np.mean(steps)) if steps else float("nan"))
        after_T2_frac.append(float(np.mean(after)) if after else 1.0)
        mw = f"{np.mean(steps):.0f}" if steps else "--"
        print(f"  coupling={cpl}: wake_fraction={wf:.2f}  mean_wake_step={mw}  "
              f"all_after_T2={after_T2_frac[-1]:.2f}")

    # ---- controls ----
    null = single_run(coupling=0.0, seed=0)
    pos = single_run(coupling=1.6, seed=0)
    print("\n=== null / sanity controls ===")
    print(f"  null (coupling=0):      wake_step={null['wake_step']}  (expect -1, never)")
    print(f"  positive (coupling=1.6): wake_step={pos['wake_step']}  woke_after_T2={pos['woke_after_T2']}")
    print(f"  pre-T2 floor max (pos): {pos['floor_t'][WARMUP:T2].max():.2f}  (expect < trigger {TRIGGER})")

    assert null["wake_step"] == -1, \
        f"null world (no cause) spuriously woke at t={null['wake_step']}"
    assert pos["wake_step"] >= 0 and pos["woke_after_T2"], \
        f"positive case did not wake after T2 (wake={pos['wake_step']})"
    assert pos["floor_t"][WARMUP:T2].max() < TRIGGER, \
        "pre-T2 residual floor exceeded the trigger (would mis-fire before the cause exists)"
    assert wake_frac[0] == 0.0, f"coupling=0 had nonzero wake fraction {wake_frac[0]}"
    assert wake_frac[-1] > 0.9, f"strong cause coupling failed to discover (frac {wake_frac[-1]})"

    # ---- save ----
    np.savez_compressed(
        out_dir / "simulation_arrays.npz",
        couplings=np.array(couplings), seeds=np.array(seeds),
        wake_fraction=np.array(wake_frac), mean_wake_step=np.array(mean_wake),
        after_T2_fraction=np.array(after_T2_frac),
        rep_floor_t=rep["floor_t"], rep_delta_F_t=rep["delta_F_t"],
        rep_n_active_t=rep["n_active_t"], rep_n_woken_t=rep["n_woken_t"],
        rep_wake_step=np.array(rep["wake_step"]), T1=np.array(T1), T2=np.array(T2),
    )
    summary = {
        "config": {"T1": T1, "T2": T2, "n_steps": N_STEPS, "sigma_o": SIGMA_O,
                   "window": WINDOW, "trigger": TRIGGER, "sustain": SUSTAIN,
                   "drives": list(DRIVES), "hub_name": HUB_NAME},
        "representative": {"coupling": 1.6, "wake_step": rep["wake_step"],
                           "woke_after_T2": rep["woke_after_T2"],
                           "f1_unwoken": rep["f1_unwoken"], "f1_woken": rep["f1_woken"],
                           "structure_recovered": rep["structure_recovered"]},
        "sweep": {"couplings": couplings, "n_seeds": len(seeds),
                  "wake_fraction": wake_frac, "mean_wake_step": mean_wake,
                  "after_T2_fraction": after_T2_frac},
        "controls": {"null_wake_step": null["wake_step"],
                     "positive_wake_step": pos["wake_step"],
                     "positive_woke_after_T2": pos["woke_after_T2"],
                     "pre_T2_floor_max": float(pos["floor_t"][WARMUP:T2].max())},
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nsaved arrays / figure / summary to {out_dir}")
    print(f"HEADLINE: the agent grows a NODE for the unconceived dark_energy AFTER it appears "
          f"(wake t={rep['wake_step']} >= T2={T2}); the null never wakes -- genuine node re-growth.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
