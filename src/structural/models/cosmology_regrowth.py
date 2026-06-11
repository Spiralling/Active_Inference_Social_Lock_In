"""Cosmology regrowth model (Lens A) -- genuine node-wake structure re-growth on the changing world.

Lens B (``models.cosmology`` / ``cosmology_tracking``) showed that in a fixed-operator
linear-Gaussian model the contested *edges* accumulate uniformly (Fisher ``HᵀH`` is operator-set,
not data-set), so theory identity rides the MEANS. This companion holds the ONE place genuine
data-driven structure re-growth happens: a hidden node the agent's menu does not contain.

In the **third** epoch (``t >= T2``) a genuinely *unconceived* ``dark_energy`` node switches on,
coupling a set of commitments the 6-node cosmology menu holds independent. A host-loop agent (the
wake GROWS the basis -- a variable-dimension move that cannot live in ``jax.lax.scan``) observes
node-wise, reads the residual floor from its recent prediction errors, and scores
``{sample, reduce, expand}``. The witness: the agent wakes a new node AFTER T2, not before, and the
null (``coupling = 0``) never wakes.

This is the **pure science library** for the cosmology-regrowth family (``cosmology_regrowth`` and
its dependent ``cosmology_poisson``): the world/agent constants, the ``single_run`` host loop, and
the helpers -- no matplotlib, no file I/O, no ``main``. The reporting, figures, and the coupling
sweep live in ``experiments/cosmology_regrowth.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp

from src.structural import world_net, action
from src.structural.belief import add_fisher
from src.structural.world import fisher_deposit
from src.structural.phlogiston import StructuralConfig
from src.structural.world_net import LatentWorldConfig, CauseSpec
from src.structural import scenarios as sc
from src.structural.landscape_presets import cosmology_basis

__all__ = [
    "T1", "T2", "N_STEPS", "SIGMA_O", "WINDOW", "TRIGGER", "SUSTAIN", "WARMUP",
    "HUB_SELF_PREC", "HUB_NAME", "DRIVES", "StepRecord", "single_run",
]

# ---- configuration: the changing world + the wake debounce ----
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


def single_run(coupling: float, *, seed: int = 0, latent_mean: float = 1.6,
               proposal_rate: float | None = None) -> dict:
    """Traverse the 3-epoch cosmology world on the fixed 6-node menu, scoring the expansion of an
    unconceived ``dark_energy`` node each step. The cause activates at ``T2`` (epoch 2) and drives
    ``DRIVES`` with a coherent shift; the agent reads the residual from its *recent* prediction
    errors (vs the current epoch's theory mean -- so the theory flips at T1/T2 are subtracted out and
    only the unconceived shift remains).

    ``proposal_rate`` selects WHEN the agent attempts a structural edit:
      * ``None`` (default): the deterministic debounce -- wake when the floor clears the trigger for
        ``SUSTAIN`` consecutive steps.
      * a float ``lambda``: structural-edit attempts arrive as a POISSON PROCESS (each step with
        prob ``1 - e^{-lambda}``) and the wake fires on the first arrival for which the floor already
        clears the trigger, so the discovery TIME is a random waiting time of rate ``lambda``. A
        SEPARATE arrival RNG leaves the world-sample stream (hence the floor trajectory) identical to
        the deterministic run -- only the wake *timing* is Poisson-gated."""
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
