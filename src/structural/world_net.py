"""The world as a true latent Bayes net — so the residual EMERGES instead of being planted.

nb37's structure-learning pipeline (action.propose_hub -> expansion_score -> wake_hub) was
exercised with a hand-injected residual (``plant_common_cause``). This module supplies the honest
alternative: a true generative ``LinearGaussianBN`` carrying a HIDDEN node the agents' paradigm
menu does not represent. The population samples observations from it; the hidden node's couplings
push a *coherent shift* onto the commitments it drives, and that shift is the residual — it grows
as data accumulate, with no planting.

Two facts about this linear-Gaussian model fix the design (see the docstring of ``action.py``):
  * ``fisher_deposit`` gives ``J = HᵀH/σ²`` — the precision deposit is set by the observation
    operator, not the data. So under NODE-WISE observation the hidden cause never appears in the
    posterior ``Π`` off-diagonal; its footprint lives in the MEANS / prediction errors. Hence the
    latent carries a non-zero mean (a coherent shift), and the trigger is read from the running
    prediction-error covariance (``action.residual_from_errors``), not from ``Π``.
  * a hidden common cause is recoverable only up to a parsimony preference. So we measure structure
    recovery on the OBSERVED-MARGINAL precision among the driven nodes (permutation-equivariant),
    comparing the learned net to the Schur marginal of the TRUE net — never on the order-dependent
    CPD ``B``.

The default driven set is three hub-neighbours the phlogiston prior holds MUTUALLY independent
(``calx``/mass-law aside, these couple only to the hidden ``phlogiston`` hub, never to each other),
so the induced correlation is genuinely *unconceived* — no edge within the menu can hold it.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from src.structural import bmr, phlogiston as ph
from src.structural.bayesnet import LinearGaussianBN
from src.structural.belief import GaussianBeliefNet
from src.structural.phlogiston import StructuralConfig


# Three commitments the phlogiston prior couples ONLY to the hidden hub, never to each other
# (verified zero mutual prior coupling in nb37) -- the clean "unconceived" set for a latent.
UNCONCEIVED_DRIVES: tuple[str, ...] = (
    "reduction_with_charcoal", "respiration_like_combustion", "air_has_capacity",
)


@dataclass(frozen=True)
class LatentWorldConfig:
    """The hidden cause the agents' menu lacks."""
    latent_coupling: float = 1.0     # weight from the latent onto each driven node
    latent_mean: float = 1.5         # the latent's mean (=> a COHERENT shift on the driven nodes)
    latent_var: float = 1.0          # the latent's residual variance (=> co-fluctuation strength)
    world_var: float = 1.0           # residual variance of the observed nodes
    drives: tuple[str, ...] = UNCONCEIVED_DRIVES
    latent_name: str = "hidden_cause"


def true_phlogiston_world(cfg: StructuralConfig, t: int,
                          lw: LatentWorldConfig) -> LinearGaussianBN:
    """The TRUE generative net: the measured commitments at their regime values, each its OWN root
    (independent), PLUS one HIDDEN node ``lw.latent_name`` parenting ``lw.drives`` at weight
    ``lw.latent_coupling``. The latent is the ONLY source of coupling among the driven nodes, so the
    induced correlation is genuinely unconceived: an agnostic agent whose model lacks the latent
    (``agnostic_prior``) holds the drives independent, and the wake that adds the latent is a clean
    structure recovery. The latent sits last; its children are at lower indices so ``B`` is not
    triangular -- fine, the world is only sampled / compiled (general solves), never ``from_info``-ed.

    Intercepts are set DIRECTLY (no ``b=(I-B)μ*`` compensation), so the latent's mean is not
    cancelled: the driven nodes' marginal means become ``phi_true_at[drive] +
    latent_coupling·latent_mean`` -- the coherent, unconceived shift the menu cannot predict.
    ``t`` selects the regime via ``phi_true_at`` (use ``t_shift > n_steps`` for a stationary world,
    so the residual is pure accumulation, not a regime flip -- the validity control)."""
    names = cfg.node_names + (lw.latent_name,)
    d = len(cfg.node_names)
    idx = {n: i for i, n in enumerate(cfg.node_names)}

    B = jnp.zeros((d + 1, d + 1))
    for n in lw.drives:                       # driven node <- latent (col d), an upper entry
        B = B.at[idx[n], d].set(lw.latent_coupling)
    phi = np.asarray(ph.phi_true_at(cfg, t))                       # (d,) regime truth
    b = jnp.asarray(np.concatenate([phi, [lw.latent_mean]]))       # NO compensation: shift stays
    s = jnp.asarray(np.concatenate([np.full(d, lw.world_var), [lw.latent_var]]))
    return LinearGaussianBN(B=B, b=b, s=s, names=names)


@dataclass(frozen=True)
class CauseSpec:
    """One hidden cause in a non-stationary schedule: it switches ON at ``activate`` and drives
    ``drives`` at ``coupling`` with mean ``mean`` (a coherent shift on its set)."""
    activate: int
    drives: tuple[str, ...]
    coupling: float = 1.2
    mean: float = 1.5
    name: str = "cause"


def scheduled_world(cfg: StructuralConfig, t: int, causes: tuple[CauseSpec, ...],
                    latent_var: float = 1.0, world_var: float = 1.0) -> LinearGaussianBN:
    """A NON-STATIONARY true world: the measured commitments (independent roots at their regime
    values) plus one hidden node PER CAUSE that has switched on by step ``t`` (``activate <= t``),
    each driving its own ``drives`` set. The true structural STATE changes as causes activate, so
    the number of hidden nodes the agent must discover grows over the schedule -- the baseline that
    asks whether structure learning *tracks a changing truth* (makes more nodes when the world does).

    Disjoint ``drives`` sets keep the causes separately identifiable. ``t`` only gates which causes
    are present; means/couplings are the cause specs. Latents are appended in cause order."""
    active = [c for c in causes if c.activate <= t]
    names = cfg.node_names + tuple(c.name for c in active)
    d = len(cfg.node_names)
    idx = {n: i for i, n in enumerate(cfg.node_names)}

    B = jnp.zeros((d + len(active), d + len(active)))
    for li, c in enumerate(active):
        for n in c.drives:
            B = B.at[idx[n], d + li].set(c.coupling)
    phi = np.asarray(ph.phi_true_at(cfg, t))
    b = jnp.asarray(np.concatenate([phi, [c.mean for c in active]]))
    s = jnp.asarray(np.concatenate([np.full(d, world_var), np.full(len(active), latent_var)]))
    return LinearGaussianBN(B=B, b=b, s=s, names=names)


def agnostic_prior(cfg: StructuralConfig, lw: LatentWorldConfig,
                   t: int = 0, prec: float = 1.0) -> GaussianBeliefNet:
    """The agent's belief: the measured commitments held INDEPENDENT at their regime means -- a model
    that simply lacks the hidden cause (the cleanest "unconceived dimension" setup). ``Pi = prec·I``,
    means ``phi_true_at(cfg, t)``. Used as both the agent's initial net and the parent ``net_prior``
    in ``expansion_score``, so recovery is well-posed: the un-woken agent has NO drive coupling, the
    true world has the latent's, and the wake fills exactly that gap."""
    d = len(cfg.node_names)
    Pi = prec * jnp.eye(d)
    mu = jnp.asarray(np.asarray(ph.phi_true_at(cfg, t)))
    return GaussianBeliefNet(Pi=Pi, h=Pi @ mu, names=cfg.node_names)


def sample_world(world: LinearGaussianBN, cfg: StructuralConfig,
                 key: jax.Array) -> jax.Array:
    """One observation of the MEASURED commitments (every node except the hidden ``phlogiston`` hub
    AND the latent), in ``ph.measured_nodes(cfg)`` row order — i.e. aligned to the rows of
    ``ph.H_observable(cfg)``. The latent is sampled and its effect carried into the driven nodes,
    then dropped: the agent sees only the shifted/correlated observables, never the cause."""
    x = world.sample(key, 1)[0]                                    # (d+1,)
    meas = ph.measured_nodes(cfg)
    meas_idx = jnp.asarray([world.names.index(n) for n in meas])
    return x[meas_idx]                                             # (m,)


# ----------------------------------------------------------------------
# Structure recovery: learned net vs the TRUE net, on the driven-node marginal.
# ----------------------------------------------------------------------

def _drive_marginal_offdiag(net: GaussianBeliefNet, drives: tuple[str, ...]) -> np.ndarray:
    """Schur-marginalize ``net`` down to ``drives`` and return the off-diagonal coupling block
    among them — the induced partial correlations a hidden common cause leaves (or fails to)."""
    drop = tuple(n for n in net.names if n not in set(drives))
    marg = bmr.schur_marginalize(net, drop) if drop else net
    # reorder to the drives order
    idx = [marg.names.index(n) for n in drives]
    P = np.asarray(marg.Pi)[np.ix_(idx, idx)].copy()
    np.fill_diagonal(P, 0.0)
    return P


def recovery_scores(learned: GaussianBeliefNet, world: LinearGaussianBN,
                    drives: tuple[str, ...], edge_threshold: float = 0.05) -> dict:
    """How well ``learned`` recovers the hidden cause's footprint, scored on the OBSERVED-MARGINAL
    coupling among ``drives`` (permutation-equivariant; never on the CPD ``B``). The comparand is
    the Schur marginal of the TRUE world over its driven nodes — the induced coupling a perfect
    learner would hold. Returns ``{precision_frobenius, edge_f1, edge_precision, edge_recall}``;
    lower Frobenius / higher F1 = better recovery of the unconceived structure."""
    true_off = _drive_marginal_offdiag(world.to_info(), drives)
    learn_off = _drive_marginal_offdiag(learned, drives)
    fro = float(np.linalg.norm(learn_off - true_off))

    iu = np.triu_indices(len(drives), 1)
    true_e = np.abs(true_off[iu]) > edge_threshold
    learn_e = np.abs(learn_off[iu]) > edge_threshold
    tp = int((true_e & learn_e).sum()); fp = int((~true_e & learn_e).sum())
    fn = int((true_e & ~learn_e).sum())
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision_frobenius": fro, "edge_f1": f1,
            "edge_precision": prec, "edge_recall": rec}
