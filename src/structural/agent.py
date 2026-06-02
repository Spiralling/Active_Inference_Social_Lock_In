"""Single-agent belief carrier and the candidate-paradigm evidence race.

The multi-agent loop lives in ``src/structural/step.py``; this module is the
clean, fusion-free driver for the *evidence race* between candidate paradigms,
the Gaussian-structural analogue of ``src/pomdp/agent_pop.py``. One agent
accumulates Fisher information from the regime-dependent world and, for each
candidate paradigm M (phlogiston, oxygen), tracks the running model
log-evidence

    logZ^M(t) = log_evidence(prior_M + accumulated Fisher) - log_evidence(prior_M),

whose drift rate is -KL(P || p_M) (note Eq. drift). The held paradigm is the
upper envelope argmax_M logZ^M(t); the crossing is the paradigm shift. Each
candidate shares the *same* accumulated Fisher (shared likelihood, the locked
design decision) and differs only by its prior --- so the whole race is driven
by how each prior meets the data.

It also runs the BMR check: phlogiston makes a falsifiable commitment (the calx
is *lighter*, mass-node mean ``mu_phlog_mass``). ``bmr_prune`` compares the
phlogiston model against a reduced model that sharpens that commitment to a
spike at its claimed value. ``Delta F > 0`` means the data *vindicate* the
commitment (sharpening it refunds complexity); ``Delta F < 0`` means the data
*refute* it. The sign flips from vindicated to refuted once the oxygen-regime
gravimetric data arrive -- the closed-form moment of falsification.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from src.structural import phlogiston as ph
from src.structural.phlogiston import StructuralConfig
from src.structural.belief import add_fisher
from src.structural.bmr import log_evidence, bmr_prune, prune_node_prior
from src.structural.world import fisher_deposit, fisher_deposit_weighted, sample_o


def evidence_race(cfg: StructuralConfig, key: jax.Array,
                  bmr_target: tuple[str, ...] = ph.DISAGREEMENT_NODES
                  ) -> dict:
    """Run a single agent through the regime schedule, tracking per-candidate
    running log-evidence and the BMR Bayes factor for abandoning phlogiston's
    mass commitment.

    Returns a dict of trajectories (length ``cfg.n_steps`` each):
      ``t``           : step index
      ``logZ_phlog``  : running model log-evidence of the phlogiston paradigm
      ``logZ_oxy``    : running model log-evidence of the oxygen paradigm
      ``gap``         : logZ_oxy - logZ_phlog (upper-envelope gap; sign = held)
      ``delta_F``     : BMR Bayes factor for sharpening phlogiston's commitment
                        on ``bmr_target`` to its claimed value (> 0 => data
                        vindicate the claim; < 0 => data refute it)
    """
    priors = ph.candidate_priors(cfg)
    posts = dict(priors)
    base = {k: float(log_evidence(v)) for k, v in priors.items()}
    phlog_prior = priors["phlogiston"]
    # sharpen phlogiston's commitment to its claimed value (calx lighter).
    reduced = prune_node_prior(phlog_prior, bmr_target, value=cfg.mu_phlog_mass)

    H = ph.H_observable(cfg)
    ts, lzp, lzo, gap, dF = [], [], [], [], []
    for t in range(cfg.n_steps):
        phi = ph.phi_true_at(cfg, t)
        key, sub = jax.random.split(key)
        o = sample_o(H, phi, cfg.sigma_o, sub)
        J, j = fisher_deposit(H, o, cfg.sigma_o)
        for k in posts:
            posts[k] = add_fisher(posts[k], J, j)

        zp = float(log_evidence(posts["phlogiston"])) - base["phlogiston"]
        zo = float(log_evidence(posts["oxygen"])) - base["oxygen"]
        d = float(bmr_prune(posts["phlogiston"], phlog_prior, reduced)["delta_F"])
        ts.append(t); lzp.append(zp); lzo.append(zo); gap.append(zo - zp); dF.append(d)

    return {
        "t": jnp.asarray(ts),
        "logZ_phlog": jnp.asarray(lzp),
        "logZ_oxy": jnp.asarray(lzo),
        "gap": jnp.asarray(gap),
        "delta_F": jnp.asarray(dF),
    }


def evidence_race_weighted(cfg: StructuralConfig, key: jax.Array,
                           rho: jax.Array,
                           bmr_target: tuple[str, ...] = ph.DISAGREEMENT_NODES
                           ) -> dict:
    """``evidence_race`` under a FIXED per-channel evidential precision ``rho`` (m,),
    shared by BOTH candidate paradigms -- the isolation experiment for the *multiplicative*
    evidential bias.

    Every deposit is ``fisher_deposit_weighted(H, o, sigma, rho)`` instead of the plain
    ``fisher_deposit``, so each observation channel contributes information scaled by its
    evidential precision ``rho_k``. Because the SAME weighted Fisher is added to both
    candidates (the locked shared-likelihood design), any difference in the running
    evidence gap is due to ``rho`` alone, not to the priors -- the cleanest isolation of
    "the agent weighs the rival's evidence using the incumbent's own precision metric".

      * ``rho = ones`` recovers ``evidence_race`` exactly (the back-compat anchor).
      * ``rho = ph.derived_channel_precision(cfg_high_g)`` is the incumbent's ``R(G)``:
        the rival's characteristic anomaly arrives on a silenced (``rho~0``) channel, so
        the oxygen-over-phlogiston gap never accumulates after the regime flip.

    Returns the same dict as ``evidence_race`` (``t, logZ_phlog, logZ_oxy, gap, delta_F``).
    """
    rho = jnp.asarray(rho)
    priors = ph.candidate_priors(cfg)
    posts = dict(priors)
    base = {k: float(log_evidence(v)) for k, v in priors.items()}
    phlog_prior = priors["phlogiston"]
    reduced = prune_node_prior(phlog_prior, bmr_target, value=cfg.mu_phlog_mass)

    H = ph.H_observable(cfg)
    ts, lzp, lzo, gap, dF = [], [], [], [], []
    for t in range(cfg.n_steps):
        phi = ph.phi_true_at(cfg, t)
        key, sub = jax.random.split(key)
        o = sample_o(H, phi, cfg.sigma_o, sub)
        J, j = fisher_deposit_weighted(H, o, cfg.sigma_o, rho)
        for k in posts:
            posts[k] = add_fisher(posts[k], J, j)

        zp = float(log_evidence(posts["phlogiston"])) - base["phlogiston"]
        zo = float(log_evidence(posts["oxygen"])) - base["oxygen"]
        d = float(bmr_prune(posts["phlogiston"], phlog_prior, reduced)["delta_F"])
        ts.append(t); lzp.append(zp); lzo.append(zo); gap.append(zo - zp); dF.append(d)

    return {
        "t": jnp.asarray(ts),
        "logZ_phlog": jnp.asarray(lzp),
        "logZ_oxy": jnp.asarray(lzo),
        "gap": jnp.asarray(gap),
        "delta_F": jnp.asarray(dF),
    }
