"""Order parameters for the categorical epistemic-POMDP population.

Mirrors the numpy-post-hoc style of ``src/observables.py`` but reads the
``step.run`` output (mean-belief trajectory + per-step ``infos`` + final beliefs)
rather than the continuous-substrate History.

Three order parameters (plan §6) plus a capture/asymmetry measure:
  1. paradigm occupancy        — fraction with MAP = a given paradigm, and the
                                 soft population mean belief.
  2. EFE / policy posterior     — population means of neg_efe, epistemic, and the
                                 belief-utility term; entropy of q(pi).
  3. polarization / bimodality  — bimodality of the per-agent belief distribution.
  4. capture / asymmetry        — did the population lock onto a paradigm, and is
                                 it the TRUE one or a wrong one (capture-against-
                                 evidence)?
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


# ----------------------------------------------------------------------
# 1. Occupancy
# ----------------------------------------------------------------------

def occupancy(final_q: np.ndarray, paradigm: int) -> float:
    """Fraction of agents whose MAP belief is ``paradigm``."""
    q = np.asarray(final_q)
    return float(np.mean(np.argmax(q, axis=1) == paradigm))


def mean_belief(final_q: np.ndarray, paradigm: int) -> float:
    """Soft population mean belief in ``paradigm``."""
    return float(np.asarray(final_q)[:, paradigm].mean())


# ----------------------------------------------------------------------
# 2. EFE / policy posterior
# ----------------------------------------------------------------------

def efe_summary(info: dict) -> dict:
    """Population-mean EFE decomposition at one step (from step.run infos[t])."""
    return {
        "neg_efe": float(np.mean(info["neg_efe"])),
        "epistemic": float(np.mean(info["epistemic"])),
        "belief_utility": float(np.mean(info["belief_utility"])),
    }


def policy_entropy(info: dict) -> float:
    """Mean Shannon entropy of the per-agent policy posterior q(pi) at one step.

    Low entropy => agents are decisive about which experiment to run (a
    motivated/committed population); high => exploratory/indifferent.
    """
    q_pi = np.asarray(info["q_pi"])                          # (N, A)
    H = -np.sum(q_pi * np.log(q_pi + EPS), axis=1)           # (N,)
    return float(np.mean(H))


def experiment_discriminability_chosen(info: dict, d: np.ndarray) -> float:
    """Mean theory-discriminability of the experiments actually chosen.

    d : (A,) per-experiment discriminability from gen_model.discriminability.
    Falling over time = the population is choosing less-informative (more
    theory-laden) experiments — a motivated-under-exploration signature.
    """
    return float(np.mean(np.asarray(d)[np.asarray(info["actions"])]))


# ----------------------------------------------------------------------
# 2b. Self-censorship (the emergent-experiment-avoidance order parameter)
# ----------------------------------------------------------------------

def discriminating_experiment_index(d: np.ndarray) -> int:
    """Index of the single most theory-separating experiment (argmax d)."""
    return int(np.argmax(np.asarray(d)))


def prob_on_experiment(info: dict, a_index: int) -> float:
    """Mean policy mass q(pi) on a specific experiment index across agents."""
    return float(np.asarray(info["q_pi"])[:, a_index].mean())


def prob_discriminating(info: dict, d: np.ndarray) -> float:
    """Mean policy mass on the most-discriminating experiment.

    This is the *derived* self-censorship readout: in the EFE model a confident
    agent assigns a low expected information gain to the decisive experiment, so
    once the experiment carries a cost this probability collapses — no gate is
    imposed. Compare across confident vs uncertain agents to see the avoidance
    emerge from belief alone.
    """
    return prob_on_experiment(info, discriminating_experiment_index(d))


# ----------------------------------------------------------------------
# 4b. Lock-in (did a confidently-wrong agent stay wrong?)
# ----------------------------------------------------------------------

def lockin_summary(run_out: dict, true_paradigm: int,
                   wrong_thresh: float = 0.5, tail: int = 1) -> dict:
    """Classify a single-bloc / population run for permanent lock-in.

    Reads the mean-belief trajectory in the TRUE paradigm:
      final_belief_true : mean q(true) over the last ``tail`` steps.
      locked_wrong      : final belief in the truth stayed below ``wrong_thresh``
                          (the population never crossed to the truth) — the
                          permanent-lock-in signature when the start was wrong.
      converged_truth   : final belief above (1 - wrong_thresh).
      crossed_at        : first step the mean belief crosses wrong_thresh upward,
                          or None if it never does (None = shift prevented).
    """
    m = np.asarray(run_out["mean_qB"])
    final = float(m[-tail:].mean())
    up = np.where(np.diff((m > wrong_thresh).astype(int)) > 0)[0]
    crossed_at = int(up[0]) + 1 if up.size else None
    return {
        "final_belief_true": final,
        "locked_wrong": bool(final < wrong_thresh),
        "converged_truth": bool(final > 1.0 - wrong_thresh),
        "crossed_at": crossed_at,
    }


# ----------------------------------------------------------------------
# 3. Polarization / bimodality
# ----------------------------------------------------------------------

def bimodality_coefficient(final_q: np.ndarray, paradigm: int = 1) -> float:
    """Sarle's bimodality coefficient of the per-agent belief in ``paradigm``.

    BC = (skew^2 + 1) / kurtosis(excess) ; BC > 5/9 ~ 0.555 suggests bimodality
    (a polarized population) vs a unimodal consensus.
    """
    x = np.asarray(final_q)[:, paradigm]
    n = x.size
    m = x.mean()
    s = x.std() + EPS
    g1 = np.mean(((x - m) / s) ** 3)                         # skewness
    g2 = np.mean(((x - m) / s) ** 4) - 3.0                   # excess kurtosis
    denom = g2 + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3) + EPS)
    return float((g1 ** 2 + 1.0) / (denom + EPS))


def polarization(final_q: np.ndarray, paradigm: int = 1) -> float:
    """Cross-agent std of belief — simple spread of opinion (0 = consensus)."""
    return float(np.asarray(final_q)[:, paradigm].std())


# ----------------------------------------------------------------------
# 4. Capture / asymmetry
# ----------------------------------------------------------------------

def capture(final_q: np.ndarray, true_paradigm: int,
            consensus_thresh: float = 0.9) -> dict:
    """Classify the population endpoint.

    Returns dict with:
      consensus  : True if >= consensus_thresh agents share a MAP paradigm.
      winner     : the MAP paradigm of the consensus (or -1 if none).
      to_truth   : winner == true_paradigm.
      against_evidence : consensus reached on a NON-true paradigm = capture
                         against the evidence (the headline failure mode).
    """
    q = np.asarray(final_q)
    maps = np.argmax(q, axis=1)
    K = q.shape[1]
    fracs = np.array([np.mean(maps == k) for k in range(K)])
    winner = int(np.argmax(fracs))
    consensus = bool(fracs[winner] >= consensus_thresh)
    return {
        "consensus": consensus,
        "winner": winner if consensus else -1,
        "winner_fraction": float(fracs[winner]),
        "to_truth": bool(consensus and winner == true_paradigm),
        "against_evidence": bool(consensus and winner != true_paradigm),
    }


def trajectory_summary(run_out: dict, cfg) -> dict:
    """One-call summary of a step.run output for quick notebook reporting."""
    final_q = run_out["final_q"]
    last = run_out["infos"][-1]
    cap = capture(final_q, cfg.true_paradigm)
    return {
        "mean_belief_true": mean_belief(final_q, cfg.true_paradigm),
        "occupancy_true": occupancy(final_q, cfg.true_paradigm),
        "polarization": polarization(final_q, cfg.true_paradigm),
        "bimodality": bimodality_coefficient(final_q, cfg.true_paradigm),
        "policy_entropy": policy_entropy(last),
        **{f"capture_{k}": v for k, v in cap.items()},
    }
