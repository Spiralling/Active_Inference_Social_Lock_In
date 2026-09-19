"""Four-coordinate phlogiston/oxygen example for the camera-ready assessment.

The two candidates share observable commitments C (combustion), G (gas uptake)
and M (mass change). Their first coordinate has a candidate-specific meaning:
P (phlogiston release) in the incumbent and O (oxygen participation) in the
rival. ``z`` is a storage label only, not a claim that P and O are the same
scientific quantity. Across-candidate comparisons must use C, G and M only.

This is a deliberately schematic historical motivation, not a reconstruction
of either historical theory. All numerical coefficients below were specified
before executing the experiment protocol. The caller selects the recovered
publication ``src`` package before importing this module; the inference engine
itself is not modified here. No belief-utility term is present.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from src.structural.bayesnet import LinearGaussianBN


NAMES = ("z", "C", "G", "M")
CANDIDATE_NAMES = ("phlogiston", "oxygen")
CANDIDATE_NODE_NAMES = (("P", "C", "G", "M"), ("O", "C", "G", "M"))
OBSERVATION_NAMES = ("C", "G", "M")
DIAGNOSTIC_ROWS = (1, 2)
OBSERVABLE_IDX = (1, 2, 3)
GATE_IDX = OBSERVABLE_IDX


def build_world() -> dict:
    """Return the fixed pair of supplied linear-Gaussian candidate priors.

    In canonical storage order ``(z,C,G,M)``, edges are P->C (+.8),
    P->G (-.8), P->M (-.8), versus O->C (+.8), O->G (+.8), G->M (+.9).
    Both explanatory coordinates have residual variance .25; all observable
    coordinates have residual variance .5. Intercepts are computed from the
    intended joint means, rather than conflating CPD intercepts with means.

    H measures only C,G,M, so sensory data contain no direct reading of either
    explanatory entity. Direct observations deposit diagonal Fisher information;
    this setup tests selection among supplied models, not novel-edge discovery.
    """
    B = np.zeros((2, 4, 4), dtype=np.float64)
    B[0, 1, 0] = .8
    B[0, 2, 0] = -.8
    B[0, 3, 0] = -.8
    B[1, 1, 0] = .8
    B[1, 2, 0] = .8
    B[1, 3, 2] = .9
    mu = np.array([[0., 1., -1., -1.], [0., 1., 1., 1.]])
    b = np.einsum("kij,kj->ki", np.eye(4)[None] - B, mu)
    s = np.tile(np.array([.25, .5, .5, .5]), (2, 1))
    H = np.eye(4, dtype=np.float64)[1:]
    bns = tuple(LinearGaussianBN(B=jnp.asarray(B[k]), b=jnp.asarray(b[k]),
                                 s=jnp.asarray(s[k]), names=CANDIDATE_NODE_NAMES[k])
                for k in range(2))
    nets = tuple(bn.to_info() for bn in bns)
    covariance = np.stack([
        np.linalg.solve(np.eye(4) - B[k], np.eye(4))
        @ np.diag(s[k])
        @ np.linalg.solve(np.eye(4) - B[k], np.eye(4)).T
        for k in range(2)
    ])
    metadata = {
        "kind": "four-variable schematic phlogiston/oxygen candidate menu",
        "candidate_order": list(CANDIDATE_NAMES),
        "storage_names": list(NAMES),
        "candidate_node_names": [list(n) for n in CANDIDATE_NODE_NAMES],
        "observable_names": list(OBSERVATION_NAMES),
        "observable_indices": list(OBSERVABLE_IDX),
        "diagnostic_observation_rows": list(DIAGNOSTIC_ROWS),
        "gate_indices": list(GATE_IDX),
        "explanatory_coordinate": {
            "phlogiston": "P: phlogiston release",
            "oxygen": "O: oxygen participation",
            "comparison_rule": "Do not pool, subtract or interpret z across candidate frames.",
        },
        "observable_semantics": {
            "C": "combustion-related commitment (unchanged at the imposed world shift)",
            "G": "signed gas uptake commitment",
            "M": "signed mass-change commitment",
        },
        "edges": {
            "phlogiston": [["P", "C", .8], ["P", "G", -.8], ["P", "M", -.8]],
            "oxygen": [["O", "C", .8], ["O", "G", .8], ["G", "M", .9]],
        },
        "joint_means": mu.tolist(),
        "cpd_intercepts": b.tolist(),
        "residual_variances": s.tolist(),
        "observation_operator": H.tolist(),
        "interpretation_limits": [
            "Dimensionless schematic commitments, not quantitative historical chemistry.",
            "The two explanatory coordinates have different meanings; only C,G,M are shared.",
            "Piecewise-constant observable world with private observation noise; not new DAG samples every round.",
            "Post-shift pattern favours the oxygen candidate's reference mean; this is not recovery of a uniquely identified causal graph.",
            "Model structures are supplied, not discovered; direct observations add no off-diagonal Fisher information.",
            "Candidate priors differ in both means and covariance; successful selection alone does not isolate an effect of topology.",
            "No exact numerical equivalence to the accepted ten-variable example is claimed.",
        ],
        "coefficients_selected": "fixed before executing this assessment protocol",
        "belief_utility": "absent",
    }
    return {
        "Pi": jnp.stack([n.Pi for n in nets]),
        "h": jnp.stack([n.h for n in nets]),
        "mu": jnp.asarray(mu), "H": jnp.asarray(H),
        "B": jnp.asarray(B), "b": jnp.asarray(b), "s": jnp.asarray(s),
        "covariance": jnp.asarray(covariance), "names": NAMES,
        "candidate_names": CANDIDATE_NAMES, "candidate_node_names": CANDIDATE_NODE_NAMES,
        "bns": bns, "nets": nets, "disc_rows": DIAGNOSTIC_ROWS,
        "gate_idx": GATE_IDX, "observable_idx": OBSERVABLE_IDX,
        "metadata": metadata,
    }


def phi(t: int, t_shift: int = 80) -> np.ndarray:
    """Imposed observable pattern; z=0 is unobserved padding, not shared truth."""
    value = -1.0 if t < t_shift else 1.0
    return np.array([0.0, 1.0, value, value], dtype=np.float64)


def wall_truth() -> np.ndarray:
    """Mixed pattern: G=-1 and M=+1, so neither reference mean is fully right.

    Two camps attending to G versus M respectively receive locally conflicting
    evidence. Different predictive variances and structural dependencies mean
    this is not guaranteed to produce balanced evidence or enduring opposition.
    """
    return np.array([0.0, 1.0, -1.0, 1.0], dtype=np.float64)


def wall_attention(n: int = 16) -> np.ndarray:
    """First equal-sized camp measures only G; the second measures only M."""
    if not isinstance(n, (int, np.integer)) or n < 2 or n % 2:
        raise ValueError("n must be an even integer of at least two")
    attention = np.zeros((n, 3), dtype=np.float64)
    attention[:n // 2, 1] = 1.0
    attention[n // 2:, 2] = 1.0
    return attention
