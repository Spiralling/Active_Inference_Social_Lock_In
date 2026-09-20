"""Combustion-mediated explanations for the revised camera-ready assessment.

The two candidates share observables C (reaction extent), G (net gas uptake)
and M (solid mass change). Their first coordinate has a candidate-specific meaning:
P (phlogiston released) in the incumbent and O (oxygen incorporated) in the
rival. ``z`` is a storage label only, not a claim that P and O are the same
scientific quantity. Across-candidate comparisons must use C, G and M only.

This is a deliberately schematic historical motivation, not a reconstruction
of either historical theory. All numerical coefficients below were specified
before executing the experiment protocol. Numerical updates use the small local kernel extracted from the figure producer. No belief-utility term is present.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .engine import to_info


NAMES = ("z", "C", "G", "M")
CANDIDATE_NAMES = ("phlogiston", "oxygen")
CANDIDATE_NODE_NAMES = (("P", "C", "G", "M"), ("O", "C", "G", "M"))
OBSERVATION_NAMES = ("C", "G", "M")
DIAGNOSTIC_ROWS = (1, 2)
OBSERVABLE_IDX = (1, 2, 3)
GATE_IDX = OBSERVABLE_IDX


def build_world() -> dict:
    """Return the fixed pair of supplied linear-Gaussian candidate priors.

    In storage order ``(z,C,G,M)``, edges are C->P (+.8),
    C->G (+.8), P->M (-.8), versus C->O (+.8), O->G (+.8), O->M (+.8).
    The topological order is C,z,G,M; storage order is retained for compatibility
    with observable-only readouts. Compilation uses the full matrix I-B.
    Both explanatory coordinates have residual variance .25; all observable
    coordinates have residual variance .5. Intercepts are computed from the
    intended joint means, rather than conflating CPD intercepts with means.

    H measures only C,G,M, so sensory data contain no direct reading of either
    explanatory entity. Direct observations deposit diagonal Fisher information;
    this setup tests selection among supplied models, not novel-edge discovery.
    """
    B = np.zeros((2, 4, 4), dtype=np.float64)
    B[0, 0, 1] = 0.8
    B[0, 2, 1] = 0.8
    B[0, 3, 0] = -0.8
    B[1, 0, 1] = 0.8
    B[1, 2, 0] = 0.8
    B[1, 3, 0] = 0.8
    mu = np.array([[0.0, 1.0, 1.0, -1.0], [0.0, 1.0, 1.0, 1.0]])
    b = np.einsum("kij,kj->ki", np.eye(4)[None] - B, mu)
    s = np.tile(np.array([0.25, 0.5, 0.5, 0.5]), (2, 1))
    H = np.eye(4, dtype=np.float64)[1:]
    compiled = tuple(
        to_info(jnp.asarray(B[k]), jnp.asarray(b[k]), jnp.asarray(s[k]))
        for k in range(2)
    )
    covariance = np.stack(
        [
            np.linalg.solve(np.eye(4) - B[k], np.eye(4))
            @ np.diag(s[k])
            @ np.linalg.solve(np.eye(4) - B[k], np.eye(4)).T
            for k in range(2)
        ]
    )
    metadata = {
        "kind": "combustion-mediated release/oxygen candidate menu",
        "candidate_order": list(CANDIDATE_NAMES),
        "storage_names": list(NAMES),
        "candidate_node_names": [list(n) for n in CANDIDATE_NODE_NAMES],
        "observable_names": list(OBSERVATION_NAMES),
        "observable_indices": list(OBSERVABLE_IDX),
        "diagnostic_observation_rows": list(DIAGNOSTIC_ROWS),
        "gate_indices": list(GATE_IDX),
        "explanatory_coordinate": {
            "phlogiston": "P: amount of hypothetical phlogiston released (centred coordinate)",
            "oxygen": "O: oxygen incorporated during the reaction (centred coordinate), not ambient oxygen availability",
            "comparison_rule": "Do not pool, subtract or interpret z across candidate frames.",
        },
        "observable_semantics": {
            "C": "extent of combustion; standardized quantity, unchanged at the imposed shift",
            "G": "net gas uptake by the reacting material; same positive reference value in both accounts",
            "M": "change in solid mass; negative release reference, positive oxygen reference",
        },
        "edges": {
            "phlogiston": [["C", "P", 0.8], ["C", "G", 0.8], ["P", "M", -0.8]],
            "oxygen": [["C", "O", 0.8], ["O", "G", 0.8], ["O", "M", 0.8]],
        },
        "topological_indices": [1, 0, 2, 3],
        "shifted_observation_rows": [2],
        "conditional_covariance_G_M_given_true_C": [0.0, 0.16],
        "joint_means": mu.tolist(),
        "cpd_intercepts": b.tolist(),
        "residual_variances": s.tolist(),
        "observation_operator": H.tolist(),
        "interpretation_limits": [
            "Standardized schematic quantities, not quantitative historical chemistry; the release account is a toy hypothesis.",
            "C indexes reaction extent, not an explanation of ignition. Both models connect C to M through their explanatory substance.",
            "G and M are independent given true C in the release prior; uncertain O induces their conditional dependence in the oxygen prior. Conditioning on noisy C is different.",
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
        "Pi": jnp.stack([v[0] for v in compiled]),
        "h": jnp.stack([v[1] for v in compiled]),
        "mu": jnp.asarray(mu),
        "H": jnp.asarray(H),
        "B": jnp.asarray(B),
        "b": jnp.asarray(b),
        "s": jnp.asarray(s),
        "covariance": jnp.asarray(covariance),
        "names": NAMES,
        "candidate_names": CANDIDATE_NAMES,
        "candidate_node_names": CANDIDATE_NODE_NAMES,
        "disc_rows": DIAGNOSTIC_ROWS,
        "gate_idx": GATE_IDX,
        "observable_idx": OBSERVABLE_IDX,
        "metadata": metadata,
    }


def phi(t: int, t_shift: int = 80) -> np.ndarray:
    """Imposed observable pattern; z=0 is unobserved padding, not shared truth."""
    value = -1.0 if t < t_shift else 1.0
    return np.array([0.0, 1.0, 1.0, value], dtype=np.float64)


def wall_truth() -> np.ndarray:
    """Oxygen-reference pattern, with separate G-only and M-only observers.

    Gas uptake is no longer assigned the incumbent's opposite sign. A G-only
    observer may distinguish predictive variances, but its evidence is not
    designed to oppose mass evidence. There is no imposed balanced conflict.
    """
    return np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float64)


def wall_attention(n: int = 16) -> np.ndarray:
    """First equal-sized camp measures only G; the second measures only M."""
    if not isinstance(n, (int, np.integer)) or n < 2 or n % 2:
        raise ValueError("n must be an even integer of at least two")
    attention = np.zeros((n, 3), dtype=np.float64)
    attention[: n // 2, 1] = 1.0
    attention[n // 2 :, 2] = 1.0
    return attention
