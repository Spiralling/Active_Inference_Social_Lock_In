"""Why the chain-entrenchment phenomenon never appeared: a 5-condition autopsy.

The expected phenomenon (Jonas): high conviction in A, with B conditioned on A
and C on B, should make C slow to revise -- changing C means changing a larger
part of the structure -- and that resistance should produce divergence between
agents. It never showed up. This demo locates why, using exactly the algebra of
src/structural/belief.py (information form: Pi' = Pi + J, h' = h + j) and
src/structural/dual_field.py (conviction U = (I - alpha W)^{-1} u, applied as
h <- h + lambda U each step, as in step._transition).

Setup: 3-node chain A--B--C, prior mean (+1,+1,+1), strong couplings.
Deterministic evidence o = -1 arrives on C ONLY, one deposit per step.
We report the analytic "pinning" of C (its marginal precision = Schur
complement onto C: how much information must accumulate before C's mean moves
halfway) next to the simulated half-time.

Conditions:
  1. chain, soft A            : the published mechanism's substrate, no conviction.
  2. chain, soft A, tilt on C : the paper's conviction (h-tilt) on an OBSERVED node.
  3. chain, soft A, tilt on A : the paper's conviction on an UNOBSERVED node.
  4. chain, STIFF A           : conviction as SELF-PRECISION on A (not in the paper).
  5. A--B edge cut, soft A    : C with the chain to A severed.

What the numbers show:
  * Edges transmit whatever the far end supplies. With a SOFT A (1), the chain
    is lubricated: revising C drags B and A along and the whole frame slides
    together -- C is ~13x EASIER to move than an isolated C (5). With a STIFF A
    (4), the same edges transmit resistance and C is as hard to move as if cut.
    Conditional structure DOES set revision speed, by an order of magnitude --
    but through the precision matrix.
  * The paper's conviction tilt never touches the precision matrix, so it
    cannot engage that mechanism. On an observed node (2) it produces a
    CONSTANT asymptotic offset (= lambda * U * sigma^2 per channel): a
    recalibration of where belief settles, with no delay, no hysteresis, no
    structure-dependence. On an unobserved node (3) the same tilt injects
    potential every step against a precision that never grows, so the mean
    RUNS AWAY linearly -- a pathology, not entrenchment.

Moral: in a Gaussian net, edges transmit revision; only precision resists it.
Conviction-as-tilt biases the destination; conviction-as-precision resists the
journey. The expected phenomenon lives in Pi, and the implemented conviction
lives in h.
"""

import numpy as np

ALPHA = 0.5           # conviction_alpha, as in StructuralConfig
SIGMA = 6.0           # observation noise on the C channel (rate 1/36 per step)
C_TRUE = -1.0         # the world disagrees with the prior about C
STEPS = 3000
SELF = 6.0            # self-precision of an ordinary node
COUPLE = -4.2         # off-diagonal precision (edge); PD limit here is 6/sqrt(2)=4.24
STIFF = 600.0         # self-precision of a conviction-stiffened node
LAM = 0.01            # conviction_tilt lambda for the tilt conditions


def chain_Pi(self_A=SELF, ab=COUPLE, bc=COUPLE):
    Pi = np.array([[self_A, ab,   0.0],
                   [ab,     SELF, bc],
                   [0.0,    bc,   SELF]])
    assert np.all(np.linalg.eigvalsh(Pi) > 0), "prior precision must be PD"
    return Pi


def pinning_of_C(Pi):
    """Marginal precision of C: Schur complement of the (A,B) block onto C."""
    P_blk, p_cross = Pi[:2, :2], Pi[:2, 2]
    return Pi[2, 2] - p_cross @ np.linalg.solve(P_blk, p_cross)


def conviction_U(Pi, u):
    """dual_field.effective_utility: (I - alpha W) U = u, W row-stochastic |Pi| off-diag."""
    d = Pi.shape[0]
    off = np.abs(Pi) * (1 - np.eye(d))
    rs = off.sum(axis=1, keepdims=True)
    W = np.divide(off, rs, out=np.zeros_like(off), where=rs > 0)
    return np.linalg.solve(np.eye(d) - ALPHA * W, u)


def run(Pi0, mu0, lam=0.0, u=None):
    Pi, h = Pi0.copy(), Pi0 @ mu0
    Hrow = np.array([0.0, 0.0, 1.0])               # evidence lands on C only
    J = np.outer(Hrow, Hrow) / SIGMA**2
    j = Hrow * C_TRUE / SIGMA**2                   # deterministic o = C_TRUE
    U = conviction_U(Pi0, u) if lam else None
    mus = np.empty((STEPS, 3))
    for t in range(STEPS):
        Pi = Pi + J
        h = h + j
        if lam:
            h = h + lam * U                        # the paper's tilt, every step
        mus[t] = np.linalg.solve(Pi, h)
    return mus


def half_time(mus, node=2):
    crossed = np.nonzero(mus[:, node] < 0.0)[0]
    return int(crossed[0]) if len(crossed) else None


if __name__ == "__main__":
    mu0 = np.ones(3)
    e_A, e_C = np.eye(3)[0], np.eye(3)[2]

    conds = [
        ("1. chain, soft A (no conviction)",        chain_Pi(),             0.0, None),
        ("2. chain, soft A, TILT on C (observed)",  chain_Pi(),             LAM, e_C),
        ("3. chain, soft A, TILT on A (unobserved)", chain_Pi(),            LAM, e_A),
        ("4. chain, STIFF A (conviction as precision)", chain_Pi(self_A=STIFF), 0.0, None),
        ("5. A--B edge CUT, soft A",                chain_Pi(ab=0.0),       0.0, None),
    ]
    print(f"deterministic evidence o={C_TRUE} on C only, channel precision "
          f"1/sigma^2 = {1/SIGMA**2:.3f}/step; prior mean +1 everywhere\n")
    print(f"{'condition':46s} {'pin(C)':>7s} {'t_half(C)':>10s}   final mu (A, B, C)")
    for name, Pi0, lam, u in conds:
        mus = run(Pi0, mu0, lam=lam, u=u)
        ht = half_time(mus)
        ht_s = f"{ht:8d}" if ht is not None else "   never"
        print(f"{name:46s} {pinning_of_C(Pi0):7.3f} {ht_s:>10s}   "
              f"({mus[-1,0]:+8.2f}, {mus[-1,1]:+6.2f}, {mus[-1,2]:+6.2f})")
