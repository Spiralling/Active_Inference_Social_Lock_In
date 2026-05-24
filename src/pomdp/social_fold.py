"""Go/no-go probe (plan step 0b): does softmax-precision social coupling on a
*categorical* paradigm belief produce a fold / pitchfork?

This is the make-or-break premise of the whole categorical-POMDP rebuild. The
diagnosis note (``notes/dynamical_systems_diagnosis.tex``) proved the *linear
Gaussian* substrate is globally monostable. The rebuild bets that moving to a
discrete belief with a softmax-precision social channel restores bistability.
We test that bet here, at the cheapest possible level (mean-field self-
consistency map + a finite-N stochastic confirmation), BEFORE building any
scaffold.

Faithfulness (avoiding the "hand-built sigmoid" objection)
----------------------------------------------------------
The coupling is *derived* from categorical Bayes, not assumed. Each agent holds
a belief b = P(paradigm = B). It receives social "reports" (a neighbour's voted
paradigm). It treats a report as evidence with a **reliability** q = P(report=B
| s=B) = P(report=A | s=A), q in (1/2, 1]. This q is exactly the paper's
**trust-as-message-precision** reading: how diagnostic the agent considers a
received signal — NOT Albarracin's attention-allocation (whom to sample). The
per-report log-likelihood-ratio is

    kappa := log( q / (1 - q) )            # >= 0, the social precision

In a homogeneous mean field with population belief m = P(B), a randomly drawn
report is B with probability m. The expected social log-odds increment per
report is therefore

    E[delta ell] = m * (+kappa) + (1 - m) * (-kappa) = kappa * (2 m - 1).

With a per-step log-odds leak lambda (geometric forgetting toward the neutral
prior — the analogue of ``InferenceConfig.posterior_rho`` in the existing
substrate, and what makes this a proper dynamical map rather than a ratchet),
the mean-field fixed point obeys

    lambda * ell* = K * (2 * sigmoid(ell*) - 1),     K := kappa / lambda

i.e.   m* = sigmoid( (K / lambda_is_absorbed) ... )  ->   ell* = K (2 sigma(ell*) - 1).

This is the mean-field Ising / tanh-feedback map (row 2 of the diagnosis note's
Table 1). It has a supercritical pitchfork at K_c = 2: for K < 2 the only fixed
point is the neutral consensus m = 1/2; for K > 2 that point goes unstable and
two symmetric stable paradigm-consensus states appear. **A clean pitchfork here
is the PASS condition for plan step 0b.**
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ----------------------------------------------------------------------
# Precision <-> reliability
# ----------------------------------------------------------------------

def reliability_to_precision(q: float | np.ndarray) -> float | np.ndarray:
    """kappa = log(q / (1 - q)); the social log-likelihood-ratio per report.

    q is trust-as-message-precision (how diagnostic a received report is).
    q = 0.5 -> kappa = 0 (reports carry no information); q -> 1 -> kappa -> inf.
    """
    q = np.asarray(q, dtype=float)
    return np.log(q / (1.0 - q))


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=float)))


# ----------------------------------------------------------------------
# Mean-field self-consistency map
# ----------------------------------------------------------------------

def fixedpoint_residual(ell: np.ndarray | float, K: float) -> np.ndarray | float:
    """g(ell) = K (2 sigma(ell) - 1) - ell. Zeros are mean-field fixed points.

    K = kappa / lambda is the effective social gain (precision per leak).
    """
    return K * (2.0 * sigmoid(ell) - 1.0) - np.asarray(ell, dtype=float)


def find_fixed_points(K: float, grid: np.ndarray | None = None,
                      refine_iter: int = 60) -> list[tuple[float, bool]]:
    """All fixed points of the log-odds map for gain K, with stability flags.

    Returns a list of (m*, stable) where m* = sigmoid(ell*) in [0, 1] and
    ``stable`` is True iff |F'(ell*)| < 1 for the iteration
    ell <- F(ell) = K (2 sigma(ell) - 1).

    Found by bracketing sign changes of ``fixedpoint_residual`` on a grid, then
    bisection refinement (robust, no derivatives needed).
    """
    if grid is None:
        # ell range generous enough to bracket the outer branches for K up to ~8
        grid = np.linspace(-12.0, 12.0, 4001)
    res = fixedpoint_residual(grid, K)
    roots: list[float] = []

    # exact-zero hits on the grid
    for i in np.where(res == 0.0)[0]:
        roots.append(float(grid[i]))

    # sign-change brackets -> bisection
    sign = np.sign(res)
    for i in np.where(sign[:-1] * sign[1:] < 0)[0]:
        lo, hi = grid[i], grid[i + 1]
        flo = fixedpoint_residual(lo, K)
        for _ in range(refine_iter):
            mid = 0.5 * (lo + hi)
            fmid = fixedpoint_residual(mid, K)
            if flo * fmid <= 0.0:
                hi = mid
            else:
                lo, flo = mid, fmid
        roots.append(0.5 * (lo + hi))

    # dedupe
    roots = sorted(roots)
    deduped: list[float] = []
    for r in roots:
        if not deduped or abs(r - deduped[-1]) > 1e-6:
            deduped.append(r)

    out: list[tuple[float, bool]] = []
    for ell_star in deduped:
        # F'(ell) = K * 2 * sigma'(ell); sigma'(ell) = sigma(1-sigma)
        s = sigmoid(ell_star)
        fprime = K * 2.0 * s * (1.0 - s)
        stable = abs(fprime) < 1.0
        out.append((float(s), bool(stable)))
    return out


def bifurcation_diagram(K_values: np.ndarray) -> dict[str, list]:
    """Sweep K and collect (K, m*, stable) for every fixed point.

    Returns dict with parallel lists 'K', 'm', 'stable' for scatter plotting.
    """
    Ks, ms, stab = [], [], []
    for K in K_values:
        for m_star, stable in find_fixed_points(float(K)):
            Ks.append(float(K))
            ms.append(m_star)
            stab.append(stable)
    return {"K": Ks, "m": ms, "stable": stab}


# ----------------------------------------------------------------------
# Finite-N stochastic confirmation (the "it's actually social" check)
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class FiniteNConfig:
    n_agents: int = 200
    q: float = 0.80                 # report reliability (trust-as-precision)
    leak: float = 0.30              # per-step log-odds leak toward neutral prior
    reports_per_step: int = 1       # social reports each agent ingests per round
    n_steps: int = 400
    private_kappa: float = 0.0      # symmetric private evidence strength (0 = none)


def simulate_finite_N(cfg: FiniteNConfig, ell0: np.ndarray,
                      rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Stochastic N-agent run of the categorical social update.

    State is per-agent log-odds ell_i. Each round:
      1. each agent votes its MAP paradigm  v_i = 1[ell_i > 0]
      2. each agent ingests ``reports_per_step`` votes drawn uniformly from the
         population, updating log-odds by +/- kappa per the report
      3. log-odds leak toward the neutral prior by factor (1 - leak)

    Returns trajectories: 'm' (mean belief over time), 'std' (cross-agent belief
    std, the polarization readout), and final 'ell'.
    """
    kappa = float(reliability_to_precision(cfg.q))
    N = cfg.n_agents
    ell = np.array(ell0, dtype=float).copy()
    m_traj = np.empty(cfg.n_steps)
    std_traj = np.empty(cfg.n_steps)

    for t in range(cfg.n_steps):
        # Agents emit a report sampled FROM their belief, v_i ~ Bernoulli(b_i),
        # not a hard MAP vote. This keeps a randomly drawn report B-valued with
        # probability E[b] = m, so the finite-N dynamics isolate the softmax-
        # precision mechanism and match the mean-field threshold K_c = 2.
        # (Hard MAP voting would add a separate threshold-pooling nonlinearity.)
        b_now = sigmoid(ell)
        votes = (rng.random(N) < b_now).astype(float)   # probabilistic report
        # each agent samples reports_per_step voters (with replacement)
        idx = rng.integers(0, N, size=(N, cfg.reports_per_step))
        sampled = votes[idx]                            # (N, reports_per_step)
        # +kappa for a B-vote (1), -kappa for an A-vote (0)
        social = kappa * (2.0 * sampled - 1.0).sum(axis=1)
        # optional symmetric private evidence: mean-zero, does not bias symmetry
        private = cfg.private_kappa * rng.standard_normal(N)
        ell = (1.0 - cfg.leak) * ell + social + private
        b = sigmoid(ell)
        m_traj[t] = float(np.mean(b))
        std_traj[t] = float(np.std(b))

    return {"m": m_traj, "std": std_traj, "ell": ell, "kappa": kappa}


def basin_selection_test(cfg: FiniteNConfig, K_effective_note: str = "",
                         n_seeds: int = 40,
                         init_bias: float = 0.15,
                         rng: np.random.Generator | None = None
                         ) -> dict[str, np.ndarray]:
    """Run many seeds from small random initial biases; record final mean belief.

    PASS signature (bistable): final m clusters into two well-separated groups
    (near 0 and near 1) -> symmetry breaking / two basins.
    FAIL signature (monostable): all seeds collapse to m ~ 0.5.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    finals = np.empty(n_seeds)
    for s in range(n_seeds):
        ell0 = init_bias * rng.standard_normal(cfg.n_agents)
        out = simulate_finite_N(cfg, ell0, rng)
        finals[s] = out["m"][-1]
    return {"final_m": finals}


# ======================================================================
# PHASE 0.5 — environment-coupled fold gate
# ----------------------------------------------------------------------
# NB15 validated the SOCIAL SUBSYSTEM with the environment off
# (private_kappa = 0). The real go/no-go for the *model* is whether the fold
# survives once the object-level evidence (A_theta) pulls every agent toward
# the true paradigm. Per the mediating-field note: "a large environment pull
# flattens the effective F; the competition is F' vs 1 once the environment is
# folded in" — and we add the belief-utility tilt as the symmetry-breaking
# field that turns symmetric polarization into asymmetric capture.
#
# Convention: paradigm B is TRUE, so +ell favours the truth. The mean-field
# log-odds fixed point gains a constant net field B_field (in leak-normalised
# units, i.e. divided by lambda):
#
#     ell* = K (2 sigma(ell*) - 1) + B_field ,     K = kappa_soc / lambda
#     B_field = d  +  h_U
#
#   d   >= 0 : ENVIRONMENT pull toward the true paradigm B. Set by theory
#             discriminability (per-row KL between paradigms' predicted
#             columns); low d = underdetermined / theory-laden data.
#   h_U       : BELIEF-UTILITY tilt. h_U > 0 favours B (aligned with truth);
#             h_U < 0 favours the WRONG paradigm A (motivated reasoning) —
#             this is the capture-against-evidence regime.
#
# Stability is unchanged by a constant field: F'(ell*) = K * 2 sigma'(ell*),
# stable iff < 1. The tilt unfolds the pitchfork into a cusp: as |B_field|
# grows the minority basin is destroyed in a saddle-node, so a sharply
# disambiguating environment (large d) flattens the fold back to monostable.
# ======================================================================

def fixedpoint_residual_env(ell, K: float, B_field: float):
    """g(ell) = K (2 sigma(ell) - 1) + B_field - ell. Zeros are fixed points."""
    return K * (2.0 * sigmoid(ell) - 1.0) + B_field - np.asarray(ell, dtype=float)


def find_fixed_points_env(K: float, B_field: float,
                          grid: np.ndarray | None = None,
                          refine_iter: int = 60) -> list[tuple[float, bool]]:
    """Fixed points (m*, stable) of the field-tilted log-odds map.

    Same bracket-and-bisect scheme as ``find_fixed_points`` with the constant
    net field ``B_field`` added (d + h_U). Stability uses the leak-normalised
    iteration derivative F'(ell)=K*2*sigma'(ell), stable iff |F'|<1.
    """
    if grid is None:
        grid = np.linspace(-16.0, 16.0, 6001)
    res = fixedpoint_residual_env(grid, K, B_field)
    roots: list[float] = [float(grid[i]) for i in np.where(res == 0.0)[0]]
    sign = np.sign(res)
    for i in np.where(sign[:-1] * sign[1:] < 0)[0]:
        lo, hi = grid[i], grid[i + 1]
        flo = fixedpoint_residual_env(lo, K, B_field)
        for _ in range(refine_iter):
            mid = 0.5 * (lo + hi)
            fmid = fixedpoint_residual_env(mid, K, B_field)
            if flo * fmid <= 0.0:
                hi = mid
            else:
                lo, flo = mid, fmid
        roots.append(0.5 * (lo + hi))
    roots = sorted(roots)
    deduped: list[float] = []
    for r in roots:
        if not deduped or abs(r - deduped[-1]) > 1e-6:
            deduped.append(r)
    out: list[tuple[float, bool]] = []
    for ell_star in deduped:
        s = sigmoid(ell_star)
        fprime = K * 2.0 * s * (1.0 - s)
        out.append((float(s), bool(abs(fprime) < 1.0)))
    return out


# regime codes for the (K, field) scan
MONOSTABLE_TRUTH = 0     # one stable fp, on the true (B, m>0.5) side
MONOSTABLE_WRONG = 1     # one stable fp, on the wrong (A, m<0.5) side — full capture
BISTABLE = 2             # two stable fps — both basins coexist (path-dependent)


def classify_regime(K: float, d: float, h_U: float) -> int:
    """Classify the (K, d, h_U) point into MONOSTABLE_TRUTH/WRONG or BISTABLE.

    truth = B (+ side). Returns one of the module-level regime codes.
    """
    fps = find_fixed_points_env(K, d + h_U)
    stable = [m for m, st in fps if st]
    if len(stable) >= 2:
        return BISTABLE
    # single stable fixed point: which basin?
    m = stable[0] if stable else 0.5
    return MONOSTABLE_TRUTH if m >= 0.5 else MONOSTABLE_WRONG


def regime_grid(K_values: np.ndarray, field_values: np.ndarray,
                d: float = 0.0, as_field: bool = True) -> np.ndarray:
    """2-D regime map.

    If ``as_field`` (default): rows = K, cols = net field B_field (= d + h_U),
    so ``d`` is ignored and ``field_values`` is the net field axis. Otherwise
    ``field_values`` is the belief-utility h_U axis at a fixed environment ``d``.
    Returns an int array of regime codes shaped (len(field_values), len(K_values)).
    """
    out = np.empty((len(field_values), len(K_values)), dtype=int)
    for j, K in enumerate(K_values):
        for i, fv in enumerate(field_values):
            B_field = fv if as_field else (d + fv)
            fps = find_fixed_points_env(float(K), float(B_field))
            stable = [m for m, st in fps if st]
            if len(stable) >= 2:
                out[i, j] = BISTABLE
            else:
                m = stable[0] if stable else 0.5
                out[i, j] = MONOSTABLE_TRUTH if m >= 0.5 else MONOSTABLE_WRONG
    return out


@dataclass(frozen=True)
class EnvConfig:
    """Finite-N config for the environment-coupled / belief-utility run.

    Extends the social knobs with:
      d_env   : per-step object-level evidence increment toward the TRUE
                paradigm B (>=0); the discriminability of the chosen experiment.
      h_util  : per-step belief-utility field (signed); <0 = motivated toward
                the wrong paradigm A.
      nu_mot  : optional motivated-sampling gain — extra sigmoidal feedback
                toward the belief-utility-preferred side (engine #1). 0 = off.
    """
    n_agents: int = 400
    q: float = 0.80
    leak: float = 0.30
    reports_per_step: int = 1
    n_steps: int = 800
    d_env: float = 0.0
    h_util: float = 0.0
    nu_mot: float = 0.0


def simulate_env(cfg: EnvConfig, ell0: np.ndarray,
                 rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Finite-N run with the environment pull and belief-utility tilt folded in.

    Per step, each agent's log-odds gets: social (sampled probabilistic reports,
    as in NB15) + environment pull (+d_env toward true B) + belief-utility field
    (h_util) + optional motivated-sampling feedback (nu_mot toward sign(h_util)),
    then the leak. truth = B (+).
    """
    kappa = float(reliability_to_precision(cfg.q))
    N = cfg.n_agents
    ell = np.array(ell0, dtype=float).copy()
    m_traj = np.empty(cfg.n_steps)
    pref = np.sign(cfg.h_util) if cfg.h_util != 0.0 else 0.0
    for t in range(cfg.n_steps):
        b_now = sigmoid(ell)
        votes = (rng.random(N) < b_now).astype(float)
        idx = rng.integers(0, N, size=(N, cfg.reports_per_step))
        social = kappa * (2.0 * votes[idx] - 1.0).sum(axis=1)
        env = cfg.d_env                       # +toward true paradigm B
        util = cfg.h_util                      # belief-utility field
        # motivated sampling: amplify evidence toward the preferred side,
        # gated by how much the agent already leans that way (positive feedback)
        motivated = cfg.nu_mot * pref * (2.0 * b_now - 1.0) * (pref * (2.0 * b_now - 1.0) > 0)
        ell = (1.0 - cfg.leak) * ell + social + env + util + motivated
        m_traj[t] = float(np.mean(sigmoid(ell)))
    return {"m": m_traj, "ell": ell, "kappa": kappa}


def capture_basin_test(cfg: EnvConfig, n_seeds: int = 40, init_bias: float = 0.10,
                       rng: np.random.Generator | None = None) -> dict[str, np.ndarray]:
    """Many seeds; record final mean belief. With truth=B and h_util<0, a cluster
    of seeds landing at m<0.5 = capture by the WRONG paradigm against evidence."""
    if rng is None:
        rng = np.random.default_rng(0)
    finals = np.empty(n_seeds)
    for s in range(n_seeds):
        ell0 = init_bias * rng.standard_normal(cfg.n_agents)
        finals[s] = simulate_env(cfg, ell0, rng)["m"][-1]
    return {"final_m": finals}
