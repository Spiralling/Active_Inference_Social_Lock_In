"""The proposal rate, made principled: Hawkes social excitation + the rate as a belief.

The paper's discovery check models the arrival of structural proposals as a constant-rate
Poisson process -- the maximally agnostic stand-in for conceiving what one has not conceived
(Stanford's problem). Its limitations section names the two refinements this module
implements, both keeping Stanford's constraint (no agent can plan toward a SPECIFIC
unconceived dimension; only the *propensity to look* is modulated):

* **Hawkes (exploration as a social process).** The per-agent rate rides the trust graph as
  a self- and mutually-exciting process: agent ``i`` receives a decaying bump whenever a
  trusted neighbour expands,

      r_i(t) = r0_i(t) + sum_j T_ij * sum_{t_j < t} beta * exp(-(t - t_j) / tau),

  with ``t_j`` the neighbours' discovery times and ``T`` the row-stochastic trust weights the
  population already fuses over. Because the exponential kernel is Markov, the excitation is
  carried recursively (``E_j <- E_j * exp(-1/tau)`` each step, ``E_j += beta`` on ``j``'s
  discovery), so the per-step cost is O(N) -- no event loop. This is the structural-proposal
  channel promoted from carrying *content* (which node to wake) to also carrying *momentum*
  (how eagerly to look): the mechanism by which a community can manufacture the
  near-simultaneity that concept survival under fusion requires.

* **Adaptive (the rate as an updatable hyperprior on model adequacy).** One explores at the
  rate one believes one's model is incomplete, and that belief is updatable: the base rate
  tracks the residual floor the discovery check already records as telemetry
  (``propose_hub(...).strength``) through an exponential moving average,

      s_i <- (1 - rho) * s_i + rho * floor_i,
      r0_i(t) = clip(r0 + kappa * max(s_i - floor_ref, 0), 0, r_max),

  so a persistent unexplained residual raises the agent's own arrival rate while a quiet
  model lets it decay back to ``r0``. The EMA is the pragmatic form of the conjugate
  Gamma-Poisson hyperprior (rate ~ Gamma(a, b) with the residual driving the pseudo-counts:
  posterior mean ``a/b`` tracks the recent evidence of inadequacy at a horizon set by the
  forgetting of the pseudo-counts -- exactly what the EMA's ``rho`` parameterizes); the EMA
  is implemented because it composes with the Hawkes excitation with no extra state.

The two compose (``mode="hawkes_adaptive"``): the adaptive term sets the *base* rate from the
agent's own inadequacy, the Hawkes term adds the community's excitement on top.

Everything here is host-side numpy (the rate gates a host-hook decision, never the jitted
belief dynamics) and strictly opt-in: the legacy single-float Poisson path in
``models/kuhn_phlogiston.py`` is untouched, and this state machine only runs when a
``RateConfig`` is supplied.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MODES = ("poisson", "hawkes", "adaptive", "hawkes_adaptive")


@dataclass(frozen=True)
class RateConfig:
    """The proposal-rate model. ``mode="poisson"`` reproduces the constant rate ``r0``.

    ``r0``           : base arrival rate (the legacy Poisson dial).
    ``hawkes_beta``  : excitation bump a discovery deposits on the discoverer (read by
                       neighbours through the trust weights). 0 => no social excitation.
    ``hawkes_tau``   : decay time (steps) of the excitation kernel.
    ``kappa``        : adaptive gain -- how strongly the residual-floor EMA lifts the base
                       rate. 0 => the rate is not a belief.
    ``floor_ema``    : EMA coefficient ``rho`` on the residual floor (the estimator horizon
                       of the inadequacy belief).
    ``floor_ref``    : the floor level read as "adequate" -- only the excess raises the rate.
    ``r_max``        : hard cap on the per-step rate (keeps ``1 - exp(-r)`` a sane gate).
    ``excite_on_prune_back`` : whether a wake that is later pruned back (the wake-then-prune
                       cycle) still counts as an exciting discovery for neighbours. True is
                       the honest default: the community got excited when the wake happened;
                       the audit verdict arrives later.
    """

    mode: str = "poisson"
    r0: float = 0.08
    hawkes_beta: float = 0.0
    hawkes_tau: float = 10.0
    kappa: float = 0.0
    floor_ema: float = 0.1
    floor_ref: float = 0.0
    r_max: float = 1.0
    excite_on_prune_back: bool = True

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"unknown rate mode {self.mode!r} (expected one of {MODES})")

    @property
    def social(self) -> bool:
        return "hawkes" in self.mode

    @property
    def adaptive(self) -> bool:
        return "adaptive" in self.mode


class ProposalRateState:
    """Per-population rate state: excitation ``E`` (N,) + inadequacy EMA ``s`` (N,).

    Call :meth:`step` once per simulation step (BEFORE this step's expansion decisions) to
    advance the kernel and read the per-agent rates; call :meth:`record_discovery` when an
    agent's expansion is accepted so trusted neighbours' rates rise from the next step on.
    """

    def __init__(self, cfg: RateConfig, n_agents: int,
                 trust_W: np.ndarray | None = None):
        if cfg.social:
            if trust_W is None:
                raise ValueError("hawkes modes need the trust weights (graph.trust_W())")
            trust_W = np.asarray(trust_W, dtype=np.float64)
            if trust_W.shape != (n_agents, n_agents):
                raise ValueError(f"trust_W shape {trust_W.shape} != ({n_agents}, {n_agents})")
        self.cfg = cfg
        self.n = int(n_agents)
        self.T = trust_W
        self.E = np.zeros(self.n)               # excitation deposited on each discoverer
        self.s = np.zeros(self.n)               # residual-floor EMA (inadequacy belief)
        self.events: list[tuple[int, int]] = []  # (agent, step) discovery log (telemetry)
        self._decay = float(np.exp(-1.0 / cfg.hawkes_tau))

    def record_discovery(self, agent: int, t: int) -> None:
        """Agent ``agent`` expanded at step ``t``: deposit the excitation bump."""
        self.events.append((int(agent), int(t)))
        if self.cfg.social:
            self.E[agent] += self.cfg.hawkes_beta

    def step(self, t: int, floors: np.ndarray | None = None) -> np.ndarray:
        """Advance one step and return the per-agent rates ``r_i(t)`` (N,).

        ``floors`` (N,): this step's residual-floor read-out (``propose_hub`` strength) --
        required for the adaptive modes, ignored otherwise. The excitation decays FIRST, so
        a discovery at ``t - 1`` contributes ``beta * exp(-1/tau)`` here, matching the
        ``t_j < t`` (strict past) convention of the kernel."""
        cfg = self.cfg
        if cfg.social:
            self.E *= self._decay
        if cfg.adaptive:
            if floors is None:
                raise ValueError("adaptive rate modes need the per-agent residual floors")
            rho = cfg.floor_ema
            self.s = (1.0 - rho) * self.s + rho * np.asarray(floors, dtype=np.float64)
            r0 = cfg.r0 + cfg.kappa * np.maximum(self.s - cfg.floor_ref, 0.0)
        else:
            r0 = np.full(self.n, cfg.r0)
        r = r0 + (self.T @ self.E if cfg.social else 0.0)
        return np.clip(r, 0.0, cfg.r_max)

    def attempt_probs(self, rates: np.ndarray) -> np.ndarray:
        """Per-step attempt probability of a rate-``r`` Poisson process: ``1 - exp(-r)``."""
        return -np.expm1(-np.asarray(rates))


def explicit_rates(cfg: RateConfig, n_agents: int, trust_W: np.ndarray | None,
                   events: list[tuple[int, int]], t: int,
                   floor_history: np.ndarray | None = None) -> np.ndarray:
    """The kernel written out as the appendix equation (the slow reference for tests):
    ``r_i(t) = r0_i(t) + sum_j T_ij sum_{t_j < t} beta exp(-(t - t_j)/tau)``.

    ``floor_history`` (t+1, N): per-step floors up to and including ``t`` (adaptive modes).
    """
    if cfg.adaptive:
        if floor_history is None:
            raise ValueError("adaptive reference needs the floor history")
        s = np.zeros(n_agents)
        for step_floors in np.asarray(floor_history, dtype=np.float64):
            s = (1.0 - cfg.floor_ema) * s + cfg.floor_ema * step_floors
        r0 = cfg.r0 + cfg.kappa * np.maximum(s - cfg.floor_ref, 0.0)
    else:
        r0 = np.full(n_agents, cfg.r0)
    excite = np.zeros(n_agents)
    if cfg.social:
        E = np.zeros(n_agents)
        for (j, t_j) in events:
            if t_j < t:
                E[j] += cfg.hawkes_beta * np.exp(-(t - t_j) / cfg.hawkes_tau)
        excite = np.asarray(trust_W, dtype=np.float64) @ E
    return np.clip(r0 + excite, 0.0, cfg.r_max)
