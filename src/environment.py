"""Multi-feature situation environment for paradigm-shift experiments.

The environment maintains a hidden ground-truth bit per feature,
``s_star ∈ {0, 1}^R``. Each timestep:

1. A situation σ_t ∈ {0, ..., R-1} is drawn (uniform). Following the paper
   we identify situations with feature indices: σ_t designates which feature
   is *live* (rewarded) this step.
2. Each of N agents picks a binary action a_i for the live feature.
3. The environment generates per-agent feedback: payoff = 1 iff a_i ==
   s_star[σ_t].
4. The hidden state evolves per the chosen regime.

The environment is intentionally agent-agnostic — it consumes a list of N
actions and returns a list of N observations. Multi-agent coupling
(prestige, trust, social mixing) lives in src/agent.py, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np


class EvolutionRegime(Enum):
    """Three modes of environmental hidden-state dynamics."""
    STATIONARY = "stationary"      # s* fixed, with rare random flips
    SLOW_DRIFT = "slow_drift"      # Bernoulli random walk per feature
    DISCRETE_SHIFT = "discrete_shift"  # deterministic flip at t = shift_time


@dataclass
class EnvConfig:
    """Configuration for the multi-feature situation environment."""
    n_agents: int = 300
    n_features: int = 4
    regime: EvolutionRegime = EvolutionRegime.STATIONARY
    # regime-specific knobs
    stationary_flip_prob: float = 0.001   # per-feature per-step flip prob
    drift_step_prob: float = 0.02         # per-feature per-step flip prob in slow drift
    shift_time: int | None = 50           # for DISCRETE_SHIFT: t at which s* flips
    shift_features: tuple[int, ...] = (0,)  # which features flip at shift_time
    seed: int = 0


class CoordinationEnvironment:
    """Multi-feature coordination environment.

    Hidden state ``s_star ∈ {0, 1}^R``. Each step samples one situation
    σ_t (= a feature index) and reveals s_star[σ_t]; agents that pick that
    bit receive payoff 1.

    The environment is deterministic given a seed; reproducibility relies
    on a single internal RNG.
    """

    def __init__(self, config: EnvConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        # Hidden state: one bit per feature.
        self.s_star: np.ndarray = self.rng.integers(
            0, 2, size=config.n_features
        ).astype(np.int64)
        self.t: int = 0

    def reset(self) -> dict:
        """Reset environment state, return initial observation dict."""
        self.t = 0
        self.s_star = self.rng.integers(0, 2, size=self.config.n_features).astype(np.int64)
        sigma_t = int(self.rng.integers(0, self.config.n_features))
        return {
            "sigma": sigma_t,
            "true_value": int(self.s_star[sigma_t]),
            "s_star": self.s_star.copy(),
            "t": self.t,
        }

    def draw_situation(self) -> int:
        """Sample (and consume) the next situation index σ_t.

        Used by ``Population.step`` so agents can condition action selection
        on the live feature *before* the env sees their actions.
        """
        return int(self.rng.integers(0, self.config.n_features))

    def step(self, actions: Sequence[int], situation: int) -> dict:
        """Compute payoffs for ``situation`` and evolve the hidden state.

        Parameters
        ----------
        actions : sequence of N binary actions (one per agent).
        situation : int — the live feature index σ_t for this step
            (typically obtained from ``draw_situation``).

        Returns
        -------
        dict with keys:
            sigma: int — the live feature this step
            true_value: int — s_star[σ_t]
            payoffs: np.ndarray (N,) — 1 if a_i == true_value else 0
            s_star: np.ndarray (R,) — full hidden state (post-evolution)
            t: int — timestep counter (post-increment)
        """
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.config.n_agents,):
            raise ValueError(
                f"Expected actions shape ({self.config.n_agents},), "
                f"got {actions.shape}"
            )

        sigma_t = int(situation)
        true_value = int(self.s_star[sigma_t])
        payoffs = (actions == true_value).astype(np.int64)

        self._evolve_hidden_state()
        self.t += 1

        return {
            "sigma": sigma_t,
            "true_value": true_value,
            "payoffs": payoffs,
            "s_star": self.s_star.copy(),
            "t": self.t,
        }

    def _evolve_hidden_state(self) -> None:
        """Advance s_star one step under the configured regime."""
        cfg = self.config

        if cfg.regime == EvolutionRegime.STATIONARY:
            flip_mask = self.rng.random(cfg.n_features) < cfg.stationary_flip_prob
            self.s_star = self.s_star ^ flip_mask.astype(np.int64)

        elif cfg.regime == EvolutionRegime.SLOW_DRIFT:
            flip_mask = self.rng.random(cfg.n_features) < cfg.drift_step_prob
            self.s_star = self.s_star ^ flip_mask.astype(np.int64)

        elif cfg.regime == EvolutionRegime.DISCRETE_SHIFT:
            if cfg.shift_time is not None and self.t == cfg.shift_time:
                for r in cfg.shift_features:
                    self.s_star[r] ^= 1

        else:
            raise ValueError(f"Unknown regime: {cfg.regime}")

    @property
    def num_agents(self) -> int:
        return self.config.n_agents

    @property
    def num_features(self) -> int:
        return self.config.n_features
