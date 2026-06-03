"""Short deterministic smoke rollout for structural production checks."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import jax

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import NetworkConfig
from src.structural.kernel import Network
from src.structural.phlogiston import StructuralConfig


def main() -> int:
    cfg = StructuralConfig(
        n_agents=24,
        n_steps=30,
        seed=0,
        network=NetworkConfig(kind="erdos_renyi", mean_degree=4),
    )
    key = jax.random.PRNGKey(0)

    m_t, attn_t = Network.init(cfg, key).run_trace()
    final_order = float(m_t[-1])
    final_attention_mean = float(attn_t[-1])

    print(f"final_order_parameter={final_order:.6f}")
    print(f"final_attention_mean={final_attention_mean:.6f}")

    if not math.isfinite(final_order) or not math.isfinite(final_attention_mean):
        print("ERROR: non-finite smoke metrics")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
