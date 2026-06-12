"""THROWAWAY probe (network_epistemology plan): does the Zollman ACCURACY tradeoff
appear (Branch A) or only a SPEED ordering (Branch B)? + the 4 lambda2 values."""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from src.structural import graphs
from src.structural import rival as rv
from src.structural.phlogiston import (
    StructuralConfig, phlogiston_bn, gravimetric_H, gravimetric_rows,
    phi_true_at, DISAGREEMENT_NODES)
from experiments.rival_kuhn import oxygen_bn

T_SHIFT, T, N = 80, 480, 24
Q_CONS = 0.9


def build(sigma):
    cfg = StructuralConfig(regime_schedule="step", t_shift=T_SHIFT, n_steps=T,
                           sigma_o=sigma)
    inc = phlogiston_bn(cfg, conviction=2.0).to_info()
    oxy = oxygen_bn(cfg).to_info()
    H = gravimetric_H(cfg)
    rows = gravimetric_rows(cfg)
    disc = [i for i, r in enumerate(rows)
            if ("mass_balance" in r) or (r in DISAGREEMENT_NODES)]
    phis = jnp.stack([phi_true_at(cfg, t) for t in range(T)])
    return inc, oxy, H, disc, phis


def topo(name):
    if name == "complete":
        return graphs.complete(N)
    if name == "cycle":
        return graphs.ring(N, mean_degree=2)
    if name == "wheel":
        return graphs.wheel(N)
    return graphs.community([N // 2, N // 2], intra=1.0, inter=0.08, seed=0)


inc, oxy, H, disc, phis = build(1.0)
m = H.shape[0]
print("lambda2 by topology (N=24):")
for name in ("cycle", "wheel", "community", "complete"):
    print(f"  {name:>10}: lambda2={graphs.algebraic_connectivity(topo(name)):.3f}")


def run_one(name, gamma, sigma, seed):
    inc, oxy, H, disc, phis = build(sigma)
    cfgr = rv.RivalConfig(Pi0=jnp.stack([inc.Pi, oxy.Pi]),
                          h0=jnp.stack([inc.h, oxy.h]), H=H, sigma_o=sigma,
                          omega=0.9, alpha_m=0.02)
    g = topo(name)
    W = g.trust_W()
    A_self = jnp.asarray(np.asarray(g.A) + np.eye(N))
    w2 = np.ones((N, m))
    w2[:, disc] = 1.0 - gamma
    r1 = rv.run_rival(cfgr, W, A_self, jnp.ones((N, m)), phis[:T_SHIFT], seed=seed)
    r2 = rv.run_rival(cfgr, W, A_self, jnp.asarray(w2), phis[T_SHIFT:],
                      seed=seed + 1000, Pi_init=r1["Pi"], h_init=r1["h"],
                      L_init=r1["L"])
    q = np.concatenate([r1["q_t"], r2["q_t"]])[:, :, 1].mean(axis=1)
    conv = q[-1] > Q_CONS
    tconv = int(np.argmax(q > Q_CONS)) if conv else T
    return conv, tconv


print("\nBranch test: P_correct & median t_conv, complete vs cycle, R=12:")
for sigma in (1.0, 0.5):
    for gamma in (0.9, 0.97, 0.99, 1.0):
        line = f"  sigma={sigma} gamma={gamma}: "
        for name in ("complete", "cycle"):
            res = [run_one(name, gamma, sigma, s) for s in range(12)]
            P = np.mean([c for c, _ in res])
            tc = [t for c, t in res if c]
            med = int(np.median(tc)) if tc else -1
            line += f"{name}: P={P:.2f} t~{med}  |  "
        print(line)
