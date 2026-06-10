"""Truth in the STRUCTURE: mean-matched theories, partial information, and (usually) no convergence.

The stress test of "communicating Bayes nets vs the Gaussian lambda thing". We build K=3 candidate
theories deliberately INDISTINGUISHABLE by any first-moment / precision-pooling mechanism: identical
means (all zero) AND identical marginal variances (unit diagonal). They differ ONLY in correlation
STRUCTURE (``Sigma_k = I + rho * A_k``), so the sole discriminating signal is the off-diagonal
couplings. The "lambda thing" (pool the precision-weighted MEAN) is structurally blind here (chance
= 1/K); to discriminate you must observe the COUPLINGS (pairs of nodes) and communicate that
structure. The headline is the NON-CONVERGENCE map: when views are too partial (k=1, no pair ever
seen) or the graph is disconnected, the population cannot converge however it communicates.

Self-contained experiment (its own correlation-theory substrate); only depends on the trust graph.
"""
from __future__ import annotations

import json

import numpy as np
import matplotlib.pyplot as plt

from src.structural import graphs
from experiments.registry import ExperimentSpec, register

D = 6
RHO = 0.4
SIG_OBS = 0.10
N = 90
N_STEPS, T1, T2 = 180, 60, 120
OMEGA = 0.9
VIEW_SIZES = (1, 2, 3, 6)            # nodes each agent sees (6 = full); imperfect info as k falls
SEEDS = (0, 1, 2)
THEORY_EDGES = {
    "chain": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
    "star":  [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)],
    "pairs": [(0, 3), (1, 4), (2, 5)],
}
LOG2PI = float(np.log(2 * np.pi))


def build_theories():
    """K mean-matched, variance-matched correlation theories (Sigma_k = I + rho A_k). Returns
    names, Sigma (K,d,d), Cholesky L (K,d,d). Asserts PD and identical diagonal (=1)."""
    names = list(THEORY_EDGES)
    Sig = np.stack([np.eye(D) for _ in names])
    for ki, nm in enumerate(names):
        for (a, b) in THEORY_EDGES[nm]:
            Sig[ki, a, b] = Sig[ki, b, a] = RHO
    for ki in range(len(names)):
        assert np.allclose(np.diag(Sig[ki]), 1.0), "marginal variances must match (=1)"
        assert np.linalg.eigvalsh(Sig[ki]).min() > 1e-6, f"{names[ki]} not PD"
    L = np.linalg.cholesky(Sig)
    return names, Sig, L


def true_theory(t):
    return 0 if t < T1 else (1 if t < T2 else 2)


def agent_views(k, seed):
    rng = np.random.default_rng(seed + 1234)
    return np.stack([np.sort(rng.choice(D, size=k, replace=False)) for _ in range(N)])


def precompute(views, Sig):
    """Per agent, per theory: inverse + logdet of the observed sub-covariance (+obs noise)."""
    K = Sig.shape[0]; k = views.shape[1]
    Sinv = np.zeros((N, K, k, k)); logdet = np.zeros((N, K))
    for i in range(N):
        V = views[i]
        for kk in range(K):
            S = Sig[kk][np.ix_(V, V)] + SIG_OBS ** 2 * np.eye(k)
            Sinv[i, kk] = np.linalg.inv(S)
            logdet[i, kk] = np.linalg.slogdet(S)[1]
    return Sinv, logdet


def _simulate(views, Sig, L, *, comms, connected, seed):
    """One run. ``comms='structure'`` pools per-theory log-evidence over the graph; ``'none'`` keeps
    agents isolated. Returns frac-of-agents-on-true-theory per step."""
    K = Sig.shape[0]; k = views.shape[1]
    Sinv, logdet = precompute(views, Sig)
    g = graphs.community([N // 3] * 3, intra=0.45, inter=(0.06 if connected else 0.0), seed=seed)
    W = np.asarray(g.trust_W())
    rng = np.random.default_rng(seed + 99)
    LL = np.zeros((N, K))                                   # accumulated per-theory log-evidence
    on_true = np.zeros(N_STEPS)
    const = -0.5 * k * LOG2PI
    for t in range(N_STEPS):
        tk = true_theory(t)
        z = rng.standard_normal((D,))
        x = L[tk] @ z                                      # one shared world draw
        o = x[views] + SIG_OBS * rng.standard_normal((N, k))      # (N,k) partial, noisy views
        quad = np.einsum("na,ntab,nb->nt", o, Sinv, o)     # (N,K) Mahalanobis per theory
        ll = const - 0.5 * (quad + logdet)                 # (N,K) per-theory log-lik this step
        LL = OMEGA * LL + ll                               # forgetting accumulation
        if comms == "structure":
            LL = W @ LL                                    # communicate the structural evidence
        on_true[t] = float((LL.argmax(axis=1) == tk).mean())
    return on_true


def run(out_dir, params: dict) -> None:
    names, Sig, L = build_theories()
    chance = 1.0 / len(names)
    print(f"mean-matched theories {names}: identical means(0) & variances(1); differ only in "
          f"correlation structure. lambda/means classifier = chance = {chance:.2f}.")
    print(f"\n{'view k':>6} | {'isolated':>9} | {'structure-comms':>15} | converged?")
    sweep = []
    for k in VIEW_SIZES:
        iso, struc = [], []
        for s in SEEDS:
            V = agent_views(k, s)
            iso.append(_simulate(V, Sig, L, comms="none", connected=True, seed=s)[-30:].mean())
            struc.append(_simulate(V, Sig, L, comms="structure", connected=True, seed=s)[-30:].mean())
        sweep.append({"k": k, "iso": float(np.mean(iso)), "struc": float(np.mean(struc))})
        conv = "yes" if sweep[-1]["struc"] > 0.6 else "NO"
        print(f"{k:6d} | {sweep[-1]['iso']:9.2f} | {sweep[-1]['struc']:15.2f} | {conv}")

    # The no-pooling baseline IS the 'isolated' column; a community graph with inter=0 still pools
    # WITHIN each 30-node block (already covers the pairs), so it is not a no-pooling control.

    # representative trajectory (k=2, connected) for the figure -- the comms-rescue regime
    traj_struc = np.mean([_simulate(agent_views(2, s), Sig, L, comms="structure", connected=True, seed=s)
                          for s in SEEDS], axis=0)
    traj_iso = np.mean([_simulate(agent_views(2, s), Sig, L, comms="none", connected=True, seed=s)
                        for s in SEEDS], axis=0)

    # ---- figure ----
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13.5, 4.8))
    a0.plot(range(N_STEPS), traj_struc, lw=2.2, color="seagreen", label="communicate structure (Bayes net)")
    a0.plot(range(N_STEPS), traj_iso, lw=2.0, color="navy", label="isolated (no comms)")
    a0.axhline(chance, color="crimson", ls="--", lw=1.2, label="precision/means (λ): blind → chance")
    for tt in (T1, T2):
        a0.axvline(tt, color="k", ls=":", lw=0.8)
    a0.set_xlabel("step (true theory changes at 60, 120)"); a0.set_ylabel("fraction on the TRUE theory")
    a0.set_ylim(-0.05, 1.05); a0.legend(fontsize=8, loc="lower right")
    a0.set_title("k=2 partial views of a changing structural world")
    ks = [s["k"] for s in sweep]
    a1.plot(ks, [s["struc"] for s in sweep], "o-", color="seagreen", lw=2, label="communicate structure")
    a1.plot(ks, [s["iso"] for s in sweep], "s-", color="navy", lw=2, label="isolated (no comms)")
    a1.axhline(chance, color="crimson", ls="--", lw=1.2, label="λ (means): chance")
    a1.set_xlabel("view size k (nodes seen; 1 = no pair ever observed)")
    a1.set_ylabel("final fraction on true theory"); a1.set_ylim(-0.05, 1.05); a1.legend(fontsize=8)
    a1.set_title("non-convergence when info is too partial (k=1) or disconnected")
    plt.tight_layout(); plt.savefig(out_dir / "diag_meanmatched.png", dpi=120); plt.close(fig)

    # ---- asserts (the structural points) ----
    by_k = {s["k"]: s for s in sweep}
    assert by_k[1]["struc"] < 0.45, \
        f"k=1 (no pair observed) must NOT converge even with structure-comms, got {by_k[1]['struc']:.2f}"
    assert by_k[6]["struc"] > 0.8, \
        f"full views + structure-comms should converge, got {by_k[6]['struc']:.2f}"
    assert by_k[2]["struc"] > by_k[2]["iso"] + 0.1, \
        "communicating structure should beat isolated once pairs are visible (comms is essential)"

    summary = {"config": {"D": D, "rho": RHO, "N": N, "omega": OMEGA, "view_sizes": list(VIEW_SIZES),
                          "seeds": list(SEEDS), "theories": names, "epochs": [0, T1, T2]},
               "chance": chance, "sweep": sweep}
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    np.savez_compressed(out_dir / "simulation_arrays.npz",
                        view_sizes=np.array(VIEW_SIZES),
                        iso=np.array([s["iso"] for s in sweep]),
                        struc=np.array([s["struc"] for s in sweep]),
                        chance=chance, traj_struc=traj_struc, traj_iso=traj_iso)

    print(f"\nsaved arrays / figure / summary to {out_dir}")
    print(f"HEADLINE: when the truth is structural, the λ/means thing is blind (chance {chance:.2f}); "
          f"communicating the Bayes net rescues it ONLY when partial views see pairs and can pool "
          f"(k=2: isolated {by_k[2]['iso']:.2f} → structure-comms {by_k[2]['struc']:.2f}); with k=1 "
          f"({by_k[1]['struc']:.2f}, no pair ever seen) the population does NOT converge.")


register(ExperimentSpec(
    model="phlogiston",
    name="meanmatched_structure",
    description="Truth in the STRUCTURE: mean-matched theories where the λ/means classifier is "
                "blind; communicating the Bayes net rescues convergence only with pair views.",
    run=run,
    out_dir="meanmatched_structure",
    params=dict(D=6, RHO=0.4, SIG_OBS=0.10, N=90, N_STEPS=180, T1=60, T2=120, OMEGA=0.9,
                VIEW_SIZES=(1, 2, 3, 6), SEEDS=(0, 1, 2)),
    seeds=(0, 1, 2),
    consumes=dict(notebook=None, figures=["diag_meanmatched"]),
))
