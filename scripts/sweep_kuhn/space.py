"""The overnight kuhn_phlogiston sweep: four panels, each feeding exactly one figure.

Pure data -- no simulation here. A *job* is a flat dict with every axis set (swept axes vary,
the rest sit at DEFAULTS = the validated kuhn_phlogiston anchor) plus ``panel`` and ``seed``.

* **A_vanguard** -- how many Lavoisiers does it take? Vanguard (open-community) fraction x
  social coupling -> whole-population conversion / oxygen adoption: the critical-mass curve.
* **B_lockin** -- the lock-in boundary, finely. The pilot showed even inter=0.01 defeats the
  gate, so the grid concentrates on VERY weak coupling; axes gate scaling and prune threshold
  ask whether any amount of conviction can shift the critical coupling.
* **C_waiting** -- discovery as a waiting time. Poisson proposal rate -> crisis-to-discovery
  delay (the 1/lambda law at population scale) + the social spread after first discovery.
* **D_robust** -- does the cycle survive the dials? Forgetting x noise, including the
  omega=1.0 memory wall. (The expansion trigger axis is gone: the model log Bayes factor is
  now the sole expansion accept test, so there is no trigger dial to sweep.)
* **E_hawkes** -- the Hawkes rescue at sweep scale: excitation beta x base rate (the
  limitations' social-excitation refinement; see experiments/hawkes_rescue.py). Long
  horizon (320): the cascade is fast but the crushed couplings need consolidation time.
* **F_fusion** -- the pooling rule as an axis: posterior vs dimension-aware masked fusion
  x proposal rate (see experiments/fusion_survival.py).

NOTE: panels E/F carry their extra keys (rate_mode/hawkes_*/fuse_mode/n_steps) inside their
own job dicts rather than in DEFAULTS, so the config hashes of the original A-D panels are
unchanged and a resumed sweep does not recompute them.
"""
from __future__ import annotations

DEFAULTS = dict(
    N=80, inter=0.0, intra=0.4, frac_open=0.5,
    lam_open=0.10, lam_dogma=0.35,
    gate_strength=1.0, s_open=0.3, s_dogma=3.0,
    omega=0.97, sigma_o=0.5, t_shift=40, n_steps=180,
    proposal_rate=0.08, snapshot_every=4,
)

A_FRACS = (0.05, 0.1, 0.2, 0.35, 0.5)
A_INTERS = (0.0, 0.002, 0.005, 0.01, 0.05, 0.2)

B_INTERS = (0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05)
B_SDOGMA = (1.0, 3.0, 6.0, 10.0)
B_LAM = (0.35, 0.8)

C_RATES = (0.01, 0.02, 0.04, 0.08, 0.16, None)        # None = deterministic attempts

D_OMEGA = (0.9, 0.95, 0.97, 0.99, 1.0)
D_SIGMA = (0.25, 0.5, 1.0)

E_BETAS = (0.0, 0.25, 0.5, 1.0, 1.5)
E_R0S = (0.005, 0.01, 0.02)          # the staggered regime (no completion without excitation)
E_STEPS = 320                        # consolidation horizon (see experiments/hawkes_rescue.py)

F_MODES = ("posterior", "posterior_masked")
F_RATES = (0.005, 0.01, 0.02)


def _job(panel, seed, **over):
    cfg = dict(DEFAULTS)
    cfg.update(over)
    cfg["panel"], cfg["seed"] = panel, seed
    return cfg


def all_jobs(seeds_a: int = 16, seeds_b: int = 16, seeds_c: int = 32,
             seeds_d: int = 8, seeds_e: int = 32, seeds_f: int = 32) -> list[dict]:
    jobs = []
    for s in range(seeds_a):
        for f in A_FRACS:
            for it in A_INTERS:
                jobs.append(_job("A_vanguard", s, frac_open=f, inter=it))
    for s in range(seeds_b):
        for it in B_INTERS:
            for sd in B_SDOGMA:
                for lm in B_LAM:
                    jobs.append(_job("B_lockin", s, inter=it, s_dogma=sd, lam_dogma=lm))
    for s in range(seeds_c):
        for rate in C_RATES:
            jobs.append(_job("C_waiting", s, proposal_rate=rate))
    for s in range(seeds_d):
        for om in D_OMEGA:
            for sg in D_SIGMA:
                jobs.append(_job("D_robust", s, omega=om, sigma_o=sg))
    for s in range(seeds_e):
        for beta in E_BETAS:
            for r0 in E_R0S:
                jobs.append(_job("E_hawkes", s, proposal_rate=r0, n_steps=E_STEPS,
                                 rate_mode=("hawkes" if beta > 0 else "poisson"),
                                 hawkes_beta=beta, hawkes_tau=10.0))
    for s in range(seeds_f):
        for fm in F_MODES:
            for rate in F_RATES:
                jobs.append(_job("F_fusion", s, proposal_rate=rate, n_steps=E_STEPS,
                                 fuse_mode=fm))
    return jobs
