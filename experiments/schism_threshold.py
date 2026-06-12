"""Incommensurability has a threshold, and confidence is the clock.

Two camps that hold opposed readings of the mass-law nodes are put in full
contact on a complete graph. Under fixed, content-blind trust the outcome never
depends on anything: fusion is a contraction, contact means merger, always.
Under content-gated trust (``cfg.social_nu``) the outcome depends on how much
evidence the camps had accumulated BEFORE contact -- their prior precision --
because the trust gate reads the calibrated disagreement
``z^2 ~ gap^2 x precision``: two camps that have merely *believed differently
with more confidence* are more mutually surprising, and past a threshold they
can no longer hear each other at all.

The result is sharp, not graded: below the threshold the gated population is
indistinguishable from the fixed-trust control (full merger); above it, most of
the disagreement survives the whole horizon. The sharpness is a positive
feedback in reverse -- any leak shrinks the gap, which reopens the gate, which
accelerates the leak (rehabilitation runaway) -- so contact either cascades to
consensus or it doesn't.

Reading the x-axis as time: in any evidence-accumulating community, precision
is the integral of exposure, so "confidence at contact" is what incubation time
*buys*. Two schools that developed apart for longer meet above the threshold:
the bridge opens onto nothing. (A direct timed-bridge version with endogenous
divergence needs per-community worlds or self-evidencing conviction -- located
future work; the first attempt here showed weak-prior camps simply dissolve
toward the shared world before the bridge opens.)

Honest scope: the gate is memoryless, so "schism" at this horizon is
metastability (the merger timescale exceeds the run), not a new fixed point --
same caveat as ``gated_divergence``, whose two-camp geometry this experiment
inherits unchanged.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from experiments.registry import ExperimentSpec, register


def _run_one(cfg, prec, n1, n2):
    """Two opposed camps with prior confidence ``prec`` in full contact from t=0.
    Returns the inter-camp distance trace D(t) on the disagreement nodes."""
    import jax
    import jax.numpy as jnp
    from src.structural import phlogiston as ph
    from src.structural import step

    N = n1 + n2
    W = step.trust_weights(jnp.ones((N, N)))
    groups = [dict(count=n1, stance=-1.0, prec_scale=prec),
              dict(count=n2, stance=+1.0, prec_scale=prec)]
    state = step.init_state(cfg, jax.random.PRNGKey(cfg.seed), groups=groups,
                            W_override=W)
    Pi_t, h_t = step.run_trace_net(cfg, state)

    mass_idx = np.asarray([cfg.node_names.index(n) for n in ph.DISAGREEMENT_NODES])
    mu_t = np.linalg.solve(np.asarray(Pi_t), np.asarray(h_t)[..., None])[..., 0]
    g1 = mu_t[:, :n1][:, :, mass_idx].mean(axis=(1, 2))
    g2 = mu_t[:, n1:][:, :, mass_idx].mean(axis=(1, 2))
    return np.abs(g1 - g2)                                  # (T,)


def fig_threshold(precs, frac_gated, frac_fixed, path) -> dict:
    fig, a0 = plt.subplots(figsize=(8.6, 4.2))
    a0.plot(precs, frac_gated, "o-", lw=2.4, color="#1a5276",
            label="inferred trust (social_nu on)")
    a0.plot(precs, frac_fixed, "s--", lw=2.0, color="#7f8c8d",
            label="fixed trust (control): contact = merger, always")
    a0.axhline(0.5, color="k", ls=":", lw=0.8)
    a0.set_xscale("log")
    a0.set_xlabel("prior confidence at contact (prec_scale = evidence accumulated apart)")
    a0.set_ylabel("surviving disagreement  $D_{final}/D_0$")
    a0.set_ylim(-0.05, 1.05); a0.legend(fontsize=8.5)
    a0.set_title("the schism threshold: camps confident enough at contact are too mutually "
                 "surprising to hear each other -- below it, the gate cannot tell them apart "
                 "from the fixed-trust control", fontsize=9)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close(fig)
    return {}


def run(out_dir, params: dict) -> None:
    import dataclasses
    from src.structural.phlogiston import StructuralConfig

    n1 = n2 = params["N_AGENTS"] // 2
    cfg_fix = StructuralConfig(n_agents=params["N_AGENTS"], n_steps=params["N_STEPS"],
                               sigma_o=params["SIGMA_O"], t_shift=10**6,
                               seed=params["SEED"])
    cfg_gat = dataclasses.replace(cfg_fix, social_nu=params["SOCIAL_NU"])

    precs = list(params["PREC_SCALES"])
    frac_gated, frac_fixed = [], []
    for prec in precs:
        Dg = _run_one(cfg_gat, float(prec), n1, n2)
        Df = _run_one(cfg_fix, float(prec), n1, n2)
        frac_gated.append(float(Dg[-1] / Dg[0]))
        frac_fixed.append(float(Df[-1] / Df[0]))
        print(f"  prec={prec:5.1f}: surviving disagreement gated {frac_gated[-1]:.3f}  "
              f"fixed {frac_fixed[-1]:.3f}", flush=True)

    fig_threshold(precs, frac_gated, frac_fixed, out_dir / "fig_schism_threshold.png")

    fg, ff = np.asarray(frac_gated), np.asarray(frac_fixed)
    # (a) fixed trust: contact = merger at every confidence level.
    assert ff.max() < 0.05, \
        f"fixed-trust control must merge at every confidence (max {ff.max():.2f})"
    # (b) gated, low confidence: merger -- the gate starts open and stays open.
    assert fg[0] < 0.05, f"low-confidence camps must merge under the gate (got {fg[0]:.2f})"
    # (c) gated, high confidence: schism -- the gate is shut at contact.
    assert fg[-1] > 0.6, f"high-confidence camps must schism (got {fg[-1]:.2f})"
    # (d) the dependence is monotone (a threshold, not noise).
    assert all(fg[i + 1] >= fg[i] - 0.05 for i in range(len(fg) - 1)), \
        f"surviving disagreement should be monotone in confidence: {fg.round(2)}"
    cross = np.nonzero(fg > 0.3)[0]
    p_star = precs[cross[0]] if cross.size else None
    assert p_star is not None and precs[0] < p_star, "the threshold must be interior"

    np.savez_compressed(out_dir / "simulation_arrays.npz",
                        prec_scales=np.array(precs, dtype=float),
                        frac_gated=fg, frac_fixed=ff)
    summary = {"config": params,
               "frac_gated": {str(p): float(f) for p, f in zip(precs, fg)},
               "frac_fixed": {str(p): float(f) for p, f in zip(precs, ff)},
               "schism_threshold_prec": p_star}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"HEADLINE: incommensurability has a threshold and confidence is the clock -- two "
          f"camps in full contact merge completely when their prior confidence at contact is "
          f"below ~{p_star} (gated indistinguishable from fixed trust), and keep "
          f"{fg[-1]*100:.0f}% of their disagreement above it, because the trust gate reads "
          f"gap^2 x precision: schools that accumulated more evidence apart are more mutually "
          f"surprising. Fixed-trust pooling merges at every confidence (max "
          f"{ff.max()*100:.1f}% survives) -- the threshold is a property of inferred trust "
          f"that content-blind pooling cannot have.")


register(ExperimentSpec(
    model="phlogiston",
    name="schism_threshold",
    description="Two opposed camps in full contact, sweeping the confidence they accumulated "
                "apart: under content-gated trust there is a sharp schism threshold (below it "
                "the camps merge exactly like the fixed-trust control; above it most "
                "disagreement survives) because the trust gate reads gap^2 x precision -- "
                "confidence at contact is what incubation time buys, so incommensurability "
                "has a timescale.",
    run=run,
    out_dir="schism_threshold",
    params=dict(N_AGENTS=20, N_STEPS=80, SIGMA_O=4.0, SOCIAL_NU=0.5,
                PREC_SCALES=(2.0, 5.0, 10.0, 20.0, 30.0, 40.0, 80.0), SEED=0),
    seeds=(0,),
    canonical=False,
    consumes=dict(figures=["fig_schism_threshold"]),
))
