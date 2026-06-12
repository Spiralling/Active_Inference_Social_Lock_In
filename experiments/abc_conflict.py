"""The ABC conflict: inferred reliability turns 'confident compromise' into crisis.

The Gaussian pathology this verifies the fix for (O'Hagan 1979): a plain
precision-accumulating agent fed persistently conflicting evidence AVERAGES it
-- the posterior mean settles on a configuration neither the model's couplings
nor the new data support, the precision keeps growing the whole time, and the
conflict leaves no trace in the belief state. The robust agent instead infers a
per-channel reliability ``lambda_k = (nu+1)/(nu+z_k^2)`` (Student-t scale
mixture; ``src/structural/reliability.py``) with an EMA-persistent surprise
``z2_ema``, and -- the load-bearing detail -- retrospectively re-scales its
ACCUMULATED per-channel deposits ``(Jd_k, jd_k)`` by the current ``lambda_bar``:

    Pi_eff = Pi0 + sum_k lambda_bar_k Jd_k ,   h_eff = h0 + sum_k lambda_bar_k jd_k.

Because PSD deposits only ever ADD, variance inflation is IMPOSSIBLE from
deposit weighting alone; only this retrospective re-scaling can give back
variance. That is why this experiment carries its own rollout (the accumulator
state) instead of reusing ``step._transition``, whose instantaneous lambda is
stateless by design (the locked decision: ``PopulationState`` carries no new
state). What the run must show, on a 3-node chain A--B--C whose channel A turns
persistently contradictory at ``T_CONFLICT``:

  * baseline: monotone confidence growth on A and a final mean strictly between
    the two sources -- the confident compromise;
  * robust: ``lambda_A`` collapses (the channel is inferred unreliable) while
    ``lambda_B, lambda_C`` stay at full trust, and ``Var_A`` shows an INTERIOR
    MAXIMUM -- confidence transiently retreats ("indecision" = crisis) before
    the belief reorganises around the new reading and reliability recovers;
  * the conflict is LOCALIZED: post-conflict edge strain on (A,B) dominates
    (B,C) -- the read-off that lets BMR be targeted rather than scanned.

Noise-free evidence (``o = H @ phi`` exactly; the agent still assumes
``SIGMA_O``) makes every assertion deterministic.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from experiments.registry import ExperimentSpec, register
from src.structural import observables as obs
from src.structural import reliability as rel

NAMES = ("A", "B", "C")
EDGES = ((0, 1), (1, 2))


def _chain_prior(params: dict) -> tuple[jnp.ndarray, jnp.ndarray]:
    Pi0 = params["DIAG"] * jnp.eye(3)
    c = params["COUPLING"]
    Pi0 = Pi0.at[0, 1].set(c).at[1, 0].set(c).at[1, 2].set(c).at[2, 1].set(c)
    return Pi0, jnp.zeros(3)


def _rollout(params: dict, robust: bool) -> dict:
    """Host-side rollout with per-channel deposit accumulators ``(Jd, jd)`` and an
    EMA surprise ``z2_ema``. ``robust=False`` runs the SAME loop with
    ``lambda_bar == 1`` (plain accumulation) -- the baseline condition."""
    T, t_conf = params["T"], params["T_CONFLICT"]
    nu, beta, sigma_o = params["NU"], params["EMA_BETA"], params["SIGMA_O"]
    Pi0, h0 = _chain_prior(params)
    H = jnp.eye(3)
    m = 3
    inv_var = 1.0 / sigma_o ** 2
    Jk = jnp.stack([jnp.outer(H[k], H[k]) * inv_var for k in range(m)])  # (m,3,3)

    Jd = jnp.zeros((m, 3, 3))
    jd = jnp.zeros((m, 3))
    z2_ema = jnp.zeros(m)

    tr = {k: [] for k in ("mu", "var_A", "lam", "Pi_AA", "strain")}
    for t in range(T):
        phi = jnp.asarray(params["PHI_PRE"] if t < t_conf else params["PHI_POST"])
        o = H @ phi                                     # noise-free evidence

        lam = rel.student_t_weight(z2_ema, nu) if robust else jnp.ones(m)
        Pi_eff = Pi0 + jnp.einsum("k,kab->ab", lam, Jd)
        h_eff = h0 + jnp.einsum("k,ka->a", lam, jd)

        # surprise against the current (pre-deposit) effective belief, EMA'd.
        z2 = rel.channel_z2(Pi_eff, h_eff, H, o, sigma_o)
        z2_ema = beta * z2_ema + (1.0 - beta) * z2

        # per-channel deposit, ATTRIBUTED (the state instantaneous lambda lacks).
        Jd = Jd + Jk
        jd = jd + H * (o * inv_var)[:, None]            # row k: H[k] * o_k / s^2

        lam = rel.student_t_weight(z2_ema, nu) if robust else jnp.ones(m)
        Pi_eff = Pi0 + jnp.einsum("k,kab->ab", lam, Jd)
        h_eff = h0 + jnp.einsum("k,ka->a", lam, jd)

        tr["mu"].append(np.asarray(jnp.linalg.solve(Pi_eff, h_eff)))
        tr["var_A"].append(float(obs.node_marginal(Pi_eff, h_eff, 0)[1]))
        tr["lam"].append(np.asarray(lam))
        tr["Pi_AA"].append(float(Pi_eff[0, 0]))
        tr["strain"].append([float(obs.edge_strain(Pi_eff, h_eff, Pi0, h0, u, v))
                             for (u, v) in EDGES])
    return {k: np.asarray(v) for k, v in tr.items()}


def _figures(base: dict, rob: dict, params: dict, out_dir) -> None:
    T, t_conf = params["T"], params["T_CONFLICT"]
    ts = np.arange(T)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for i, n in enumerate(NAMES):
        ax.plot(ts, rob["mu"][:, i], lw=2.0, label=f"robust  mu_{n}")
        ax.plot(ts, base["mu"][:, i], lw=1.2, ls="--", alpha=0.7,
                label=f"baseline mu_{n}")
    ax.axvline(t_conf, color="k", lw=0.8, ls=":")
    ax.set_xlabel("step"); ax.set_ylabel("posterior mean")
    ax.set_title("the confident compromise vs the robust reorganisation")
    ax.legend(fontsize=7.5, ncol=2)
    plt.tight_layout(); plt.savefig(out_dir / "fig_abc_means.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for i, n in enumerate(NAMES):
        ax.plot(ts, rob["lam"][:, i], lw=2.0, label=f"lambda_{n}")
    ax.axvline(t_conf, color="k", lw=0.8, ls=":")
    ax.set_xlabel("step"); ax.set_ylabel("inferred reliability lambda")
    ax.set_title("channel A is inferred unreliable; B and C keep full trust")
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out_dir / "fig_abc_lambda.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(ts, base["var_A"], lw=1.4, ls="--", color="#922b21",
            label="baseline (monotone -- conflict leaves no trace)")
    ax.plot(ts, rob["var_A"], lw=2.0, color="#1e8449",
            label="robust (interior maximum -- indecision/crisis)")
    ax.axvline(t_conf, color="k", lw=0.8, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel("step"); ax.set_ylabel("marginal Var(A)")
    ax.set_title("conflict shows up as transient variance inflation")
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out_dir / "fig_abc_variance.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    im = ax.imshow(np.log10(rob["strain"].T + 1e-12), aspect="auto",
                   cmap="inferno", interpolation="nearest")
    ax.set_yticks([0, 1], ["A-B", "B-C"])
    ax.axvline(t_conf, color="w", lw=0.8, ls=":")
    ax.set_xlabel("step"); ax.set_title("log10 edge strain: the conflict is localized")
    fig.colorbar(im, ax=ax, shrink=0.85)
    plt.tight_layout(); plt.savefig(out_dir / "fig_abc_strain.png", dpi=130)
    plt.close(fig)


def run(out_dir, params: dict) -> None:
    T, t_conf, nu = params["T"], params["T_CONFLICT"], params["NU"]
    base = _rollout(params, robust=False)
    rob = _rollout(params, robust=True)

    # (a) baseline: the confident compromise -- monotone confidence, in-between mean.
    assert (np.diff(base["Pi_AA"]) >= -1e-6).all(), \
        "baseline Pi[A,A] must be non-decreasing"
    mu_A_final = float(base["mu"][-1, 0])
    assert -2.5 < mu_A_final < -1.0, \
        f"baseline final mu_A must sit between the sources (got {mu_A_final:.2f})"
    assert (np.diff(base["var_A"]) <= 1e-9).all(), \
        "baseline Var(A) must be monotone non-increasing (no trace of conflict)"

    # (b) robust: channel A inferred unreliable, B/C keep trust.
    lam_A, lam_B, lam_C = rob["lam"][:, 0], rob["lam"][:, 1], rob["lam"][:, 2]
    assert lam_A[t_conf:].min() < 0.35, \
        f"post-conflict lambda_A must collapse (min {lam_A[t_conf:].min():.2f})"
    below = lam_A < 0.5
    runs = np.diff(np.flatnonzero(np.diff(np.concatenate(
        [[0], below.astype(int), [0]]))).reshape(-1, 2), axis=1)
    assert runs.size and runs.max() >= 15, \
        "lambda_A must stay below 0.5 for >= 15 consecutive steps"
    assert lam_B.min() > 0.9 and lam_C.min() > 0.9, \
        "the unconflicted channels must keep full trust throughout"

    # (c) crisis: Var(A) has an interior maximum (impossible without the
    # retrospective deposit re-scaling -- PSD adds are monotone).
    win = rob["var_A"][t_conf:T - 5 + 1]
    peak = int(win.argmax())
    assert 0 < peak < len(win) - 1, "Var(A) peak must be interior to (T_CONFLICT, T-5)"
    assert win[peak] > 1.1 * win[0] and win[peak] > 1.1 * win[-1], \
        "the variance peak must clear both endpoints by 10%"

    # (d) localization: the conflicted edge carries the strain.
    s_ab = rob["strain"][t_conf:, 0].mean()
    s_bc = rob["strain"][t_conf:, 1].mean()
    assert s_ab >= 2.0 * s_bc, \
        f"post-conflict strain must localize on A-B ({s_ab:.3f} vs {s_bc:.3f})"

    _figures(base, rob, params, out_dir)
    np.savez_compressed(
        out_dir / "simulation_arrays.npz",
        mu_baseline=base["mu"], mu_robust=rob["mu"],
        var_A_baseline=base["var_A"], var_A_robust=rob["var_A"],
        lambda_robust=rob["lam"], Pi_AA_baseline=base["Pi_AA"],
        strain_robust=rob["strain"], strain_baseline=base["strain"],
    )
    summary = {
        "config": {k: v for k, v in params.items()},
        "baseline_final_mu_A": mu_A_final,
        "robust_min_lambda_A": float(lam_A[t_conf:].min()),
        "robust_var_A_peak": float(win[peak]),
        "robust_var_A_peak_step": int(t_conf + peak),
        "post_conflict_strain_AB": float(s_ab),
        "post_conflict_strain_BC": float(s_bc),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2),
                                          encoding="utf-8")
    print(f"HEADLINE: a plain Gaussian agent averages a persistent contradiction "
          f"into a confident compromise (final mu_A={mu_A_final:.2f}, variance "
          f"monotone -- no trace of conflict); the same agent with inferred "
          f"reliability instead enters a legible crisis: lambda_A collapses to "
          f"{lam_A[t_conf:].min():.2f} while lambda_B,C hold >0.9, Var(A) inflates "
          f"{float(win[peak] / win[0]):.1f}x at step {int(t_conf + peak)} before the "
          f"belief reorganises, and the strain localizes on the conflicted edge "
          f"(A-B {s_ab:.1e} vs B-C {s_bc:.1e}, {s_ab / s_bc:.1f}x) -- conflict becomes a STATE the "
          f"model can react to, not an artifact it absorbs.")


register(ExperimentSpec(
    model="phlogiston",
    name="abc_conflict",
    description="Inferred reliability (Student-t lambda) on a 3-node chain with a "
                "persistently contradictory channel: the baseline's confident "
                "compromise vs the robust agent's crisis -- lambda collapse, "
                "transient variance inflation, and strain localized on the "
                "conflicted edge.",
    run=run,
    out_dir="abc_conflict",
    params=dict(
        T=120, T_CONFLICT=30, NU=4.0, EMA_BETA=0.9, SIGMA_O=0.5,
        DIAG=1.0, COUPLING=-0.6,
        PHI_PRE=(1.0, 1.0, 1.0), PHI_POST=(-3.0, 1.0, 1.0),
    ),
    seeds=(),
    canonical=False,
    consumes=dict(figures=["fig_abc_means", "fig_abc_lambda",
                           "fig_abc_variance", "fig_abc_strain"]),
))
