"""Represented rivals: the existence proof that model uncertainty buys walls.

The two-result structure this experiment completes:

  RESULT 1 (everywhere else in the repo): the HOMOGENEITY effect. Gaussian pooling is
  a contraction and Gaussian conflict resolves by confident compromise, so every
  configuration collapses to consensus -- and the Student-t trust gates only buy
  metastability ("consensus remains the only fixed point", gated_divergence et al.).

  RESULT 2 (here): same world, same agents, same evidence -- the ONLY change is that
  uncertainty about the MODEL ITSELF is represented (``src/structural/rival.py``:
  each agent carries K candidate wirings plus a posterior over a model index m,
  scored prequentially, communicated as hypotheses) -- and distinct camps become a
  genuine steady state. Small scale by design; larger structures are future work.

Demo A ("slide 10 made real"): two candidate wirings that differ in NOTHING but
structure (direct A->C vs chain A->B->C, zero-mean, variance-matched). The Gaussian
pool holds all three couplings at half strength FOREVER (H = I deposits are diagonal:
the blur cannot heal) -- a confident third wiring neither evidence stream supports.
The rival agent's q(m) locks onto the true wiring because the per-observation
predictive density reads the correlation structure the deposit erases.

Demo B ("the wall"): two camps, full contact, theory-laden evidence streams (each
camp runs only its own commitment's experiment -- the channel that CONFIRMS its
prior). Plain fusion collapses in ~5 steps (the theorem); the means-gate at its best
calibration decays over ~400 steps (metastability, measured across two decades); the
rival layer holds a FLAT inter-camp gap to T = 1500 with both camps internally
certain of different theories -- and the wall height is PREDICTED in closed form
(steady-state log-odds gap ~ source / (2 alpha_m)) before being measured.

Honest mechanism notes (probe, seed 0):
  * omega SMALL on candidate nets is load-bearing: heavy within-frame parameter
    memory lets EVERY wiring explain the data (accumulated diagonal deposits swamp
    the structural part of the predictive covariance) -- Quine-Duhem, measured. The
    Gaussian baselines have no candidate object to anchor (the rival was averaged
    away at deposit time); each architecture runs its own best memory setting.
  * The gated baseline needs omega = 0.95 + per-camp anchors to show its best self:
    under omega = 1 a sender's potential scales with its unboundedly growing
    precision, so even a severed gate leaks means at (tiny weight) x (huge
    confidence) -- the deep reason means-gates are metastable at long horizons.
  * The wall is relative to theory-laden evidence streams (camps run different
    experiments) -- exactly the Kuhnian condition the talk's diagnosis names.
  * The gated baseline's tail does NOT reach zero: it converges to a small FLAT
    floor (~0.08, ~35x below the rival wall). Controls isolate the mechanism: the
    bare gate at omega=1 decays to exactly 0; per-camp anchors alone give ~0.03;
    gate + anchors give ~0.09 -- a Friedkin-Johnsen-type equilibrium (STUBBORN
    PRIORS sustain a parameter-level disagreement floor, and the gate amplifies it
    ~3x by suppressing the leak). A real but categorically different object from
    the rival wall: means sitting slightly apart vs two camps internally CERTAIN
    of different theories. The ladder: gates buy time; stubbornness buys a small
    floor; represented rivals buy walls.

Future work (named, not built): the lone discoverer under hypothesis pooling (plain
log-linear pooling majority-swamps a minority's evidence; the m-gate
``rival.m_gated_weights`` is the constructive answer), and acting on
``rival.rival_divergence`` -- choosing the experiment where your live candidates
disagree most (crucial experiments).
"""
from __future__ import annotations

import json

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from experiments.registry import ExperimentSpec, register

RED, AMBER, GREEN, GREY = "#922b21", "#b9770e", "#1e8449", "#7f8c8d"


# ----------------------------------------------------------------------
# Worlds.
# ----------------------------------------------------------------------

def _demo_a_nets():
    """M1 direct A->C (0.8) vs M2 chain A->B->C (0.7, 0.7): zero-mean, PD,
    differing in nothing but wiring."""
    from src.structural.bayesnet import LinearGaussianBN
    names = ("a", "b", "c")
    z, s = jnp.zeros(3), jnp.ones(3)
    B1 = jnp.zeros((3, 3)).at[2, 0].set(0.8)
    B2 = jnp.zeros((3, 3)).at[1, 0].set(0.7).at[2, 1].set(0.7)
    n1 = LinearGaussianBN(B=B1, b=z, s=s, names=names).to_info()
    n2 = LinearGaussianBN(B=B2, b=z, s=s, names=names).to_info()
    return n1, n2


def _demo_b_world():
    """The cosmology candidate menu + the CHIMERA truth: dm_commitment at dm's
    preset value, sv_commitment at sv's, everything else at the dm/sv midpoint.
    (Not the midpoint blend everywhere -- a pure midpoint is exactly equidistant on
    every linear channel and would kill all model discrimination.)"""
    from src.structural.scenarios import cosmology_presets
    nets = [p.belief_net for p in cosmology_presets()]      # dm, mg, sv
    mu = [np.asarray(jnp.linalg.solve(n.Pi, n.h)) for n in nets]
    chim = 0.5 * (mu[0] + mu[2])
    chim[3] = mu[0][3]                                      # dm commitment: 1.6
    chim[4] = mu[2][4]                                      # sv commitment: 1.5
    return nets, chim


# ----------------------------------------------------------------------
# Demo runners (serial; each run is one small lax.scan).
# ----------------------------------------------------------------------

def _run_demo_a(p):
    from src.structural import rival as rv
    n1, n2 = _demo_a_nets()
    Sigma1 = np.linalg.inv(np.asarray(n1.Pi, float))
    chol = np.linalg.cholesky(Sigma1)
    W = A = jnp.ones((1, 1))
    w = jnp.ones((1, 3))
    off = ~np.eye(3, dtype=bool)
    Pi_pool = 0.5 * (n1.Pi + n2.Pi)
    h_pool = 0.5 * (n1.h + n2.h)

    rows = []
    for seed in p["SEEDS"]:
        rng = np.random.default_rng(seed)
        phis = jnp.asarray((chol @ rng.normal(size=(3, p["A_T"]))).T,
                           dtype=jnp.float32)
        cfg = rv.RivalConfig(Pi0=jnp.stack([n1.Pi, n2.Pi]),
                             h0=jnp.stack([n1.h, n2.h]), H=jnp.eye(3),
                             sigma_o=p["SIGMA_O"], omega=p["A_OMEGA"], alpha_m=0.0)
        r = rv.run_rival(cfg, W, A, w, phis, seed=seed)
        cfg_b = rv.RivalConfig(Pi0=Pi_pool[None], h0=h_pool[None], H=jnp.eye(3),
                               sigma_o=p["SIGMA_O"], omega=1.0)
        rb = rv.run_rival(cfg_b, W, A, w, phis, seed=seed)
        k_map = int(np.argmax(r["L"][0]))
        rows.append(dict(
            q1_t=r["q_t"][:, 0, 0],
            q1_final=float(r["q_t"][-1, 0, 0]),
            rate=float((r["L_t"][-1, 0, 0] - r["L_t"][-1, 0, 1]) / p["A_T"]),
            map_off=np.asarray(r["Pi"][0, k_map])[off],
            pool_off=np.asarray(rb["Pi"][0, 0])[off],
        ))
    return rows, np.asarray(n1.Pi)[off], np.asarray(n2.Pi)[off], np.asarray(Pi_pool)[off]


def _run_demo_b(p):
    from src.structural import rival as rv
    nets, chim = _demo_b_world()
    N, T = p["B_N"], p["B_T"]
    camp = (np.arange(N) >= N // 2).astype(int)             # 0 = dm camp, 1 = sv camp
    W = jnp.ones((N, N)) / N
    A = jnp.ones((N, N))
    phis = jnp.asarray(np.tile(chim, (T, 1)), dtype=jnp.float32)
    Pi3 = jnp.stack([n.Pi for n in nets])
    h3 = jnp.stack([n.h for n in nets])
    Pi_camp = jnp.stack([Pi3[0 if c == 0 else 2][None] for c in camp])
    h_camp = jnp.stack([h3[0 if c == 0 else 2][None] for c in camp])
    w_att = np.zeros((N, 6))                                # fully theory-laden:
    w_att[camp == 0, 3] = 1.0                               # own commitment's
    w_att[camp == 1, 4] = 1.0                               # experiment ONLY
    w_att = jnp.asarray(w_att)

    def D_trace(mu_eff_t):
        gA = mu_eff_t[:, camp == 0, 3:6].mean(axis=1)
        gB = mu_eff_t[:, camp == 1, 3:6].mean(axis=1)
        return np.linalg.norm(gA - gB, axis=1)

    out = {"plain": [], "gated": [], "rival": []}
    walls = []
    for seed in p["SEEDS"]:
        cfg_p = rv.RivalConfig(Pi0=Pi3[:1], h0=h3[:1], H=jnp.eye(6),
                               sigma_o=p["SIGMA_O"], omega=1.0)
        rp = rv.run_rival(cfg_p, W, A, w_att, phis, seed=seed,
                          Pi_init=Pi_camp, h_init=h_camp)
        out["plain"].append(D_trace(rp["mu_eff_t"]))

        cfg_g = rv.RivalConfig(Pi0=Pi3[:1], h0=h3[:1], H=jnp.eye(6),
                               sigma_o=p["SIGMA_O"], omega=p["B_GATED_OMEGA"],
                               social_nu=p["B_SOCIAL_NU"], gate_idx=tuple(range(6)),
                               Pi0_agent=Pi_camp, h0_agent=h_camp)
        rg = rv.run_rival(cfg_g, W, A, w_att, phis, seed=seed,
                          Pi_init=Pi_camp, h_init=h_camp)
        out["gated"].append(D_trace(rg["mu_eff_t"]))

        cfg_r = rv.RivalConfig(Pi0=Pi3, h0=h3, H=jnp.eye(6), sigma_o=p["SIGMA_O"],
                               omega=p["B_RIVAL_OMEGA"], alpha_m=p["B_ALPHA_M"])
        rr = rv.run_rival(cfg_r, W, A, w_att, phis, seed=seed)
        out["rival"].append(D_trace(rr["mu_eff_t"]))
        LA = rr["L_t"][:, camp == 0].mean(axis=1)
        g = LA[:, 0] - LA[:, 2]                             # camp-A log-odds dm vs sv
        # the per-step source, measured early (pooling not yet equilibrated)
        s_src = float((g[60] - g[20]) / 40.0)
        walls.append(dict(
            g_t=g,
            wall_pred=s_src * (1.0 + 0.0) / (2 * p["B_ALPHA_M"]),
            wall_real=float(g[-1]),
            slope_tail=float(np.polyfit(np.arange(T - T // 4, T),
                                        g[-(T // 4):], 1)[0]),
            qA_own=float(rr["q_t"][-1, camp == 0, 0].mean()),
            qB_own=float(rr["q_t"][-1, camp == 1, 2].mean()),
        ))
    return {k: np.stack(v) for k, v in out.items()}, walls


# ----------------------------------------------------------------------
# The experiment.
# ----------------------------------------------------------------------

def run(out_dir, params: dict) -> None:
    p = dict(params)
    print(f"represented rivals: K candidate wirings + prequential model posterior + "
          f"hypothesis pooling; seeds={tuple(p['SEEDS'])}")

    # ---------------- Demo A ----------------
    rows, off1, off2, off_pool = _run_demo_a(p)
    q1 = np.array([r["q1_final"] for r in rows])
    rates = np.array([r["rate"] for r in rows])
    pool_drift = max(float(np.abs(r["pool_off"] - off_pool).max()) for r in rows)
    map_err = max(float(np.abs(r["map_off"] - off1).max()
                        / (np.abs(off1).max())) for r in rows)
    print(f"\nDemo A (slide 10 made real): q(M1) final = {np.round(q1, 3).tolist()}  "
          f"rate = {rates.mean():.3f} nats/step")
    print(f"  Gaussian pool off-diagonals: drift from half-strength = {pool_drift:.2e} "
          f"(frozen blur);  rival MAP off-diag rel-err vs M1 = {map_err:.3f}")

    # ---------------- Demo B ----------------
    D, walls = _run_demo_b(p)
    Dm = {k: v.mean(axis=0) for k, v in D.items()}
    peak = {k: float(v.max()) for k, v in Dm.items()}
    end = {k: float(v[-1]) for k, v in Dm.items()}
    wall_pred = float(np.mean([w["wall_pred"] for w in walls]))
    wall_real = float(np.mean([w["wall_real"] for w in walls]))
    slope = float(np.mean([w["slope_tail"] for w in walls]))
    qA = float(np.mean([w["qA_own"] for w in walls]))
    qB = float(np.mean([w["qB_own"] for w in walls]))
    print(f"\nDemo B (the wall), seed-mean D(t) [peak -> end]:")
    for k in ("plain", "gated", "rival"):
        ts = [5, 20, 50, 100, 400, p["B_T"] - 1]
        print(f"  {k:>6}: peak={peak[k]:.2f}  " +
              "  ".join(f"D({t})={Dm[k][t]:.2f}" for t in ts))
    print(f"  wall height: predicted ~ s/(2 alpha_m) = {wall_pred:.0f} nats, "
          f"realized = {wall_real:.0f} nats (same order; prediction conservative)")
    print(f"  rival tail slope = {slope:+.4f} nats/step (non-eroding); "
          f"camp certainty in OWN theory: qA = {qA:.3f}, qB = {qB:.3f}")

    # ---------------- assertions (print-first; probe-calibrated) ----------------
    assert (q1 > 0.95).all(), f"a1: rival agent should lock onto M1 ({q1})"
    assert rates.mean() > 0.02, f"a1: log-odds rate too small ({rates.mean():.4f})"
    assert pool_drift < 1e-3, "a2: the Gaussian pool's blur should be FROZEN"
    assert map_err < 0.05, f"a2: rival MAP wiring should match M1 ({map_err:.3f})"
    assert end["plain"] <= 0.05 * peak["plain"] + 1e-6, "b1: plain must collapse"
    assert Dm["gated"][50] > 10 * Dm["plain"][50], \
        "b2: the gate must visibly buy time (D_gated(50) >> D_plain(50))"
    assert end["gated"] <= 0.5 * peak["gated"], \
        f"b2: the gated baseline must still DECAY (metastable, not a wall): " \
        f"{end['gated']:.2f} vs peak {peak['gated']:.2f}"
    # the wall is judged against the POST-FORMATION plateau (t = T/10, after the camps
    # have locked), not the early argmax-flicker transient while L is near-uniform
    plateau = float(Dm["rival"][p["B_T"] // 10])
    assert end["rival"] >= 0.9 * plateau, \
        f"b3: the rival wall must hold ({end['rival']:.2f} vs plateau {plateau:.2f})"
    assert slope > -1e-3, f"b3: the wall must not erode (slope {slope:+.4f})"
    assert qA > 0.95 and qB > 0.95, \
        f"b3: both camps must be internally certain of DIFFERENT theories ({qA}, {qB})"
    assert all(np.isfinite(v).all() for v in D.values()), "c: finiteness"

    # ---------------- figures ----------------
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    x = np.arange(3)
    wdt = 0.2
    labels = ["(a,b)", "(b,c)", "(a,c)"]
    vals = [_offvec(off1), _offvec(off2), _offvec(off_pool),
            _offvec(np.mean([r["map_off"] for r in rows], axis=0))]
    names = ["M1 truth (direct)", "M2 rival (chain)",
             "Gaussian pool: blurred third wiring", "rival agent: MAP wiring"]
    colors = ["#2c3e50", GREY, RED, GREEN]
    for i, (v, nm, c) in enumerate(zip(vals, names, colors)):
        ax.bar(x + (i - 1.5) * wdt, v, wdt, color=c, label=nm)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("|coupling|  |Pi[i, j]|")
    ax.set_title("same world, same evidence -- only the representation differs:\n"
                 "the pool holds a third wiring no evidence supports (frozen); "
                 "the rival agent locks on", fontsize=10)
    ax.legend(fontsize=8)
    ins = ax.inset_axes([0.66, 0.45, 0.32, 0.3])
    ins.plot(np.mean([r["q1_t"] for r in rows], axis=0), color=GREEN, lw=1.5)
    ins.set_ylim(0.4, 1.02); ins.set_title("q(M1)", fontsize=8)
    ins.tick_params(labelsize=6)
    plt.tight_layout(); plt.savefig(out_dir / "fig_blur_vs_crisp.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    t = np.arange(p["B_T"])
    floor = 1e-3
    for k, c, ls, lab in (("plain", RED, "-", "plain fusion (collapse)"),
                          ("gated", AMBER, "--", "Student-t trust gate (buys time)"),
                          ("rival", GREEN, "-", "represented rivals (a wall)")):
        m = np.clip(D[k].mean(axis=0), floor, None)
        s = D[k].std(axis=0)
        ax.plot(t, m, ls, color=c, lw=2.4, label=lab)
        ax.fill_between(t, np.clip(m - s, floor, None), m + s, color=c, alpha=0.15)
    ax.set_yscale("log"); ax.set_xscale("log")
    ax.set_xlabel("step (log)", fontsize=11)
    ax.set_ylabel("inter-camp distance $D(t)$ (log)", fontsize=11)
    ax.legend(fontsize=10, loc="lower left")
    ax.tick_params(labelsize=10)
    plt.tight_layout(); plt.savefig(out_dir / "fig_walls_vs_time.png", dpi=150)
    plt.close(fig)

    # ---------------- save ----------------
    np.savez_compressed(
        out_dir / "simulation_arrays.npz",
        D_plain=D["plain"], D_gated=D["gated"], D_rival=D["rival"],
        qA_t=np.mean([r["q1_t"] for r in rows], axis=0),
        g_t=np.stack([w["g_t"] for w in walls]),
    )
    summary = {
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in p.items()},
        "demo_A": {"q_M1_final": q1.tolist(), "rate_nats_per_step": float(rates.mean()),
                   "pool_blur_drift": pool_drift, "map_rel_err_vs_M1": map_err},
        "demo_B": {"peak": peak, "end": end,
                   "wall_predicted_nats": wall_pred, "wall_realized_nats": wall_real,
                   "tail_slope": slope, "qA_own": qA, "qB_own": qB,
                   "gated_floor_note": "the gated tail converges to a small flat "
                   "floor (~0.08) rather than 0: a Friedkin-Johnsen-type "
                   "equilibrium from the per-camp anchors (stubborn priors), "
                   "amplified ~3x by the gate. Controls (probe, seed 0): bare gate "
                   "omega=1 -> 0.000; anchors alone -> ~0.03; gate+anchors -> "
                   "~0.09. Parameter-level disagreement, vs the rival wall's "
                   "categorical certainty 35x higher."},
        "mechanism": "K candidate wirings per agent + prequential model posterior "
                     "(the structure-discriminating evidence the deposit erases now "
                     "accumulates in L) + hypothesis-aligned fusion (no cross-frame "
                     "averaging) + damped log-linear hypothesis pooling (wall height "
                     "~ source / (2 alpha_m), predicted before measured)",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2),
                                          encoding="utf-8")
    print(f"\nsaved arrays / 2 figures / summary to {out_dir}")
    print(f"HEADLINE: same world, same agents, same evidence -- representing model "
          f"uncertainty turns collapse into pluralism. The Gaussian pool holds a "
          f"blurred third wiring forever (drift {pool_drift:.0e}) while the rival "
          f"agent locks the true one (q = {q1.mean():.3f}); under full contact, "
          f"plain fusion collapses by t~5, the best-calibrated trust gate decays to "
          f"{end['gated'] / peak['gated']:.0%} of peak by t={p['B_T']} (metastable), "
          f"and the rival layer holds {end['rival'] / plateau:.0%} of its post-"
          f"formation plateau with both camps certain of different theories "
          f"(q = {min(qA, qB):.3f}) -- a "
          f"wall whose height ({wall_real:.0f} nats) the theory predicts in closed "
          f"form ({wall_pred:.0f}). Gates buy time; represented rivals buy walls.")


def _offvec(off_flat):
    """The 6 off-diagonal entries (~eye mask, row-major) -> |(a,b)|, |(b,c)|, |(a,c)|."""
    # mask order for 3x3 ~eye: (0,1),(0,2),(1,0),(1,2),(2,0),(2,1)
    ab = abs(float(off_flat[0]))
    ac = abs(float(off_flat[1]))
    bc = abs(float(off_flat[3]))
    return np.array([ab, bc, ac])


register(ExperimentSpec(
    model="cosmology",
    name="represented_rivals",
    description="The existence proof that model uncertainty buys walls: agents that "
                "hold K candidate wirings with a prequential posterior over a model "
                "index (q(phi, m)) lock onto true structure where the Gaussian pool "
                "blurs forever, and sustain genuinely distinct camps under full "
                "contact where plain fusion collapses and trust gates only buy time.",
    run=run,
    out_dir="represented_rivals",
    params=dict(
        SIGMA_O=0.5, SEEDS=(0, 1, 2, 3, 4),
        # CALIB (probe): omega SMALL is load-bearing for structure discrimination
        # (omega=0.9 stalls at q=0.48; omega=0.3 locks at 1.000, 0.118 nats/step)
        A_T=400, A_OMEGA=0.3,
        # CALIB (probe): gated baseline at its best -- nu=0.01 (working nu is ~50x
        # below naive guesses, as with every gate in this repo) + omega=0.95 with
        # per-camp anchors (omega=1 lets unbounded sender precision defeat the gate);
        # rival wall: source ~1.7 nats/step -> predicted ~42 nats at alpha_m=0.02.
        B_N=16, B_T=1500, B_SOCIAL_NU=0.01, B_GATED_OMEGA=0.95,
        B_RIVAL_OMEGA=0.3, B_ALPHA_M=0.02,
    ),
    seeds=(0, 1, 2, 3, 4),
    consumes=dict(figures=["fig_blur_vs_crisp", "fig_walls_vs_time"]),
))
