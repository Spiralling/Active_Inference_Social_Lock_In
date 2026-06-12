"""The Kuhn cycle replicated in q(phi, m): crisis is now a STATE, not a read-out.

The paradigm-shift phenomena of the phlogiston world, re-run on the represented-rival
engine (``src/structural/rival.py``): each agent holds BOTH paradigms as candidate
nets -- the incumbent ``phlogiston_bn`` (hub common-cause wiring) and an oxygen CPD
twin (no hub principle, same belt mass-balance, oxygen means) -- with a prequential
posterior over the model index and hypothesis communication. The phenomena become
native states instead of ledger flags:

  * NORMAL SCIENCE forms endogenously: from a uniform start, the phlogiston-regime
    data lock q(m) onto the incumbent (~21 nats by the shift).
  * CRISIS is the posterior going TORN after the world turns: the entropy of q(m)
    spikes for a measurable interval -- "suspended judgment under conflict", the slot
    the talk's diagnosis says the single-Gaussian form lacks, now occupied.
  * REVOLUTION is the q-flip; NEW NORMAL its stable aftermath.
  * DENIAL / LOCK-IN: communities that stop attending the disconfirming channel
    after the shift (an experiment you don't run discriminates nothing -- a w = 0 row
    contributes a candidate-independent constant) delay or never reach revolution.
  * SOCIAL REVOLUTION: with hypothesis pooling, a 25% full-attention vanguard
    converts a 75% conservative majority that NEVER runs the disconfirming
    experiment -- the revolution travels through communication of hypotheses, not
    through data. Without pooling the conservatives stay locked forever.

Design note (probe round 1, the honest reason for the TWO-PHASE runs): censoring the
disconfirming channel from t = 0 weakens the phlogiston LOCK as much as the overturn
(pre-shift those channels read -1, which SUPPORTS the incumbent), inverting the
denial story -- censored agents flipped EARLIER. Kuhnian denial bites the anomaly
era: all communities share a full-attention normal-science phase to t_shift, and the
attention regimes differentiate when the world turns.

Honest findings (probe, seed 0):
  * The pre-registered prediction "flip delay ~ 1/(1 - gamma)" is REFUTED at mid
    gamma: the delay stays flat (~74-93 steps) up to gamma = 0.9 and then explodes
    (356 at 0.99, never at 1.0). Mechanism: under high attention the INCUMBENT
    candidate adapts to the anomalous data (Quine-Duhem accommodation, the same
    omega-mediated effect measured in represented_rivals Demo A), eating the
    discrimination rate -- so lock-in is a sharp threshold, not a smooth divergence.
  * The hand-poked ``phlogiston_prior`` is indefinite (out of the engine's PD
    contract); both candidates here are CPD compiles (PD by construction, min
    eigenvalues asserted).

Future work: endogenous denial (the Student-t channel gate of crisis_trace driving
attention, instead of a fixed gamma), and the conviction tilt as a q(m) prior.
"""
from __future__ import annotations

import json

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from experiments.registry import ExperimentSpec, register

RED, AMBER, GREEN, GREY = "#922b21", "#b9770e", "#1e8449", "#7f8c8d"


# ----------------------------------------------------------------------
# The candidate menu + world.
# ----------------------------------------------------------------------

def oxygen_bn(cfg, conviction: float = 2.0):
    """The oxygen paradigm as ``phlogiston_bn``'s CPD twin: NO hub couplings (the
    theory has no phlogiston principle -- a genuinely structural difference), the
    same belt mass-balance edges, and the disagreement-node means at
    ``mu_oxy_mass``. PD by the CPD compile, like the incumbent."""
    from src.structural.bayesnet import LinearGaussianBN
    from src.structural.phlogiston import DISAGREEMENT_NODES, HUB, _idx
    names = cfg.node_names
    d = len(names)
    idx = _idx(cfg)
    B = jnp.zeros((d, d))
    anom = idx["calx_heavier_than_metal"]
    for n in ("mass_change_sign", "gas_consumed"):
        B = B.at[anom, idx[n]].set(cfg.mass_coupling)
    target = jnp.full((d,), cfg.mu_agree).at[idx[HUB]].set(0.0)
    for n in DISAGREEMENT_NODES:
        target = target.at[idx[n]].set(cfg.mu_oxy_mass)
    b = (jnp.eye(d) - B) @ target
    s = jnp.full((d,), 1.0 / conviction)
    return LinearGaussianBN(B=B, b=b, s=s, names=names)


def _build(p, n_steps):
    from src.structural.phlogiston import (
        StructuralConfig, phlogiston_bn, gravimetric_H, gravimetric_rows,
        phi_true_at, DISAGREEMENT_NODES)
    cfg = StructuralConfig(regime_schedule="step", t_shift=p["T_SHIFT"],
                           n_steps=n_steps, sigma_o=p["SIGMA_O"])
    inc = phlogiston_bn(cfg, conviction=2.0).to_info()
    oxy = oxygen_bn(cfg, conviction=2.0).to_info()
    H = gravimetric_H(cfg)
    rows = gravimetric_rows(cfg)
    disc = [i for i, r in enumerate(rows)
            if ("mass_balance" in r) or (r in DISAGREEMENT_NODES)]
    phis = jnp.stack([phi_true_at(cfg, t) for t in range(n_steps)])
    e_inc = float(np.linalg.eigvalsh(np.asarray(inc.Pi)).min())
    e_oxy = float(np.linalg.eigvalsh(np.asarray(oxy.Pi)).min())
    assert e_inc > 0 and e_oxy > 0, f"PD contract: {e_inc}, {e_oxy}"
    return inc, oxy, H, disc, phis


def _entropy(q):
    return -(q * np.log(q + 1e-12)).sum(axis=-1)


def _flip(q_oxy_mean, thresh):
    hit = q_oxy_mean > thresh
    return int(np.argmax(hit)) if hit.any() else -1


# ----------------------------------------------------------------------
# Runners.
# ----------------------------------------------------------------------

def _run(p, n_steps, N, w_att, alpha_m, seed, two_phase=False):
    """Single- or two-phase rival run; two-phase shares a full-attention
    normal-science era to t_shift, then switches to ``w_att``."""
    from src.structural import rival as rv
    inc, oxy, H, disc, phis = _build(p, n_steps)
    rcfg = rv.RivalConfig(Pi0=jnp.stack([inc.Pi, oxy.Pi]),
                          h0=jnp.stack([inc.h, oxy.h]), H=H,
                          sigma_o=p["SIGMA_O"], omega=p["OMEGA"], alpha_m=alpha_m)
    W = jnp.ones((N, N)) / N
    A = jnp.ones((N, N))
    m = H.shape[0]
    if not two_phase:
        r = rv.run_rival(rcfg, W, A, w_att, phis, seed=seed)
        return r["q_t"], r["D_top2_t"]
    ts = p["T_SHIFT"]
    r1 = rv.run_rival(rcfg, W, A, jnp.ones((N, m)), phis[:ts], seed=seed)
    r2 = rv.run_rival(rcfg, W, A, w_att, phis[ts:], seed=seed + 1000,
                      Pi_init=r1["Pi"], h_init=r1["h"], L_init=r1["L"])
    return (np.concatenate([r1["q_t"], r2["q_t"]]),
            np.concatenate([r1["D_top2_t"], r2["D_top2_t"]]))


def _disc_attention(p, N, gamma, n_van=0):
    """(N, m) attention: full everywhere except the disconfirming rows at 1-gamma;
    the first ``n_van`` agents (the vanguard) keep full attention."""
    _, _, H, disc, _ = _build(p, 8)
    w = np.ones((N, H.shape[0]))
    w[n_van:, disc] = 1.0 - gamma
    return jnp.asarray(w)


# ----------------------------------------------------------------------
# The experiment.
# ----------------------------------------------------------------------

def run(out_dir, params: dict) -> None:
    p = dict(params)
    seeds = tuple(p["SEEDS"])
    ts = p["T_SHIFT"]
    print(f"rival Kuhn cycle: K=2 {{phlogiston (hub wiring), oxygen (no hub)}}, "
          f"prequential q(m), hypothesis pooling; world steps at t={ts}; "
          f"seeds={seeds}")

    # ---------------- Demo 1: the cycle ----------------
    N1, T1 = p["N_CYCLE"], p["T_CYCLE"]
    m_att = _disc_attention(p, N1, 0.0)
    q_runs, D_runs = zip(*[_run(p, T1, N1, m_att, p["ALPHA_M"], s) for s in seeds])
    q_oxy1 = np.stack([q[:, :, 1].mean(axis=1) for q in q_runs])      # (S, T)
    Hq1 = np.stack([_entropy(q).mean(axis=1) for q in q_runs])        # (S, T)
    qm, Hm = q_oxy1.mean(axis=0), Hq1.mean(axis=0)
    normal_ok = float(qm[ts // 2:ts].max())
    torn = np.where(Hm[ts:] > p["H_CRISIS"])[0]
    crisis_on = int(ts + torn[0]) if torn.size else -1
    crisis_w = int(torn[-1] - torn[0] + 1) if torn.size else 0
    flip1 = _flip(qm, p["Q_LOCK"])
    print(f"\nDemo 1 (the cycle): normal science max q_oxy = {normal_ok:.3f}; "
          f"crisis onset t={crisis_on}, torn width = {crisis_w} steps; "
          f"revolution t={flip1}; new normal q_oxy(end) = {qm[-1]:.3f}")

    # ---------------- Demo 2: denial / lock-in ----------------
    T2, N2 = p["T_LOCK"], p["N_CYCLE"]
    delays = {}
    for g in p["GAMMAS"]:
        w2 = _disc_attention(p, N2, g)
        qs = np.stack([_run(p, T2, N2, w2, p["ALPHA_M"], s, two_phase=True)[0]
                       [:, :, 1].mean(axis=1) for s in seeds])
        f = _flip(qs.mean(axis=0), p["Q_LOCK"])
        delays[g] = (f - ts) if f > 0 else -1
        print(f"Demo 2: gamma={g}: revolution delay = "
              f"{delays[g] if delays[g] > 0 else f'NEVER (q_end={qs.mean(0)[-1]:.3f})'}")

    # ---------------- Demo 3: social revolution ----------------
    N3, T3, nv = p["N_MIXED"], p["T_LOCK"], int(p["N_MIXED"] * p["VANGUARD_FRAC"])
    w3 = _disc_attention(p, N3, p["GAMMA_CONS"], n_van=nv)
    social = {}
    for am in (0.0, p["ALPHA_M"]):
        qs = [_run(p, T3, N3, w3, am, s, two_phase=True)[0] for s in seeds]
        qv = np.stack([q[:, :nv, 1].mean(axis=1) for q in qs]).mean(axis=0)
        qc = np.stack([q[:, nv:, 1].mean(axis=1) for q in qs]).mean(axis=0)
        social[am] = dict(qv=qv, qc=qc, fv=_flip(qv, p["Q_LOCK"]),
                          fc=_flip(qc, p["Q_LOCK"]))
        print(f"Demo 3: alpha_m={am}: vanguard flip t={social[am]['fv']}, "
              f"conservatives flip t={social[am]['fc']} "
              f"(q_cons end = {qc[-1]:.3f})")

    # ---------------- assertions (print-first, probe-calibrated) ----------------
    assert normal_ok < 0.1, f"k1: normal science must form (q_oxy {normal_ok:.3f})"
    assert crisis_on >= ts and crisis_w >= 5, \
        f"k2: a visible post-shift torn interval is required ({crisis_on}, {crisis_w})"
    assert flip1 > 0 and (qm[flip1:] > 0.85).all(), \
        "k3: revolution must complete and the new normal must be stable"
    g_lo, g_hi = min(p["GAMMAS"]), 0.99
    assert delays[g_hi] < 0 or delays[g_hi] > 3 * delays[g_lo], \
        f"d1: gamma={g_hi} must delay the revolution >= 3x ({delays})"
    assert delays[1.0] < 0, "d2: gamma=1 must be locked in forever (no source)"
    am = p["ALPHA_M"]
    assert social[am]["qc"][-1] > 0.9 and social[0.0]["qc"][-1] < 0.5, \
        "s1: conservatives must convert WITH hypothesis pooling and ONLY with it"
    assert 0 < social[am]["fv"] < social[am]["fc"], \
        "s2: the vanguard must flip before the conservatives (the lag is the spread)"
    assert np.isfinite(qm).all() and np.isfinite(Hm).all(), "health"

    # ---------------- figures ----------------
    t1 = np.arange(T1)
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.plot(t1, qm, color=GREEN, lw=2.2, label="population q(oxygen)")
    ax.fill_between(t1, q_oxy1.min(axis=0), q_oxy1.max(axis=0), color=GREEN,
                    alpha=0.15)
    ax2 = ax.twinx()
    ax2.fill_between(t1, 0, Hm, color=GREY, alpha=0.35)
    ax2.set_ylabel("posterior entropy H(q)  (nats)", color=GREY)
    ax2.set_ylim(0, np.log(2) * 1.6)
    ax.axvline(ts, color="gray", ls=":", lw=1.5)
    if torn.size:
        ax.axvspan(crisis_on, crisis_on + crisis_w, color=AMBER, alpha=0.18)
    for x, lab in ((ts * 0.45, "NORMAL\nSCIENCE"),
                   (crisis_on + crisis_w / 2, "CRISIS\n(torn)"),
                   ((flip1 + T1) / 2, "NEW NORMAL")):
        ax.annotate(lab, (x, 0.5), ha="center", fontsize=8, color="#444444")
    ax.set_xlabel("step"); ax.set_ylabel("q(oxygen)"); ax.set_ylim(-0.02, 1.02)
    ax.set_title("the Kuhn cycle with represented rivals: crisis is a STATE --\n"
                 "the model posterior goes torn between paradigms before it flips",
                 fontsize=10)
    ax.legend(fontsize=8, loc="center left")
    plt.tight_layout(); plt.savefig(out_dir / "fig_kuhn_cycle.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    gs = [g for g in p["GAMMAS"] if delays[g] > 0]
    ax.plot(gs, [delays[g] for g in gs], "o-", color=AMBER, lw=2.2)
    nevers = [g for g in p["GAMMAS"] if delays[g] < 0]
    if nevers:
        ax.plot(nevers, [p["T_LOCK"] - ts] * len(nevers), "o", mfc="none",
                color=RED, ms=10, label=f"never within horizon ({p['T_LOCK'] - ts})")
    ax.set_yscale("log")
    ax.set_xlabel("post-shift self-censorship gamma on the disconfirming channel")
    ax.set_ylabel("revolution delay (steps after the shift, log)")
    ax.set_title("denial as an evidence-source cut: flat, then explosion, then never\n"
                 "(the 1/(1-gamma) prediction is refuted -- the incumbent ADAPTS at "
                 "high attention)", fontsize=9)
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out_dir / "fig_denial_lockin.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    t3 = np.arange(T3)
    ax.plot(t3, social[am]["qv"], color=GREY, lw=1.6,
            label="vanguard (runs the experiment)")
    ax.plot(t3, social[am]["qc"], color=GREEN, lw=2.4,
            label=f"conservatives, hypothesis pooling on (alpha_m={am})")
    ax.plot(t3, social[0.0]["qc"], color=RED, lw=2.2, ls="--",
            label="conservatives, no hypothesis communication")
    ax.axvline(ts, color="gray", ls=":", lw=1.5)
    ax.set_xlabel("step"); ax.set_ylabel("q(oxygen)"); ax.set_ylim(-0.02, 1.02)
    ax.set_title("the revolution travels through hypothesis communication:\n"
                 "conservatives convert without ever running the experiment",
                 fontsize=10)
    ax.legend(fontsize=8, loc="center right")
    plt.tight_layout(); plt.savefig(out_dir / "fig_social_revolution.png", dpi=130)
    plt.close(fig)

    # ---------------- save ----------------
    np.savez_compressed(
        out_dir / "simulation_arrays.npz",
        q_oxy_cycle=q_oxy1, entropy_cycle=Hq1,
        gammas=np.asarray(p["GAMMAS"], dtype=float),
        delays=np.asarray([delays[g] for g in p["GAMMAS"]], dtype=float),
        q_vanguard=social[am]["qv"], q_cons_pooled=social[am]["qc"],
        q_cons_isolated=social[0.0]["qc"],
    )
    summary = {
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in p.items()},
        "cycle": {"normal_max_q_oxy": normal_ok, "crisis_onset": crisis_on,
                  "crisis_width": crisis_w, "revolution_t": flip1,
                  "new_normal_q": float(qm[-1])},
        "denial_lockin": {str(g): delays[g] for g in p["GAMMAS"]},
        "social_revolution": {"vanguard_flip": social[am]["fv"],
                              "conservatives_flip": social[am]["fc"],
                              "lag": social[am]["fc"] - social[am]["fv"],
                              "q_cons_no_pooling": float(social[0.0]["qc"][-1])},
        "honest_findings": [
            "two-phase design required: censoring the disconfirming channel from "
            "t=0 weakens the LOCK as much as the overturn (pre-shift the channel "
            "supports the incumbent) and inverts the denial story",
            "the 1/(1-gamma) delay prediction is refuted at mid gamma: delay is "
            "flat to gamma=0.9 then explodes -- the incumbent candidate ADAPTS to "
            "the anomaly under high attention (Quine-Duhem accommodation), so "
            "lock-in is a sharp threshold, not a smooth divergence",
        ],
        "mechanism": "K=2 candidate paradigms per agent (hub wiring vs no-hub), "
                     "prequential q(m), damped log-linear hypothesis pooling; "
                     "crisis = the entropy of q(m); denial = attention cutting the "
                     "evidence source; social revolution = the vanguard's log-odds "
                     "propagating to agents who never ran the experiment",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2),
                                          encoding="utf-8")
    print(f"\nsaved arrays / 3 figures / summary to {out_dir}")
    print(f"HEADLINE: the Kuhn cycle replicates inside the rival representation, "
          f"with crisis as a measurable STATE: normal science forms endogenously "
          f"(q_oxy < {normal_ok:.2f} before the shift), the posterior goes torn for "
          f"{crisis_w} steps after the world turns (entropy spike), revolution "
          f"completes at t={flip1}, and the new normal is stable. Communities that "
          f"stop running the disconfirming experiment delay the revolution "
          f"({delays[0.99] if delays[0.99] > 0 else 'forever'} at gamma=0.99 vs "
          f"{delays[min(p['GAMMAS'])]} at gamma=0) and gamma=1 locks in forever; a "
          f"{int(p['VANGUARD_FRAC'] * 100)}% vanguard converts the locked majority "
          f"through hypothesis communication alone "
          f"(lag {social[am]['fc'] - social[am]['fv']} steps; without pooling the "
          f"conservatives never move).")


register(ExperimentSpec(
    model="phlogiston",
    name="rival_kuhn",
    description="The Kuhn cycle on the represented-rival engine: normal science, "
                "crisis (the model posterior going torn -- entropy spike), "
                "revolution and stable new normal, denial/lock-in via post-shift "
                "self-censorship, and a vanguard converting a locked majority "
                "through hypothesis communication alone.",
    run=run,
    out_dir="rival_kuhn",
    params=dict(
        SIGMA_O=1.0, T_SHIFT=80, T_CYCLE=240, T_LOCK=480,
        N_CYCLE=16, N_MIXED=24,
        # CALIB (probe, seed 0): omega=0.9 gives a ~29-step torn interval (smaller
        # omega flips too sharply for a visible crisis); normal science locks ~21
        # nats by the shift; gamma=1 never flips (no structure-drift escape).
        OMEGA=0.9, ALPHA_M=0.02,
        GAMMAS=(0.0, 0.5, 0.9, 0.99, 1.0),
        VANGUARD_FRAC=0.25, GAMMA_CONS=0.99,
        H_CRISIS=0.35, Q_LOCK=0.9,
        SEEDS=(0, 1, 2),
    ),
    seeds=(0, 1, 2),
    consumes=dict(figures=["fig_kuhn_cycle", "fig_denial_lockin",
                           "fig_social_revolution"]),
))
