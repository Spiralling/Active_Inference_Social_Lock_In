"""Dynamic conviction: value accretes onto entrenched commitments (Lakatos, implemented).

The appendix's commitment list ends with: "the conviction source ``u`` is exogenous and
static... agents do not revise ``u`` itself. Motivated agents who re-derive what they value
is a different (interesting) model." And the limitations name what that model would buy:
"Letting ``u`` accrete onto structurally entrenched commitments -- value aligning with
conservatism as a programme matures -- would recover Lakatos's progressive and degenerating
problemshifts."

This experiment implements that model (``simulation.ConvictionDynamics``): each agent's
intrinsic utility gains a per-node accretion state driven by its OWN entrenchment (the
accumulated Fisher deposit, max-normalized), ``u_eff = u_base (1 + g)``,
``g <- clip(g + eps*entrenchment - decay*g, 0, g_max)``. Value follows where the programme
has piled its evidence -- and because the conviction field feeds the endogenous gate (E2),
a maturing programme increasingly silences its own disconfirming channels.

What it must show (and asserts):
* the gain ``g`` accretes over time and loads on the dogmatic community's entrenched
  commitments (``u_gain_t`` rises as the programme matures);
* the realized conviction gate ``gamma`` strengthens monotonically with the accretion rate
  ``eps`` -- conservatism grows with maturity, not by assignment;
* at the lock-in boundary (weak coupling) the accretion DEGENERATES the programme: the
  dogmatic community's conversion falls and its crisis fraction drops as ``eps`` rises --
  the progressive -> degenerating problemshift as a phase diagram in ``(eps, decay)``.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np

from experiments.registry import ExperimentSpec, register


def _one_job(cfg: dict) -> dict:
    from src.structural.models.kuhn_phlogiston import single_run

    r = single_run(N=cfg["N"], inter=cfg["INTER"], omega=cfg["OMEGA"],
                   sigma_o=cfg["SIGMA_O"], gate_strength=cfg["GATE_STRENGTH"],
                   s_dogma=cfg["S_DOGMA"], t_shift=cfg["T_SHIFT"],
                   n_steps=cfg["N_STEPS"], proposal_rate=cfg["PROPOSAL_RATE"],
                   conviction_eps=cfg["eps"], conviction_decay=cfg["decay"],
                   seed=cfg["seed"], snapshot_every=20)
    d = r["community"] == 1
    row = dict(eps=cfg["eps"], decay=cfg["decay"], seed=cfg["seed"],
               dogma_oxy_end=float(r["oxy_index_sc"][-1, 1]),
               dogma_crisis_frac=float((r["crisis_step"][d] >= 0).mean()),
               gamma_final=float(r["gamma_t"][-1]))
    if "u_gain_t" in r:
        g = r["u_gain_t"]                                # (S, N, d)
        row["gain_dogma_t"] = g[:, d, :].mean(axis=(1, 2)).tolist()
    return row


def fig_lakatos(rows, epss, decays, gain_traces, path) -> dict:
    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(13.6, 4.0))

    for (eps, tr) in gain_traces.items():
        a0.plot(np.linspace(0, 1, len(tr)), tr, lw=2.0,
                label=f"$\\epsilon$={eps}")
    a0.set_xlabel("run fraction"); a0.set_ylabel("mean accreted gain $g$ (dogmatic)")
    a0.set_title("value accretes as the programme matures:\nconviction follows the "
                 "deposit, not a decree")
    a0.legend(fontsize=8)

    g_by_eps = [(e, np.mean([r["gamma_final"] for r in rows
                             if r["eps"] == e and r["decay"] == decays[0]]))
                for e in epss]
    a1.plot([e for e, _ in g_by_eps], [g for _, g in g_by_eps], "o-", lw=2.2,
            color="#922b21")
    a1.set_xlabel(r"accretion rate $\epsilon$")
    a1.set_ylabel(r"realized gate $\gamma$ (final)")
    a1.set_title("the maturing programme silences its own\ndisconfirming channels "
                 "(E2, endogenously strengthened)")

    M = np.zeros((len(decays), len(epss)))
    for di, dc in enumerate(decays):
        for ei, e in enumerate(epss):
            M[di, ei] = np.mean([r["dogma_oxy_end"] for r in rows
                                 if r["eps"] == e and r["decay"] == dc])
    im = a2.imshow(M, cmap="inferno", vmin=0, vmax=1, aspect="auto", origin="lower")
    a2.set_xticks(range(len(epss))); a2.set_xticklabels(epss)
    a2.set_yticks(range(len(decays))); a2.set_yticklabels(decays)
    a2.set_xlabel(r"accretion $\epsilon$"); a2.set_ylabel("decay")
    a2.set_title("progressive $\\to$ degenerating:\ndogmatic conversion at the "
                 "lock-in boundary")
    plt.colorbar(im, ax=a2, shrink=0.85)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close(fig)
    return dict(conversion=M.tolist(),
                gamma_by_eps={str(e): float(g) for e, g in g_by_eps})


def run(out_dir, params: dict) -> None:
    epss, decays, seeds = list(params["EPSS"]), list(params["DECAYS"]), params["SEEDS"]
    jobs = [dict(params, eps=float(e), decay=float(dc), seed=int(s))
            for e in epss for dc in decays for s in seeds]
    rows = []
    with ProcessPoolExecutor(max_workers=params["WORKERS"]) as pool:
        for row in pool.map(_one_job, jobs, chunksize=1):
            rows.append(row)
            print(f"  eps={row['eps']} decay={row['decay']} seed={row['seed']}: "
                  f"conv {row['dogma_oxy_end']:.2f} gamma {row['gamma_final']:.3f}",
                  flush=True)

    gain_traces = {}
    for e in epss:
        if e == 0.0:
            continue
        trs = [r["gain_dogma_t"] for r in rows
               if r["eps"] == e and r["decay"] == decays[0] and "gain_dogma_t" in r]
        if trs:
            gain_traces[e] = np.mean(np.array(trs), axis=0)

    info = fig_lakatos(rows, epss, decays, gain_traces,
                       out_dir / "fig_lakatos_conviction.png")

    # ---- the claims, asserted (at the reference decay) ----
    dc0 = decays[0]
    conv = [np.mean([r["dogma_oxy_end"] for r in rows
                     if r["eps"] == e and r["decay"] == dc0]) for e in epss]
    gam = [np.mean([r["gamma_final"] for r in rows
                    if r["eps"] == e and r["decay"] == dc0]) for e in epss]
    for tr in gain_traces.values():
        assert tr[-1] > tr[0], "the gain must accrete as the programme matures"
    assert all(gam[i + 1] >= gam[i] - 1e-3 for i in range(len(gam) - 1)), \
        f"the realized gate must strengthen with eps (got {gam})"
    assert conv[-1] < conv[0] - 0.1, \
        f"strong accretion must degenerate the programme (conversion {conv})"

    np.savez_compressed(
        out_dir / "simulation_arrays.npz",
        epss=np.array(epss), decays=np.array(decays),
        conversion=np.array(info["conversion"]),
        gamma_by_eps=np.array(gam),
        gain_traces=np.array([gain_traces[e] for e in epss if e in gain_traces]),
    )
    summary = {"config": params, "fig": info,
               "conversion_by_eps": [float(c) for c in conv],
               "gamma_by_eps": [float(g) for g in gam]}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"HEADLINE: the conviction source is no longer exogenous -- value accretes onto "
          f"the commitments the programme's own evidence has entrenched, and Lakatos falls "
          f"out: at the lock-in boundary the dogmatic community's conversion falls from "
          f"{conv[0]:.2f} (static u) to {conv[-1]:.2f} (eps={epss[-1]}) while its realized "
          f"self-silencing gate climbs from {gam[0]:.3f} to {gam[-1]:.3f} -- a programme "
          f"that matures into conservatism (degenerating problemshift), derived from its "
          f"own deposit rather than assigned.")


register(ExperimentSpec(
    model="phlogiston",
    name="lakatos_conviction",
    description="Dynamic conviction u (limitations + appendix commitment, implemented): "
                "intrinsic utility accretes onto structurally entrenched commitments via "
                "the agent's own Fisher deposit, recovering Lakatos's progressive vs "
                "degenerating problemshifts as an (eps, decay) phase diagram at the "
                "lock-in boundary.",
    run=run,
    out_dir="lakatos_conviction",
    params=dict(
        N=40, T_SHIFT=40, N_STEPS=180, OMEGA=0.97, SIGMA_O=0.5,
        GATE_STRENGTH=1.0, S_DOGMA=3.0, PROPOSAL_RATE=0.08, INTER=0.01,
        EPSS=(0.0, 0.1, 0.3, 1.0), DECAYS=(0.01, 0.05),
        SEEDS=(0, 1, 2, 3), WORKERS=6,
    ),
    seeds=(0, 1, 2, 3),
    canonical=False,
    consumes=dict(figures=["fig_lakatos_conviction"]),
))
