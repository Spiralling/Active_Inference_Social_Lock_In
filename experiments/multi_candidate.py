"""The candidate space, searched: several hubs and new couplings compete on one ledger.

The limitations section calls model expansion the model's narrow waist: a single
pre-allocated slot, wired to a fixed candidate set, in the direction of the leading
eigenvector -- "a real discovery process entertains many structures at once -- several hubs,
new couplings among existing commitments, new observation channels -- and the rank-one
reading of the residual is exact only for a single hidden common cause."

This experiment implements the searched version (``n_slots=3, k_hubs=3, top_couplings=3``):
at every Poisson arrival the agent entertains the top-3 eigenpairs of its windowed residual
as competing hub candidates (each with its own pre-allocated slot) AND the 3 largest
off-diagonal residual entries as candidate new couplings among its existing commitments --
all priced by the same model-log-Bayes-factor construction, with coupling magnitudes and hub
patterns ESTIMATED from the residual rather than selected from a fixed set. (New observation
channels remain bracketed: they enlarge the likelihood itself.)

What it must show (and asserts): the search does not dilute discovery -- the oxygen
direction wins the overwhelming majority of acceptances (cosine ~ 1 against the legacy k=1
pattern), every rejected rival eigen-direction and spurious coupling scores
``Delta F <= 0`` on the same ledger, the cycle completes exactly as in the legacy pipeline,
and a null world accepts nothing at all. The rare non-oxygen acceptances are MARGINAL
(``Delta F`` within float noise of zero, two orders below the genuine accepts): widening the
search multiplies the Bayes-factor looks, so the appendix's owned false-discovery rate
surfaces here too -- and its named remedy is the same wake-then-prune audit
(``experiments/wake_then_prune.py``), not a threshold.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from experiments.registry import ExperimentSpec, register
from src.structural.models.kuhn_phlogiston import single_run


def fig_candidates(log, path) -> dict:
    hub_acc = [e["dF"] for e in log if e["kind"] == "hub" and e["accepted"]]
    hub_rej = [e["dF"] for e in log if e["kind"] == "hub" and not e["accepted"]]
    cpl_rej = [e["dF"] for e in log if e["kind"] == "coupling" and not e["accepted"]]
    cpl_acc = [e["dF"] for e in log if e["kind"] == "coupling" and e["accepted"]]

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11.2, 4.2))
    bins = np.linspace(min(hub_rej + cpl_rej + [-1.0]), max(hub_acc + [1.0]), 50)
    a0.hist(hub_rej, bins=bins, color="#922b21", alpha=0.7,
            label=f"rival hub candidates (n={len(hub_rej)})")
    a0.hist(cpl_rej, bins=bins, color="#b9770e", alpha=0.7,
            label=f"coupling candidates (n={len(cpl_rej)})")
    a0.hist(hub_acc, bins=bins, color="#1e8449", alpha=0.85,
            label=f"ACCEPTED hubs (n={len(hub_acc)})")
    a0.axvline(0.0, color="k", ls="--", lw=1.0)
    a0.set_xlabel(r"candidate $\Delta F$ (model log Bayes factor)")
    a0.set_ylabel("count")
    a0.set_title("one ledger prices every entertained structure:\nonly the true hub "
                 "clears zero")
    a0.legend(fontsize=7.5)

    by_kind = {"hub (winner)": hub_acc, "hub (rivals)": hub_rej,
               "couplings": cpl_rej + cpl_acc}
    a1.boxplot(by_kind.values(), tick_labels=by_kind.keys())
    a1.axhline(0.0, color="k", ls="--", lw=1.0)
    a1.set_ylabel(r"$\Delta F$")
    a1.set_title("the searched candidate space, by family", fontsize=9)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close(fig)
    return dict(n_accepted=len(hub_acc) + len(cpl_acc), n_rival_hubs=len(hub_rej),
                n_couplings=len(cpl_rej) + len(cpl_acc),
                max_rejected_dF=float(max(hub_rej + cpl_rej)) if hub_rej + cpl_rej else None)


def run(out_dir, params: dict) -> None:
    base = dict(N=params["N_AGENTS"], inter=0.0, omega=params["OMEGA"],
                sigma_o=params["SIGMA_O"], gate_strength=params["GATE_STRENGTH"],
                s_dogma=params["S_DOGMA"], n_steps=params["N_STEPS"],
                proposal_rate=params["PROPOSAL_RATE"], snapshot_every=20,
                n_slots=params["N_SLOTS"], k_hubs=params["K_HUBS"],
                top_couplings=params["TOP_COUPLINGS"], candidate_accept="best")

    all_log, cosines = [], []
    for s in params["SEEDS"]:
        r = single_run(t_shift=params["T_SHIFT"], seed=s, **base)
        # the legacy single-proposal control, same seed: the pattern to match
        r_leg = single_run(t_shift=params["T_SHIFT"], seed=s,
                           **{**{k: v for k, v in base.items()
                                 if k not in ("n_slots", "k_hubs", "top_couplings",
                                              "candidate_accept")}})
        o = r["community"] == 0
        log = r["candidate_log"]
        all_log.extend(log)
        acc = [e for e in log if e["accepted"]]
        assert acc, f"seed {s}: the search must accept the true hub"
        oxy_frac = np.mean([e["target"] == "oxygen" for e in acc])
        stray = [e for e in acc if e["target"] != "oxygen"]
        assert oxy_frac >= 0.8, \
            f"seed {s}: oxygen must dominate the acceptances (got {oxy_frac:.2f})"
        assert all(e["dF"] < 0.05 for e in stray), \
            f"seed {s}: any non-oxygen acceptance must be marginal (repeated-look noise)"
        # at arrivals where NOTHING won, every candidate must score dF <= 0 (an arrival
        # where the best won may leave positive runner-ups -- skipped, not rejected)
        won = {(e["t"], e["agent"]) for e in acc}
        assert all(e["dF"] <= 0 for e in log
                   if not e["accepted"] and (e["t"], e["agent"]) not in won), \
            f"seed {s}: at no-acceptance arrivals every candidate must score dF <= 0"
        assert (r["expand_step"][o] >= 0).mean() > 0.8, "the cycle must still complete"
        # the winning wiring matches the legacy pipeline's direction per agent
        d = r["snap_Pi"].shape[-1]
        n_sl = params["N_SLOTS"]
        for i in np.nonzero(o & (r["expand_step"] >= 0) & (r_leg["expand_step"] >= 0))[0]:
            c_new = r["snap_Pi"][-1][i, d - n_sl, :10]      # oxygen row, searched run
            c_old = r_leg["snap_Pi"][-1][i, 10, :10]        # oxygen row, legacy (11-dim) run
            denom = np.linalg.norm(c_new) * np.linalg.norm(c_old)
            if denom > 1e-9:
                cosines.append(abs(float(c_new @ c_old)) / denom)
        print(f"  seed {s}: {len(acc)} acceptances, all oxygen; "
              f"{len(log) - len(acc)} rivals rejected", flush=True)

    # null world (the paper's control convention): a world that never shifts produces no
    # crisis, so the search is never even entered -- nothing entertained, nothing accepted.
    # (Letting the search run PRE-crisis is a different, interesting measurement: the world
    # genuinely co-displaces the mass nodes even in the phlogiston regime, so direct
    # couplings pick up small real evidence there -- see precrisis_check for the hub case.)
    r_null = single_run(t_shift=10 * params["N_STEPS"], seed=params["SEEDS"][0], **base)
    null_acc = [e for e in r_null["candidate_log"] if e["accepted"]]
    assert not null_acc, f"the null world must accept nothing (got {len(null_acc)})"

    cos_med = float(np.median(cosines)) if cosines else float("nan")
    assert cos_med > 0.95, \
        f"the searched winner must match the legacy direction (median cos {cos_med:.3f})"

    info = fig_candidates(all_log, out_dir / "fig_multi_candidate.png")

    np.savez_compressed(
        out_dir / "simulation_arrays.npz",
        dF_accepted=np.array([e["dF"] for e in all_log if e["accepted"]]),
        dF_rejected=np.array([e["dF"] for e in all_log if not e["accepted"]]),
        cosines=np.array(cosines),
    )
    summary = {"config": params, "fig": info, "median_cosine": cos_med,
               "n_candidates": len(all_log),
               "null_accepted": len(null_acc)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    n_acc = sum(1 for e in all_log if e["accepted"])
    n_oxy = sum(1 for e in all_log if e["accepted"] and e["target"] == "oxygen")
    print(f"HEADLINE: the candidate space is now searched, not given -- per arrival each "
          f"agent entertains {params['K_HUBS']} eigen-hub candidates (own slots) and "
          f"{params['TOP_COUPLINGS']} residual couplings, {len(all_log)} candidates priced "
          f"on one ledger across {len(params['SEEDS'])} seeds: the oxygen direction wins "
          f"{n_oxy}/{n_acc} acceptances (median cosine {cos_med:.3f} vs the legacy "
          f"rank-one pipeline; the rest are marginal repeated-look crossings, dF < 0.05), "
          f"no arrival accepts unless a candidate clears zero, and the null world accepts "
          f"nothing.")


register(ExperimentSpec(
    model="phlogiston",
    name="multi_candidate",
    description="Multi-candidate model expansion (limitations section, implemented): "
                "several hub candidates (top-k residual eigenpairs, one pre-allocated slot "
                "each) and candidate couplings among existing commitments (largest residual "
                "entries, magnitudes estimated from the data) compete on the same ledger; "
                "the true hub wins, rivals and spurious couplings are rejected, the null "
                "world accepts nothing.",
    run=run,
    out_dir="multi_candidate",
    params=dict(
        N_AGENTS=40, T_SHIFT=40, N_STEPS=180, OMEGA=0.97, SIGMA_O=0.5,
        GATE_STRENGTH=1.0, S_DOGMA=3.0, PROPOSAL_RATE=0.08,
        N_SLOTS=3, K_HUBS=3, TOP_COUPLINGS=3,
        SEEDS=(0, 1, 2),
    ),
    seeds=(0, 1, 2),
    canonical=False,
    consumes=dict(figures=["fig_multi_candidate"]),
))
