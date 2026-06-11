"""The wake-then-prune cycle: structure entertained on expiring credit, kept by evidence.

The appendix owns a consequence of having no tuned threshold: repeated Bayes-factor looks
give the one-shot discovery check a small false-discovery rate, and names its principled
remedy -- "not a reinstated threshold but the wake-then-prune cycle ... in which a spuriously
woken node is pruned back by the same reduction that scored it." The limitations add that the
single-step accept "collapses into one step what is more naturally a wake-then-prune cycle in
which a community entertains structures on credit before the data settle them."

This experiment implements that cycle (``wake_credit=True``):

* **Wake on credit.** A proposal is accepted on the FULL ledger ``Delta F + Delta G > 0``:
  the epistemic gain (positive for any new latent) pays for entertaining the structure, so
  hubs wake even on noise -- curiosity is allowed to be wrong.
* **The credit expires.** Every woken hub is audited each round in the same currency that
  scored it (the bordered-model log Bayes factor against the pre-wake anchor at the stored
  coupling), plus the epistemic credit decayed per audit: verdict_k = dF_now + dG * decay^k.
  The data must take over before the credit runs out.
* **Prune-back is exact.** ``prune_patience`` consecutive prune votes re-pin the slot to the
  original pinned form -- and the agent may wake again later (the cycle).

What it must show (and asserts): in a NULL world (no regime flip; pre-crisis attempts
allowed, since no crisis ever fires) the credit wakes hubs on pure noise and the audit prunes
essentially ALL of them back -- the false-discovery rate is controlled by the cycle, not by a
threshold. In the FLIP world the same machinery wakes the genuine oxygen hub and never
prunes it -- the residual keeps paying the evidence before the credit expires.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from experiments.registry import ExperimentSpec, register
from src.structural.models.kuhn_phlogiston import single_run


def fig_cycle(r_null, r_flip, path) -> dict:
    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(13.4, 4.2))

    for ax, r, title, color in ((a0, r_null, "NULL world (no flip): noise wakes", "#922b21"),
                                (a1, r_flip, "FLIP world: the genuine hub", "#1e8449")):
        tr = r["hub_dF_prune_tn"]
        tt = np.arange(tr.shape[0])
        with np.errstate(all="ignore"):
            m = np.nanmean(tr, axis=1)
        ax.plot(tt, m, lw=2.0, color=color, label="audit verdict (mean over woken)")
        ax.fill_between(tt, np.nanmin(tr, axis=1), np.nanmax(tr, axis=1),
                        color=color, alpha=0.15, label="min-max over agents")
        ax.axhline(0.0, color="k", ls="--", lw=1.0, label="prune vote below 0")
        ax.set_xlabel("step"); ax.set_ylabel(r"audit: $\Delta F$ + expiring credit")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)

    on, of = r_null["community"] == 0, r_flip["community"] == 0
    es_n, ps_n = r_null["expand_step"][on], r_null["prune_back_step"][on]
    es_f, ps_f = r_flip["expand_step"][of], r_flip["prune_back_step"][of]
    woke_n, woke_f = es_n >= 0, es_f >= 0
    bars = [woke_n.mean(), (ps_n[woke_n] >= 0).mean() if woke_n.any() else 0.0,
            woke_f.mean(), (ps_f[woke_f] >= 0).mean() if woke_f.any() else 0.0]
    a2.bar(range(4), bars, color=["#c0392b", "#7b241c", "#27ae60", "#145a32"])
    a2.set_xticks(range(4))
    a2.set_xticklabels(["null:\nwoke", "null:\npruned back", "flip:\nwoke",
                        "flip:\npruned back"], fontsize=8)
    a2.set_ylim(0, 1.05)
    a2.set_title("curiosity wakes on noise AND on signal;\nthe expiring credit keeps only "
                 "what the data hold", fontsize=9)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close(fig)
    return dict(null_woke=float(bars[0]), null_pruned=float(bars[1]),
                flip_woke=float(bars[2]), flip_pruned=float(bars[3]))


def run(out_dir, params: dict) -> None:
    base = dict(N=params["N_AGENTS"], inter=0.0, omega=params["OMEGA"],
                sigma_o=params["SIGMA_O"], gate_strength=params["GATE_STRENGTH"],
                s_dogma=params["S_DOGMA"], n_steps=params["N_STEPS"],
                proposal_rate=params["PROPOSAL_RATE"], snapshot_every=20,
                wake_credit=True, prune_grace=params["PRUNE_GRACE"],
                prune_patience=params["PRUNE_PATIENCE"],
                credit_decay=params["CREDIT_DECAY"])

    rows = []
    r_null_show = r_flip_show = None
    for s in params["SEEDS"]:
        r_null = single_run(t_shift=10 * params["N_STEPS"], allow_precrisis_apply=True,
                            seed=s, **base)
        r_flip = single_run(t_shift=params["T_SHIFT"], seed=s, **base)
        if r_null_show is None:
            r_null_show, r_flip_show = r_null, r_flip
        for label, r in (("null", r_null), ("flip", r_flip)):
            o = r["community"] == 0          # the open community (the dogmatic half never
            es, ps = r["expand_step"][o], r["prune_back_step"][o]   # reaches crisis at inter=0)
            woke = es >= 0
            rows.append(dict(
                world=label, seed=s, woke_frac=float(woke.mean()),
                pruned_frac=float((ps[woke] >= 0).mean()) if woke.any() else np.nan,
                med_prune_delay=float(np.median((ps - es)[woke & (ps >= 0)]))
                if (woke & (ps >= 0)).any() else np.nan))
            print(f"  {label} seed {s}: woke {rows[-1]['woke_frac']:.2f} "
                  f"pruned {rows[-1]['pruned_frac']:.2f}", flush=True)

    info = fig_cycle(r_null_show, r_flip_show, out_dir / "fig_wake_then_prune.png")

    null_pruned = np.nanmean([r["pruned_frac"] for r in rows if r["world"] == "null"])
    null_woke = np.mean([r["woke_frac"] for r in rows if r["world"] == "null"])
    flip_pruned = np.nanmean([r["pruned_frac"] for r in rows if r["world"] == "flip"])
    flip_woke = np.mean([r["woke_frac"] for r in rows if r["world"] == "flip"])
    assert null_woke > 0.5, "the credit must wake hubs even on noise"
    assert null_pruned > 0.95, \
        f"the audit must prune essentially all noise wakes (got {null_pruned:.2f})"
    assert flip_woke > 0.8 and flip_pruned < 0.05, \
        f"the genuine hub must survive the audit (woke {flip_woke:.2f}, pruned {flip_pruned:.2f})"

    np.savez_compressed(
        out_dir / "simulation_arrays.npz",
        null_audit_tn=r_null_show["hub_dF_prune_tn"],
        flip_audit_tn=r_flip_show["hub_dF_prune_tn"],
        null_expand=r_null_show["expand_step"], null_prune=r_null_show["prune_back_step"],
        flip_expand=r_flip_show["expand_step"], flip_prune=r_flip_show["prune_back_step"],
        woke_pruned=np.array([[r["woke_frac"], r["pruned_frac"]] for r in rows]),
    )
    summary = {"config": params, "fig": info, "rows": rows,
               "null_pruned_frac": float(null_pruned),
               "flip_pruned_frac": float(flip_pruned)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"HEADLINE: the wake-then-prune cycle controls the discovery check's "
          f"false-discovery rate with no threshold: on pure noise, curiosity (the epistemic "
          f"credit) wakes hubs in {null_woke:.0%} of agents and the expiring-credit audit "
          f"prunes {null_pruned:.0%} of them back to the exact pinned form; in the flip "
          f"world the genuine oxygen hub is woken ({flip_woke:.0%}) and never pruned "
          f"({flip_pruned:.0%}) -- entertaining is paid by curiosity, keeping by evidence.")


register(ExperimentSpec(
    model="phlogiston",
    name="wake_then_prune",
    description="The wake-then-prune cycle (limitations + appendix, implemented): wakes "
                "accepted on the full ledger (epistemic credit pays for entertaining), "
                "audited every round by the re-evaluated wake evidence plus expiring credit; "
                "spurious wakes re-pinned exactly, genuine hubs kept -- the principled "
                "false-discovery remedy with no tuned threshold.",
    run=run,
    out_dir="wake_then_prune",
    params=dict(
        N_AGENTS=40, T_SHIFT=40, N_STEPS=260, OMEGA=0.97, SIGMA_O=0.5,
        GATE_STRENGTH=1.0, S_DOGMA=3.0, PROPOSAL_RATE=0.1,
        PRUNE_GRACE=15, PRUNE_PATIENCE=5, CREDIT_DECAY=0.85,
        SEEDS=(0, 1, 2),
    ),
    seeds=(0, 1, 2),
    canonical=False,
    consumes=dict(figures=["fig_wake_then_prune"]),
))
