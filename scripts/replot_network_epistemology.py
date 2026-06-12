"""Re-plot existing rival-experiment results in STANDARD network-epistemology figure
conventions -- no re-running, pure read-off of the saved ``simulation_arrays.npz``.

The rival experiments (represented_rivals, rival_kuhn, kuhnian_transition) save their
per-seed arrays; this script reframes them in the field's canonical visual idioms so
the work is legible to the network-epistemology / social-epistemology-simulation
literature:

  1. ZOLLMAN accuracy-vs-connectivity (Zollman 2007, Phil. Sci.): the order parameter
     against algebraic connectivity lambda_2 -- the field's actual x-axis -- instead
     of raw bridge density. (kuhnian_transition.)
  2. Paradigm-shift dose-response (Rodriguez-Sickert et al. 2015, PLoS ONE Fig 3):
     time-to-revolution vs a control parameter. (rival_kuhn: self-censorship gamma.)
  3. The HEGSELMANN-KRAUSE trajectory fan (HK 2002, JASSS): per-seed opinion lines
     converging / holding apart over time. (represented_rivals camp distance; the
     Lakatos ladder q(m).)

HONEST SCOPE (printed on every ensemble-style panel): these reuse 3-5 seeds. The
field reserves probability/rate axes for many-run ensembles (R ~ 100-10^4). Where a
panel would be a *rate*, it is drawn as a small-sample curve with the seed spread
shown and the R = n_seeds caveat in the caption -- the Tier-2 seed sweep is what
makes those axes publication-grade. Single-run *shapes* (fans, trajectories) are
already honest at this seed count.

Run: ``python scripts/replot_network_epistemology.py`` -> results/network_epistemology_figs/.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (repo root on sys.path + utf-8 stdout)

import numpy as np
import matplotlib.pyplot as plt

from src.repro.paths import ROOT

RED, AMBER, GREEN, GREY, BLUE = "#922b21", "#b9770e", "#1e8449", "#7f8c8d", "#2c6fbb"
RESULTS = ROOT / "results"
OUT = RESULTS / "network_epistemology_figs"


def _load(name):
    return np.load(RESULTS / name / "simulation_arrays.npz")


def _caption_R(ax, n):
    ax.text(0.99, 0.02, f"R = {n} seeds (illustrative; ensemble axes need R >> 1)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
            color="#999999", style="italic")


# ----------------------------------------------------------------------
# 1. Zollman accuracy vs algebraic connectivity lambda_2.
# ----------------------------------------------------------------------

def fig_zollman_accuracy():
    z = _load("kuhnian_transition")
    l2 = z["lambda2"]                                   # (B,)
    qc = z["q_end_clean"]                               # (B, S) order parameter
    qt = z["q_end_trickle"]
    S = qc.shape[1]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for q, color, lab in ((qc, GREEN, "zero-source conservatives"),
                          (qt, AMBER, "trickle-source conservatives (gamma=0.99)")):
        m, s = q.mean(axis=1), q.std(axis=1)
        ax.errorbar(l2, m, yerr=s, marker="o", lw=2.2, capsize=3, color=color,
                    label=lab)
    ax.set_xlabel(r"algebraic connectivity  $\lambda_2$  (Fiedler value)")
    ax.set_ylabel("P(community adopts the better paradigm)")
    ax.set_ylim(-0.04, 1.04)
    ax.set_title("Zollman-style accuracy vs connectivity\n"
                 "(kuhnian_transition, re-plotted on the field's connectivity axis)",
                 fontsize=10)
    ax.legend(fontsize=8, loc="center left")
    _caption_R(ax, S)
    plt.tight_layout(); plt.savefig(OUT / "fig_zollman_accuracy.png", dpi=130)
    plt.close(fig)
    return {"lambda2": l2.tolist(),
            "q_clean": qc.mean(axis=1).tolist(),
            "q_trickle": qt.mean(axis=1).tolist()}


# ----------------------------------------------------------------------
# 2. Paradigm-shift dose-response: time-to-revolution vs self-censorship.
# ----------------------------------------------------------------------

def fig_doseresponse():
    z = _load("rival_kuhn")
    g, d = z["gammas"], z["delays"].astype(float)      # delays: -1 == never
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ok = d > 0
    ax.plot(g[ok], d[ok], "o-", lw=2.2, color=AMBER, ms=8)
    never = g[~ok]
    if never.size:
        ymax = d[ok].max() * 1.25
        ax.plot(never, [ymax] * never.size, "o", mfc="none", color=RED, ms=11,
                label="locked in (no revolution)")
        for x in never:
            ax.annotate("never", (x, ymax), ha="center", va="bottom", fontsize=8,
                        color=RED)
    ax.set_yscale("log")
    ax.set_xlabel(r"self-censorship $\gamma$ on the disconfirming channel")
    ax.set_ylabel("time to revolution (steps after the shift, log)")
    ax.set_title("paradigm-shift dose-response\n"
                 "(Rodriguez-Sickert et al. 2015 Fig 3 idiom: control parameter vs "
                 "time-to-shift)", fontsize=10)
    if never.size:
        ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout(); plt.savefig(OUT / "fig_doseresponse.png", dpi=130)
    plt.close(fig)
    return {"gamma": g.tolist(), "delay": d.tolist()}


# ----------------------------------------------------------------------
# 3a. HK trajectory fan: the schism that holds vs the consensus that collapses.
# ----------------------------------------------------------------------

def fig_hk_fan():
    z = _load("represented_rivals")
    t = np.arange(z["D_plain"].shape[1])
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for arr, color, lab in (("D_plain", RED, "plain fusion (consensus)"),
                            ("D_gated", AMBER, "Student-t gate (metastable)"),
                            ("D_rival", GREEN, "represented rivals (schism holds)")):
        D = z[arr]                                      # (S, T) inter-camp distance
        for s in range(D.shape[0]):                     # the FAN: one line per seed
            ax.plot(t, np.clip(D[s], 1e-3, None), color=color, lw=0.8, alpha=0.45)
        ax.plot(t, np.clip(D.mean(axis=0), 1e-3, None), color=color, lw=2.4,
                label=lab)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("step (log)")
    ax.set_ylabel("inter-camp distance (log)")
    ax.set_title("Hegselmann-Krause trajectory fan: consensus vs schism\n"
                 "(represented_rivals; one faint line per seed, bold = mean)",
                 fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    _caption_R(ax, z["D_plain"].shape[0])
    plt.tight_layout(); plt.savefig(OUT / "fig_hk_fan.png", dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------
# 3b. Stickiness of each model over time (Devezer et al. 2019 idiom).
# ----------------------------------------------------------------------

def fig_model_stickiness():
    z = _load("kuhnian_transition")
    q = z["q_ladder"]                                   # (S, T, 3) full/artic/oxy
    t = np.arange(q.shape[1])
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for k, (color, lab) in enumerate((
            (GREY, "M_full (incumbent)"),
            (AMBER, "M_artic (articulated incumbent)"),
            (GREEN, "M_oxy (rival core)"))):
        for s in range(q.shape[0]):
            ax.plot(t, q[s, :, k], color=color, lw=0.7, alpha=0.4)
        ax.plot(t, q[:, :, k].mean(axis=0), color=color, lw=2.4, label=lab)
    ax.set_xlabel("step"); ax.set_ylabel("stickiness  q(m)  (community model posterior)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("model 'stickiness' over time (Devezer et al. 2019 idiom):\n"
                 "the Lakatos ladder as a sequence of dominant models", fontsize=10)
    ax.legend(fontsize=8, loc="center right")
    _caption_R(ax, q.shape[0])
    plt.tight_layout(); plt.savefig(OUT / "fig_model_stickiness.png", dpi=130)
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    info = {}
    info["zollman"] = fig_zollman_accuracy()
    info["doseresponse"] = fig_doseresponse()
    fig_hk_fan()
    fig_model_stickiness()
    print(f"wrote 4 figures to {OUT.relative_to(ROOT)}:")
    for f in ("fig_zollman_accuracy", "fig_doseresponse", "fig_hk_fan",
              "fig_model_stickiness"):
        print(f"  {f}.png")
    print("\nall re-plotted from existing npz -- no experiment was re-run.")
    print("Zollman accuracy curve (lambda2 -> P adopt better paradigm, clean): "
          + "  ".join(f"{l:.2f}->{q:.2f}" for l, q in
                      zip(info["zollman"]["lambda2"], info["zollman"]["q_clean"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
