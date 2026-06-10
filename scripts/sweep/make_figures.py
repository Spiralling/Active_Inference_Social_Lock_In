"""Curated, shareable summary figures from the big sweep (vs. the 28 raw phase maps).

Reads results/big_sweep/rows.jsonl and writes a handful of presentation-quality PNGs to
results/big_sweep/shared/ -- the headline findings for collaborators:
  1. structural pluralism: the three ingredients (selectivity, disconnection, value-diversity);
  2. forgetting health (use omega <= 0.9);
  3. lock-in is a narrow corner (conviction x weak-evidence).

    python scripts/sweep/make_figures.py
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 120, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False})

OUT = ROOT / "results" / "big_sweep"
SHARED = OUT / "shared"
BLUE, RED, PURPLE, GREEN = "#2c6fbb", "#c0392b", "#6c3483", "#1e8449"


def load_clean():
    rows = [json.loads(l) for l in (OUT / "rows.jsonl").open(encoding="utf-8") if l.strip()]
    return [r for r in rows if not r.get("error") and not r.get("diverged")]


def _curve(rows, job, xax, metric, fixed=None):
    d = defaultdict(list)
    for r in rows:
        if job and r.get("job") != job:
            continue
        if fixed and any(r.get(k) != v for k, v in fixed.items()):
            continue
        m = r.get(metric)
        if m is not None and np.isfinite(m):
            d[r[xax]].append(m)
    xs = sorted(d, key=lambda x: (isinstance(x, str), x))
    return xs, [float(np.mean(d[x])) for x in xs], [float(np.std(d[x])) for x in xs]


def fig_pluralism(rows):
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(13.5, 4.0))

    # (a) selectivity (job E, blend=1.0)
    x, y, e = _curve(rows, "E_blend_beta", "attention_beta", "final_struct_dist", {"blend_ratio": 1.0})
    a.errorbar(range(len(x)), y, yerr=e, fmt="o-", color=BLUE, lw=2.3, capsize=3)
    a.set_xticks(range(len(x))); a.set_xticklabels(x)
    a.set_xlabel("attention selectivity  β (W_hi/W_lo)")
    a.set_ylabel("structural divergence")
    a.set_title("Selective attention is necessary", fontsize=11)
    a.set_ylim(-0.05, 1.5)

    # (b) connectivity (job A): even category spacing, labelled by inter (→ λ₂)
    xi, yi, ei = _curve(rows, "A_omega_inter", "inter", "final_struct_dist")
    _, lam, _ = _curve(rows, "A_omega_inter", "inter", "lambda2")
    b.errorbar(range(len(xi)), yi, yerr=ei, fmt="o-", color=PURPLE, lw=2.3, capsize=3)
    b.set_xticks(range(len(xi)))
    b.set_xticklabels([f"{it}\n(λ₂={l:.0f})" for it, l in zip(xi, lam)], fontsize=7.5)
    b.annotate("more connected →", xy=(0.5, 0.92), xycoords="axes fraction",
               ha="center", fontsize=8, color="0.4")
    b.set_xlabel("inter-community connectivity")
    b.set_ylabel("structural divergence")
    b.set_title("Fusion collapses it → pluralism needs disconnection", fontsize=10)
    b.set_ylim(-0.05, 1.5)

    # (c) value-diversity (job B, disconnected)
    xn, yn, en = _curve(rows, "B_inter_ncomm", "n_communities", "final_struct_dist", {"inter": 0.0})
    c.bar([str(v) for v in xn], yn, yerr=en, color=GREEN, capsize=4, width=0.6)
    c.set_xlabel("# value-communities")
    c.set_ylabel("structural divergence")
    c.set_title("≥2 distinct values needed", fontsize=11)
    c.set_ylim(0, 1.5)

    fig.suptitle("Structural pluralism = selective attention × disconnection × value-diversity",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(SHARED / "fig1_structural_pluralism.png", bbox_inches="tight"); plt.close(fig)


def fig_forgetting(rows_all):
    by = defaultdict(list)
    for r in rows_all:
        by[r.get("omega")].append(1 if r.get("diverged") else 0)
    xs = sorted(k for k in by if k is not None)
    frac = [np.mean(by[k]) for k in xs]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    cols = [RED if f > 0.05 else BLUE for f in frac]
    ax.bar([str(x) for x in xs], frac, color=cols, width=0.6)
    for i, f in enumerate(frac):
        ax.text(i, f + 0.02, f"{f:.0%}", ha="center", fontsize=9)
    ax.axhline(0.05, color="0.4", ls="--", lw=1)
    ax.set_xlabel("forgetting  ω"); ax.set_ylabel("fraction of runs that diverge (|Π| → ∞)")
    ax.set_title("Use ω ≤ 0.9: precision stays bounded; ω = 1.0 blows up", fontsize=11)
    ax.set_ylim(0, 1.0)
    fig.tight_layout(); fig.savefig(SHARED / "fig2_forgetting_health.png", bbox_inches="tight"); plt.close(fig)


def fig_lockin(rows):
    ce = [r for r in rows if r.get("world_mode") == "changing_epochs"]
    sigmas = sorted({r.get("sigma_o") for r in ce})
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    cmap = {0.25: BLUE, 0.5: PURPLE, 1.0: RED}
    for sg in sigmas:
        x, y, _ = _curve([r for r in ce if r.get("sigma_o") == sg], None,
                         "conviction_tilt", "lockin_frac")
        if x:
            ax.plot(x, y, "o-", color=cmap.get(sg, "0.4"), lw=2.2, label=f"σ_obs = {sg}")
    ax.set_xlabel("conviction strength  λ (motivated update)")
    ax.set_ylabel("lock-in fraction\n(communities frozen on a superseded theory)", fontsize=9)
    ax.set_title("Lock-in is a narrow corner: needs strong conviction AND weak evidence",
                 fontsize=10.5)
    ax.set_ylim(-0.03, 1.0); ax.legend()
    fig.tight_layout(); fig.savefig(SHARED / "fig3_lockin_corner.png", bbox_inches="tight"); plt.close(fig)


def main():
    SHARED.mkdir(parents=True, exist_ok=True)
    clean = load_clean()
    rows_all = [json.loads(l) for l in (OUT / "rows.jsonl").open(encoding="utf-8") if l.strip()]
    fig_pluralism(clean)
    fig_forgetting(rows_all)
    fig_lockin(clean)
    print(f"wrote 3 shared figures to {SHARED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
