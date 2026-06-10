"""Data-driven analysis pass: phase-map heatmaps + anomaly flags + REPORT.md.

Reads ``results/big_sweep/rows.jsonl`` and, without any per-experiment cherry-picking, emits:
  * one heatmap per (Tier-1 job, metric) over its two swept axes,
  * health flags (which regimes diverge -- e.g. omega=1.0),
  * per-axis importance from the Tier-2 random sample (which knobs move each metric most),
  * a REPORT.md summary.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.sweep.space import JOB_AXES

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "big_sweep"
FIG = OUT / "figures"
METRICS = ("final_struct_dist", "final_comm_diversity", "lockin_frac", "max_pi")
AXES_ALL = ("N", "inter", "omega", "n_communities", "attention_beta",
            "conviction_tilt", "world_mode", "blend_ratio", "sigma_o", "fuse_mode")


def load_rows():
    rows = []
    with (OUT / "rows.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return [r for r in rows if not r.get("error")]


def _levels(vals):
    u = sorted(set(vals), key=lambda x: (isinstance(x, str), x))
    return u, {v: i for i, v in enumerate(u)}


def _grid(rows, ax1, ax2, metric):
    pts = [(r[ax1], r[ax2], r.get(metric)) for r in rows
           if r.get(metric) is not None and np.isfinite(r.get(metric, np.nan))]
    l1, p1 = _levels([p[0] for p in pts])
    l2, p2 = _levels([p[1] for p in pts])
    acc = {}
    for v1, v2, m in pts:
        acc.setdefault((p1[v1], p2[v2]), []).append(m)
    Z = np.full((len(l2), len(l1)), np.nan)
    for (i, j), vals in acc.items():
        Z[j, i] = np.mean(vals)
    return l1, l2, Z


def phase_maps(rows):
    FIG.mkdir(parents=True, exist_ok=True)
    made = []
    for job, (ax1, ax2) in JOB_AXES.items():
        jrows = [r for r in rows if r.get("job") == job]
        if not jrows:
            continue
        for metric in METRICS:
            l1, l2, Z = _grid(jrows, ax1, ax2, metric)
            if Z.size == 0 or np.isnan(Z).all():
                continue
            fig, ax = plt.subplots(figsize=(1.2 + 0.7 * len(l1), 1.5 + 0.5 * len(l2)))
            im = ax.imshow(Z, origin="lower", aspect="auto", cmap="viridis")
            ax.set_xticks(range(len(l1))); ax.set_xticklabels([str(x) for x in l1], rotation=45, fontsize=7)
            ax.set_yticks(range(len(l2))); ax.set_yticklabels([str(x) for x in l2], fontsize=7)
            ax.set_xlabel(ax1); ax.set_ylabel(ax2)
            ax.set_title(f"{job}: {metric}", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046)
            plt.tight_layout()
            fn = FIG / f"phase_{job}_{metric}.png"
            plt.savefig(fn, dpi=110); plt.close(fig)
            made.append(fn.name)
    return made


def health_report(rows):
    lines = ["## Health\n"]
    n = len(rows)
    div = [r for r in rows if r.get("diverged")]
    lines.append(f"- {len(div)}/{n} runs flagged diverged (max|Pi| > 1e3 or non-finite).")
    by_om = {}
    for r in rows:
        by_om.setdefault(r.get("omega"), []).append(1 if r.get("diverged") else 0)
    lines.append("- diverged fraction by forgetting omega:")
    for om in sorted(by_om, key=lambda x: (x is None, x)):
        v = by_om[om]
        lines.append(f"    omega={om}: {np.mean(v):.2f} ({sum(v)}/{len(v)})")
    return "\n".join(lines) + "\n"


def axis_importance(rows):
    """Tier-2 random sample: spread of per-level metric means = how much each axis moves it."""
    t2 = [r for r in rows if r.get("tier") == 2 and not r.get("diverged")]
    lines = ["## Axis importance (Tier-2 random sample, diverged runs excluded)\n"]
    if not t2:
        return "## Axis importance\n- no clean Tier-2 rows yet.\n"
    for metric in ("final_struct_dist", "lockin_frac", "final_comm_diversity"):
        ranks = []
        for ax in AXES_ALL:
            by = {}
            for r in t2:
                m = r.get(metric)
                if m is not None and np.isfinite(m):
                    by.setdefault(r.get(ax), []).append(m)
            if len(by) >= 2:
                means = [np.mean(v) for v in by.values()]
                ranks.append((float(np.nanmax(means) - np.nanmin(means)), ax))
        ranks.sort(reverse=True)
        lines.append(f"- **{metric}** most-moved-by: " +
                     ", ".join(f"{ax}({spread:.2f})" for spread, ax in ranks[:5]))
    return "\n".join(lines) + "\n"


def interesting_flags(rows):
    lines = ["## Flags (diverged runs excluded)\n"]
    clean = [r for r in rows if not r.get("diverged")]
    valid = [r for r in clean if r.get("final_struct_dist") is not None]
    if valid:
        top = max(valid, key=lambda r: r.get("final_struct_dist", 0))
        lines.append(f"- max structural divergence {top['final_struct_dist']:.2f} at "
                     f"N={top.get('N')} inter={top.get('inter')} omega={top.get('omega')} "
                     f"n_comm={top.get('n_communities')} beta={top.get('attention_beta')} "
                     f"world={top.get('world_mode')}.")
        lock = [r for r in clean if r.get("world_mode") == "changing_epochs"
                and r.get("lockin_frac") is not None]
        if lock:
            tl = max(lock, key=lambda r: r.get("lockin_frac", 0))
            lines.append(f"- max lock-in {tl['lockin_frac']:.2f} (changing-epochs) at "
                         f"omega={tl.get('omega')} tilt={tl.get('conviction_tilt')} "
                         f"inter={tl.get('inter')}.")
    return "\n".join(lines) + "\n"


def write_csv(rows):
    """Flatten the JSONL rows to a tidy rows.csv (config axes first, then metrics)."""
    pref = ["hash", "tier", "job", "seed", "N", "inter", "omega", "n_communities",
            "attention_beta", "conviction_tilt", "world_mode", "blend_ratio", "sigma_o",
            "fuse_mode", "n_steps", "error"]
    keys = set().union(*(r.keys() for r in rows))
    fields = [k for k in pref if k in keys] + sorted(k for k in keys if k not in pref)
    with (OUT / "rows.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    return len(rows), len(fields)


def main():
    rows = load_rows()
    if not rows:
        print("no rows to analyze yet."); return 0
    # CSV covers ALL rows (incl. errored); phase maps use only successful ones.
    import json as _json
    all_rows = [_json.loads(l) for l in (OUT / "rows.jsonl").open("r", encoding="utf-8") if l.strip()]
    nr, nf = write_csv(all_rows)
    print(f"wrote rows.csv ({nr} rows x {nf} cols)")
    made = phase_maps(rows)
    report = (f"# Big sweep report\n\n{len(rows)} successful runs.\n\n"
              + health_report(rows) + "\n" + axis_importance(rows) + "\n"
              + interesting_flags(rows) + "\n## Phase maps\n"
              + "\n".join(f"- figures/{m}" for m in made) + "\n")
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"wrote {len(made)} phase maps + REPORT.md to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
