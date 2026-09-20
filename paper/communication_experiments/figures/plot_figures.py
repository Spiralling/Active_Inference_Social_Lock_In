#!/usr/bin/env python3
"""Real-LaTeX publication figures from frozen, portable per-seed reductions."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
os.environ["PATH"] = "/Library/TeX/texbin:" + os.environ.get("PATH", "")
import publication_style as style
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BLUE, ORANGE, GREY = style.BLUE, style.ORANGE, style.GREY
GREEN = style.orx.PALETTE["green"]
PURPLE = style.orx.PALETTE["purple"]
ARM = {"00": (GREY, ":", "Neither"), "10": (ORANGE, "-", "Conditional only"),
       "01": (BLUE, "--", "Weights only"), "11": (GREEN, "-.", "Both")}
CONTROLS = ["C10", "C10_beta50", "C10_omega90", "C10_gasvar_matched"]
CONTROL_LABELS = ["Primary", r"$\beta=0.5$", r"$\omega=0.9$", "Gas variance matched"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Data:
    def __init__(self, root):
        self.root = root
        self.a = np.load(root/"plot_data.npz", allow_pickle=False)
        self.summary = json.loads((root/"summary.json").read_text())
        idx = self.a["bootstrap_indices"]
        self.weights = np.stack([np.bincount(row, minlength=idx.shape[1]) for row in idx]) / idx.shape[1]
        self.cache = {}

    def get(self, name, key):
        return self.a[f"{name}::{key}"]

    def ci(self, x):
        x = np.asarray(x, float)
        b = self.weights @ x
        lo, hi = np.quantile(b, [.025, .975], axis=0)
        return x.mean(0), lo, hi

    def curve(self, ax, name, metric, color, ls="-", state_min=1, state_max=None, offset=0):
        key = (name, metric)
        x = self.get(name, "states" if metric in ("q_A", "q_B", "primary_opposition", "secondary_opposition") else "aux_states")
        if key not in self.cache:
            self.cache[key] = self.ci(self.get(name, metric))
        mean, lo, hi = self.cache[key]
        mask = x >= state_min
        if state_max is not None:
            mask &= x <= state_max
        ax.fill_between(x[mask]-offset, lo[mask], hi[mask], color=color, alpha=.16, linewidth=0)
        return ax.plot(x[mask]-offset, mean[mask], color=color, linestyle=ls, linewidth=1.05)[0]


def panel(ax, letter, text=""):
    ax.text(0, 1.06, r"\textbf{"+letter+r"}"+(r"\quad "+text if text else ""),
            transform=ax.transAxes, ha="left", va="bottom")


def qaxis(ax, xmax, xmin=0, branch=False):
    ax.set_ylim(0, 1)
    ax.set_yticks([0, .5, 1])
    ax.set_ylabel(r"Oxygen probability $q(O)$")
    ax.set_xlim(xmin, xmax)
    ticks = [0, 50, 100] if xmax == 100 else ([0, 300, 600] if xmax == 600 else ([0, 1750, 3500] if xmax == 3500 else [0, 2500, 5000]))
    ax.set_xticks(ticks, [f"{t:,}" for t in ticks])
    ax.set_xlabel("Updates after intervention" if branch else "Completed updates")


def arm_legend(fig, y=.01, title=None):
    fig.legend([Line2D([], [], color=c, ls=l) for c, l, _ in ARM.values()],
               [s for _, _, s in ARM.values()], ncol=2, loc="lower center",
               bbox_to_anchor=(.53, y), frameon=False, columnspacing=1.4, title=title,
               title_fontsize=8)


def point_ci(ax, x, y, lo, hi, color=BLUE, marker="o", horizontal=False):
    if horizontal:
        ax.plot([lo, hi], [y, y], color=color, lw=1)
        ax.plot(x, y, marker, color=color, ms=3.4)
    else:
        ax.plot([x, x], [lo, hi], color=color, lw=1)
        ax.plot(x, y, marker, color=color, ms=3.4)


def save(fig, stem, records):
    records[stem] = style.save(fig, HERE/stem)
    print("Rendered", stem, flush=True)


def network(data, records):
    fig, axes = plt.subplots(2, 2, figsize=(4.8, 4.2))
    fig.subplots_adjust(left=.125, right=.97, bottom=.28, top=.92, hspace=.7, wspace=.42)
    for ax, group, label, letter in zip(axes[0], ("A", "B"), ("Full observation", "Restricted observation"), "ab"):
        for suffix, (color, ls, _) in ARM.items():
            data.curve(ax, "N"+suffix, "q_"+group, color, ls)
        ax.axvline(80, color=GREY, lw=.7, ls=":")
        qaxis(ax, 600)
        panel(ax, letter, label)
    for ax, metric, ylabel, letter in zip(axes[1], ("mse", "nll"), ("Full-signal MSE", "Predictive NLL (nats)"), "cd"):
        for i, suffix in enumerate(ARM):
            for letter_g, color, marker, off in (("A", BLUE, "o", -.11), ("B", ORANGE, "s", .11)):
                mean, lo, hi = data.ci(data.get("N"+suffix, f"{metric}_{letter_g}_600_tail"))
                point_ci(ax, i+off, mean, lo, hi, color, marker)
        ax.set_xticks(range(4), ["Neither", "Cond.", "Weights", "Both"], rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_xlim(-.4, 3.4)
        panel(ax, letter)
    arm_legend(fig, .008)
    fig.legend([Line2D([], [], color=BLUE, marker="o", ls=""), Line2D([], [], color=ORANGE, marker="s", ls="")],
               ["Full", "Restricted"], loc="lower center", bbox_to_anchor=(.52, .115), ncol=2)
    save(fig, "fig_network_channels", records)


def ecdf(ax, values, color, ls):
    x = np.sort(values.ravel())
    # Descriptive agent distribution. Agents within a seed are dependent.
    ax.step(np.r_[0, x, 1], np.r_[0, np.arange(1, len(x)+1)/len(x), 1], where="post", color=color, ls=ls, lw=1)


def strip(ax, position, values, color):
    """All dependent agent values; fixed display jitter never changes q."""
    y = values.ravel()
    jitter = .18*(2*((np.arange(len(y))*.6180339887498949) % 1)-1)
    ax.plot(position+jitter, y, ".", color=color, ms=1.5, alpha=.5, ls="")


def strip_axis(ax, group):
    ax.set_xlim(-.4, 3.4)
    ax.set_ylim(-.04, 1.04)
    ax.set_yticks([0, .5, 1])
    ax.set_xticks(range(4), ["Neither", "Cond.", "Weights", "Both"], rotation=25, ha="right")
    ax.set_ylabel(r"Individual "+group+r"-observer $q(O)$")


def camps(data, records):
    fig, axes = plt.subplots(2, 2, figsize=(4.8, 4.0))
    fig.subplots_adjust(left=.125, right=.97, bottom=.24, top=.92, hspace=.65, wspace=.42)
    for ax, group, label, letter in zip(axes[0], ("A", "B"), ("Gas observers", "Mass observers"), "ab"):
        for suffix, (color, ls, _) in ARM.items():
            data.curve(ax, "C"+suffix, "q_"+group, color, ls)
        ax.axvline(1500, color=GREY, lw=.7, ls=":")
        qaxis(ax, 5000)
        panel(ax, letter, label)
    for ax, horizon, letter in zip(axes[1], (1500, 5000), "cd"):
        for i, (suffix, (color, ls, _)) in enumerate(ARM.items()):
            strip(ax, i, data.get("C"+suffix, f"individual_q_{horizon}")[:, :8], color)
        strip_axis(ax, "gas")
        panel(ax, letter, f"State {horizon:,}")
    arm_legend(fig)
    save(fig, "fig_camp_channels", records)


def diagnostics(data, records, protocol):
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.9), sharey=True, gridspec_kw={"width_ratios": [1.05, 1]})
    fig.subplots_adjust(left=.31, right=.97, bottom=.3, top=.86, wspace=.34)
    for y, name in zip(np.arange(4)[::-1], CONTROLS):
        mean, lo, hi = data.ci(data.get(name, "score_A_5000_tail"))
        point_ci(axes[0], 1000*mean, y+.07, 1000*lo, 1000*hi, BLUE, "o", True)
        pred = protocol["stationary_forecasts"][name][0]
        axes[0].plot(1000*pred, y-.1, "|", color=GREY, ms=8, mew=1.2)
        for horizon, color, marker, off in ((1500, ORANGE, "o", -.1), (5000, BLUE, "s", .1)):
            mean, lo, hi = data.ci(data.get(name, f"q_A_{horizon}_endpoint"))
            point_ci(axes[1], mean, y+off, lo, hi, color, marker, True)
    axes[0].set_yticks(np.arange(4)[::-1], CONTROL_LABELS)
    axes[0].set_ylabel("Control", labelpad=3)
    axes[0].axvline(0, color=GREY, ls=":", lw=.7)
    axes[0].set_xlabel("Gas score difference\n"+r"($10^{-3}$ nats per update)")
    axes[1].set_xlabel(r"Gas-group mean $q(O)$")
    axes[1].set_xlim(-.02, 1.02)
    axes[1].set_xticks([0, .5, 1])
    for ax, letter in zip(axes, "ab"):
        ax.set_ylim(-.45, 3.45)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        panel(ax, letter)
    fig.legend([Line2D([], [], color=BLUE, marker="o", ls=""), Line2D([], [], color=GREY, marker="|", ls="", ms=7),
                Line2D([], [], color=ORANGE, marker="o", ls=""), Line2D([], [], color=BLUE, marker="s", ls="")],
               ["Empirical score", "Frozen forecast", "State 1,500", "State 5,000"],
               loc="lower center", bbox_to_anchor=(.57, .012), ncol=2)
    save(fig, "fig_reference_diagnostics", records)


def branches(data, records):
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.4))
    fig.subplots_adjust(left=.125, right=.97, bottom=.3, top=.84, wspace=.42)
    for ax, maximum, letter, text in zip(axes, (3500, 100), "ab", ("Gas: full continuation", "Gas: first 100 updates")):
        for name, color, ls in (("S_keep_private", GREY, "--"), ("S_enable_weights", BLUE, "-")):
            data.curve(ax, name, "q_A", color, ls, state_min=1500, state_max=1500+maximum, offset=1500)
        qaxis(ax, maximum, branch=True)
        panel(ax, letter, text)
    fig.legend([Line2D([], [], color=GREY, ls="--"), Line2D([], [], color=BLUE)],
               ["Keep weights private", "Enable weight sharing"], loc="lower center", bbox_to_anchor=(.53, .015), ncol=2)
    save(fig, "fig_weight_sharing_intervention", records)


def all_controls(data, records):
    names = ["C00", "C10", "C01", "C11", "C10_beta50", "C10_omega90", "C10_gasvar_matched"]
    labels = ["Neither", "Conditional only", "Weights only", "Both", r"Conditional: $\beta=0.5$", r"Conditional: $\omega=0.9$", "Gas variance matched"]
    fig, axes = plt.subplots(4, 2, figsize=(4.8, 6.8))
    fig.subplots_adjust(left=.125, right=.97, bottom=.12, top=.96, hspace=.87, wspace=.42)
    for ax, name, label, letter in zip(axes.flat, names, labels, "abcdefg"):
        data.curve(ax, name, "q_A", BLUE)
        data.curve(ax, name, "q_B", ORANGE, "--")
        qaxis(ax, 5000)
        panel(ax, letter, label)
    ax = axes.flat[-1]
    for name, color, ls in (("C10", BLUE, "-"), ("C11", ORANGE, "--")):
        data.curve(ax, name, "primary_opposition", color, ls)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 5000)
    ax.set_xticks([0, 2500, 5000], ["0", "2,500", "5,000"])
    ax.set_ylabel("Opposing-seed fraction")
    ax.set_xlabel("Completed updates")
    panel(ax, "h", "Conditional only / both")
    fig.legend([Line2D([], [], color=BLUE), Line2D([], [], color=ORANGE, ls="--")],
               ["Gas observers (a--g)", "Mass observers (a--g)"], loc="lower center", bbox_to_anchor=(.53, .025), ncol=2)
    save(fig, "fig_all_camp_conditions", records)


def performance(data, records):
    fig, axes = plt.subplots(2, 2, figsize=(4.8, 3.85))
    fig.subplots_adjust(left=.125, right=.97, bottom=.14, top=.92, wspace=.5, hspace=.68)
    for ax, metric, label, letter in zip(list(axes.flat)[:3], ("q", "mse", "nll"), (r"$\Delta q(O)$", r"$\Delta$ full MSE", r"$\Delta$ NLL (nats)"), "abc"):
        for i, (group, color) in enumerate((("A", BLUE), ("B", ORANGE))):
            mean, lo, hi = data.ci(data.a[f"branch::{metric}_{group}_tail"])
            point_ci(ax, i, mean, lo, hi, color)
        ax.axhline(0, color=GREY, lw=.7, ls=":")
        ax.set_xticks([0, 1], ["Gas", "Mass"])
        ax.set_xlim(-.4, 1.4)
        ax.set_ylabel(label)
        panel(ax, letter)
    ax = axes.flat[-1]
    for name, color, ls in (("S_keep_private", GREY, "--"), ("S_enable_weights", BLUE, "-")):
        data.curve(ax, name, "q_B", color, ls, state_min=1500, offset=1500)
    qaxis(ax, 3500, branch=True)
    panel(ax, "d", "Mass: full continuation")
    save(fig, "fig_intervention_prediction", records)


def individual_and_mechanism(data, records):
    fig, axes = plt.subplots(2, 2, figsize=(4.8, 3.9))
    fig.subplots_adjust(left=.13, right=.97, bottom=.24, top=.92, hspace=.8, wspace=.44)
    for ax, horizon, letter in zip(axes[0], (1500, 5000), "ab"):
        for i, (suffix, (color, ls, _)) in enumerate(ARM.items()):
            strip(ax, i, data.get("C"+suffix, f"individual_q_{horizon}")[:, 8:], color)
        strip_axis(ax, "mass")
        ax.set_ylabel(r"Individual $q(O)$")
        panel(ax, letter, f"Mass: state {horizon:,}")
    fixed = "C10"
    x = data.get(fixed, "states")
    qi = data.get(fixed, "illustrative_individual_q")
    for agent in range(16):
        axes[1, 0].plot(x, qi[:, agent], color=BLUE if agent < 8 else ORANGE,
                         ls="-" if agent < 8 else "--", lw=.65, alpha=.65)
    qaxis(axes[1, 0], 5000)
    panel(axes[1, 0], "c", "Fixed seed 10,000")
    for key, color, ls in (("H_individual", BLUE, "-"), ("disagreement", ORANGE, "--")):
        data.curve(axes[1, 1], fixed, key, color, ls)
    axes[1, 1].set_ylim(0, np.log(2)*1.04)
    axes[1, 1].set_ylabel("Entropy (nats)")
    axes[1, 1].set_xlabel("Completed updates")
    axes[1, 1].set_xlim(0, 5000)
    axes[1, 1].set_xticks([0, 2500, 5000], ["0", "2,500", "5,000"])
    panel(axes[1, 1], "d", "Individual / disagreement")
    arm_legend(fig, title="Conditions in (a,b)")
    save(fig, "fig_individual_and_uncertainty", records)


def captions(data):
    common = (r"Unless otherwise specified, ensemble curves and summary points summarize 50 paired seeds. Shading and error bars are pointwise 95\% percentile intervals "
              r"from 2,000 whole-seed bootstrap resamples (analysis seed 20260920); they are neither ranges of agents nor simultaneous confidence bands. ")
    return {
      "fig_network_channels": r"\textbf{Separate effects of the two communication channels.} "
        r"Four arms share an 80-update fully observed history, then the mass signal changes from $-1$ to $+1$ (dotted line). "
        r"Four agents retain full attention; twenty have weights $(1,.01,.01)$. The latter receive weak gas and mass evidence, not zero evidence. "
        r"(a,b) Group-average oxygen probability. (c,d) Mean full-signal MSE and held-out mixture NLL over completed states 451--600, "
        r"averaged within each observation group before averaging seeds; lower is better. Conditional exchange is uniform over 24 agents when enabled; "
        r"model-weight exchange uses $\alpha_m=.02$ when enabled. $\omega=.9$, $\sigma_o=1$. "
        r"The existing network seed set 0--49 is reused for paired extension, not independent confirmation of its three archived arms. "+common,
      "fig_camp_channels": r"\textbf{Communication channels and opposed account preferences.} "
        r"Eight gas observers and eight mass observers receive private noisy measurements of the same constant observable signal $(1,1,1)$. "
        r"(a,b) Group-average oxygen probability under four channel combinations. The dotted line marks the first reporting horizon, state 1,500. "
        r"(c,d) Descriptive strip distributions of individual gas-observer probabilities at states 1,500 and 5,000, "
        r"including all eight agents in each of 50 seeds (400 values per arm, dependent within seed). These distributions have no inferential bands. "
        r"Fixed horizontal jitter separates coincident points and does not change probabilities. "
        r"$\beta=.05$ is the total cross-group conditional weight; $\alpha_m=.02$ when model sharing is enabled; $\omega=.3$, $\sigma_o=.5$. "
        r"Reference means and observation groups are identical across arms; scores accumulate without decay. "+common,
      "fig_reference_diagnostics": r"\textbf{Diagnostics of the small gas-score preference.} "
        r"All four arms share conditional beliefs and retain private model weights. Primary settings are $\beta=.05$, $\omega=.3$, $\sigma_o=.5$. "
        r"Each additional arm changes only the named setting; gas-variance matching changes the release gas residual from .5 to .5448, matching the "
        r"reference gas marginal variance (.8648), not the full covariance or topology. (a) Oxygen-minus-release predictive score for gas observers, "
        r"averaged over completed states 3,751--5,000. Grey ticks are stationary expectations supplied before this confirmation; they are not fitted to the runs "
        r"and do not predict finite-time adoption exactly. (b) Gas-group mean oxygen probability at the two endpoints, distinct from window averages. "+common,
      "fig_weight_sharing_intervention": r"\textbf{A fixed-time intervention on model-weight communication.} "
        r"Every primary conditional-only seed branches from its exact state 1,500: keep model weights private or enable $\alpha_m=.02$. "
        r"Conditional sharing, attention, references and all future training and evaluation draws remain paired; no seed is selected by its outcome. "
        r"Gas-group mean oxygen probability over (a) all 3,500 continuation updates and (b) the first 100 updates, "
        r"a zoom specified before these runs using the expected rapid transfer of accumulated support. "
        r"Mass-group probabilities remain at one to stored numerical precision; their full continuation and predictive outcomes appear in the intervention supplement. "
        r"$\beta=.05$, $\omega=.3$, $\sigma_o=.5$. "
        r"The origin is the common pre-intervention state. Conditional trajectories and direct score streams are unchanged by the intervention; probability-dependent predictions can change. "+common,
      "fig_all_camp_conditions": r"\textbf{All seven camp conditions over their full horizon.} "
        r"(a--g) Gas- and mass-group mean oxygen probabilities. The three diagnostic arms change one primary conditional-only setting at a time. "
        r"(h) Fraction of seeds whose gas-group mean probability is below .1 and mass-group mean is above .9, comparing conditional-only (blue solid) "
        r"with both-channel communication (orange dashed). This is a statewise fraction, not the separate seed-pass criterion requiring 90\% occupancy in a reporting window. "+common,
      "fig_intervention_prediction": r"\textbf{Paired predictive consequences of enabling model-weight sharing.} "
        r"Each point is an enable-minus-keep difference, formed within each of the 50 seeds before aggregation. "
        r"(a) Group-mean oxygen probability. (b) Full-signal mixture-mean MSE. (c) Held-out mixture NLL (nats per full observation). "
        r"All three are averaged over completed states 3,751--5,000, separately for gas and mass observers. "
        r"Negative predictive differences indicate improvement; a probability change alone does not establish improved prediction. "
        r"(d) Full mass-group probability continuation: private (grey dashed) and shared (blue solid) weights overlap at one to stored numerical precision. "+common,
      "fig_individual_and_uncertainty": r"\textbf{Individual probabilities and uncertainty diagnostics.} "
        r"(a,b) Individual mass-observer probability distributions for all four arms; strips show all 400 dependent agent values per arm, not 400 independent replications. "
        r"Fixed horizontal jitter is for display only. "
        r"(c) All 16 individual trajectories in the primary conditional-only condition for the prespecified illustrative seed 10,000; blue denotes gas and orange dashed mass. "
        r"This seed is an illustration, not an inferential sample selected for a clear effect. "
        r"(d) Within-seed average individual entropy (blue solid) and between-agent disagreement $H(\langle q_i\rangle)-\langle H(q_i)\rangle$ "
        r"(orange dashed), then averaged across seeds. They distinguish uncertainty within agents from diversity between them. "
        r"Entropy is displayed every ten updates without smoothing; summaries use every raw update. "+common,
    }


def build_pack(caps, records):
    body = []
    for stem in records:
        body.append(r"\begin{center}\includegraphics[width=4.8in]{"+stem+r".pdf}\end{center}"+"\n"+caps[stem]+r"\clearpage")
    tex = (r"\documentclass[10pt]{article}"+"\n"+r"\usepackage[T1]{fontenc}\usepackage{graphicx}\usepackage{amsmath}"+"\n"
           +r"\usepackage[paperwidth=6.2in,paperheight=10.8in,left=.7in,right=.7in,top=.5in,bottom=.5in]{geometry}"+"\n"
           +r"\pagestyle{empty}\setlength{\parindent}{0pt}\begin{document}"+"\n"+"\n".join(body)+"\n"+r"\end{document}"+"\n")
    (HERE/"figure_pack.tex").write_text(tex)
    result = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "figure_pack.tex"], cwd=HERE, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout[-6000:]+result.stderr)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis", type=Path, default=STUDY/"analysis")
    args = p.parse_args()
    style.use_style()
    data = Data(args.analysis)
    protocol = json.loads((STUDY/"protocol.json").read_text())
    records = {}
    network(data, records)
    camps(data, records)
    diagnostics(data, records, protocol)
    branches(data, records)
    all_controls(data, records)
    performance(data, records)
    individual_and_mechanism(data, records)
    caps = captions(data)
    (HERE/"captions.json").write_text(json.dumps(caps, indent=2)+"\n")
    (HERE/"captions.md").write_text("# Publication figure captions\n\n"+"\n\n".join("## "+stem+"\n\n"+text for stem, text in caps.items())+"\n")
    (HERE/"integration_blocks.tex").write_text("\n\n".join(
        r"\begin{figure}[t]\centering"+"\n"+r"\includegraphics[width=\textwidth]{figures/"+stem+r".pdf}"+"\n"+r"\caption{"+caps[stem]+r"}\label{fig:"+stem+r"}\end{figure}" for stem in records)+"\n")
    build_pack(caps, records)
    manifest = {"style": style.provenance(), "plot_script_sha256": sha(__file__),
                "plot_data_sha256": sha(args.analysis/"plot_data.npz"), "protocol_sha256": sha(STUDY/"protocol.json"),
                "figures": records, "captions_sha256": sha(HERE/"captions.json"),
                "pack_sha256": sha(HERE/"figure_pack.pdf"), "pack_pages": len(records),
                "statistical_unit": "seed", "bootstrap": "2000 shared whole-seed resamples, seed20260920, pointwise95% percentile",
                "illustrative_seed": 10000, "branch_zoom": "first100updates, prespecified before production outcomes",
                "plot_selection": "Four prespecified main candidates; three supplements retain all conditions, predictive differences, and individual diagnostics"}
    (HERE/"figure_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
