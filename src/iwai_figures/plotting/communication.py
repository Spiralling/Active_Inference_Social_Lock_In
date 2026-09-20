"""Four communication-channel plots used by the layout revision."""
from pathlib import Path
import numpy as np
from . import style
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from .data import PlotData
HERE=Path(__file__).resolve().parent
BLUE, ORANGE, GREY = style.BLUE, style.ORANGE, style.GREY
GREEN=style.orx.PALETTE["green"]
ARM = {"00": (GREY, ":", "Neither"), "10": (ORANGE, "-", "Conditional only"),
       "01": (BLUE, "--", "Weights only"), "11": (GREEN, "-.", "Both")}
CONTROLS=["C10", "C10_beta50", "C10_omega90", "C10_gasvar_matched"]
CONTROL_LABELS=["Primary", r"$\beta=0.5$", r"$\omega=0.9$", "Gas variance matched"]
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
