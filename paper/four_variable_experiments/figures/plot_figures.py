"""Render frozen four-variable experiments at their final LNCS print size.

The renderer never simulates, selects seeds, smooths, or modifies results.
Ribbons are pointwise percentile-bootstrap intervals over whole seed curves.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from publication_style import use_style, save, provenance, WIDTH, BLUE, ORANGE, GREY
from orx_figstyle import PALETTE
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import xlogy
import fitz

BOOTSTRAP_SEED = 20260919
BOOTSTRAP_REPLICATES = 2000


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path):
    return {"path": str(Path(path).resolve()), "sha256": sha256(path)}


def interval(runs):
    runs = np.asarray(runs, dtype=float)
    if runs.ndim != 2 or len(runs) < 3 or not np.isfinite(runs).all():
        raise ValueError("Expected at least three finite seed-level curves")
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, len(runs), (BOOTSTRAP_REPLICATES, len(runs)))
    weights = np.zeros((BOOTSTRAP_REPLICATES, len(runs)))
    np.add.at(weights, (np.arange(BOOTSTRAP_REPLICATES)[:, None], indices), 1)
    bootstrap = weights @ runs / len(runs)
    low, high = np.quantile(bootstrap, [.025, .975], axis=0, method="linear")
    return runs.mean(axis=0), low, high


def curve(ax, runs, color, label=None, linestyle="-", x=None, individual=False):
    runs = np.asarray(runs)
    x = np.arange(runs.shape[1]) if x is None else x
    mean, low, high = interval(runs)
    if individual:
        for trajectory in runs:
            ax.plot(x, trajectory, color=color, lw=.35, alpha=.14)
    ax.fill_between(x, low, high, color=color, alpha=.15, lw=0)
    ax.plot(x, mean, color=color, label=label, ls=linestyle, lw=1.5)
    return {"seed_count": len(runs), "final_mean": float(mean[-1]),
            "final_interval": [float(low[-1]), float(high[-1])],
            "interval_bounds": [float(low.min()), float(high.max())]}


def entropy(q):
    q = np.asarray(q, dtype=float)
    if np.any((q < 0) | (q > 1)):
        raise ValueError("Probability outside [0,1]")
    return -xlogy(q, q)-xlogy(1-q, 1-q)


def panel(ax, label):
    ax.annotate(rf"\textbf{{({label})}}", (0, 1), xycoords="axes fraction",
                xytext=(0, 6), textcoords="offset points", ha="left", va="bottom")


def grid(ax):
    ax.grid(axis="y", color="#E1E1E1", lw=.4)
    ax.set_axisbelow(True)


def band_caption(n):
    return (f"Lines average {n} seeds; ribbons are pointwise 95% percentile-bootstrap "
            "intervals for the seed mean (2,000 resamples of complete seed curves). "
            "No smoothing or interval clipping is used.")


def compile_tex(filename):
    process = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-no-shell-escape", filename],
                             cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError(process.stdout.decode(errors="replace")[-6000:])
    if "Overfull" in process.stdout.decode(errors="replace"):
        raise ValueError("LaTeX figure has an overfull box")


def check_pdf(path):
    with fitz.open(path) as pdf:
        if len(pdf) != 1 or abs(pdf[0].rect.width-72*WIDTH) > .01:
            raise ValueError("Figure must be one page at exact LNCS width")
        fonts = sorted({item[3].split("+")[-1] for item in pdf[0].get_fonts()})
        if not fonts or any(not font.upper().startswith(("CM", "SF")) for font in fonts):
            raise ValueError(f"Unexpected manuscript font family: {fonts}")
        if pdf[0].get_images():
            raise ValueError("Unexpected raster image in scientific vector figure")
        for block in pdf[0].get_text("blocks"):
            if not pdf[0].rect.contains(fitz.Rect(block[:4])):
                raise ValueError("PDF text outside page bounds")
    return fonts


def model_figure():
    compile_tex("model.tex")
    target = HERE / "fig01_model.pdf"
    target.write_bytes((HERE / "model.pdf").read_bytes())
    fonts = check_pdf(target)
    with fitz.open(target) as pdf:
        (HERE / "fig01_model.svg").write_text(pdf[0].get_svg_image(text_as_path=True))
        pdf[0].get_pixmap(matrix=fitz.Matrix(220/72, 220/72)).save(HERE / "fig01_model.png")
    return {"outputs": {ext: record(HERE / f"fig01_model.{ext}") for ext in ("pdf", "svg", "png")},
            "embedded_fonts": fonts, "source": record(HERE / "model.tex"),
            "layout_audit": "TikZ: exact LNCS width; fonts/vector/text bounds checked",
            "caption": "Two supplied prior theories organize the same three observable commitments. "
            "The hidden explanatory variable is interpreted as phlogiston release P in (a), or oxygen "
            "participation O in (b). Observable commitments are combustion C, gas uptake G and mass "
            "change M. The phlogiston candidate links its core directly to all three; the oxygen "
            "candidate links its core to C and G, and G to M. Arrows are linear-Gaussian belief "
            "dependencies, not a historically calibrated chemical model. Signs show coefficient "
            "directions; magnitudes are 0.8 at the core and 0.9 for G to M. Dashed nodes are unobserved. "
            "The same array coordinate holds the candidate-specific hidden variable; it is not shared "
            "evidence that P and O denote one physical quantity. Single-Gaussian controls operate "
            "on the three shared observable commitments. No aggregate of background commitments remains."}


class Data:
    def __init__(self, directory):
        self.directory = directory
        self.cache = {}
        self.inputs = {}
        manifest = directory / "results_manifest.json"
        self.expected = json.loads(manifest.read_text()) if manifest.exists() else {}

    def __getitem__(self, name):
        if name not in self.cache:
            choices = [self.directory / f"{name}.npz", self.directory / "conditions" / f"{name}.npz"]
            path = next((p for p in choices if p.exists()), None)
            if path is None:
                raise FileNotFoundError(f"No frozen result for {name}: {choices}")
            if name in self.expected and sha256(path) != self.expected[name]["sha256"]:
                raise ValueError(f"Frozen condition checksum mismatch: {name}")
            with np.load(path, allow_pickle=False) as arrays:
                self.cache[name] = {key: arrays[key] for key in arrays.files}
            self.inputs[name] = record(path)
        return self.cache[name]


def oxygen(data, name):
    return np.asarray(data[name]["q"])[..., 1]


def cycle_figure(data, protocol):
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 3.55), sharex=True)
    fig.subplots_adjust(left=.14, right=.975, bottom=.20, top=.935, hspace=.36)
    summary = {}
    for key, label, color, linestyle in [("cycle", "Changed world", BLUE, "-"),
                                         ("cycle_no_shift", "Unchanged world", GREY, "--")]:
        q = oxygen(data, key)
        summary[key] = {"q": curve(axes[0], q.mean(axis=2), color, label, linestyle),
                        "entropy": curve(axes[1], entropy(q).mean(axis=2), color, label, linestyle)}
    for ax, letter in zip(axes, ["a", "b"]):
        panel(ax, letter)
        ax.axvline(protocol["shift"], color=ORANGE, lw=.9, ls=":")
        grid(ax)
        ax.set(xlim=(0, q.shape[1]-1))
    axes[0].set(ylabel=r"Mean $q(\mathrm{oxygen})$", ylim=(-.02, 1.02), yticks=[0, .5, 1])
    axes[1].set(ylabel="Individual entropy (nats)", xlabel="Simulation step", ylim=(-.015, .71),
                yticks=[0, .2, .4, .6], xticks=[0, 40, 80, 120, 160, 200, 240])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.54, .01), ncol=2)
    return fig, {"summary": summary, "caption":
                "The cycle experiment tests switching between two supplied theories. "
                "Sixteen agents on a complete graph start with equal candidate probabilities. The "
                "observable world changes at step 80 (dotted line): C stays at +1 while G and M "
                "switch from -1 to +1. The grey control does not change. "
                "(a) Probability assigned to oxygen, averaged over agents within each seed. "
                "(b) Individual posterior entropy, computed before averaging agents and seeds. "
                "Parameter memory is 0.9, model-score sharing 0.02 and observation noise standard "
                "deviation 1. Both candidates remain available throughout; no theory is generated. "
                "The first plotted point follows the first update. " + band_caption(len(q))}


def topology_figure(data, graphs, protocol):
    from publication_topology import figure9
    view = {"seeds": np.asarray(protocol["network_seeds"]),
            "vanguard_nodes": np.asarray(protocol["informed_nodes"])}
    for name in protocol["network_graphs"]:
        adjacency = np.asarray(graphs[f"A_{name}"])
        view[f"A_{name}"] = adjacency
        view[f"q_{name}"] = oxygen(data, "network_"+name)
        view[f"lambda2_{name}"] = np.linalg.eigvalsh(np.diag(adjacency.sum(axis=1))-adjacency)[1]
    params = {"Q_CONS": protocol["threshold"], "T_SHIFT": protocol["shift"]}
    fig, summary = figure9(view, params, interval)
    fig.axes[0].set_ylabel(r"Restricted agents' $q(\mathrm{oxygen})$")
    summary["band"] = band_caption(len(protocol["network_seeds"]))
    summary["individual_agent_traces"]="All 20 restricted agents from seed 0; these are not seed-level uncertainty."
    summary["graph_drawing"]="All frozen nodes and edges; black outlines identify the four full-attention agents."
    for entry in summary["graphs"].values():
        entry["censoring"]="Noncrossings assigned horizon 600 for this descriptive median only; noncrossing count also reported."
    return fig, {"summary": summary, "caption":
                "Contact topology changes how the community responds to the same evidence. "
                "All 24 nodes and edges are shown; black outlines mark four full-attention agents. "
                "After the world changes at step 80, the other 20 agents assign weight 0.01 to "
                "gas-uptake and mass-change observations; combustion remains fully attended. "
                "Thin lines show every restricted agent from seed 0; heavy lines average restricted "
                "agents within each seed and then average seeds. The graph definitions, informed "
                "positions and row-normalized self-plus-neighbour weights match the accepted "
                "experiment. Memory is 0.9 and model-score sharing 0.02. Lambda2 is the unnormalized "
                "graph Laplacian's second eigenvalue. Density and source placement vary alongside "
                "topology; this is not an isolated test of that eigenvalue. "
                + band_caption(len(protocol["network_seeds"]))}


def wall_distance(result):
    mu = np.asarray(result["mu_map"])[..., 1:]
    half = mu.shape[2] // 2
    return np.linalg.norm(mu[:, :, :half].mean(axis=2)-mu[:, :, half:].mean(axis=2), axis=-1)


def wall_figure(data, matched=False):
    fig = plt.figure(figsize=(WIDTH, 4.4))
    gs = fig.add_gridspec(2, 2, left=.14, right=.975, top=.935, bottom=.235,
                          height_ratios=[1.1, 1], hspace=.53, wspace=.42)
    top = fig.add_subplot(gs[0, :])
    lower = [fig.add_subplot(gs[1, i]) for i in range(2)]
    suffix = "matched" if matched else "original"
    summary = {}
    for name, label, color, ls in [("plain", "Single Gaussian", GREY, ":"),
                                    ("gated", "Trust gate", ORANGE, "--"),
                                    ("rivals", "Two candidates", BLUE, "-")]:
        key = f"wall_{name}_{suffix}"
        if not matched:
            label += {"plain":r" ($\omega=1$)", "gated":r" ($\omega=0.95$)",
                      "rivals":r" ($\omega=0.3$)"}[name]
        distance = wall_distance(data[key])
        if np.any(distance <= 0):
            raise ValueError("Logarithmic wall distance contains a nonpositive value")
        summary[key] = curve(top, distance, color, label, ls)
    q = oxygen(data, f"wall_rivals_{suffix}")
    half = q.shape[2] // 2
    for ax, sl, letter, label in zip(lower, [slice(None,half),slice(half,None)], ["b", "c"],
                                     ["Gas observers", "Mass observers"]):
        curve(ax, q[:, :, sl].mean(axis=2), BLUE)
        ax.set(ylim=(-.02,1.02), yticks=[0,.5,1], xlim=(0,q.shape[1]-1), xticks=[0,750,1500],
               xlabel="Simulation step", ylabel=r"$q(\mathrm{oxygen})$")
        panel(ax, letter)
        ax.text(.60, 1.07, label, transform=ax.transAxes, ha="center", va="bottom", fontsize=8)
        grid(ax)
    top.set(yscale="log", ylabel="Camp-mean distance", xlim=(0,q.shape[1]-1),
            xticks=[0,500,1000,1500], xlabel="Simulation step")
    panel(top, "a"); grid(top)
    handles, labels = top.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.54,.002), ncol=1 if not matched else 2)
    recipe = ("All three conditions use memory 0.9, but the gated condition has camp-specific "
              "anchors and the candidate condition has candidate-specific anchors; this is not a "
              "pure intervention on representation alone. " if matched else
              "The inherited comparison uses memory 1 for the plain Gaussian, 0.95 for the gate "
              "and 0.3 for the two candidates. Anchors also differ, so this comparison does not "
              "isolate representation alone. ")
    return fig, {"summary": summary, "caption":
                "A persistent parameter gap does not preserve opposing theories in this four-variable world. "
                "Two camps of eight agents attend G or M in the fixed mixed world (C,G,M)=(1,-1,1) and "
                "communicate on a complete graph. The gate downweights social influence from agents "
                "with differing observable means; it is not a sensory-reliability update. "
                "(a) Euclidean distance between camp-mean commitments "
                "on C, G and M; the two-candidate condition reads out each agent's highest-probability "
                "candidate. The logarithmic axis displays all positive values without clipping. "
                "(b,c) The camps' oxygen probabilities in that candidate condition, allowing a "
                "parameter gap to be distinguished from sustained opposing model choices. "
                + recipe + band_caption(len(q))}


def social_controls_figure(data, graphs, protocol):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.1))
    fig.subplots_adjust(left=.14,right=.975,bottom=.31,top=.86,wspace=.44)
    groups = [[("complete","Both routes",BLUE,"-"),
               ("hypothesis_off","Parameters only",ORANGE,"--"),
               ("all_social_off","No communication",GREY,":")],
              [("wheel","Hub informed",ORANGE,"--"),("wheel_rim","Rim informed",BLUE,"-")]]
    summaries={}
    for ax, conditions, letter in zip(axes, groups, ["a", "b"]):
        for name,label,color,linestyle in conditions:
            q=oxygen(data,"network_"+name)
            sources=np.asarray(graphs[f"informed_{name}"])
            restricted=np.setdiff1d(np.arange(q.shape[2]),sources)
            summaries[name]=curve(ax,q[:,:,restricted].mean(axis=2),color,label,linestyle)
        ax.axvline(protocol["shift"],color=GREY,lw=.8,ls=":")
        ax.set(xlim=(0,q.shape[1]-1),ylim=(-.02,1.02),xticks=[0,200,400,600],yticks=[0,.5,1],
               xlabel="Simulation step",ylabel=r"Restricted $q(\mathrm{oxygen})$")
        panel(ax,letter);grid(ax)
    handles,labels=[],[]
    for ax in axes:
        h,l=ax.get_legend_handles_labels();handles.extend(h);labels.extend(l)
    fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.54,.01),ncol=2,fontsize=8)
    return fig,{"summary":summaries,"caption":
                "Communication routes and informed-agent placement are separate interventions. "
                "(a) On the complete graph, both information-sharing routes operate, only parameters "
                "are shared, or all social sharing is disabled. Restricted agents still receive "
                "gas and mass evidence at weight 0.01. (b) The wheel's full-attention agents include "
                "the hub or occupy four rim nodes. Each curve averages the other 20 agents within "
                "each seed. In (a), all conditions branch at step 80 from exactly the same "
                "pre-change state; the communication intervention and attention restriction start "
                "together. " + band_caption(len(q))}


def memory_figure(data, protocol):
    fig,axes=plt.subplots(1,2,figsize=(WIDTH,2.8))
    fig.subplots_adjust(left=.14,right=.975,bottom=.27,top=.87,wspace=.44)
    summary={}
    for name,label,color,ls in [("cycle",r"Memory $0.9$",BLUE,"-"),
                                ("cycle_no_forgetting",r"Memory $1$",ORANGE,"--")]:
        q=oxygen(data,name)
        summary[name]=curve(axes[0],q.mean(axis=2),color,label,ls)
        curve(axes[1],np.asarray(data[name]["mse_obs"]).mean(axis=2),color,label,ls)
    summary["single_gaussian_mse"]=curve(axes[1],np.asarray(data["cycle_single_gaussian"]["mse_obs"]).mean(axis=2),
                                         GREY,"Single Gaussian",":")
    for ax,letter in zip(axes,["a","b"]):
        ax.axvline(protocol["shift"],color=GREY,lw=.8,ls=":")
        ax.set(xlim=(0,q.shape[1]-1),xticks=[0,80,160,240],xlabel="Simulation step")
        panel(ax,letter);grid(ax)
    axes[0].set(ylabel=r"Mean $q(\mathrm{oxygen})$",ylim=(-.02,1.02),yticks=[0,.5,1])
    axes[1].set(ylabel="Prediction MSE",ylim=(0,None))
    h,l=axes[1].get_legend_handles_labels()
    fig.legend(h,l,loc="lower center",bbox_to_anchor=(.54,.01),ncol=2)
    return fig,{"summary":summary,"caption":
                "A memory control tests whether adaptation depends on forgetting. "
                "The world changes at step 80. (a) Mean candidate probability with parameter memory "
                "0.9 or 1, retaining the same priors, observations and communication settings. "
                "(b) Noiseless observable prediction MSE adds a single-Gaussian comparator, formed "
                "by information pooling of the candidates' observable marginal priors, with memory "
                "0.9. This comparator has no candidate probability. "
                "A finite run identifies delay or failure to switch "
                "within its horizon; it cannot establish permanent lock-in. "+band_caption(len(q))}


def direct_evidence_figure(data, protocol):
    fig,ax=plt.subplots(figsize=(WIDTH,2.8))
    fig.subplots_adjust(left=.14,right=.975,bottom=.28,top=.91)
    summary={}
    for name,label,color,ls in [("network_zero_direct",r"Diagnostic weight $0$",GREY,":"),
                                ("network_complete",r"Diagnostic weight $0.01$",BLUE,"-"),
                                ("network_full_attention",r"Diagnostic weight $1$",ORANGE,"--")]:
        q=oxygen(data,name)
        summary[name]=curve(ax,q[:,:,4:].mean(axis=2),color,label,ls)
    ax.axvline(protocol["shift"],color=GREY,lw=.8,ls=":")
    ax.set(xlim=(0,q.shape[1]-1),xticks=[0,100,200,300,400,500,600],xlabel="Simulation step",
           ylabel=r"Restricted agents' $q(\mathrm{oxygen})$",ylim=(-.02,1.02),yticks=[0,.25,.5,.75,1])
    grid(ax)
    h,l=ax.get_legend_handles_labels()
    fig.legend(h,l,loc="lower center",bbox_to_anchor=(.54,.01),ncol=2)
    return fig,{"summary":summary,"caption":
                "A direct-evidence control separates reduced attention from no diagnostic observations. "
                "Twenty agents on a complete graph assign the indicated weights to gas and mass "
                "observations after step 80; four source agents always assign weight 1. All conditions "
                "retain full combustion attention and both communication routes. They share the "
                "same pre-change state and private noise streams. Weight zero removes the restricted "
                "agents' diagnostic input but not information arriving through neighbours. "
                +band_caption(len(q))}


def adaptation_figure(data, protocol):
    fig=plt.figure(figsize=(WIDTH,4.0))
    gs=fig.add_gridspec(2,2,left=.14,right=.975,top=.91,bottom=.17,hspace=.72,wspace=.49)
    upper=[fig.add_subplot(gs[0,i]) for i in range(2)]
    lower=[fig.add_subplot(gs[1,i]) for i in range(2)]
    summary={}
    for index,(name,label,color,ls) in enumerate([("adaptation_social","Networked",ORANGE,"-"),
                                ("adaptation_isolated","Isolated",GREY,"--"),
                                ("adaptation_reference","Full information",BLUE,":")]):
        q=oxygen(data,name)
        half=q.shape[2]//2
        summary[name]={}
        for ax,sl,camp in zip(upper,[slice(None,half),slice(half,None)],["gas","mass"]):
            summary[name][camp]=curve(ax,q[:,:,sl].mean(axis=2),color,label,ls)
        for ax,key in zip(lower,["mse_obs","heldout_nll"]):
            runs=np.asarray(data[name][key])[:,-200:].mean(axis=(1,2))[:,None]
            mean,low,high=interval(runs)
            ax.errorbar(index,mean[0],yerr=[[mean[0]-low[0]],[high[0]-mean[0]]],
                        fmt="o",color=color,markersize=4,capsize=3,lw=1)
            summary[name][key]={"tail_updates":200,"mean":float(mean[0]),"interval":[float(low[0]),float(high[0])]}
    for ax,letter,label in zip(upper,["a","b"],["Gas observers","Mass observers"]):
        ax.set(ylabel=r"$q(\mathrm{oxygen})$",ylim=(-.02,1.02),yticks=[0,.5,1])
        ax.text(.60,1.07,label,transform=ax.transAxes,ha="center",va="bottom",fontsize=8)
        panel(ax,letter)
    lower[0].set(ylabel="Prediction MSE",ylim=(0,None));panel(lower[0],"c")
    lower[1].set(ylabel="Held-out NLL (nats)");panel(lower[1],"d")
    for ax in lower:
        ax.set(xticks=[0,1,2],xticklabels=["Network","Isolated","Full"],xlim=(-.45,2.45))
        grid(ax)
    for ax in upper:
        ax.axvline(protocol["shift"],color=GREY,lw=.8,ls=":")
        ax.set(xlim=(0,500),xticks=[0,100,200,300,400,500],xlabel="Simulation step")
        grid(ax)
    h,l=upper[0].get_legend_handles_labels()
    fig.legend(h,l,loc="lower center",bbox_to_anchor=(.54,.01),ncol=3)
    return fig,{"summary":summary,"caption":
                "Social sharing accelerates correction, while full information gives the best predictive density. "
                "At step 80 the world changes to the oxygen reference pattern. Agents branch from "
                "one shared pre-change state into networked selective observation, isolated selective "
                "observation, or a reference with both diagnostic channels available individually. "
                "(a,b) Oxygen probabilities for the gas- and mass-observing camps during the first "
                "500 of 1,500 updates; reference agents "
                "in both plotted groups observe both channels. Points in (c,d) average the last 200 "
                "updates within each seed; whiskers use the same whole-seed bootstrap as the ribbons. "
                "(c) Mean squared error of the posterior "
                "mixture mean on C,G,M against the noiseless world vector, averaged over coordinates "
                "and agents within each seed. Sensor noise variance is excluded from this error; "
                "it is not a held-out log-likelihood. (d) Negative log density of the full posterior "
                "predictive mixture on four independent withheld observations per agent and update, "
                "in nats per three-coordinate observation. Lower is better in (c,d); the NLL axis "
                "is expanded to show differences. The reference is not a centralized observer "
                "pooling all agents' observations. These data test adaptation to a prescribed mean "
                "pattern, not recovery of the true causal wiring. "+band_caption(len(q))}


def wall_controls_figure(data, protocol):
    conditions=protocol["wall_conditions"]
    labels={"plain_original":r"Gaussian, $\omega=1$", "gated_original":r"Gate, $\omega=.95$",
            "rivals_original":r"Candidates, $\omega=.3$", "plain_matched":r"Gaussian, $\omega=.9$",
            "gated_matched":r"Gate, $\omega=.9$", "rivals_matched":r"Candidates, $\omega=.9$",
            "plain_shared":"Gaussian, shared anchor", "gated_shared":"Gate, shared anchor",
            "rivals_no_forgetting":r"Candidates, $\omega=1$", "rivals_hypothesis_off":"Parameters only",
            "rivals_all_social_off":"No communication"}
    fig,axes=plt.subplots(1,2,figsize=(WIDTH,4.1),sharey=True)
    fig.subplots_adjust(left=.40,right=.97,bottom=.18,top=.92,wspace=.33)
    summary={}
    for row,condition in enumerate(conditions):
        key="wall_"+condition["name"]
        distance=wall_distance(data[key])[:,-1:]
        mean,low,high=interval(distance)
        axes[0].errorbar(mean[0],row,xerr=[[mean[0]-low[0]],[high[0]-mean[0]]],
                         fmt="o",color=GREY,markersize=3,capsize=2,lw=.8)
        summary[key]={"distance":{"mean":float(mean[0]),"interval":[float(low[0]),float(high[0])]}}
        if condition["K"]>1:
            q=oxygen(data,key);half=q.shape[2]//2
            for offset,sl,marker,color,camp in [(-.12,slice(None,half),"o",BLUE,"gas"),
                                               (.12,slice(half,None),"^",ORANGE,"mass")]:
                mean,low,high=interval(q[:,-1:,sl].mean(axis=2))
                axes[1].errorbar(mean[0],row+offset,xerr=[[mean[0]-low[0]],[high[0]-mean[0]]],
                                 fmt=marker,color=color,markersize=3,capsize=2,lw=.8)
                summary[key][camp]={"mean":float(mean[0]),"interval":[float(low[0]),float(high[0])]}
    axes[0].set(yticks=np.arange(len(conditions)),yticklabels=[labels[c["name"]] for c in conditions],
                ylim=(len(conditions)-.5,-.5),xlabel="Camp distance",ylabel="Condition",xlim=(0,None))
    axes[1].set(xlabel=r"Camp $q(\mathrm{oxygen})$",xlim=(-.05,1.05),xticks=[0,.5,1])
    for ax,letter in zip(axes,["a","b"]):
        panel(ax,letter)
        ax.grid(axis="x",color="#E1E1E1",lw=.4)
        for y in [2.5,5.5,7.5]: ax.axhline(y,color="#DDDDDD",lw=.5)
    from matplotlib.lines import Line2D
    handles=[Line2D([],[],marker="o",color=BLUE,lw=0,label="Gas observers"),
             Line2D([],[],marker="^",color=ORANGE,lw=0,label="Mass observers")]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.58,.01),ncol=2)
    return fig,{"summary":summary,"caption":
                "The full disagreement-control matrix reports parameter and candidate outcomes separately. "
                "Each row summarizes step 1499: (a) distance between camp-mean MAP observable commitments; "
                "(b) each camp's mean oxygen probability where candidates exist. Empty probability "
                "rows are single-Gaussian conditions, not missing runs. Memory is shown explicitly; "
                "the shared-anchor Gaussian controls use memory 0.9, while communication controls "
                "use candidate memory 0.3. Other Gaussian controls retain camp-specific anchors. "
                "Points average 20 seeds; bars are pointwise 95% percentile-bootstrap intervals "
                "(2,000 whole-seed resamples)."}


def persistence_figure(data, protocol):
    base=data["wall_rivals_original"]
    continuation=data["wall_rivals_continuation"]
    q=np.concatenate((base["q"],continuation["q"]),axis=1)[...,1]
    distance=np.concatenate((wall_distance(base),wall_distance(continuation)),axis=1)
    fig,axes=plt.subplots(3,1,figsize=(WIDTH,4.65))
    fig.subplots_adjust(left=.15,right=.97,bottom=.18,top=.94,hspace=.65)
    summary={"distance":curve(axes[0],distance,BLUE)}
    axes[0].set(ylabel="Camp distance",ylim=(0,None))
    half=q.shape[2]//2
    for sl,color,label,ls in [(slice(None,half),BLUE,"Gas observers","-"),
                              (slice(half,None),ORANGE,"Mass observers","--")]:
        summary[label]=curve(axes[1],q[:,:,sl].mean(axis=2),color,label,ls)
    axes[1].set(ylabel=r"Camp $q(\mathrm{oxygen})$",ylim=(-.02,1.02),yticks=[0,.5,1])
    for ax in axes[:2]:
        ax.axvline(protocol["wall_T"],color=GREY,lw=.8,ls=":")
        ax.set(xlim=(0,q.shape[1]-1),xticks=[0,1500,3000,4500],xlabel="Simulation step")
    off=data["wall_sources_off"]
    odds=np.asarray(off["L"])[...,1]-np.asarray(off["L"])[...,0]
    gap=odds[:,:,:half].mean(axis=2)-odds[:,:,half:].mean(axis=2)
    start=np.asarray(base["L"])[:,-1,:,1]-np.asarray(base["L"])[:,-1,:,0]
    start=start[:,:half].mean(axis=1)-start[:,half:].mean(axis=1)
    predicted=start[:,None]*(1-protocol["alpha_m"])**np.arange(1,gap.shape[1]+1)
    summary["source_off_gap"]=curve(axes[2],gap,BLUE,"Measured gap",x=np.arange(1,gap.shape[1]+1))
    axes[2].plot(np.arange(1,gap.shape[1]+1),predicted.mean(axis=0),color=GREY,lw=1.1,ls="--",label="Pooling prediction")
    summary["pooling_recurrence_max_error"]=float(np.abs(gap-predicted).max())
    axes[2].set(xlabel="Updates after evidence is removed",ylabel="Log-odds gap (nats)",xlim=(1,gap.shape[1]))
    for ax,letter in zip(axes,["a","b","c"]):panel(ax,letter);grid(ax)
    h,l=axes[1].get_legend_handles_labels();h2,l2=axes[2].get_legend_handles_labels()
    fig.legend(h+h2,l+l2,loc="lower center",bbox_to_anchor=(.54,.01),ncol=2)
    return fig,{"summary":summary,"caption":
                "A longer run separates a distance plateau from persistent competing model choices. "
                "The candidate condition with memory 0.3 continues to 5,000 updates without changing "
                "its mixed world or selective evidence; the dotted line marks the saved continuation "
                "boundary, not an intervention. (a) MAP commitment distance. (b) Candidate probabilities "
                "in the two camps. (c) A separate branch from update 1499 removes all incoming evidence "
                "and sets memory to 1. The signed camp log-odds gap follows the pure-pooling recurrence "
                "gap(t) = gap(0) times (1 - alpha_m)^t. This tests the source of disagreement, not "
                "adaptation to a new world. "+band_caption(len(q))}


def latex_escape(text):
    replacement = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                   "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacement.get(character, character) for character in text)


def build_review(figures):
    contents = []
    for number, entry in figures.items():
        contents.append(r"\section*{Four-variable experiment: figure " + str(number) + "}\n" +
                        r"\noindent\includegraphics[width=\linewidth]{" + Path(entry["outputs"]["pdf"]["path"]).name + "}\n\n" +
                        r"\small\noindent " + latex_escape(entry["caption"]) + "\n" + r"\clearpage")
    preamble = r"""\documentclass[10pt]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{graphicx}
\usepackage[a4paper,textwidth=4.8in,top=1in,bottom=1in]{geometry}
\usepackage[expansion=false]{microtype}
\pagestyle{plain}
\begin{document}
"""
    (HERE / "figure_review.tex").write_text(preamble+"\n".join(contents)+"\n\\end{document}\n")
    compile_tex("figure_review.tex")
    return record(HERE / "figure_review.pdf")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path)
    parser.add_argument("--only",choices=["model","all"],default="all")
    args=parser.parse_args()
    use_style()
    manifest={"script":record(__file__),"style":provenance(),
              "topology_renderer":record(HERE/"publication_topology.py"),
              "figures":{"1":model_figure()}}
    if args.only != "model":
        if args.run_dir is None:
            parser.error("--run-dir required for numerical figures")
        protocol=json.loads((args.run_dir/"protocol.json").read_text())
        data=Data(args.run_dir)
        with np.load(args.run_dir/"graphs.npz",allow_pickle=False) as arrays:
            graphs={key:arrays[key] for key in arrays.files}
        plotters=[("3","fig03_walls",lambda:wall_figure(data)),
                  ("4","fig04_cycle",lambda:cycle_figure(data,protocol)),
                  ("9","fig09_topology",lambda:topology_figure(data,graphs,protocol)),
                  ("S1","figS1_matched_memory",lambda:wall_figure(data,matched=True)),
                  ("S2","figS2_social_controls",lambda:social_controls_figure(data,graphs,protocol)),
                  ("S3","figS3_memory",lambda:memory_figure(data,protocol)),
                  ("S4","figS4_direct_evidence",lambda:direct_evidence_figure(data,protocol)),
                  ("S5","figS5_adaptation",lambda:adaptation_figure(data,protocol)),
                  ("S6","figS6_wall_controls",lambda:wall_controls_figure(data,protocol)),
                  ("S7","figS7_persistence",lambda:persistence_figure(data,protocol))]
        for number,name,plotter in plotters:
            fig,entry=plotter()
            entry["outputs"]=save(fig,HERE/name)
            entry["embedded_fonts"]=check_pdf(entry["outputs"]["pdf"]["path"])
            entry["layout_audit"]="clean"
            manifest["figures"][number]=entry
        manifest["inputs"]={"conditions":data.inputs,**{name:record(args.run_dir/name) for name in
                            ["protocol.json","graphs.npz","world.npz","world.json","summary.json",
                             "environment.json","accepted_engine_hashes.json","results_manifest.json"]}}
    manifest["bootstrap"]={"replicates":BOOTSTRAP_REPLICATES,"seed":BOOTSTRAP_SEED,
                            "resampling_unit":"whole seed-level trajectory","quantiles":[.025,.975],
                            "quantile_method":"linear","smoothing":"none","clipping":"none"}
    manifest["review"]=build_review(manifest["figures"])
    (HERE/"figure_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (HERE/"captions.md").write_text("\n\n".join(f"**Figure {n}.** {e['caption']}" for n,e in manifest["figures"].items())+"\n")


if __name__ == "__main__":
    main()
