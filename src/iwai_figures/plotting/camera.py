"""Exact camera-ready figure layouts; only their output/data paths are new."""
from pathlib import Path
import json
import subprocess
import numpy as np
import fitz
from scipy.special import xlogy
from .style import use_style, save, provenance, WIDTH, BLUE, ORANGE, GREY
from .orx_figstyle import PALETTE
import matplotlib.pyplot as plt
from .data import CameraData as Data
HERE = Path(__file__).resolve().parent
GREEN=PALETTE['green']

import hashlib
BOOTSTRAP_SEED=20260919
BOOTSTRAP_REPLICATES=2000
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


def oxygen(data, name):
    return np.asarray(data[name]["oxygen"])


def topology_figure(data, graphs, protocol):
    from .topology import figure9
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
    for name in protocol["network_graphs"]:
        restricted=np.setdiff1d(np.arange(protocol["network_N"]),view["vanguard_nodes"])
        means=view[f"q_{name}"][:,:,restricted].mean(axis=2)
        hits=means>protocol["threshold"]
        hits[:,:protocol["shift"]]=False
        event=hits.any(axis=1)
        times=hits.argmax(axis=1)[event]
        summary["graphs"][name]["group_crossings"]={
            "events":int(event.sum()),"total":int(len(event)),"censored":int((~event).sum()),
            "median_among_events":float(np.median(times)) if len(times) else None}
    cycle_events=summary["graphs"]["cycle"]["group_crossings"]
    return fig, {"summary": summary, "caption":
                "Contact topology changes adoption timing, with incomplete adoption on the cycle. "
                "All 24 nodes and edges are shown; black outlines mark four full-attention agents. "
                "After the world changes at step 80, the other 20 agents assign weight 0.01 to "
                "gas-uptake and mass-change observations; combustion remains fully attended. "
                "Thin lines show every restricted agent from seed 0; heavy lines average restricted "
                "agents within each seed and then average seeds. The graph definitions, informed "
                "positions and row-normalized self-plus-neighbour weights match the accepted "
                "experiment. Memory is 0.9 and model-score sharing 0.02. Lambda2 is the unnormalized "
                "graph Laplacian's second eigenvalue. Density and source placement vary alongside "
                "topology; this is not an isolated test of that eigenvalue. "
                f"On the cycle, {cycle_events['events']} of {cycle_events['total']} seeds reach a "
                f"restricted-group mean above {protocol['threshold']} within {protocol['network_T']} "
                "updates; noncrossing runs are right-censored, not assigned a crossing at the horizon. "
                + band_caption(len(protocol["network_seeds"]))}


def wall_distance(result):
    if "distance" in result:
        return result["distance"]
    mu = np.asarray(result["mu_map"])[..., 1:]
    half = mu.shape[2] // 2
    return np.linalg.norm(mu[:, :, :half].mean(axis=2)-mu[:, :, half:].mean(axis=2), axis=-1)


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
            runs=np.asarray(data[name][key+"_tail"])[:,None]
            mean,low,high=interval(runs)
            ax.errorbar(index,mean[0],yerr=[[mean[0]-low[0]],[high[0]-mean[0]]],
                        fmt="o",color=color,markersize=4,capsize=3,lw=1)
            summary[name][key]={"tail_updates":375,"mean":float(mean[0]),"interval":[float(low[0]),float(high[0])]}
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
                "Social sharing allows gas observers to adopt oxygen, while their isolated counterparts "
                "retain the release account over the observed horizon. "
                "At step 80 the world changes to the oxygen reference pattern. Agents branch from "
                "one shared pre-change state into networked selective observation, isolated selective "
                "observation, or a reference with both diagnostic channels available individually. "
                "(a,b) Oxygen probabilities for the gas- and mass-observing camps during the first "
                "500 of 1,500 updates; reference agents "
                "in both plotted groups observe both channels. Points in (c,d) average the last 375 "
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


def structural_figure(data):
    result = data["structural_check"]
    covariance = np.asarray(result["observable_covariance"])
    scale = np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1))
    correlation = covariance / (scale[:, :, None] * scale[:, None, :])
    fig = plt.figure(figsize=(WIDTH, 4.8))
    matrix_axes = [fig.add_axes([.13,.61,.28,.27]),
                   fig.add_axes([.53,.61,.28,.27])]
    color_ax = fig.add_axes([.87,.61,.025,.27])
    ax = fig.add_axes([.14,.23,.815,.27])
    for index, (mat_ax, letter, label) in enumerate(zip(matrix_axes, ["a","b"], ["Release","Oxygen"])):
        # Vector cells, not an imshow bitmap embedded in the publication PDF.
        edges=np.arange(4)-.5
        im = mat_ax.pcolormesh(edges,edges,correlation[index],vmin=-1,vmax=1,
                              cmap="RdBu_r",shading="flat",rasterized=False)
        mat_ax.set(xlim=(-.5,2.5),ylim=(2.5,-.5),aspect="equal")
        for (row,col), value in np.ndenumerate(correlation[index]):
            rgba = im.cmap(im.norm(value))
            luminance = .299*rgba[0]+.587*rgba[1]+.114*rgba[2]
            mat_ax.text(col,row,f"{value:.2f}",ha="center",va="center",fontsize=8,
                        color="black" if luminance>.6 else "white")
        mat_ax.set(xticks=[0,1,2],yticks=[0,1,2],xticklabels=["C","G","M"],
                   yticklabels=["C","G","M"],xlabel="Observable",ylabel="Observable")
        mat_ax.text(.65,1.11,label,transform=mat_ax.transAxes,ha="center",va="bottom",fontsize=8.5)
        panel(mat_ax,letter)
    colorbar=fig.colorbar(im,cax=color_ax,label="Correlation",ticks=[-1,0,1])
    colorbar.solids.set_rasterized(False)
    x=np.asarray(result["sample_counts"])
    q=np.asarray(result["q_at_sample_counts"])
    summary={"observable_correlation":correlation.tolist(),"accuracy":{}}
    for i,(sigma,color,ls) in enumerate(zip(result["sigmas"],
                                            [BLUE,PALETTE["green"],ORANGE,PALETTE["red"]],
                                            ["-","--","-.",":"])):
        selected=q[i]  # generating candidate, seed, prefix, candidate
        tied=np.isclose(selected,selected.max(axis=-1,keepdims=True),rtol=0,atol=1e-10)
        votes=tied/tied.sum(axis=-1,keepdims=True)
        correct=np.stack([votes[k,...,k] for k in range(2)]).mean(axis=0)
        np.testing.assert_allclose(correct.mean(axis=0),result["accuracy"][i],atol=1e-14)
        summary["accuracy"][str(float(sigma))]=curve(ax,correct,color,rf"$\sigma={sigma:g}$",ls,x=x)
        ax.plot(x,correct.mean(axis=0),color=color,marker="o",ls="none",markersize=3)
    control=np.asarray(result["product_marginals_accuracy"])
    np.testing.assert_allclose(control,np.broadcast_to(control[0],control.shape),atol=1e-14)
    ax.plot(x,control[0],color=GREY,ls="--",lw=1,label="Marginals only")
    ax.set(xscale="log",xlim=(1,500),ylim=(.45,1.025),yticks=[.5,.75,1],
           xticks=x,xticklabels=[str(value) for value in x],xlabel="Independent observations (log scale)",
           ylabel="Correct candidate fraction")
    ax.minorticks_off()
    panel(ax,"c");grid(ax)
    h,l=ax.get_legend_handles_labels()
    fig.legend(h,l,loc="lower center",bbox_to_anchor=(.55,.025),ncol=3)
    summary["bootstrap_unit"]="Seed index, retaining both independently generated source candidates (200 balanced source pairs)."
    return fig,{"summary":summary,"caption":
                "Dependence alone can distinguish the supplied candidates under a fixed known-model classifier. "
                "(a,b) Observable correlation matrices of the generating models before measurement noise. "
                "Both models have zero means and equal observable marginal variances; the release model's "
                "G residual variance is adjusted to 0.5448 for this check only. (c) Fraction choosing the "
                "generating candidate after the indicated numbers of independent full observations, for "
                "measurement-noise standard deviations sigma. Each candidate generates 200 independent "
                "sequences. Lines join measured sample counts; ribbons are pointwise 95% percentile-bootstrap "
                "intervals (2,000 resamples of seed indices, retaining both source candidates). A "
                "product-of-marginals classifier stays at 0.5; posterior ties count half-correct. "
                "The classifier has fixed known Gaussian parameters, equal model priors and no social "
                "pooling or parameter adaptation. This is a discrimination check for this two-model menu, "
                "not graph discovery or a structure-only ablation of the adaptive simulations. "
                "Both edge patterns and coefficient signs contribute to the differing dependence."}

def diagram(stem, caption):
    compile_tex(stem+'.tex')
    pdf=HERE/(stem+'.pdf')
    fonts=check_pdf(pdf)
    with fitz.open(pdf) as doc:
        page=doc[0]
        (HERE/(stem+'.svg')).write_text(page.get_svg_image(text_as_path=True))
        page.get_pixmap(matrix=fitz.Matrix(220/72,220/72)).save(HERE/(stem+'.png'))
    return {'caption':caption,'outputs':{ext:record(HERE/(stem+'.'+ext)) for ext in ['pdf','svg','png']},
            'embedded_fonts':fonts,'source':record(HERE/(stem+'.tex')),
            'layout_audit':'TikZ: vector, fonts, exact LNCS width and text bounds checked'}


def precision_figure(world):
    """Exact observable marginal precision, not blocks of the joint precision."""
    precision=np.linalg.inv(world['covariance'][:,1:,1:])
    values=[precision[0],precision[1],precision.mean(axis=0)]
    labels=[r'$\Pi_P^{\rm obs}$',r'$\Pi_O^{\rm obs}$',r'$\frac12(\Pi_P^{\rm obs}+\Pi_O^{\rm obs})$']
    lines=[]
    for k,(matrix,label) in enumerate(zip(values,labels)):
        suffix=str(k)
        placement='' if k==0 else f',right=29mm of c{k-1}'
        lines += [rf'\node[var{placement}] (c{suffix}) {{$C$}};',
                  rf'\node[var,below left=12mm and 6mm of c{suffix}] (g{suffix}) {{$G$}};',
                  rf'\node[var,below right=12mm and 6mm of c{suffix}] (m{suffix}) {{$M$}};',
                  rf'\node[above=3mm of c{suffix}] {{{label}}};']
        for i,j,u,v,location in [(0,1,'c','g','above left'),(0,2,'c','m','above right'),(1,2,'g','m','below')]:
            value=matrix[i,j]
            if abs(value)>1e-12:
                # Labels are entries of Pi, not regression signs or correlations.
                color='accent' if value<0 else 'orange'
                style='solid' if value<0 else 'dashed'
                lines += [rf'\draw[{color},{style},line width=.8pt] ({u}{suffix})--node[edge,{location}] {{{value:+.3f}}}({v}{suffix});']
        lines += [rf'\node[below=8mm of g{suffix},anchor=west] {{\textbf{{({chr(97+k)})}} '+['Release','Oxygen','Equal-weight pool'][k]+'};']
    preamble=r'''\documentclass[10pt]{article}
\usepackage[T1]{fontenc}
\usepackage[paperwidth=4.8in,paperheight=1.75in,margin=0pt]{geometry}
\usepackage{amsmath,tikz}\usetikzlibrary{positioning}
\definecolor{accent}{HTML}{0072B2}\definecolor{orange}{HTML}{D55E00}
\pagestyle{empty}\setlength\parindent{0pt}
\tikzset{every node/.style={font=\fontsize{8.5}{10}\selectfont},
var/.style={circle,draw=black!60,minimum size=6.5mm,inner sep=0pt},
edge/.style={font=\fontsize{7.5}{9}\selectfont,inner sep=1pt}}
\begin{document}\begin{minipage}[c][1.75in][c]{4.8in}\centering
\begin{tikzpicture}
'''
    (HERE/'fig02_precision_pool.tex').write_text(preamble+'\n'.join(lines)+'\n\\end{tikzpicture}\\end{minipage}\\end{document}\n')
    entry=diagram('fig02_precision_pool',
        'Equal-weight Gaussian pooling combines existing precision entries. '
        'The inputs are the exact observable marginal priors on C, G and M in the revised '
        'release and oxygen candidates; hidden P and O coordinates are integrated out before '
        'comparison. Undirected links are labelled by off-diagonal precision entries '
        '(blue solid: negative; orange dashed: positive), rounded to three decimals. '
        'Diagonal entries and information vectors are omitted from the drawing but included '
        'in the single-Gaussian comparator. The G--M entry present only in the oxygen input '
        'is halved; opposing C--M entries nearly cancel. In general, entries zero in both '
        'inputs remain zero. These are conditional-dependence links, not the directed '
        'causal arrows of the candidate graphs. This closed-form illustration is not a simulation.')
    entry['observable_precision']=precision.tolist();entry['pooled_precision']=values[-1].tolist()
    return entry


def camps(data):
    fig,ax=plt.subplots(figsize=(WIDTH,2.75))
    fig.subplots_adjust(left=.15,right=.975,bottom=.20,top=.96)
    summary={}
    for name,label,color,ls in [
        ('plain',r'Plain fusion ($\omega=1$)',PALETTE['red'],'-'),
        ('gated',r'Mean-disagreement gate ($\omega=.95$)',ORANGE,'--'),
        ('rivals',r'Represented rivals ($\omega=.3$)',GREEN,'-.')]:
        key='wall_'+name+'_original';d=wall_distance(data[key]);x=np.arange(1,d.shape[1]+1)
        if np.any(d<=0):raise ValueError('Log distance requires positive values; no clipping allowed')
        summary[key]=curve(ax,d,color,label,ls,x=x)
    ax.set(xscale='log',yscale='log',xlim=(1,1500),xlabel='Completed updates (log scale)',
           ylabel=r'Inter-camp distance $D_t$ (log scale)')
    ax.grid(axis='y',color='#E1E1E1',lw=.4);ax.set_axisbelow(True)
    ax.legend(loc='upper right',fontsize=7.5,frameon=False)
    return fig,{'summary':summary,'caption':
        'Different update recipes retain different parameter distances, without sustained opposing candidate choices. '
        'Two camps of eight agents observe G or M in the oxygen-reference world on a complete contact graph. '
        'D_t is the Euclidean distance between camp means on C,G,M; the rival condition uses each agent\'s '
        'highest-probability candidate. Both axes are logarithmic; all positive values are shown without clipping. '
        'Memory values are given in the legend; prior anchors also differ, so this is not an isolated test '
        'of representation. Both camps in the represented-rival condition adopt oxygen. '
        +band_caption(20)}


def cycle(data,protocol):
    q=oxygen(data,'cycle');qm=q.mean(axis=2);hm=entropy(q).mean(axis=2)
    fig,ax=plt.subplots(figsize=(WIDTH,2.9));fig.subplots_adjust(left=.14,right=.855,bottom=.19,top=.94)
    axh=ax.twinx();axh.spines['right'].set_visible(True)
    t=np.arange(q.shape[1]);mean,lo,hi=interval(hm)
    high_uncertainty=(t>=protocol['shift']) & (mean>.35)
    indices=np.flatnonzero(high_uncertainty)
    if not len(indices) or np.any(np.diff(indices)!=1):
        raise ValueError('Expected one contiguous post-change entropy interval')
    phase_start,phase_stop=int(indices[0]),int(indices[-1])
    axh.axvspan(phase_start,phase_stop+1,color=PALETTE['orange'],alpha=.13,lw=0)
    axh.fill_between(t,0,mean,color='#B5BEC3',alpha=.50,lw=0,zorder=0)
    axh.fill_between(t,lo,hi,color=GREY,alpha=.24,lw=0)
    axh.plot(t,mean,color=GREY,lw=.8,ls='--',label='Individual entropy')
    ax.set_zorder(axh.get_zorder()+1);ax.patch.set_visible(False)
    qs=curve(ax,qm,GREEN,'Oxygen probability')
    ax.axvline(protocol['shift'],color=GREY,ls=':',lw=.9)
    ax.grid(False);axh.grid(False)
    ax.set(xlim=(0,q.shape[1]-1),ylim=(0,1.02),xticks=[0,40,80,120,160,200,240],
           yticks=[0,.25,.5,.75,1],xlabel='Simulation step',ylabel=r'Mean $q(\mathrm{oxygen})$')
    axh.set(ylim=(0,np.log(2)*1.02),yticks=[0,.2,.4,np.log(2)],
            yticklabels=['0','0.2','0.4',r'$\log 2$'],ylabel='Mean individual entropy (nats)')
    ax.text(36,.52,'Release\nfavoured',ha='center',va='center',fontsize=8,color=GREY)
    ax.text((phase_start+phase_stop)/2,.91,'High uncertainty',ha='center',va='center',fontsize=8,color=GREY)
    ax.text(215,.52,'Oxygen\nfavoured',ha='center',va='center',fontsize=8,color=GREY)
    hs,ls=ax.get_legend_handles_labels();he,le=axh.get_legend_handles_labels()
    ax.legend(hs+he,ls+le,loc='upper left',fontsize=8,frameon=False)
    return fig,{'summary':{'oxygen':qs,'entropy_maximum':float(mean.max()),
        'entropy_band_bounds':[float(lo.min()),float(hi.max())],
        'display_entropy_threshold':.35,'post_change_display_interval':[phase_start,phase_stop]},'caption':
        'The represented-rival transition passes through uncertainty before oxygen gains support. '
        'Green shows population-mean oxygen probability (left axis); grey fill shows posterior entropy '
        'computed per agent before averaging (right axis, nats; maximum log 2). Narrow ribbons around '
        'both curves are pointwise 95% percentile-bootstrap intervals for the seed mean '
        '(50 seeds; 2,000 resamples of complete seed curves), not min--max ranges. '
        'The vertical dotted line marks the first changed-world update at step 80; C and G stay +1 '
        'while M changes from -1 to +1. Sixteen agents communicate on a complete graph, with '
        'memory .9, model-score sharing .02 and sensor noise standard deviation 1. '
        'The first point follows the first update. As in the original composition, the orange span '
        'marks post-change mean individual entropy above 0.35 nats, here steps '
        f'{phase_start}--{phase_stop}. This inherited display threshold is a descriptive guide, '
        'not a fitted or validated crisis boundary. The entropy axis ends at the binary maximum '
        'log 2, rather than the original 1.1 nats. No smoothing or clipping is applied. '
        'The main text reports the unchanged-world and memory controls.'}


def camp_choices(data):
    q=oxygen(data,'wall_rivals_original');half=q.shape[2]//2
    fig,axes=plt.subplots(1,2,figsize=(WIDTH,2.65),sharey=True)
    fig.subplots_adjust(left=.13,right=.98,bottom=.28,top=.89,wspace=.22)
    summary={}
    for sl,color,label,ls in [(slice(None,half),BLUE,'Gas observers','-'),
                            (slice(half,None),ORANGE,'Mass observers','--')]:
        for ax in axes:
            result=curve(ax,q[:,:,sl].mean(axis=2),color,label,ls)
        summary[label]=result
    for ax,letter,limit in zip(axes,['a','b'],[150,1499]):
        ax.set(xlim=(0,limit),ylim=(0,1.02),yticks=[0,.5,1],xlabel='Simulation step')
        ax.text(0,1.05,rf'\textbf{{({letter})}}',transform=ax.transAxes,va='bottom')
        ax.grid(axis='y',color='#E1E1E1',lw=.4)
    axes[0].set_ylabel(r'Camp mean $q(\mathrm{oxygen})$')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.54,.005),ncol=2)
    return fig,{'summary':summary,'caption':
        'Both selectively observing camps adopt oxygen in the fixed-world represented-rival comparison. '
        '(a) The first 150 updates resolve the transition; (b) the entire 1,500-update run shows '
        'that the late parameter distance in the companion distance plot is not sustained '
        'opposition between theories. Curves average candidate probabilities within the '
        'gas- or mass-observing camp. These are the exact same memory-.3 runs as the green '
        'distance curve, with a constant oxygen-reference world from the outset; there is '
        'no imposed change at step 80 in this experiment. '+band_caption(20)}
