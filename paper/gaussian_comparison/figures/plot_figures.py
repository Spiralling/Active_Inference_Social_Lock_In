#!/usr/bin/env python3
"""Two matched-comparison figures and a revised nine-page publication pack.

All curves use every completed update; uncertainty resamples entire paired
seeds. Existing seven figure assets are embedded without regenerating them.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
OLD_FIGURES = STUDY.parent / 'communication_experiments' / 'figures'
os.environ['PATH'] = '/Library/TeX/texbin:' + os.environ.get('PATH', '')
import publication_style as style
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ORANGE, RED = style.ORANGE, style.orx.PALETTE['red']
GREEN, BLUE = style.orx.PALETTE['green'], style.BLUE
GREY, PURPLE = style.GREY, style.orx.PALETTE['purple']
MODELS = [('G_plain_info', ORANGE, '-', 'Gaussian'),
          ('G_gated_info', RED, '-.', 'Gated Gaussian'),
          ('C10', GREEN, '-', 'Candidates: private weights'),
          ('C11', BLUE, '--', 'Candidates: shared weights')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Data:
    def __init__(self, root):
        self.root = root
        self.a = np.load(root / 'plot_data.npz', allow_pickle=False)
        self.summary = json.loads((root / 'summary.json').read_text())
        idx = self.a['bootstrap_indices']
        self.weights = np.stack([np.bincount(row, minlength=idx.shape[1]) for row in idx]) / idx.shape[1]
        self.cache = {}
        self.curve_rows = []

    def ci(self, x):
        x = np.asarray(x, float)
        if x.ndim == 1:
            b = self.weights @ x
            lo, hi = np.quantile(b, [.025, .975])
        else:
            # Bound memory while preserving all states and identical bootstrap draws.
            bands = []
            for start in range(0, x.shape[1], 500):
                b = self.weights @ x[:, start:start + 500]
                bands.append(np.quantile(b, [.025, .975], axis=0))
            lo, hi = np.concatenate(bands, axis=1)
        return x.mean(0), lo, hi

    def curve(self, ax, name, metric, color, ls='-', lw=1.05):
        key = (name, metric)
        if key not in self.cache:
            self.cache[key] = self.ci(self.a[name + '::' + metric])
            for state, mean, lo, hi in zip(self.a['states'], *self.cache[key]):
                self.curve_rows.append({'condition': name, 'metric': metric, 'completed_state': int(state),
                                        'mean': float(mean), 'lo': float(lo), 'hi': float(hi)})
        mean, lo, hi = self.cache[key]
        x = self.a['states']
        ax.fill_between(x, lo, hi, color=color, alpha=.14, linewidth=0)
        return ax.plot(x, mean, color=color, ls=ls, lw=lw)[0]

    def metric(self, name, key, horizon=5000):
        return self.summary[name]['windows'][str(horizon)]['metrics'][key]


def panel(ax, letter):
    ax.text(0, 1.045, r'\textbf{' + letter + '}', transform=ax.transAxes, ha='left', va='bottom')


def time_axis(ax, log=False):
    if log:
        ax.set_xscale('log')
        ax.set_xlim(1, 5000)
        ax.set_xticks([1, 100, 5000], ['1', '100', '5,000'])
        ax.minorticks_off()
        ax.set_xlabel('Completed updates (log scale)')
    else:
        ax.set_xlim(0, 5000)
        ax.set_xticks([0, 2500, 5000], ['0', '2,500', '5,000'])
        ax.set_xlabel('Completed updates')


def points(ax, x, metric, color, marker='o', hollow=False):
    if metric['lo'] is not None and metric['hi'] is not None:
        ax.plot([x, x], [metric['lo'], metric['hi']], color=color, lw=1.1)
    ax.plot(x, metric['mean'], marker, color=color, ms=3.5,
            markerfacecolor='white' if hollow else color, markeredgewidth=.9)


def comparison(data, records):
    fig = plt.figure(figsize=(4.8, 4.9))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.12, 1], left=.13, right=.97,
                            top=.94, bottom=.30, hspace=.69, wspace=.43)
    top = fig.add_subplot(grid[0, :])
    prob = fig.add_subplot(grid[1, 0])
    loss = fig.add_subplot(grid[1, 1])
    for name, color, ls, label in MODELS:
        data.curve(top, name, 'mixture_distance', color, ls)
    top.set_ylabel('Camp predictive distance')
    top.set_ylim(bottom=0)
    time_axis(top, log=True)
    panel(top, 'a')
    for name, color in [('C10', GREEN), ('C11', BLUE)]:
        data.curve(prob, name, 'q_A', color, '-')
        data.curve(prob, name, 'q_B', color, ':')
    time_axis(prob)
    prob.set_ylim(0, 1)
    prob.set_yticks([0, .5, 1])
    prob.set_ylabel(r'Oxygen probability $q(O)$')
    panel(prob, 'b')
    names = ['G_plain_info', 'G_gated_info', 'C10', 'C11', 'C10_gasvar_matched']
    for x, name, color in zip(range(5), names, [ORANGE, RED, GREEN, BLUE, GREY]):
        for group, offset, marker in [('A', -.10, 'o'), ('B', .10, 's')]:
            points(loss, x + offset, data.metric(name, 'mse_' + group + '_tail'), color, marker, group == 'B')
    loss.set_xticks(range(5), ['Plain', 'Gate', 'Priv.', 'Share', 'Var.'], rotation=45, ha='right', fontsize=7.5)
    loss.set_xlim(-.4, 4.4)
    loss.set_ylim(bottom=-.05)
    loss.set_yticks([0, .5, 1])
    loss.set_ylabel('Full-signal MSE')
    panel(loss, 'c')
    fig.legend([Line2D([], [], color=c, ls=ls) for _, c, ls, _ in MODELS],
               [label for _, _, _, label in MODELS], ncol=2, loc='lower center',
               bbox_to_anchor=(.54, .105), columnspacing=1.2, handlelength=2.3)
    fig.legend([Line2D([], [], color=GREY, ls='-', marker='o', markersize=3.5),
                Line2D([], [], color=GREY, ls=':', marker='s', markersize=3.5, markerfacecolor='white')],
               ['Gas observers (b,c)', 'Mass observers (b,c)'], ncol=2,
               loc='lower center', bbox_to_anchor=(.53, .035), columnspacing=1.5)
    records['fig_matched_camp_models'] = style.save(fig, HERE / 'fig_matched_camp_models')
    print('Rendered fig_matched_camp_models', flush=True)


def diagnostics(data, records):
    fig, axes = plt.subplots(2, 2, figsize=(4.8, 5.8))
    fig.subplots_adjust(left=.13, right=.97, bottom=.27, top=.94, wspace=.47, hspace=.68)
    a, b, c, d = axes.flat
    for prefix, color in [('G_plain', ORANGE), ('G_gated', RED)]:
        for variant, ls in [('info', '-'), ('moment', '--')]:
            data.curve(a, prefix + '_' + variant, 'mixture_distance', color, ls)
    a.set_ylabel('Camp predictive distance')
    a.set_ylim(bottom=0)
    time_axis(a, log=True)
    for variant, color in [('info', PURPLE), ('moment', GREY)]:
        for group, ls in [('A', '-'), ('B', ':')]:
            data.curve(b, 'G_gated_' + variant, 'cross_' + group, color, ls)
    b.axhline(.05, color=GREY, lw=.7, ls='--')
    b.set_ylabel('Cross-group pooling weight')
    cross_max = max(np.max(data.a['G_gated_' + variant + '::cross_' + group]) for variant in ['info', 'moment'] for group in 'AB')
    b.set_ylim(0, 1.08 * max(.05, cross_max))
    time_axis(b, log=True)
    for name, color in [('C10', GREEN), ('C11', BLUE)]:
        for metric, ls in [('mixture_distance', '-'), ('map_distance', '--')]:
            data.curve(c, name, metric, color, ls)
    c.set_ylabel('Camp predictive distance')
    c.set_ylim(bottom=0)
    time_axis(c)
    for x, name in enumerate(['G_plain_info', 'G_gated_info', 'C10', 'C11']):
        for key, offset, marker, color in [
            ('mixture_distance_tail', -.18, 'o', GREY),
            ('mixture_distance_of_time_means_tail', 0, 's', BLUE),
            ('mixture_norm_of_across_seed_time_mean_difference', .18, '^', ORANGE)]:
            points(d, x + offset, data.metric(name, key), color, marker, marker == '^')
    d.set_xticks(range(4), ['Plain', 'Gate', 'Priv.', 'Share'], rotation=45, ha='right', fontsize=7.5)
    d.set_xlim(-.4, 3.4)
    d.set_ylim(bottom=-.055)
    d.set_yticks([0, .5, 1, 1.5, 2])
    d.set_ylabel('Late-window distance')
    for ax, letter in zip(axes.flat, 'abcd'):
        panel(ax, letter)
    # One figure legend with explicit panel scope keeps overloaded line styles honest.
    handles = [Line2D([], [], color=ORANGE), Line2D([], [], color=RED),
               Line2D([], [], color=PURPLE), Line2D([], [], color=GREY),
               Line2D([], [], color=GREEN), Line2D([], [], color=BLUE),
               Line2D([], [], color=GREY, marker='o', ls=''),
               Line2D([], [], color=BLUE, marker='s', ls=''),
               Line2D([], [], color=ORANGE, marker='^', ls='', markerfacecolor='white')]
    labels = ['Gaussian (a)', 'Gated Gaussian (a)', 'Info. reference (b)', 'Moment reference (b)',
              'Private weights (c)', 'Shared weights (c)', 'Instantaneous norm (d)', 'Time-mean norm (d)', 'Ensemble-mean norm (d)']
    fig.legend(handles, labels, ncol=2, loc='lower center', bbox_to_anchor=(.53, .015),
               columnspacing=1.3, handlelength=1.8, fontsize=7.5)
    records['fig_matched_camp_diagnostics'] = style.save(fig, HERE / 'fig_matched_camp_diagnostics')
    print('Rendered fig_matched_camp_diagnostics', flush=True)


def captions():
    common = (r'Means and pointwise 95\% percentile intervals use 2,000 whole-seed bootstrap resamples of all 50 paired seeds (10,000--10,049). '
              r'The candidate trajectories reuse the previous runs with exactly the same observation streams: this is a paired extension, not an independent replication. '
              r'Every displayed update is retained without smoothing. All arms have 16 agents, fixed gas/mass observation groups, '
              r'$\beta=.05$, $\omega=.3$, $\sigma_o=.5$, and the constant full signal $(C,G,M)=(1,1,1)$. ')
    return {
        'fig_matched_camp_models':
            r'\textbf{Numerical disagreement persists with a single Gaussian.} '
            r'(a) Distance $\|\langle\mu\rangle_{\mathrm{gas}}-\langle\mu\rangle_{\mathrm{mass}}\|_2$ in the common observable coordinates $(C,G,M)$. '
            r'The horizontal axis in (a) is logarithmic; the distance axis includes zero. The two Gaussian arms share one information-pooled reference; the gated arm reweights the same social matrix using entering-mean disagreement ($\nu=.01$). '
            r'Candidate predictions use the posterior mixture mean. Model weights are private in green and shared with $\alpha_m=.02$ in blue; conditional sharing is enabled throughout. '
            r'(b) Gas (solid) and mass (dotted) group-mean oxygen probabilities for the two candidate arms; the single Gaussian has no corresponding theory probability. '
            r'(c) Full-signal predictive MSE averaged over states 3,751--5,000: gas circles and open mass squares. '
            r'Plain and Gate denote Gaussian and gated Gaussian; Priv. and Share denote private and shared candidate weights. Var. is the candidate/private arm with only the release gas residual variance changed from .5 to .5448. '
            r'Distinct single-Gaussian predictions under sustained selective observations are not evidence of distinct theoretical commitments. ' + common,
        'fig_matched_camp_diagnostics':
            r'\textbf{Reference compression and readout choices affect the comparison.} '
            r'(a) Gaussian (orange) and gated Gaussian (red) distances: information-pool (solid) or equal-mixture moment-matched (dashed) references, including between-candidate mean covariance. '
            r'Panels (a,b) use logarithmic time. '
            r'(b) Realized cross-group weight before each observation: information-pool reference (purple), moment reference (grey); gas (solid), mass (dotted). '
            r'Dashed horizontal: ungated weight .05. '
            r'(c) Candidate/private (green) and candidate/shared (blue) distances using mixture means (solid) and selected-candidate/MAP means (dashed). '
            r'(d) Three different quantities over states 3,751--5,000: the mean instantaneous norm (grey circle), '
            r'the seed-average norm after time-averaging the signed camp difference (blue square), and the norm after both time and seed averaging (open orange triangle). '
            r'Plain, Gate, Priv. and Share denote Gaussian, gated Gaussian, candidates/private and candidates/shared, respectively. The last two distance definitions distinguish time-averaged directional differences from fluctuations. The orange triangle is a descriptive point without an interval: a norm alone cannot establish a nonzero ensemble mean. '
            r'All hidden candidate-specific coordinates are excluded. ' + common,
    }


def build_pack(caps, records):
    oldcaps = json.loads((OLD_FIGURES / 'captions.json').read_text())
    oldmanifest = json.loads((OLD_FIGURES / 'figure_manifest.json').read_text())
    entries = [(stem, '../../communication_experiments/figures/' + stem + '.pdf', oldcaps[stem]) for stem in oldmanifest['figures']]
    entries += [(stem, stem + '.pdf', caps[stem]) for stem in records]
    body = []
    for stem, path, caption in entries:
        body.append(r'\begin{center}\includegraphics[width=4.8in]{' + path + r'}\end{center}' + '\n' + caption + r'\clearpage')
    tex = (r'\documentclass[10pt]{article}' + '\n' +
           r'\usepackage[T1]{fontenc}\usepackage{graphicx}\usepackage{amsmath}\usepackage{textcomp}' + '\n' +
           r'\usepackage[paperwidth=6.2in,paperheight=10.8in,left=.7in,right=.7in,top=.5in,bottom=.5in]{geometry}' + '\n' +
           r'\pagestyle{empty}\setlength{\parindent}{0pt}\begin{document}' + '\n' + '\n'.join(body) + '\n' + r'\end{document}' + '\n')
    (HERE / 'figure_pack.tex').write_text(tex)
    result = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'figure_pack.tex'], cwd=HERE, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout[-6000:] + result.stderr)
    return {stem: {'pdf': sha(OLD_FIGURES / (stem + '.pdf')), 'caption': oldcaps[stem]} for stem in oldmanifest['figures']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, default=STUDY / 'analysis')
    args = parser.parse_args()
    style.use_style()
    data = Data(args.analysis)
    records = {}
    comparison(data, records)
    diagnostics(data, records)
    caps = captions()
    (HERE / 'captions.json').write_text(json.dumps(caps, indent=2) + '\n')
    (HERE / 'captions.md').write_text('# Matched camp comparison captions\n\n' + '\n\n'.join('## ' + name + '\n\n' + value for name, value in caps.items()) + '\n')
    (HERE / 'integration_blocks.tex').write_text('\n\n'.join(
        r'\begin{figure}[t]\centering' + '\n' + r'\includegraphics[width=\textwidth]{figures/' + name + r'.pdf}' + '\n' +
        r'\caption{' + caps[name] + r'}\label{fig:' + name + r'}\end{figure}' for name in records) + '\n')
    old_records = build_pack(caps, records)
    with (HERE / 'curve_statistics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['condition', 'metric', 'completed_state', 'mean', 'lo', 'hi'])
        writer.writeheader()
        writer.writerows(data.curve_rows)
    manifest = {'style': style.provenance(), 'plot_script_sha256': sha(__file__),
                'plot_data_sha256': sha(args.analysis / 'plot_data.npz'), 'protocol_sha256': sha(STUDY / 'protocol.json'),
                'figures': records, 'captions_sha256': sha(HERE / 'captions.json'), 'pack_sha256': sha(HERE / 'figure_pack.pdf'),
                'pack_pages': 9, 'unchanged_prior_figures': old_records,
                'curve_statistics_sha256': sha(HERE / 'curve_statistics.csv'),
                'bootstrap': '2000 shared whole-seed resamples, seed20260920, pointwise95percentile; all50pairedseeds',
                'curve_display': 'all5000updates,no smoothing', 'new_figure_count': len(records)}
    (HERE / 'figure_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Built revised nine-page figure pack', flush=True)


if __name__ == '__main__':
    main()
