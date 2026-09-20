#!/usr/bin/env python3
"""Paired whole-seed reductions, independent of simulator/plotting imports.

Public times are completed updates: raw round t is state t+1. The primary
camp distance is the norm of the difference of camp-averaged observable
mixture means. It is not a norm in candidate-specific hidden coordinates.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
NEW = ['G_plain_info', 'G_gated_info', 'G_plain_moment', 'G_gated_moment']
OLD = ['C10', 'C11', 'C10_gasvar_matched']
NAMES = NEW + OLD
VECTORS = ['mixture_difference', 'map_difference']
SCALARS = ['mixture_distance', 'map_distance', 'mse_A', 'mse_B', 'nll_A', 'nll_B', 'q_A', 'q_B', 'cross_A', 'cross_B']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_indices(n, replicates, seed):
    return np.random.default_rng(seed).integers(0, n, size=(replicates, n))


def estimate(x, idx):
    x = np.asarray(x, float)
    if x.ndim != 1 or len(x) != idx.shape[1] or not np.isfinite(x).all():
        raise ValueError('Invalid complete-seed bootstrap input')
    lo, hi = np.quantile(x[idx].mean(1), [.025, .975])
    return {'mean': float(x.mean()), 'lo': float(lo), 'hi': float(hi), 'n': len(x)}


def vector_norm_estimate(x, idx):
    """Norm after across-seed averaging, distinct from average of seed norms."""
    x = np.asarray(x, float)
    assert x.shape == (idx.shape[1], 3) and np.isfinite(x).all()
    # A percentile interval for this nonlinear norm is nonregular at zero and
    # can exclude zero even under a zero-vector null. Report only the descriptive
    # point; the signed-coordinate means have ordinary whole-seed intervals.
    return {'mean': float(np.linalg.norm(x.mean(0))), 'lo': None, 'hi': None, 'n': len(x),
            'interval': 'none: descriptive nonlinear aggregate; do not infer a nonzero directional gap',
            'estimand': 'norm of the across-seed mean of within-seed time-averaged signed camp differences'}


def load_condition(raw, name, records, expected_seeds):
    meta = json.loads((raw / name / 'setup.json').read_text())
    seeds = meta['seeds']
    if seeds != expected_seeds:
        raise ValueError('Seed identities/order changed: ' + name)
    lookup = {seed: i for i, seed in enumerate(seeds)}
    S, T, N = len(seeds), meta['T'], meta['N']
    if (S, T, N) != (50, 5000, 16):
        raise ValueError('Unexpected frozen dimensions: ' + name)
    groups = [np.asarray(meta['group_indices'][g], int) for g in ['gas', 'mass']]
    if [g.tolist() for g in groups] != [list(range(8)), list(range(8, 16))]:
        raise ValueError('Observation groups changed')
    values = {k: np.full((S, T), np.nan) for k in SCALARS}
    values.update({k: np.full((S, T, 3), np.nan) for k in VECTORS})
    coverage = np.zeros((S, T), np.uint8)
    files = []
    for row in sorted(records, key=lambda r: (r['start_step'], r['seeds'])):
        path = raw / row['file']
        if sha(path) != row['sha256']:
            raise ValueError('Raw hash mismatch: ' + str(path))
        si = np.array([lookup[s] for s in row['seeds']])
        sl = slice(row['start_step'], row['start_step'] + row['steps'])
        with np.load(path, allow_pickle=False) as a:
            for source, metric in [('mu_mixture', 'mixture'), ('mu_map', 'map')]:
                means = a[source]
                if means.shape != (len(si), row['steps'], N, 3):
                    raise ValueError('Unexpected observable mean shape')
                difference = means[:, :, groups[0]].mean(2) - means[:, :, groups[1]].mean(2)
                values[metric + '_difference'][si, sl] = difference
                values[metric + '_distance'][si, sl] = np.linalg.norm(difference, axis=-1)
            if name in OLD:
                q = a['q']
                if q.shape[-1] != 2 or not np.isfinite(q).all() or np.any(q < 0) or np.any(q > 1) or not np.allclose(q.sum(-1), 1, atol=2e-14, rtol=0):
                    raise ValueError('Invalid candidate probabilities')
            for letter, group in zip('AB', groups):
                for source, metric in [('mse_full', 'mse'), ('nll_full', 'nll')]:
                    values[metric + '_' + letter][si, sl] = a[source][:, :, group].mean(2)
                # K=1 has no probability of the oxygen candidate. Do not invent it.
                if name in OLD:
                    values['q_' + letter][si, sl] = q[:, :, group, 1].mean(2)
                    # For these no-gate arms, the recorded row-stochastic W is the
                    # actual conditional pooling operator. There is no alpha factor.
                    W = np.asarray(meta['W'], float)
                    other = groups[1] if letter == 'A' else groups[0]
                    values['cross_' + letter][si, sl] = W[np.ix_(group, other)].sum(1).mean()
                else:
                    values['cross_' + letter][si, sl] = a['gate_crossweight'][:, :, group].mean(2)
        coverage[si, sl] += 1
        files.append({'file': row['file'], 'sha256': row['sha256'], 'seeds': row['seeds'],
                      'start_step': row['start_step'], 'steps': row['steps']})
    if not np.all(coverage == 1):
        raise ValueError('Missing or duplicated raw state: ' + name)
    if name in NEW:
        for key in ['q_A', 'q_B']:
            del values[key]
    if any(not np.isfinite(v).all() for v in values.values()):
        raise ValueError('Nonfinite derived metric: ' + name)
    for letter in 'AB':
        assert np.min(values['cross_' + letter]) >= 0 and np.max(values['cross_' + letter]) <= 1
    return meta, values, files


def seed_window_metrics(values, start, end):
    """Inclusive public window [start,end], all raw states, no curve thinning."""
    sl = slice(start - 1, end)
    result = {}
    for key, x in values.items():
        if x.ndim == 2:
            result[key + '_tail'] = x[:, sl].mean(1)
            result[key + '_endpoint'] = x[:, end - 1]
        else:
            tail = x[:, sl].mean(1)
            for d, coordinate in enumerate(['C', 'G', 'M']):
                result[key + '_' + coordinate + '_tail'] = tail[:, d]
                result[key + '_' + coordinate + '_endpoint'] = x[:, end - 1, d]
            result[key.replace('_difference', '') + '_distance_of_time_means_tail'] = np.linalg.norm(tail, axis=-1)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--candidate-raw', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=HERE / 'analysis')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((HERE / 'protocol.json').read_text())
    idx = bootstrap_indices(len(protocol['seeds']), protocol['analysis']['bootstrap_replicates'], protocol['analysis']['bootstrap_seed'])
    manifests = {key: json.loads((path / 'manifest.json').read_text()) for key, path in [('new', args.raw), ('old', args.candidate_raw)]}
    if set(r['condition'] for r in manifests['new']) != set(NEW):
        raise ValueError('New manifest differs from frozen four conditions')
    summary, seed_rows, plot, raw_records, condition_metrics = {}, [], {'bootstrap_indices': idx, 'states': np.arange(1, 5001)}, {}, {}
    for name in NAMES:
        root, manifest = (args.raw, manifests['new']) if name in NEW else (args.candidate_raw, manifests['old'])
        meta, values, files = load_condition(root, name, [r for r in manifest if r['condition'] == name], protocol['seeds'])
        raw_records[name] = {'setup_sha256': sha(root / name / 'setup.json'), 'files': files}
        summary[name] = {'seeds': meta['seeds'], 'settings': {k: meta.get(k) for k in ['K', 'N', 'T', 'omega', 'beta', 'sigma_o', 'gate_nu', 'reference_variant']}, 'windows': {}}
        condition_metrics[name] = {}
        # Full, unthinned per-seed scientific inputs for all trajectory figures.
        # Performance trajectories are reduced only into reported per-seed windows.
        for key, value in values.items():
            if not key.startswith(('mse_', 'nll_')):
                plot[name + '::' + key] = value
        for horizon, window in protocol['analysis']['windows'].items():
            start, end = window
            metrics = seed_window_metrics(values, start, end)
            condition_metrics[name][horizon] = metrics
            window_summary = {'first_state': start, 'last_state': end, 'n_states': end - start + 1,
                              'metrics': {key: estimate(x, idx) for key, x in metrics.items()}}
            for kind in ['mixture', 'map']:
                difference = values[kind + '_difference'][:, start - 1:end].mean(1)
                window_summary['metrics'][kind + '_norm_of_across_seed_time_mean_difference'] = vector_norm_estimate(difference, idx)
            summary[name]['windows'][horizon] = window_summary
            for s, seed in enumerate(meta['seeds']):
                seed_rows.append({'condition': name, 'seed': seed, 'horizon': int(horizon),
                                  'window_first_state': start, 'window_last_state': end,
                                  **{key: float(value[s]) for key, value in metrics.items()}})
            for key, value in metrics.items():
                plot[name + '::' + key + '_' + horizon] = value
        print('Reduced', name, 'all 50 seeds and 5,000 states', flush=True)
    contrast_spec = [
        ('gated_minus_plain_info', 'G_gated_info', 'G_plain_info'),
        ('gated_minus_plain_moment', 'G_gated_moment', 'G_plain_moment'),
        ('moment_minus_info_plain', 'G_plain_moment', 'G_plain_info'),
        ('moment_minus_info_gated', 'G_gated_moment', 'G_gated_info'),
        ('shared_minus_private_candidates', 'C11', 'C10'),
        ('gasvar_matched_minus_original_candidates', 'C10_gasvar_matched', 'C10')]
    contrasts, contrast_rows = {}, []
    for cname, plus, minus in contrast_spec:
        contrasts[cname] = {'plus': plus, 'minus': minus, 'windows': {}}
        for horizon in protocol['analysis']['windows']:
            mplus, mminus = condition_metrics[plus][horizon], condition_metrics[minus][horizon]
            metrics = {key: mplus[key] - mminus[key] for key in mplus.keys() & mminus.keys()}
            contrasts[cname]['windows'][horizon] = {key: estimate(value, idx) for key, value in metrics.items()}
            for s, seed in enumerate(protocol['seeds']):
                contrast_rows.append({'contrast': cname, 'seed': seed, 'horizon': int(horizon),
                                      **{key: float(value[s]) for key, value in sorted(metrics.items())}})
    write_csv(args.out / 'seed_metrics.csv', seed_rows)
    write_csv(args.out / 'paired_seed_differences.csv', contrast_rows)
    dump(args.out / 'summary.json', summary)
    dump(args.out / 'paired_contrasts.json', contrasts)
    np.savez_compressed(args.out / 'plot_data.npz', **plot)
    dump(args.out / 'analysis_manifest.json', {
        'protocol_sha256': sha(HERE / 'protocol.json'), 'analysis_script_sha256': sha(__file__),
        'new_manifest_sha256': sha(args.raw / 'manifest.json'), 'candidate_manifest_sha256': sha(args.candidate_raw / 'manifest.json'),
        'source_records': raw_records,
        'output_sha256': {p.name: sha(p) for p in sorted(args.out.iterdir()) if p.name != 'analysis_manifest.json' and p.is_file()},
        'bootstrap': {'n': 50, 'replicates': 2000, 'seed': 20260920, 'unit': 'whole seed', 'paired': True, 'coverage': 'pointwise95percentile'},
        'curve_input': 'all 5,000 completed states; no smoothing or outcome-based filtering',
        'windows': protocol['analysis']['windows'],
        'data_relationship': 'Gaussian extension uses existing paired streams. Candidate outcomes reused, not newly independently confirmed.',
        'distance_definitions': {'mean_instantaneous_norm': 'E_seed E_time ||mean_gas(mu)-mean_mass(mu)||',
                                 'mean_distance_of_time_means': 'E_seed ||E_time(mean_gas(mu)-mean_mass(mu))||',
                                 'norm_of_across_seed_time_mean_difference': '||E_seed E_time(mean_gas(mu)-mean_mass(mu))||'},
        'undefined_K1_oxygen_probability': 'omitted, not set to zero or one',
    })
    print('Wrote paired analysis to', args.out, flush=True)


if __name__ == '__main__':
    main()
