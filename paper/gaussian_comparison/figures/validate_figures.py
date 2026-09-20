#!/usr/bin/env python3
"""Validate comparison vector figures and unchanged seven prior assets."""
import hashlib
import json
from pathlib import Path
import fitz
import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
OLD = STUDY.parent / 'communication_experiments' / 'figures'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    m = json.loads((HERE / 'figure_manifest.json').read_text())
    assert sha(STUDY / 'analysis/plot_data.npz') == m['plot_data_sha256']
    assert sha(HERE / 'plot_figures.py') == m['plot_script_sha256']
    assert sha(STUDY / 'protocol.json') == m['protocol_sha256']
    assert sha(HERE / 'orx_figstyle.py') == m['style']['vendor_sha256']
    assert sha(HERE / 'publication_style.py') == m['style']['adapter_sha256']
    assert sha(HERE / 'captions.json') == m['captions_sha256']
    assert sha(HERE / 'figure_pack.pdf') == m['pack_sha256']
    assert sha(HERE / 'curve_statistics.csv') == m['curve_statistics_sha256']
    fonts = {}
    for stem, outputs in m['figures'].items():
        for ext, item in outputs.items():
            assert sha(HERE / (stem + '.' + ext)) == item['sha256']
        with fitz.open(HERE / (stem + '.pdf')) as doc:
            assert len(doc) == 1
            p = doc[0]
            assert abs(p.rect.width - 4.8 * 72) < .01
            assert not p.get_images()
            fonts[stem] = sorted({f[3].split('+')[-1] for f in p.get_fonts()})
            assert fonts[stem] and all(f.upper().startswith(('CM', 'SF')) for f in fonts[stem]), fonts[stem]
            assert all(p.rect.contains(fitz.Rect(b[:4])) for b in p.get_text('blocks')), stem
    for stem, record in m['unchanged_prior_figures'].items():
        assert sha(OLD / (stem + '.pdf')) == record['pdf']
    with np.load(STUDY / 'analysis/plot_data.npz', allow_pickle=False) as a:
        count = 0
        for key in a.files:
            if key.endswith(('::q_A', '::q_B', '::cross_A', '::cross_B')):
                assert np.isfinite(a[key]).all() and np.min(a[key]) >= 0 and np.max(a[key]) <= 1, key
                count += 1
            if key.endswith(('::mixture_distance', '::map_distance')):
                assert np.isfinite(a[key]).all() and np.min(a[key]) >= 0, key
            if key.startswith('G_'):
                assert '::q_' not in key, 'K=1 must not acquire a fake theory probability'
    with fitz.open(HERE / 'figure_pack.pdf') as doc:
        assert len(doc) == m['pack_pages'], len(doc)
        assert all(all(p.rect.contains(fitz.Rect(b[:4])) for b in p.get_text('blocks')) for p in doc)
    report = {'status': 'passed', 'new_figures': len(fonts), 'prior_figures_unchanged': len(m['unchanged_prior_figures']),
              'pack_pages': m['pack_pages'], 'width_inches': 4.8, 'fonts': fonts,
              'bounded_probability_or_influence_arrays': count, 'vector': 'no raster images in plot PDFs',
              'plotting_style': 'orx with real LaTeX Computer Modern', 'uncertainty': m['bootstrap']}
    (HERE / 'figure_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
