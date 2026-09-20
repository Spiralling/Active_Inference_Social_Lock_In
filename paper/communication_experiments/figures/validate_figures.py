#!/usr/bin/env python3
"""Check portable figure hashes, physical size, vector content, fonts and bounds."""
import hashlib
import json
from pathlib import Path
import fitz
import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    m = json.loads((HERE/"figure_manifest.json").read_text())
    assert sha(STUDY/"analysis/plot_data.npz") == m["plot_data_sha256"]
    assert sha(HERE/"plot_figures.py") == m["plot_script_sha256"]
    assert sha(STUDY/"protocol.json") == m["protocol_sha256"]
    assert sha(HERE/"orx_figstyle.py") == m["style"]["vendor_sha256"]
    assert sha(HERE/"publication_style.py") == m["style"]["adapter_sha256"]
    assert sha(HERE/"captions.json") == m["captions_sha256"]
    assert sha(HERE/"figure_pack.pdf") == m["pack_sha256"]
    fonts = {}
    for stem, outputs in m["figures"].items():
        for ext, item in outputs.items():
            assert sha(HERE/(stem+"."+ext)) == item["sha256"]
        with fitz.open(HERE/(stem+".pdf")) as doc:
            assert len(doc) == 1
            page = doc[0]
            assert abs(page.rect.width-4.8*72) < .01
            assert not page.get_images(), stem
            fonts[stem] = sorted({f[3].split("+")[-1] for f in page.get_fonts()})
            assert fonts[stem] and all(f.upper().startswith(("CM", "SF")) for f in fonts[stem]), fonts[stem]
            assert all(page.rect.contains(fitz.Rect(b[:4])) for b in page.get_text("blocks")), stem
    with np.load(STUDY/"analysis/plot_data.npz", allow_pickle=False) as a:
        count = 0
        for key in a.files:
            if key.endswith(("::q_A", "::q_B")) or "::individual_q_" in key:
                x = a[key]
                assert np.isfinite(x).all() and (x >= 0).all() and (x <= 1).all(), key
                count += 1
            if key.endswith(("::H_individual", "::H_population", "::disagreement")):
                assert np.min(a[key]) >= -1e-13 and np.max(a[key]) <= np.log(2)+1e-13, key
    with fitz.open(HERE/"figure_pack.pdf") as doc:
        assert len(doc) == m["pack_pages"]
        assert all(all(p.rect.contains(fitz.Rect(b[:4])) for b in p.get_text("blocks")) for p in doc)
        pack_pages = len(doc)
    report = {"status": "passed", "figure_count": len(fonts), "pack_pages": pack_pages,
              "width_inches": 4.8, "vector": "No plot PDF contains raster images", "fonts": fonts,
              "bounded_probability_arrays": count, "uncertainty": "pointwise whole-seed percentile bootstrap, bounded by construction"}
    (HERE/"figure_validation.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
