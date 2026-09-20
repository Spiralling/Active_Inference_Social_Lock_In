"""LNCS adaptation of the vendored orx academic plotting style.

The vendor module is unmodified. This adapter sets the actual 122-mm venue
width (rounded to 4.8 inches), renders all text and mathematics with the
manuscript's LaTeX fonts, and writes deterministic vector exports.
PNG files are inspection previews, never manuscript plot assets.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
if Path("/Library/TeX/texbin").is_dir():
    os.environ["PATH"] = "/Library/TeX/texbin:" + os.environ.get("PATH", "")
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "iwai-figure-matplotlib"))

import matplotlib as mpl
import matplotlib.pyplot as plt
from . import orx_figstyle as orx

WIDTH = 4.8
BLUE, ORANGE, GREY = orx.PALETTE["blue"], orx.PALETTE["orange"], orx.BASELINE
LATEX_PREAMBLE = r"\usepackage[T1]{fontenc}"


def use_style():
    orx.TEXT = WIDTH  # The venue is LNCS, not the vendor's 5.5-inch article.
    orx.use_style(family="serif")
    # Use the actual manuscript renderer, including T1/CM-Super text, for
    # every figure. Matplotlib's font-manager fallback is not used by TeX.
    mpl.rcParams.update({"text.usetex": True, "font.family": "serif",
                         "font.serif": ["Computer Modern Roman"],
                         "text.latex.preamble": LATEX_PREAMBLE,
                         "font.size": 8.5, "axes.labelsize": 8.5,
                         "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "svg.hashsalt": "iwai-orx-utility-free-v1", "savefig.dpi": 300})


def typography():
    return json.loads((ROOT / "font_reference.json").read_text())


def provenance():
    return {"skill": "orx-figures", "reference": "orx skill figures",
            "vendor_sha256": hashlib.sha256(Path(orx.__file__).read_bytes()).hexdigest(),
            "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "font": typography(),
            "font_reference_sha256": hashlib.sha256((ROOT / "font_reference.json").read_bytes()).hexdigest(),
            "width_inches": WIDTH, "source_date_epoch": os.environ["SOURCE_DATE_EPOCH"],
            "png_role": "inspection preview only"}


def save(fig, stem):
    """Run orx's audit and export; refuse layout defects. Return hash records."""
    stem = Path(stem)
    paths = orx.save(fig, str(stem), close=False)
    issues = orx._audit(fig)
    defects = [issue for issue in issues if not issue.startswith("no publication font")]
    if defects:
        plt.close(fig)
        raise ValueError("Publication figure audit failed: " + "; ".join(defects))
    preview = stem.with_suffix(".png")
    fig.savefig(preview, dpi=220, facecolor="white")
    paths.append(str(preview))
    plt.close(fig)
    return {Path(p).suffix.lstrip("."): {"path": str(Path(p).resolve()),
             "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in paths}
