"""Compare freshly rendered figures with the two supplied paper archives.

The archived PDFs are verification targets only. They are not drawing inputs.
Pixel equality ignores incidental PDF metadata while checking the whole image.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import fitz
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_figures(reference: Path, generated: Path, output: Path) -> dict:
    """Require exact 216-dpi pixels, page dimensions and extracted text."""
    records = {}
    for target in sorted(reference.glob("*.pdf")):
        actual = generated / target.name
        record = {"target_sha256": sha256(target), "generated": actual.name}
        if not actual.is_file():
            records[target.name] = dict(record, status="missing")
            continue
        record["generated_sha256"] = sha256(actual)
        record["byte_identical"] = record["target_sha256"] == record["generated_sha256"]
        with fitz.open(target) as expected, fitz.open(actual) as observed:
            if len(expected) != 1 or len(observed) != 1:
                records[target.name] = dict(record, status="wrong_page_count")
                continue
            a, b = expected[0], observed[0]
            record["page_dimensions_equal"] = list(a.rect) == list(b.rect)
            record["text_equal"] = a.get_text() == b.get_text()
            pa = a.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False)
            pb = b.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False)
            record["raster_dimensions_equal"] = (pa.width, pa.height, pa.n) == (pb.width, pb.height, pb.n)
            if record["raster_dimensions_equal"]:
                aa = np.frombuffer(pa.samples, dtype=np.uint8).astype(np.int16)
                bb = np.frombuffer(pb.samples, dtype=np.uint8).astype(np.int16)
                difference = np.abs(aa - bb)
                record.update(pixel_identical=bool(np.all(difference == 0)),
                              changed_channel_values=int(np.count_nonzero(difference)),
                              max_channel_error=int(difference.max()),
                              mean_channel_error=float(difference.mean()))
            else:
                record["pixel_identical"] = False
            record["fonts"] = sorted({f[3].split("+")[-1] for f in b.get_fonts()})
            record["raster_images_in_vector_pdf"] = len(b.get_images())
            record["text_within_page"] = all(b.rect.contains(fitz.Rect(x[:4])) for x in b.get_text("blocks"))
            record["status"] = "passed" if all(record[k] for k in
                ("page_dimensions_equal", "text_equal", "pixel_identical", "text_within_page")) else "different"
        records[target.name] = record
    report = {"status": "passed" if records and all(r["status"] == "passed" for r in records.values()) else "failed",
              "comparison_dpi": 216, "figures": records,
              "total": len(records), "byte_identical": sum(r.get("byte_identical", False) for r in records.values()),
              "pixel_identical": sum(r.get("pixel_identical", False) for r in records.values())}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report
