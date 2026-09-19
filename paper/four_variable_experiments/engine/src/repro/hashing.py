"""Deterministic content hashing for arrays and ``.npz`` archives.

The single source of truth for "how we fingerprint a result", shared by the
provenance manifest writer (:mod:`src.repro.manifest`) and the byte-identity
regression goldens (``tests/golden_array_hashes.json``). One implementation means
a manifest hash and a golden hash of the same array are directly comparable.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def hash_array(arr) -> str:
    """SHA-256 of one array's content, stable across runs and processes.

    Numeric arrays hash their raw bytes (plus dtype + shape, so a reshape or
    dtype change is a different hash) -- exact bit-for-bit reproducibility, which
    is what byte-identity regression needs. Object arrays (e.g. anything saved
    under ``allow_pickle``) hash a canonical JSON of their nested-list form, since
    Python object bytes are not reproducible across processes.
    """
    a = np.asarray(arr)
    h = hashlib.sha256()
    h.update(a.dtype.str.encode())
    h.update(repr(a.shape).encode())
    if a.dtype == object:
        h.update(json.dumps(a.tolist(), sort_keys=True, ensure_ascii=False,
                             default=str).encode("utf-8"))
    else:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def hash_npz(path) -> dict[str, str]:
    """``{array_name: sha256}`` for every array in a ``.npz`` archive (name-sorted)."""
    with np.load(Path(path), allow_pickle=True) as npz:
        return {name: hash_array(npz[name]) for name in sorted(npz.files)}


def hash_file(path) -> str:
    """SHA-256 of a file's raw bytes (for non-array outputs like ``summary.json``)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
