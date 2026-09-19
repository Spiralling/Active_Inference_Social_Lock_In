"""Reproducibility spine: content hashing, provenance manifests, headless plotting.

A pure-plumbing layer *around* the (untouched) simulation engine. Nothing here
changes any numerics -- it records *what produced a result* (git SHA, environment,
seeds, resolved params, per-array hashes) and provides the shared bootstrap and
headless helpers the experiment runner uses.
"""
