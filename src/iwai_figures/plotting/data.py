"""Lazy, hash-checked portable inputs; all statistical replicates are seeds."""

from pathlib import Path
import hashlib
import json
import numpy as np


class Arrays:
    """Mapping of namespace::array to small per-condition NPZ files."""

    def __init__(self, root):
        self.root = Path(root)
        self.cache = {}
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        for name, digest in self.manifest["files"].items():
            if (
                name.endswith(".json")
                and hashlib.sha256((self.root / name).read_bytes()).hexdigest()
                != digest
            ):
                raise ValueError("Plot metadata checksum mismatch: " + name)

    def __getitem__(self, key):
        group, name = key.split("::", 1) if "::" in key else ("common", key)
        if group not in self.cache:
            path = self.root / (group + ".npz")
            if (
                hashlib.sha256(path.read_bytes()).hexdigest()
                != self.manifest["files"][path.name]
            ):
                raise ValueError("Plot input checksum mismatch: " + str(path))
            with np.load(path, allow_pickle=False) as a:
                self.cache[group] = {k: a[k] for k in a.files}
        return self.cache[group][name]


class CameraData:
    def __init__(self, directory):
        self.arrays = Arrays(directory)

    def __getitem__(self, name):
        # Populate the condition cache without relying on an arbitrary array key.
        if name not in self.arrays.cache:
            path = self.arrays.root / (name + ".npz")
            if (
                hashlib.sha256(path.read_bytes()).hexdigest()
                != self.arrays.manifest["files"][path.name]
            ):
                raise ValueError("Camera plot input checksum mismatch: " + str(path))
            with np.load(path, allow_pickle=False) as a:
                self.arrays.cache[name] = {k: a[k] for k in a.files}
        return self.arrays.cache[name]


class PlotData:
    def __init__(self, root, gaussian=False):
        self.a = Arrays(root)
        summary_path = Path(root) / "summary.json"
        if (
            hashlib.sha256(summary_path.read_bytes()).hexdigest()
            != self.a.manifest["files"]["summary.json"]
        ):
            raise ValueError("Plot summary checksum mismatch")
        self.summary = json.loads(summary_path.read_text())
        indices = self.a["bootstrap_indices"]
        self.weights = (
            np.stack([np.bincount(row, minlength=indices.shape[1]) for row in indices])
            / indices.shape[1]
        )
        self.cache = {}
        self.gaussian = gaussian

    def get(self, name, key):
        return self.a[name + "::" + key]

    def ci(self, x):
        x = np.asarray(x, float)
        if self.gaussian and x.ndim > 1:
            bands = []
            for start in range(0, x.shape[1], 500):
                b = self.weights @ x[:, start : start + 500]
                bands.append(np.quantile(b, [0.025, 0.975], axis=0))
            lo, hi = np.concatenate(bands, axis=1)
        else:
            lo, hi = np.quantile(self.weights @ x, [0.025, 0.975], axis=0)
        return x.mean(0), lo, hi

    def curve(
        self,
        ax,
        name,
        metric,
        color,
        ls="-",
        state_min=1,
        state_max=None,
        offset=0,
        lw=1.05,
    ):
        key = (name, metric)
        if key not in self.cache:
            self.cache[key] = self.ci(self.get(name, metric))
        mean, lo, hi = self.cache[key]
        x = self.a["states"] if self.gaussian else self.get(name, "states")
        mask = x >= state_min
        if state_max is not None:
            mask &= x <= state_max
        ax.fill_between(
            x[mask] - offset,
            lo[mask],
            hi[mask],
            color=color,
            alpha=0.14 if self.gaussian else 0.16,
            linewidth=0,
        )
        return ax.plot(
            x[mask] - offset, mean[mask], color=color, linestyle=ls, linewidth=lw
        )[0]

    def metric(self, name, key, horizon=5000):
        return self.summary[name]["windows"][str(horizon)]["metrics"][key]
