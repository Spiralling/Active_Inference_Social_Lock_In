"""Known-model discrimination from dependence alone, separate from social runs.

Unlike the adaptive simulations' persistent signals, this check draws a new
linear-Gaussian state for every observation. Both fixed candidates have zero
joint means and identical observable univariate variances. Their multivariate
relations differ. A Bayesian classifier evaluates their *known*, unchanging
observable joint densities; it does not discover a graph or fit parameters.

This is an identifiability check for this supplied two-model menu, not a claim
of unique causal identification or recovery by the adaptive social engine.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.special import logsumexp
from scipy.stats import multivariate_normal


OBSERVABLE_IDX = np.array([1, 2, 3])
RNG_NAMESPACE = 260919


def matched_models(world: dict) -> dict:
    """Center the proposed models and match only the G residual variance.

    Storage order is (P/O,C,G,M), so the DAG is not lower triangular in
    storage order. Solving (I-B)x=epsilon correctly draws its joint state.
    The incumbent G variance becomes .5448 + .8**2*.5 = .8648,
    matching oxygen's .5 + .8**2*(.25 + .8**2*.5).
    """
    B = np.asarray(world["B"], dtype=np.float64).copy()
    s = np.asarray(world["s"], dtype=np.float64).copy()
    expected = np.zeros((2, 4, 4), dtype=np.float64)
    expected[0, 0, 1] = expected[0, 2, 1] = 0.8
    expected[0, 3, 0] = -0.8
    expected[1, 0, 1] = expected[1, 2, 0] = expected[1, 3, 0] = 0.8
    np.testing.assert_allclose(B, expected, atol=0, rtol=0)
    np.testing.assert_allclose(s, [[0.25, 0.5, 0.5, 0.5]] * 2, atol=0, rtol=0)
    s[0, 2] = 0.5448
    transforms = np.linalg.inv(np.eye(4)[None] - B)
    covariance = np.stack(
        [transforms[k] @ np.diag(s[k]) @ transforms[k].T for k in range(2)]
    )
    observable_covariance = covariance[:, 1:, 1:]
    variances = np.diagonal(observable_covariance, axis1=-2, axis2=-1)
    np.testing.assert_allclose(
        variances, [[0.5, 0.8648, 0.8648]] * 2, atol=2e-15, rtol=0
    )
    conditional_covariance = (
        observable_covariance[:, 1, 2]
        - observable_covariance[:, 1, 0]
        * observable_covariance[:, 0, 2]
        / observable_covariance[:, 0, 0]
    )
    np.testing.assert_allclose(conditional_covariance, [0, 0.16], atol=2e-15, rtol=0)
    if np.linalg.eigvalsh(covariance).min() <= 0:
        raise ValueError("Structural-check joint covariances must be positive definite")
    return {
        "B": B,
        "s": s,
        "mu": np.zeros((2, 4)),
        "transforms": transforms,
        "covariance": covariance,
        "observable_covariance": observable_covariance,
        "observable_variances": variances,
        "conditional_covariance_G_M_given_C": conditional_covariance,
    }


def _posterior(log_density: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Uniform prior, independent observations, no adaptation or pooling."""
    L = np.cumsum(log_density, axis=-2)
    shifted = L - L.max(axis=-1, keepdims=True)
    q = np.exp(shifted - logsumexp(shifted, axis=-1, keepdims=True))
    return L, q


def _kl(covariance_p: np.ndarray, covariance_q: np.ndarray) -> float:
    """KL[N(0, P)||N(0, Q)] in nats for one fresh observation."""
    d = covariance_p.shape[0]
    return float(
        0.5
        * (
            np.trace(np.linalg.solve(covariance_q, covariance_p))
            - d
            + np.linalg.slogdet(covariance_q)[1]
            - np.linalg.slogdet(covariance_p)[1]
        )
    )


def _prefix_summary(q: np.ndarray, sample_counts: np.ndarray) -> dict:
    """q axes: noise, generating candidate, seed, observation, candidate.

    A posterior tie (within 1e-10 floating-point tolerance) contributes half
    to each predicted category. This avoids
    reporting a spurious preference for candidate 0 in the negative controls.
    """
    selected = q[..., sample_counts - 1, :]
    # noise, prefix, true candidate, seed, predicted candidate
    selected = selected.transpose(0, 3, 1, 2, 4)
    maxima = selected.max(axis=-1, keepdims=True)
    tied = np.isclose(selected, maxima, rtol=0, atol=1e-10)
    votes = tied / tied.sum(axis=-1, keepdims=True)
    counts = votes.sum(axis=-2)
    confusion = counts / q.shape[2]
    accuracy_by_source = np.diagonal(confusion, axis1=-2, axis2=-1)
    return {
        "confusion_counts": counts,
        "confusion_probability": confusion,
        "mean_q": selected.mean(axis=-2),
        "accuracy_by_source": accuracy_by_source,
        "accuracy": accuracy_by_source.mean(axis=-1),
    }


def run_check(world: dict, protocol: dict, outdir: Path | str) -> dict:
    """Write ``structural_check.npz/json`` and return their JSON summary.

    ``protocol['structural_check']`` specifies ``seeds``, ``sigmas`` and
    ``sample_counts``. This routine has no outcome-dependent tuning or stopping.
    Noise levels share the same latent states and standard measurement noise;
    different source candidates and seeds use independent reproducible streams.
    """
    config = protocol["structural_check"]
    seeds = np.asarray(config["seeds"], dtype=np.int64)
    sigmas = np.asarray(config["sigmas"], dtype=np.float64)
    sample_counts = np.asarray(config["sample_counts"], dtype=np.int64)
    if (
        seeds.ndim != 1
        or len(seeds) == 0
        or len(np.unique(seeds)) != len(seeds)
        or np.any(seeds < 0)
    ):
        raise ValueError(
            "seeds must be a nonempty sequence of unique nonnegative integers"
        )
    if (
        sigmas.ndim != 1
        or len(sigmas) == 0
        or not np.isfinite(sigmas).all()
        or np.any(sigmas < 0)
        or len(np.unique(sigmas)) != len(sigmas)
    ):
        raise ValueError(
            "sigmas must be distinct finite nonnegative noise standard deviations"
        )
    if (
        sample_counts.ndim != 1
        or len(sample_counts) == 0
        or np.any(sample_counts < 1)
        or np.any(np.diff(sample_counts) <= 0)
    ):
        raise ValueError("sample_counts must be increasing positive integers")
    models = matched_models(world)
    steps = int(sample_counts[-1])
    latent = np.empty((2, len(seeds), steps, 4), dtype=np.float64)
    noise = np.empty((2, len(seeds), steps, 3), dtype=np.float64)
    for source in range(2):
        for i, seed in enumerate(seeds):
            sequence = np.random.SeedSequence([RNG_NAMESPACE, source, int(seed)])
            latent_sequence, noise_sequence = sequence.spawn(2)
            innovations = np.random.default_rng(latent_sequence).standard_normal(
                (steps, 4)
            )
            innovations *= np.sqrt(models["s"][source])
            latent[source, i] = innovations @ models["transforms"][source].T
            noise[source, i] = np.random.default_rng(noise_sequence).standard_normal(
                (steps, 3)
            )
    observations = (
        latent[None, ..., OBSERVABLE_IDX]
        + sigmas[:, None, None, None, None] * noise[None]
    )
    predictive_covariance = (
        models["observable_covariance"][None]
        + sigmas[:, None, None, None] ** 2 * np.eye(3)[None, None]
    )
    shape = observations.shape[:-1] + (2,)
    log_density = np.empty(shape, dtype=np.float64)
    marginal_log_density = np.empty_like(log_density)
    analytic_kl = np.empty((len(sigmas), 2), dtype=np.float64)
    for i in range(len(sigmas)):
        for candidate in range(2):
            cov = predictive_covariance[i, candidate]
            log_density[i, ..., candidate] = multivariate_normal.logpdf(
                observations[i], mean=np.zeros(3), cov=cov
            )
            marginal_log_density[i, ..., candidate] = multivariate_normal.logpdf(
                observations[i], mean=np.zeros(3), cov=np.diag(np.diag(cov))
            )
            analytic_kl[i, candidate] = _kl(
                cov, predictive_covariance[i, 1 - candidate]
            )
    L, q = _posterior(log_density)
    L_marginal, q_marginal = _posterior(marginal_log_density)
    duplicate_log_density = np.repeat(log_density[..., :1], 2, axis=-1)
    L_duplicate, q_duplicate = _posterior(duplicate_log_density)
    _, q_swapped = _posterior(log_density[..., ::-1])
    summaries = _prefix_summary(q, sample_counts)
    marginal_summaries = _prefix_summary(q_marginal, sample_counts)
    duplicate_summaries = _prefix_summary(q_duplicate, sample_counts)
    checks = {
        "matched_observable_variance_max_error": float(
            np.abs(
                models["observable_variances"][0] - models["observable_variances"][1]
            ).max()
        ),
        "product_marginals_q_max_error_from_half": float(
            np.abs(q_marginal - 0.5).max()
        ),
        "duplicate_candidates_q_max_error_from_half": float(
            np.abs(q_duplicate - 0.5).max()
        ),
        "product_marginals_confusion_max_error_from_half": float(
            np.abs(marginal_summaries["confusion_probability"] - 0.5).max()
        ),
        "duplicate_confusion_max_error_from_half": float(
            np.abs(duplicate_summaries["confusion_probability"] - 0.5).max()
        ),
        "label_swap_q_max_error": float(np.abs(q_swapped[..., ::-1] - q).max()),
        "posterior_normalization_max_error": float(np.abs(q.sum(axis=-1) - 1).max()),
        "all_finite": all(
            np.isfinite(x).all()
            for x in (
                latent,
                noise,
                observations,
                log_density,
                L,
                q,
                L_marginal,
                q_marginal,
                L_duplicate,
                q_duplicate,
                analytic_kl,
            )
        ),
    }
    # These checks concern mathematical invariances, never favorable outcomes.
    if not checks["all_finite"]:
        raise FloatingPointError("Nonfinite structural-check output")
    for name, value in checks.items():
        if name.endswith("error") or name.endswith("half"):
            if value > 1e-10:
                raise AssertionError(f"Structural-check invariant {name}: {value}")
    arrays = {
        "seeds": seeds,
        "sigmas": sigmas,
        "sample_counts": sample_counts,
        "generating_candidate": np.arange(2),
        "latent_states": latent,
        "measurement_noise_standard": noise,
        "observations": observations,
        "log_density": log_density,
        "L": L,
        "q": q,
        "L_product_marginals": L_marginal,
        "q_product_marginals": q_marginal,
        "L_duplicate": L_duplicate,
        "q_duplicate": q_duplicate,
        "predictive_covariance": predictive_covariance,
        "analytic_kl_nats_per_observation": analytic_kl,
        **models,
        **summaries,
        **{f"product_marginals_{k}": v for k, v in marginal_summaries.items()},
        **{f"duplicate_{k}": v for k, v in duplicate_summaries.items()},
    }
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    arrays_path = outdir / "structural_check.npz"
    np.savez_compressed(arrays_path, **arrays)
    summary = {
        "kind": "Known-model discrimination using multivariate dependence only",
        "candidate_order": ["phlogiston", "oxygen"],
        "observable_order": ["C", "G", "M"],
        "config": {
            "seeds": seeds.tolist(),
            "sigmas": sigmas.tolist(),
            "sample_counts": sample_counts.tolist(),
        },
        "number_of_independent_seed_source_pairs": int(2 * len(seeds)),
        "noise_pairing": "Each sigma uses the same latent states and standard measurement noise for each source/seed pair.",
        "observations": "Fresh independent DAG state and independent Gaussian measurement noise for every observation.",
        "measurement_noise": "Independent noise on each of C,G,M; sigma is standard deviation.",
        "classifier": "Equal prior probabilities; cumulative joint Gaussian log density under fixed known candidates; no parameter estimation, adaptation, forgetting or social pooling.",
        "matching": "Both joint means are zero. Incumbent G residual variance is .5448 instead of .5; all other residuals and all edges match the revised main model. Observable univariate variances are [.5,.8648,.8648] for both candidates before measurement noise.",
        "conditional_covariance_G_M_given_true_C": models[
            "conditional_covariance_G_M_given_C"
        ].tolist(),
        "conditional_covariance_caveat": "The [0,.16] result conditions on true C; conditioning on a noisy measurement need not remove the incumbent dependence.",
        "confusion_convention": "Rows generating candidate, columns predicted candidate; posterior ties within absolute tolerance 1e-10 contribute one half to each column; probabilities divide counts by number of seeds.",
        "analytic_kl_convention": "Entry [noise,source] is KL(source observable joint density || other candidate), nats per fresh observation.",
        "interpretation_limits": [
            "This establishes distinguishability only within the supplied two-model menu at the tested coefficients and noise levels.",
            "Dependence differs through coefficients and their signs as well as graph structure; this is not an ablation isolating topology.",
            "Distinct latent causal models may share these observable densities; no unique causal graph identification is claimed.",
            "The adaptive social engine is not run in this check and does not receive these fresh DAG samples in the main experiments.",
            "The main persistent-signal experiments retain unequal means and are not converted into structure-recovery tests by this supplemental check.",
        ],
        "checks": checks,
        "analytic_kl_nats_per_observation": analytic_kl.tolist(),
        **{k: v.tolist() for k, v in summaries.items()},
        "product_marginals_accuracy": marginal_summaries["accuracy"].tolist(),
        "duplicate_accuracy": duplicate_summaries["accuracy"].tolist(),
        "array_file": arrays_path.name,
        "array_sha256": hashlib.sha256(arrays_path.read_bytes()).hexdigest(),
        "arrays": {
            k: {"shape": list(v.shape), "dtype": str(v.dtype)}
            for k, v in arrays.items()
        },
        "axis_contract": {
            "q_L_and_log_density": [
                "sigma",
                "generating_candidate",
                "seed",
                "observation",
                "candidate",
            ],
            "observations": [
                "sigma",
                "generating_candidate",
                "seed",
                "observation",
                "C_G_M",
            ],
            "latent_states": ["generating_candidate", "seed", "observation", "z_C_G_M"],
            "confusion_and_mean_q": [
                "sigma",
                "sample_count",
                "generating_candidate",
                "candidate",
            ],
            "accuracy": ["sigma", "sample_count"],
        },
        "provenance": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "rng": "NumPy PCG64 via SeedSequence([260919, source_candidate, seed]); independent spawned latent/noise streams.",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    (outdir / "structural_check.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "experiment": "structural_check",
                "scope": "Fixed known-model classifier on fresh independent DAG samples; separate from adaptive social engine.",
                "sigmas": sigmas.tolist(),
                "sample_counts": sample_counts.tolist(),
                "accuracy": summaries["accuracy"].tolist(),
                "analytic_kl_nats_per_observation": analytic_kl.tolist(),
                "checks": checks,
            },
            allow_nan=False,
        ),
        flush=True,
    )
    return summary
