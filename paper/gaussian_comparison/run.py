#!/usr/bin/env python3
"""Matched single-Gaussian and socially gated baselines for the camp study.

Every update calls the archived rival engine. The only new dynamical operation
is reweighting the *weighted* base communication matrix by the archived social
disagreement gate. Both Gaussian compression rules use observable marginals,
never a common interpretation of the candidates' hidden P/O coordinates.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import sys
from threading import Lock
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT_SOURCE = ROOT / "paper/communication_experiments/run.py"
_spec = importlib.util.spec_from_file_location("gaussian_parent_runner", PARENT_SOURCE)
parent = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(parent)
jax, jnp, rv, wm = parent.jax, parent.jnp, parent.rv, parent.wm
ENGINE, WORLD_SOURCE = parent.ENGINE, parent.WORLD_SOURCE
sha, array_sha, dump = parent.sha, parent.array_sha, parent.dump
STATE_KEYS = parent.STATE_KEYS
HISTORY_KEYS = parent.HISTORY_KEYS + ("gate_crossweight",)
CONDITION_NAMES = ("G_plain_info", "G_gated_info", "G_plain_moment", "G_gated_moment")


def compress_reference(world, variant):
    """Return a K=1 reference, plus observable compression diagnostics.

    Information pooling is the equal log-linear pool of the two *marginal*
    observable Gaussians. Moment matching is the equal mixture's first two
    moments, including the between-component mean variance. Independent dummy
    padding keeps the engine's four-coordinate interface without identifying
    phlogiston with oxygen.
    """
    candidate_pi = np.asarray(world["Pi"], dtype=np.float64)
    candidate_h = np.asarray(world["h"], dtype=np.float64)
    joint_cov = np.linalg.solve(candidate_pi, np.broadcast_to(np.eye(4), candidate_pi.shape))
    joint_mu = np.linalg.solve(candidate_pi, candidate_h[..., None])[..., 0]
    covariance = joint_cov[:, 1:, 1:]
    means = joint_mu[:, 1:]
    marginal_pi = np.linalg.solve(covariance, np.broadcast_to(np.eye(3), covariance.shape))
    marginal_h = np.einsum("kij,kj->ki", marginal_pi, means)
    if variant == "info":
        precision = marginal_pi.mean(axis=0)
        information = marginal_h.mean(axis=0)
        common_cov = np.linalg.solve(precision, np.eye(3))
        common_mu = np.linalg.solve(precision, information)
    elif variant == "moment":
        common_mu = means.mean(axis=0)
        offset = means - common_mu
        common_cov = (covariance + np.einsum("ki,kj->kij", offset, offset)).mean(axis=0)
        precision = np.linalg.solve(common_cov, np.eye(3))
        information = precision @ common_mu
    else:
        raise ValueError(f"Unknown reference compression {variant!r}")
    if np.linalg.eigvalsh(precision).min() <= 0:
        raise ValueError("Compressed reference is not positive definite")
    padded_pi = np.zeros((1, 4, 4), dtype=np.float64)
    padded_h = np.zeros((1, 4), dtype=np.float64)
    padded_pi[0, 0, 0] = 4.0
    padded_pi[0, 1:, 1:] = precision
    padded_h[0, 1:] = information
    metadata = dict(reference_variant=variant,
                    reference_observable_mean=common_mu.tolist(),
                    reference_observable_covariance=common_cov.tolist(),
                    original_observable_means=means.tolist(),
                    original_observable_covariances=covariance.tolist(),
                    original_marginal_precisions=marginal_pi.tolist(),
                    original_marginal_information=marginal_h.tolist(),
                    compression_weights=[.5, .5],
                    dummy_coordinate=dict(mean=0., variance=.25, independent=True,
                                          meaning="unobserved storage padding; neither P nor O"))
    return jnp.asarray(padded_pi), jnp.asarray(padded_h), metadata


def make_setup(condition, N=None):
    """Return cfg,W,A,attention,constant signal,metadata for one baseline."""
    c = {"name": condition} if isinstance(condition, str) else dict(condition)
    name = c.get("name", c.get("id"))
    if name not in CONDITION_NAMES:
        raise ValueError(f"Unknown condition {name!r}")
    _, gate_type, variant = name.split("_")
    if c.get("reference_variant", variant) != variant:
        raise ValueError("Reference variant must agree with condition name")
    N = c.get("N", 16) if N is None else N
    beta, omega, sigma = c.get("beta", .05), c.get("omega", .3), c.get("sigma_o", .5)
    gate_nu = c.get("gate_nu", .01) if gate_type == "gated" else None
    if gate_nu is not None and gate_nu <= 0:
        raise ValueError("Social gate degrees of freedom must be positive")
    if not 0 <= omega <= 1 or sigma <= 0:
        raise ValueError("Invalid retention or observation noise")
    world = wm.build_world()
    Pi, h, reference = compress_reference(world, variant)
    W = parent.weighted_matrix(beta, N)
    A = (W > 0).astype(np.float64)  # Interface placeholder; no gate uses A.
    attention = wm.wall_attention(N)
    phi = np.array([0., 1., 1., 1.], dtype=np.float64)
    cfg = rv.RivalConfig(Pi0=Pi, h0=h, H=world["H"], sigma_o=sigma, omega=omega,
                         alpha_m=0., rho_L=0., fuse_nets=True, social_nu=None,
                         nu_m=None, jitter=0., Pi0_agent=None, h0_agent=None)
    metadata = dict(condition=name, id=name, family="gaussian_camp", N=N, K=1,
                    group_indices={"gas": list(range(N // 2)), "mass": list(range(N // 2, N))},
                    omega=omega, alpha_m=0., sigma_o=sigma, fuse_nets=True, beta=beta,
                    gate_nu=gate_nu, gate_indices=[1, 2, 3],
                    gate_definition="archived reliability.social_gamma; weighted W*gamma then exact row normalization",
                    gate_placement="entering conditional states before communication and observations",
                    engine_social_nu=None, gate_normalization_epsilon=0.,
                    rho_L=0., model_gate="disabled", observation_gate="disabled", utility="absent",
                    W=W.tolist(), attention=attention.tolist(), phi=phi.tolist(),
                    Pi0=np.asarray(Pi).tolist(), h0=np.asarray(h).tolist(), H=np.asarray(cfg.H).tolist(),
                    observable_names=["C", "G", "M"],
                    coordinate_names=["independent unobserved padding", "C", "G", "M"],
                    common_reference_for_all_agents=True, initial_camp_difference=False,
                    candidate_names=["single Gaussian " + variant],
                    training_random="key,sk=split(key); agent_keys=split(sk,N); sample_o per agent",
                    evaluation_random="fold_in(entering_key,314159); normal((N,4,3),float64)",
                    evaluation_logdet="2*sum(log(diag(cholesky(predictive_covariance))))",
                    **reference)
    return cfg, jnp.asarray(W), jnp.asarray(A), jnp.asarray(attention), jnp.asarray(phi), metadata


def effective_W(Pi, h, W, gate_nu):
    """Preserve base edge strengths when applying the archived means gate."""
    if gate_nu is None:
        return W
    gamma = rv.rel.social_gamma(Pi[:, 0], h[:, 0], jnp.asarray([1, 2, 3]), gate_nu)
    weighted = W * gamma
    # Positive base self weights and gamma_ii=(nu+1)/nu guarantee positive sums.
    return weighted / weighted.sum(axis=-1, keepdims=True)


def make_batch(cfg, W, A, attention, draws=4, gate_nu=None):
    """Compile actual-engine updates with paired readouts and exact restarts."""
    if cfg.n_candidates != 1 or cfg.social_nu is not None or cfg.nu_m is not None or cfg.rho_L != 0.:
        raise ValueError("Only externally gated K=1 configurations are supported")
    if not cfg.fuse_nets or cfg.Pi0_agent is not None or cfg.h0_agent is not None:
        raise ValueError("Matched baselines require fusion and one shared reference")
    N = W.shape[0]
    if N < 2 or N % 2 or draws < 1:
        raise ValueError("Need equal-sized groups and at least one held-out draw")
    cross = (jnp.arange(N)[:, None] // (N // 2) != jnp.arange(N)[None, :] // (N // 2))

    def one(Pi, h, L, key, phis):
        def step(carry, phi):
            Pi, h, L, key = carry
            W_eff = effective_W(Pi, h, W, gate_nu)
            pp, hh = rv.fuse_within_frame(Pi, h, W_eff)
            _, observation_key = jax.random.split(key)
            keys = jax.random.split(observation_key, N)
            obs = jax.vmap(lambda k: parent.sample_o(cfg.H, phi, cfg.sigma_o, k))(keys)
            ell = jax.vmap(lambda p, h, o, a: rv.score_candidates(p, h, cfg.H, o, cfg.sigma_o, a, cfg.jitter))(pp, hh, obs, attention)
            state, _ = rv._step(carry, phi, cfg, W_eff, A, attention, track_means=False)
            pn, hn, ln, _ = state
            q = jax.nn.softmax(ln, axis=-1)
            mu = jnp.linalg.solve(pn, hn[..., None])[..., 0][..., 1:]
            mixture = jnp.einsum("nk,nkd->nd", q, mu)
            selected = mu[:, 0]
            mse = jnp.mean((mixture - phi[1:]) ** 2, axis=-1)
            cov = jnp.linalg.inv(pn)[..., 1:, 1:] + cfg.sigma_o ** 2 * jnp.eye(3)
            cf = jnp.linalg.cholesky(cov)
            eval_key = jax.random.fold_in(key, 314159)
            eval_obs = phi[1:] + cfg.sigma_o * jax.random.normal(eval_key, (N, draws, 3), dtype=pn.dtype)
            residual = eval_obs[:, :, None, :] - mu[:, None, :, :]
            quadratic = jnp.einsum("nrki,nkij,nrkj->nrk", residual, jnp.linalg.inv(cov), residual)
            logdet = 2 * jnp.log(jnp.diagonal(cf, axis1=-2, axis2=-1)).sum(-1)
            logp = -.5 * (3 * jnp.log(2 * jnp.pi) + logdet[:, None, :] + quadratic)
            nll = -jax.scipy.special.logsumexp(jax.nn.log_softmax(ln)[:, None, :] + logp, axis=-1).mean(-1)
            crossweight = jnp.sum(jnp.where(cross, W_eff, 0.), axis=-1)
            return state, (ln, q, ell, obs, mu, mixture, selected, mse, nll, crossweight, pn, eval_obs)
        return jax.lax.scan(step, (Pi, h, L, key), phis)

    compiled = jax.jit(jax.vmap(one, in_axes=(0, 0, 0, 0, None)))

    def call(phis, seeds, initial=None):
        entering = parent.initial_state(cfg, N, seeds) if initial is None else initial
        if set(STATE_KEYS) - set(entering):
            raise ValueError("Restart requires exact Pi,h,L,key")
        if any(np.asarray(entering[k]).shape[0] != len(seeds) for k in STATE_KEYS):
            raise ValueError("Restart batch dimension differs from seed count")
        final, out = compiled(*(jnp.asarray(entering[k]) for k in STATE_KEYS), jnp.asarray(phis))
        jax.block_until_ready(out)
        result = {name: np.asarray(value) for name, value in zip(HISTORY_KEYS + ("Pi_history", "eval_obs"), out)}
        result.update({"initial_" + k: np.asarray(entering[k]) for k in STATE_KEYS})
        result.update({"final_" + k: np.asarray(value) for k, value in zip(STATE_KEYS, final)})
        if not all(np.isfinite(value).all() for value in result.values()):
            raise FloatingPointError("Nonfinite engine or predictive output")
        if not np.array_equal(result["q"], np.ones_like(result["q"])):
            raise AssertionError("Single-candidate probability is not identically one")
        previous = np.concatenate((result["initial_L"][:, None], result["L"][:, :-1]), axis=1)
        if not np.allclose(result["L"], previous + result["ell"], rtol=1e-11, atol=1e-8):
            raise AssertionError("Instrumented scores differ from engine updates")
        if np.any(result["gate_crossweight"] < 0) or np.any(result["gate_crossweight"] > 1):
            raise AssertionError("Cross-group row weights are outside [0,1]")
        return result
    return call


def batch(cfg, W, A, attention, phis, seeds, initial=None, draws=4, gate_nu=None):
    return make_batch(cfg, W, A, attention, draws, gate_nu)(phis, seeds, initial)


def final_state(result):
    return {k: result["final_" + k] for k in STATE_KEYS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=HERE / "protocol.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    p = json.loads(args.protocol.read_text())
    conditions = [dict(c) if isinstance(c, dict) else {"name": c} for c in p["conditions"]]
    if len(conditions) != 4 or {c.get("name", c.get("id")) for c in conditions} != set(CONDITION_NAMES):
        raise ValueError("Production requires all four fixed conditions exactly once")
    seeds = p.get("seeds", p.get("camp_seeds"))
    if seeds != list(range(10000, 10050)):
        raise ValueError("Production requires the 50 paired camp seeds 10000..10049")
    T = p.get("T", p.get("camp_T", 5000))
    chunk_steps, batch_seeds = p.get("chunk_steps", 250), p.get("batch_seeds", 5)
    draws, workers = p.get("heldout_draws", 4), p.get("condition_workers", 2)
    if (T, chunk_steps, batch_seeds, draws) != (5000, 250, 5, 4) or workers not in (1, 2):
        raise ValueError("Production timing, batching or predictive-draw settings differ from the fixed design")
    for c in conditions:
        _, _, _, _, _, m = make_setup(c)
        if (m["N"], m["beta"], m["omega"], m["sigma_o"]) != (16, .05, .3, .5):
            raise ValueError("Production condition deviates from the matched camp setup")
        if m["gate_nu"] not in (None, .01):
            raise ValueError("Production gate parameter differs from .01")
    out = args.output or HERE / p.get("output", "runs/matched-2026-09-20")
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    shutil.copy2(args.protocol, out / "protocol.json")
    source = out / "source"
    source.mkdir()
    for path in HERE.glob("*.py"):
        shutil.copy2(path, source / path.name)
    shutil.copy2(PARENT_SOURCE, source / "communication_parent_run.py")
    shutil.copy2(WORLD_SOURCE, source / "combustion_world.py")
    engine_hashes = {str(f.relative_to(ENGINE)): sha(f) for f in ENGINE.rglob("*.py")}
    dump(out / "engine_hashes.json", engine_hashes)
    dump(out / "environment.json", dict(python=sys.version, platform=platform.platform(),
            backend=jax.default_backend(), jax_enable_x64=jax.config.jax_enable_x64,
            jax_default_prng_impl=str(jax.config.jax_default_prng_impl),
            jax_devices=[str(device) for device in jax.devices()],
            engine_file=str(Path(rv.__file__).resolve()),
            world_sha256=sha(WORLD_SOURCE), runner_sha256=sha(__file__), parent_runner_sha256=sha(PARENT_SOURCE),
            packages={name: importlib.metadata.version(name) for name in
                      ["jax", "jaxlib", "numpy", "scipy", "equinox", "networkx", "matplotlib"]}))
    print("PROTOCOL_SHA256 " + sha(args.protocol), flush=True)
    manifest, summaries = [], {}
    lock = Lock()

    def persist(result, name, ss, start, steps):
        file = out / name / f"seeds-{ss[0]}-{ss[-1]}_updates-{start + 1:05d}-{start + steps:05d}.npz"
        stored = {k: v for k, v in result.items() if k != "eval_obs"}
        np.savez_compressed(file, **stored)
        return dict(condition=name, file=str(file.relative_to(out)), seeds=ss,
                    start_step=start, steps=steps, sha256=sha(file),
                    observation_sha256=array_sha(result["obs"]),
                    evaluation_observation_sha256=array_sha(result["eval_obs"]),
                    precision_storage="Pi_history in this chunk for every seed; no deduplication",
                    array_shapes={k: list(v.shape) for k, v in stored.items()})

    def run_condition(c):
        cfg, W, A, att, phi, meta = make_setup(c)
        name = meta["condition"]
        directory = out / name
        directory.mkdir()
        meta.update(seeds=seeds, T=T, start_step=0, final_state=T, heldout_draws=draws,
                    array_time="entry j is state start_step+j+1 after its observation; gate weights are entering-state weights for this update",
                    observations="all three channels sampled regardless of attention")
        dump(directory / "setup.json", meta)
        print("SETUP " + json.dumps({key: meta[key] for key in
            ("condition", "N", "T", "reference_variant", "omega", "sigma_o", "beta", "gate_nu")}), flush=True)
        runner = make_batch(cfg, W, A, att, draws, meta["gate_nu"])
        records, endpoints = [], []
        for si in range(0, len(seeds), batch_seeds):
            ss = seeds[si:si + batch_seeds]
            state = None
            for t in range(0, T, chunk_steps):
                count = min(chunk_steps, T - t)
                result = runner(np.broadcast_to(np.asarray(phi), (count, 4)), ss, state)
                state = final_state(result)
                records.append(persist(result, name, ss, t, count))
                if t + count == T:
                    for j, seed in enumerate(ss):
                        means = result["mu_mixture"][j, -1]
                        gas, mass = means[:len(W)//2].mean(0), means[len(W)//2:].mean(0)
                        endpoints.append(dict(seed=seed, observable_camp_distance=float(np.linalg.norm(gas - mass)),
                            mse_gas=float(result["mse_full"][j, -1, :len(W)//2].mean()),
                            mse_mass=float(result["mse_full"][j, -1, len(W)//2:].mean()),
                            nll_gas=float(result["nll_full"][j, -1, :len(W)//2].mean()),
                            nll_mass=float(result["nll_full"][j, -1, len(W)//2:].mean()),
                            gate_crossweight=float(result["gate_crossweight"][j, -1].mean())))
            print("BATCH " + json.dumps(dict(condition=name, seeds=ss, elapsed_seconds=round(time.time() - started, 2))), flush=True)
        summary = dict(seed_count=len(seeds), final_state=T, endpoint_only=True,
                       **{"endpoint_mean_" + key: float(np.mean([r[key] for r in endpoints]))
                          for key in endpoints[0] if key != "seed"})
        dump(directory / "endpoint_by_seed.json", endpoints)
        with lock:
            manifest.extend(records)
            summaries[name] = summary
            dump(out / "manifest.partial.json", manifest)
            dump(out / "summary.partial.json", summaries)
        print("CONDITION " + json.dumps(dict(condition=name, elapsed_seconds=round(time.time() - started, 2), **summary)), flush=True)

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_condition, c) for c in conditions]
            for future in futures:
                future.result()
        if engine_hashes != {str(f.relative_to(ENGINE)): sha(f) for f in ENGINE.rglob("*.py")}:
            raise AssertionError("Archived engine changed during execution")
        dump(out / "manifest.json", sorted(manifest, key=lambda r: (r["condition"], r["seeds"][0], r["start_step"])))
        dump(out / "summary.json", summaries)
        dump(out / "completion.json", dict(status="complete", conditions=len(summaries),
            chunks=len(manifest), seed_condition_histories=len(seeds)*len(conditions),
            independent_seed_streams=len(seeds), elapsed_seconds=time.time()-started,
            protocol_sha256=sha(args.protocol)))
        print("COMPLETE " + json.dumps(dict(conditions=len(summaries), seconds=time.time()-started)), flush=True)
    except Exception:
        dump(out / "failure.json", dict(traceback=traceback.format_exc(), elapsed_seconds=time.time()-started))
        raise


if __name__ == "__main__":
    main()
