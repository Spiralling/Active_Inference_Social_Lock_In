#!/usr/bin/env python3
"""Frozen communication experiments, using the archived engine at every update.

Random observations and predictive scores are instrumented without changing the
engine's arithmetic or consuming its random stream. Production output is chunked
lossless float64 data; deterministic precision histories are stored once per
condition/time chunk only after exact equality across seeds has been checked.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
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
ENGINE = ROOT / "paper/four_variable_experiments/engine"
WORLD_SOURCE = ROOT / "paper/combustion_experiments/world.py"
sys.path.insert(0, str(ENGINE))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from src.structural import rival as rv
from src.structural.bayesnet import LinearGaussianBN
from src.structural.world import sample_o

if not Path(rv.__file__).resolve().is_relative_to(ENGINE):
    raise RuntimeError("The imported rival engine is not the frozen local source")
_spec = importlib.util.spec_from_file_location("communication_reference_world", WORLD_SOURCE)
wm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wm)

STATE_KEYS = ("Pi", "h", "L", "key")
HISTORY_KEYS = ("L", "q", "ell", "obs", "mu_conditional", "mu_mixture", "mu_map", "mse_full", "nll_full")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def weighted_matrix(beta, N=16):
    if N < 2 or N % 2 or not 0 <= beta <= 1:
        raise ValueError("Two equal groups and beta in [0,1] are required")
    n = N // 2
    J = np.ones((n, n), dtype=np.float64) / n
    return np.block([[(1 - beta) * J, beta * J], [beta * J, (1 - beta) * J]])


def make_setup(condition, N=None):
    """Return cfg,W,A,attention,constant signal, and fully resolved metadata.

    Strings select the fixed brief conditions. Dictionaries may override fields
    for isolated implementation tests; production passes the frozen protocol.
    Network setup describes the post-change phase; main separately generates the
    common full-attention prehistory once for each seed.
    """
    c = {"name": condition} if isinstance(condition, str) else dict(condition)
    name = c.get("name", c.get("id"))
    if not name:
        raise ValueError("A condition name is required")
    family = c.get("family", "network" if name.startswith("N") else "camp")
    network = family == "network"
    if family not in ("network", "camp"):
        raise ValueError(f"Unknown family {family}")
    N = c.get("N", 24 if network else 16) if N is None else N
    suffix = name[1:3] if name.startswith(("N", "C")) else "10"
    fuse = c.get("fuse_nets", suffix[0] == "1")
    alpha = c.get("alpha_m", .02 if suffix[1] == "1" or name == "S_enable_weights" else 0.)
    omega = c.get("omega", .9 if network or name == "C10_omega90" else .3)
    sigma = c.get("sigma_o", 1. if network else .5)
    beta = c.get("beta", .5 if name == "C10_beta50" else .05)
    matched = c.get("gasvar_matched", name == "C10_gasvar_matched")
    if c.get("rho_L", 0.) != 0. or c.get("nu_m") is not None or c.get("social_nu") is not None:
        raise ValueError("The frozen study disables model-score decay and all gates")
    world = wm.build_world()
    Pi, h = world["Pi"], world["h"]
    residuals = np.array(world["s"], copy=True)
    if matched:
        residuals[0, 2] = .5448
        reference = LinearGaussianBN(B=world["B"][0], b=world["b"][0],
                                     s=jnp.asarray(residuals[0]), names=world["candidate_node_names"][0]).to_info()
        Pi = Pi.at[0].set(reference.Pi)
        h = h.at[0].set(reference.h)
    W = np.ones((N, N), dtype=np.float64) / N if network else weighted_matrix(beta, N)
    A = (W > 0).astype(np.float64)  # Gate support only; fixed W is never renormalized.
    attention = np.ones((N, 3), dtype=np.float64) if network else wm.wall_attention(N)
    if network:
        attention[4:, 1:] = .01
    phi = np.array([0., 1., 1., 1.])
    cfg = rv.RivalConfig(Pi0=Pi, h0=h, H=world["H"], sigma_o=sigma, omega=omega,
                         alpha_m=alpha, rho_L=0., fuse_nets=fuse, social_nu=None, nu_m=None)
    groups = {"fully_observing": list(range(min(4, N))), "restricted": list(range(min(4, N), N))} if network else {"gas": list(range(N // 2)), "mass": list(range(N // 2, N))}
    covariance = np.linalg.solve(np.asarray(Pi), np.broadcast_to(np.eye(4), np.asarray(Pi).shape))
    metadata = dict(condition=name, id=name, family=family, N=N, K=2, group_indices=groups,
                    omega=omega, alpha_m=alpha, sigma_o=sigma, fuse_nets=fuse,
                    beta=None if network else beta, rho_L=0., gates="disabled", utility="absent",
                    reference_variant="release gas residual .5448" if matched else "original combustion references",
                    Pi0=np.asarray(Pi).tolist(), h0=np.asarray(h).tolist(), H=np.asarray(cfg.H).tolist(),
                    reference_means=np.asarray(world["mu"]).tolist(), reference_covariance=covariance.tolist(),
                    B=np.asarray(world["B"]).tolist(), b=np.asarray(world["b"]).tolist(), residual_variances=residuals.tolist(),
                    W=W.tolist(), effective_conditional_W=(W if fuse else np.eye(N)).tolist(),
                    effective_model_W=((1 - alpha) * np.eye(N) + alpha * W).tolist(),
                    attention=attention.tolist(), phi=phi.tolist(), observable_names=["C", "G", "M"],
                    candidate_names=["release/phlogiston", "oxygen"],
                    coordinate_names=["candidate-specific P/O (not shared)", "C", "G", "M"],
                    training_random="key,sk=split(key); agent_keys=split(sk,N); sample_o per agent",
                    evaluation_random="fold_in(entering_key,314159); normal((N,4,3),float64)",
                    evaluation_logdet="2*sum(log(diag(cholesky(component_predictive_covariance))))")
    return cfg, jnp.asarray(W), jnp.asarray(A), jnp.asarray(attention), jnp.asarray(phi), metadata


def initial_state(cfg, N, seeds):
    values = rv.init_uniform(cfg, N)
    result = {name: np.broadcast_to(np.asarray(value), (len(seeds),) + value.shape).copy()
              for name, value in zip(STATE_KEYS[:3], values)}
    result["key"] = np.stack([np.asarray(jax.random.PRNGKey(int(seed))) for seed in seeds])
    return result


def make_batch(cfg, W, A, attention, draws=4):
    """Create a compiled actual-engine batch with non-state-changing readouts."""
    if cfg.nu_m is not None or cfg.social_nu is not None or cfg.rho_L != 0.:
        raise ValueError("Instrumentation is specified only for ungated, no-score-decay runs")
    N = W.shape[0]

    def one(Pi, h, L, key, phis):
        def step(carry, phi):
            Pi, h, L, key = carry
            pp, hh = rv.fuse_within_frame(Pi, h, W) if cfg.fuse_nets else (Pi, h)
            _, observation_key = jax.random.split(key)
            keys = jax.random.split(observation_key, N)
            obs = jax.vmap(lambda k: sample_o(cfg.H, phi, cfg.sigma_o, k))(keys)
            ell = jax.vmap(lambda p, h, o, w: rv.score_candidates(p, h, cfg.H, o, cfg.sigma_o, w, cfg.jitter))(pp, hh, obs, attention)
            state, _ = rv._step(carry, phi, cfg, W, A, attention, track_means=False)
            pn, hn, ln, _ = state
            q = jax.nn.softmax(ln, axis=-1)
            mu = jnp.linalg.solve(pn, hn[..., None])[..., 0][..., 1:]
            mixture = jnp.einsum("nk,nkd->nd", q, mu)
            selected = jnp.take_along_axis(mu, jnp.argmax(ln, axis=-1)[:, None, None], axis=1)[:, 0, :]
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
            return state, (ln, q, ell, obs, mu, mixture, selected, mse, nll, pn, eval_obs)
        return jax.lax.scan(step, (Pi, h, L, key), phis)

    compiled = jax.jit(jax.vmap(one, in_axes=(0, 0, 0, 0, None)))

    def call(phis, seeds, initial=None):
        entering = initial_state(cfg, N, seeds) if initial is None else initial
        if set(STATE_KEYS) - set(entering):
            raise ValueError("Restart requires exact Pi,h,L,key, including the random state")
        final, out = compiled(*(jnp.asarray(entering[k]) for k in STATE_KEYS), jnp.asarray(phis))
        jax.block_until_ready(out)
        result = {name: np.asarray(value) for name, value in zip(HISTORY_KEYS + ("Pi_history", "eval_obs"), out)}
        result.update({"initial_" + k: np.asarray(entering[k]) for k in STATE_KEYS})
        result.update({"final_" + k: np.asarray(value) for k, value in zip(STATE_KEYS, final)})
        if not all(np.isfinite(value).all() for value in result.values()):
            raise FloatingPointError("Nonfinite output from actual engine or predictive evaluation")
        if not np.allclose(result["q"].sum(-1), 1., rtol=0., atol=5e-15):
            raise AssertionError("Invalid candidate probability normalization")
        previous = np.concatenate((result["initial_L"][:, None], result["L"][:, :-1]), axis=1)
        pooled = (1 - cfg.alpha_m) * previous + cfg.alpha_m * np.einsum("ij,stjk->stik", np.asarray(W), previous)
        residual = result["L"] - pooled - result["ell"]
        if not np.allclose(result["L"], pooled + result["ell"], rtol=1e-11, atol=1e-8):
            raise AssertionError(f"Direct-score recurrence failed: {np.max(np.abs(residual))}")
        return result
    return call


def batch(cfg, W, A, attention, phis, seeds, initial=None, draws=4):
    return make_batch(cfg, W, A, attention, draws)(phis, seeds, initial)


def final_state(result):
    return {k: result["final_" + k] for k in STATE_KEYS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=HERE / "protocol.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    p = json.loads(args.protocol.read_text())
    out = args.output or HERE / p.get("output", "runs/confirmation-2026-09-20")
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    dump(out / "protocol.json", p)
    source = out / "source"
    source.mkdir()
    for path in HERE.glob("*.py"):
        shutil.copy2(path, source / path.name)
    engine_hashes = {str(f.relative_to(ENGINE)): sha(f) for f in ENGINE.rglob("*.py")}
    dump(out / "engine_hashes.json", engine_hashes)
    dump(out / "environment.json", dict(python=sys.version, platform=platform.platform(),
                                        backend=jax.default_backend(), jax_enable_x64=jax.config.jax_enable_x64,
                                        jax_default_prng_impl=str(jax.config.jax_default_prng_impl),
                                        jax_devices=[str(device) for device in jax.devices()],
                                        engine_file=str(Path(rv.__file__).resolve()),
                                        world_sha256=sha(WORLD_SOURCE), runner_sha256=sha(__file__),
                                        packages={name: importlib.metadata.version(name) for name in
                                                  ["jax", "jaxlib", "numpy", "scipy", "equinox", "networkx", "matplotlib"]}))
    print("PROTOCOL_SHA256 " + sha(args.protocol), flush=True)
    manifest, summaries, restart_checks = [], {}, []
    lock = Lock()
    chunk_steps, batch_seeds = p.get("chunk_steps", 250), p.get("batch_seeds", 5)
    draws, branch_start = p.get("heldout_draws", 4), p.get("branch_start", 1500)
    network_T, shift, camp_T = p.get("network_T", 600), p.get("network_shift", 80), p.get("camp_T", 5000)
    conditions = p.get("conditions", [{"name": n} for n in ["N00", "N10", "N01", "N11", "C00", "C10", "C01", "C11", "C10_beta50", "C10_omega90", "C10_gasvar_matched"]])
    conditions = [dict(c) if isinstance(c, dict) else {"name": c} for c in conditions]
    by_name = {c.get("name", c.get("id")): c for c in conditions}
    if set(by_name) != {"N00", "N10", "N01", "N11", "C00", "C10", "C01", "C11", "C10_beta50", "C10_omega90", "C10_gasvar_matched"}:
        raise ValueError("The production condition set differs from the required frozen suite")
    if branch_start % chunk_steps:
        raise ValueError("Chunk boundaries must include the fixed branch state")

    def persist(result, name, seeds, start, steps):
        directory = out / name
        directory.mkdir(exist_ok=True)
        precision = result["Pi_history"]
        if not np.array_equal(precision, np.broadcast_to(precision[:1], precision.shape)):
            raise AssertionError("Precision is not deterministic across seeds")
        pfile = directory / f"precision_updates-{start + 1:05d}-{start + steps:05d}.npz"
        if pfile.exists():
            with np.load(pfile) as archived:
                if not np.array_equal(precision[0], archived["Pi_history"]):
                    raise AssertionError("Precision differs between seed batches")
        else:
            np.savez(pfile, Pi_history=precision[0])
        file = directory / f"seeds-{seeds[0]}-{seeds[-1]}_updates-{start + 1:05d}-{start + steps:05d}.npz"
        stored = {k: v for k, v in result.items() if k not in ("Pi_history", "eval_obs")}
        np.savez(file, **stored)
        record = dict(condition=name, file=str(file.relative_to(out)), seeds=seeds,
                      start_step=start, steps=steps, sha256=sha(file),
                      precision_file=str(pfile.relative_to(out)), precision_sha256=sha(pfile),
                      observation_sha256=array_sha(result["obs"]), evaluation_observation_sha256=array_sha(result["eval_obs"]),
                      deterministic_precision_seed_checks=len(seeds),
                      array_shapes={k: list(v.shape) for k, v in stored.items()})
        return record

    def publish(name, records, summary):
        with lock:
            manifest.extend(records)
            summaries[name] = summary
            dump(out / "manifest.partial.json", manifest)
            dump(out / "summary.partial.json", summaries)
        print("CONDITION " + json.dumps(dict(condition=name, elapsed_seconds=round(time.time() - started, 2), **summary)), flush=True)

    def metadata(c, seeds, T, start=0):
        cfg, W, A, att, phi, meta = make_setup(c)
        meta.update(seeds=seeds, T=T, start_step=start, final_state=T, heldout_draws=draws,
                    array_time="entry j is state start_step+j+1 after its observation", observations="all three channels sampled regardless of attention")
        if meta["family"] == "network":
            meta["prehistory"] = dict(rounds=[0, shift - 1], both_channels=True, attention=[1., 1., 1.], phi=[0., 1., 1., -1.], seed_key="PRNGKey(seed)")
            meta["post_change"] = dict(first_round=shift, state_key="PRNGKey(seed+1000)", common_checkpoint=True)
        directory = out / meta["condition"]
        directory.mkdir(exist_ok=True)
        dump(directory / "setup.json", meta)
        print("SETUP " + json.dumps({key: meta[key] for key in
              ("condition", "family", "N", "T", "start_step", "seeds", "fuse_nets", "alpha_m", "omega", "sigma_o", "beta", "reference_variant")}), flush=True)
        return cfg, W, A, att, phi, meta

    pre_records, pre_states, branch_states, c10_records = [], {}, {}, []

    def run_condition(c):
        name = c.get("name", c.get("id"))
        is_network = name.startswith("N")
        is_branch = name == "S_enable_weights"
        seeds = p["network_seeds"] if is_network else p["camp_seeds"]
        start = shift if is_network else branch_start if is_branch else 0
        T = network_T if is_network else camp_T
        cfg, W, A, att, phi, meta = metadata(c, seeds, T, start)
        runner = make_batch(cfg, W, A, att, draws)
        records = [dict(r, condition=name, shared_from="network_prehistory") for r in pre_records] if is_network else []
        if is_branch:
            records = [dict(r, condition=name, shared_from="C10") for r in c10_records if r["start_step"] < branch_start]
        final_rows = []
        for si in range(0, len(seeds), batch_seeds):
            ss = seeds[si:si + batch_seeds]
            state = None
            if is_network or is_branch:
                states = pre_states if is_network else branch_states
                state = {k: np.stack([states[s][k] for s in ss]) for k in STATE_KEYS}
            for t in range(start, T, chunk_steps):
                count = min(chunk_steps, T - t)
                result = runner(np.broadcast_to(np.asarray(phi), (count, 4)), ss, state)
                state = final_state(result)
                record = persist(result, name, ss, t, count)
                records.append(record)
                if name == "C10" and t + count == branch_start:
                    for j, seed in enumerate(ss):
                        branch_states[seed] = {k: v[j].copy() for k, v in state.items()}
                if t + count == T:
                    midpoint = 4 if is_network else len(W) // 2
                    for j, seed in enumerate(ss):
                        final_rows.append(dict(seed=seed, q_oxygen_group1=float(result["q"][j, -1, :midpoint, 1].mean()),
                                               q_oxygen_group2=float(result["q"][j, -1, midpoint:, 1].mean()),
                                               mse=float(result["mse_full"][j, -1].mean()), nll=float(result["nll_full"][j, -1].mean())))
            print("BATCH " + json.dumps(dict(condition=name, seeds=ss, elapsed_seconds=round(time.time() - started, 2))), flush=True)
        if name == "C10":
            c10_records.extend(records)
        summary = dict(seed_count=len(seeds), final_state=T, endpoint_only=True,
                       endpoint_mean_q_group1=float(np.mean([r["q_oxygen_group1"] for r in final_rows])),
                       endpoint_mean_q_group2=float(np.mean([r["q_oxygen_group2"] for r in final_rows])),
                       endpoint_mean_mse=float(np.mean([r["mse"] for r in final_rows])),
                       endpoint_mean_nll=float(np.mean([r["nll"] for r in final_rows])))
        dump(out / name / "endpoint_by_seed.json", final_rows)
        publish(name, records, summary)

    def run_phase(names):
        with ThreadPoolExecutor(max_workers=p.get("condition_workers", 3)) as pool:
            futures = [pool.submit(run_condition, by_name[name]) for name in names]
            for future in futures:
                future.result()

    try:
        # One common prehistory per seed; its final key is archived, then the
        # legacy second-phase key is reset identically across all four arms.
        c = dict(by_name["N11"], fuse_nets=True, alpha_m=.02)
        cfg, W, A, att, phi, meta = make_setup(c)
        runner = make_batch(cfg, W, A, jnp.ones_like(att), draws)
        truth = np.broadcast_to([0., 1., 1., -1.], (shift, 4))
        for si in range(0, len(p["network_seeds"]), batch_seeds):
            ss = p["network_seeds"][si:si + batch_seeds]
            result = runner(truth, ss)
            pre_records.append(persist(result, "network_prehistory", ss, 0, shift))
            state = final_state(result)
            for j, seed in enumerate(ss):
                pre_states[seed] = {k: v[j].copy() for k, v in state.items()}
                pre_states[seed]["key"] = np.asarray(jax.random.PRNGKey(seed + 1000))
        run_phase(["N00", "N10", "N01", "N11"])
        run_phase(["C00", "C10", "C01", "C11", "C10_beta50", "C10_omega90", "C10_gasvar_matched"])
        if len(branch_states) != len(p["camp_seeds"]):
            raise AssertionError("Not every seed has its state-1500 checkpoint")
        # Verify every unchanged continuation from the exact saved checkpoint.
        # This is a numerical equivalence check, not a new independent outcome.
        cfg, W, A, att, phi, meta = make_setup(by_name["C10"])
        runner = make_batch(cfg, W, A, att, draws)
        for si in range(0, len(p["camp_seeds"]), batch_seeds):
            ss = p["camp_seeds"][si:si + batch_seeds]
            state = {k: np.stack([branch_states[s][k] for s in ss]) for k in STATE_KEYS}
            checked = 0
            for record in sorted((r for r in c10_records if r["seeds"] == ss and r["start_step"] >= branch_start), key=lambda r: r["start_step"]):
                count = record["steps"]
                result = runner(np.broadcast_to(np.asarray(phi), (count, 4)), ss, state)
                state = final_state(result)
                with np.load(out / record["file"]) as saved:
                    for key in saved.files:
                        if not np.array_equal(result[key], saved[key]):
                            raise AssertionError(f"Restart not byte-identical: {record['file']} {key}")
                        checked += 1
                with np.load(out / record["precision_file"]) as saved:
                    if not np.array_equal(result["Pi_history"], np.broadcast_to(saved["Pi_history"], result["Pi_history"].shape)):
                        raise AssertionError("Restart precision differs")
                if array_sha(result["eval_obs"]) != record["evaluation_observation_sha256"]:
                    raise AssertionError("Restart evaluation stream differs")
            restart_checks.append(dict(seeds=ss, start_state=branch_start, final_state=camp_T, checked_arrays=checked, byte_identical=True))
            dump(out / "restart_equivalence.json", restart_checks)
        # Alias unchanged data only after the full continuation equality passes.
        alias = dict(by_name["C10"], name="S_keep_private")
        metadata(alias, p["camp_seeds"], camp_T, branch_start)
        publish("S_keep_private", [dict(r, condition="S_keep_private", shared_from="C10") for r in c10_records], dict(summaries["C10"], reused_after_exact_restart_check=True))
        run_condition(dict(by_name["C10"], name="S_enable_weights", alpha_m=.02))
        dump(out / "manifest.json", manifest)
        dump(out / "summary.json", summaries)
        if engine_hashes != {str(f.relative_to(ENGINE)): sha(f) for f in ENGINE.rglob("*.py")}:
            raise AssertionError("Frozen engine files changed during execution")
        dump(out / "completion.json", dict(status="complete", conditions=len(summaries), logical_chunks=len(manifest),
                                           unique_data_files=len({r["file"] for r in manifest}),
                                           elapsed_seconds=time.time() - started, protocol_sha256=sha(args.protocol)))
        print("COMPLETE " + json.dumps(dict(conditions=len(summaries), seconds=time.time() - started)), flush=True)
    except Exception:
        dump(out / "failure.json", dict(traceback=traceback.format_exc(), elapsed_seconds=time.time() - started))
        raise


if __name__ == "__main__":
    main()
