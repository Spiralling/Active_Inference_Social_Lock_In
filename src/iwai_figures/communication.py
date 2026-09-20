"""Communication-channel factorial and paired support-sharing intervention."""

from __future__ import annotations
from pathlib import Path
import json
import time
import traceback
import numpy as np
from . import engine as rv
from . import world as wm

jax, jnp = rv.jax, rv.jnp
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from .simulation import (
    STATE_KEYS,
    CONFIGS,
    sha,
    array_sha,
    dump,
    weighted_matrix,
    make_batch,
    final_state,
    begin_run,
    finish_source_check,
)


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
    alpha = c.get(
        "alpha_m", 0.02 if suffix[1] == "1" or name == "S_enable_weights" else 0.0
    )
    omega = c.get("omega", 0.9 if network or name == "C10_omega90" else 0.3)
    sigma = c.get("sigma_o", 1.0 if network else 0.5)
    beta = c.get("beta", 0.5 if name == "C10_beta50" else 0.05)
    matched = c.get("gasvar_matched", name == "C10_gasvar_matched")
    if (
        c.get("rho_L", 0.0) != 0.0
        or c.get("nu_m") is not None
        or c.get("social_nu") is not None
    ):
        raise ValueError("The frozen study disables model-score decay and all gates")
    world = wm.build_world()
    Pi, h = world["Pi"], world["h"]
    residuals = np.array(world["s"], copy=True)
    if matched:
        residuals[0, 2] = 0.5448
        reference_pi, reference_h = rv.to_info(
            world["B"][0], world["b"][0], jnp.asarray(residuals[0])
        )
        Pi = Pi.at[0].set(reference_pi)
        h = h.at[0].set(reference_h)
    W = np.ones((N, N), dtype=np.float64) / N if network else weighted_matrix(beta, N)
    A = (W > 0).astype(np.float64)  # Gate support only; fixed W is never renormalized.
    attention = np.ones((N, 3), dtype=np.float64) if network else wm.wall_attention(N)
    if network:
        attention[4:, 1:] = 0.01
    phi = np.array([0.0, 1.0, 1.0, 1.0])
    cfg = rv.RivalConfig(
        Pi0=Pi,
        h0=h,
        H=world["H"],
        sigma_o=sigma,
        omega=omega,
        alpha_m=alpha,
        rho_L=0.0,
        fuse_nets=fuse,
        social_nu=None,
        nu_m=None,
    )
    groups = (
        {
            "fully_observing": list(range(min(4, N))),
            "restricted": list(range(min(4, N), N)),
        }
        if network
        else {"gas": list(range(N // 2)), "mass": list(range(N // 2, N))}
    )
    covariance = np.linalg.solve(
        np.asarray(Pi), np.broadcast_to(np.eye(4), np.asarray(Pi).shape)
    )
    metadata = dict(
        condition=name,
        id=name,
        family=family,
        N=N,
        K=2,
        group_indices=groups,
        omega=omega,
        alpha_m=alpha,
        sigma_o=sigma,
        fuse_nets=fuse,
        beta=None if network else beta,
        rho_L=0.0,
        gates="disabled",
        utility="absent",
        reference_variant="release gas residual .5448"
        if matched
        else "original combustion references",
        Pi0=np.asarray(Pi).tolist(),
        h0=np.asarray(h).tolist(),
        H=np.asarray(cfg.H).tolist(),
        reference_means=np.asarray(world["mu"]).tolist(),
        reference_covariance=covariance.tolist(),
        B=np.asarray(world["B"]).tolist(),
        b=np.asarray(world["b"]).tolist(),
        residual_variances=residuals.tolist(),
        W=W.tolist(),
        effective_conditional_W=(W if fuse else np.eye(N)).tolist(),
        effective_model_W=((1 - alpha) * np.eye(N) + alpha * W).tolist(),
        attention=attention.tolist(),
        phi=phi.tolist(),
        observable_names=["C", "G", "M"],
        candidate_names=["release/phlogiston", "oxygen"],
        coordinate_names=["candidate-specific P/O (not shared)", "C", "G", "M"],
        training_random="key,sk=split(key); agent_keys=split(sk,N); sample_o per agent",
        evaluation_random="fold_in(entering_key,314159); normal((N,4,3),float64)",
        evaluation_logdet="2*sum(log(diag(cholesky(component_predictive_covariance))))",
    )
    return (
        cfg,
        jnp.asarray(W),
        jnp.asarray(A),
        jnp.asarray(attention),
        jnp.asarray(phi),
        metadata,
    )


def run(output, protocol=None):
    """Run all required factorial arms and both exact-checkpoint continuations."""
    protocol_path = (
        Path(protocol) if protocol is not None else CONFIGS / "communication.json"
    )
    p = json.loads(protocol_path.read_text())
    started = time.time()
    out, source_hashes = begin_run(output, p, "communication")
    print("PROTOCOL_SHA256 " + sha(protocol_path), flush=True)
    manifest, summaries, restart_checks = [], {}, []
    lock = Lock()
    chunk_steps, batch_seeds = p.get("chunk_steps", 250), p.get("batch_seeds", 5)
    draws, branch_start = p.get("heldout_draws", 4), p.get("branch_start", 1500)
    network_T, shift, camp_T = (
        p.get("network_T", 600),
        p.get("network_shift", 80),
        p.get("camp_T", 5000),
    )
    conditions = p.get(
        "conditions",
        [
            {"name": n}
            for n in [
                "N00",
                "N10",
                "N01",
                "N11",
                "C00",
                "C10",
                "C01",
                "C11",
                "C10_beta50",
                "C10_omega90",
                "C10_gasvar_matched",
            ]
        ],
    )
    conditions = [dict(c) if isinstance(c, dict) else {"name": c} for c in conditions]
    by_name = {c.get("name", c.get("id")): c for c in conditions}
    if set(by_name) != {
        "N00",
        "N10",
        "N01",
        "N11",
        "C00",
        "C10",
        "C01",
        "C11",
        "C10_beta50",
        "C10_omega90",
        "C10_gasvar_matched",
    }:
        raise ValueError(
            "The production condition set differs from the required frozen suite"
        )
    if branch_start % chunk_steps:
        raise ValueError("Chunk boundaries must include the fixed branch state")

    def persist(result, name, seeds, start, steps):
        directory = out / name
        directory.mkdir(exist_ok=True)
        precision = result["Pi_history"]
        if not np.array_equal(
            precision, np.broadcast_to(precision[:1], precision.shape)
        ):
            raise AssertionError("Precision is not deterministic across seeds")
        pfile = directory / f"precision_updates-{start + 1:05d}-{start + steps:05d}.npz"
        if pfile.exists():
            with np.load(pfile) as archived:
                if not np.array_equal(precision[0], archived["Pi_history"]):
                    raise AssertionError("Precision differs between seed batches")
        else:
            np.savez(pfile, Pi_history=precision[0])
        file = (
            directory
            / f"seeds-{seeds[0]}-{seeds[-1]}_updates-{start + 1:05d}-{start + steps:05d}.npz"
        )
        stored = {
            k: v for k, v in result.items() if k not in ("Pi_history", "eval_obs")
        }
        np.savez(file, **stored)
        record = dict(
            condition=name,
            file=str(file.relative_to(out)),
            seeds=seeds,
            start_step=start,
            steps=steps,
            sha256=sha(file),
            precision_file=str(pfile.relative_to(out)),
            precision_sha256=sha(pfile),
            observation_sha256=array_sha(result["obs"]),
            evaluation_observation_sha256=array_sha(result["eval_obs"]),
            deterministic_precision_seed_checks=len(seeds),
            array_shapes={k: list(v.shape) for k, v in stored.items()},
        )
        return record

    def publish(name, records, summary):
        with lock:
            manifest.extend(records)
            summaries[name] = summary
            dump(out / "manifest.partial.json", manifest)
            dump(out / "summary.partial.json", summaries)
        print(
            "CONDITION "
            + json.dumps(
                dict(
                    condition=name,
                    elapsed_seconds=round(time.time() - started, 2),
                    **summary,
                )
            ),
            flush=True,
        )

    def metadata(c, seeds, T, start=0):
        cfg, W, A, att, phi, meta = make_setup(c)
        meta.update(
            seeds=seeds,
            T=T,
            start_step=start,
            final_state=T,
            heldout_draws=draws,
            array_time="entry j is state start_step+j+1 after its observation",
            observations="all three channels sampled regardless of attention",
        )
        if meta["family"] == "network":
            meta["prehistory"] = dict(
                rounds=[0, shift - 1],
                both_channels=True,
                attention=[1.0, 1.0, 1.0],
                phi=[0.0, 1.0, 1.0, -1.0],
                seed_key="PRNGKey(seed)",
            )
            meta["post_change"] = dict(
                first_round=shift,
                state_key="PRNGKey(seed+1000)",
                common_checkpoint=True,
            )
        directory = out / meta["condition"]
        directory.mkdir(exist_ok=True)
        dump(directory / "setup.json", meta)
        print(
            "SETUP "
            + json.dumps(
                {
                    key: meta[key]
                    for key in (
                        "condition",
                        "family",
                        "N",
                        "T",
                        "start_step",
                        "seeds",
                        "fuse_nets",
                        "alpha_m",
                        "omega",
                        "sigma_o",
                        "beta",
                        "reference_variant",
                    )
                }
            ),
            flush=True,
        )
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
        records = (
            [
                dict(r, condition=name, shared_from="network_prehistory")
                for r in pre_records
            ]
            if is_network
            else []
        )
        if is_branch:
            records = [
                dict(r, condition=name, shared_from="C10")
                for r in c10_records
                if r["start_step"] < branch_start
            ]
        final_rows = []
        for si in range(0, len(seeds), batch_seeds):
            ss = seeds[si : si + batch_seeds]
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
                        final_rows.append(
                            dict(
                                seed=seed,
                                q_oxygen_group1=float(
                                    result["q"][j, -1, :midpoint, 1].mean()
                                ),
                                q_oxygen_group2=float(
                                    result["q"][j, -1, midpoint:, 1].mean()
                                ),
                                mse=float(result["mse_full"][j, -1].mean()),
                                nll=float(result["nll_full"][j, -1].mean()),
                            )
                        )
            print(
                "BATCH "
                + json.dumps(
                    dict(
                        condition=name,
                        seeds=ss,
                        elapsed_seconds=round(time.time() - started, 2),
                    )
                ),
                flush=True,
            )
        if name == "C10":
            c10_records.extend(records)
        summary = dict(
            seed_count=len(seeds),
            final_state=T,
            endpoint_only=True,
            endpoint_mean_q_group1=float(
                np.mean([r["q_oxygen_group1"] for r in final_rows])
            ),
            endpoint_mean_q_group2=float(
                np.mean([r["q_oxygen_group2"] for r in final_rows])
            ),
            endpoint_mean_mse=float(np.mean([r["mse"] for r in final_rows])),
            endpoint_mean_nll=float(np.mean([r["nll"] for r in final_rows])),
        )
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
        c = dict(by_name["N11"], fuse_nets=True, alpha_m=0.02)
        cfg, W, A, att, phi, meta = make_setup(c)
        runner = make_batch(cfg, W, A, jnp.ones_like(att), draws)
        truth = np.broadcast_to([0.0, 1.0, 1.0, -1.0], (shift, 4))
        for si in range(0, len(p["network_seeds"]), batch_seeds):
            ss = p["network_seeds"][si : si + batch_seeds]
            result = runner(truth, ss)
            pre_records.append(persist(result, "network_prehistory", ss, 0, shift))
            state = final_state(result)
            for j, seed in enumerate(ss):
                pre_states[seed] = {k: v[j].copy() for k, v in state.items()}
                pre_states[seed]["key"] = np.asarray(jax.random.PRNGKey(seed + 1000))
        run_phase(["N00", "N10", "N01", "N11"])
        run_phase(
            [
                "C00",
                "C10",
                "C01",
                "C11",
                "C10_beta50",
                "C10_omega90",
                "C10_gasvar_matched",
            ]
        )
        if len(branch_states) != len(p["camp_seeds"]):
            raise AssertionError("Not every seed has its state-1500 checkpoint")
        # Verify every unchanged continuation from the exact saved checkpoint.
        # This is a numerical equivalence check, not a new independent outcome.
        cfg, W, A, att, phi, meta = make_setup(by_name["C10"])
        runner = make_batch(cfg, W, A, att, draws)
        for si in range(0, len(p["camp_seeds"]), batch_seeds):
            ss = p["camp_seeds"][si : si + batch_seeds]
            state = {k: np.stack([branch_states[s][k] for s in ss]) for k in STATE_KEYS}
            checked = 0
            for record in sorted(
                (
                    r
                    for r in c10_records
                    if r["seeds"] == ss and r["start_step"] >= branch_start
                ),
                key=lambda r: r["start_step"],
            ):
                count = record["steps"]
                result = runner(np.broadcast_to(np.asarray(phi), (count, 4)), ss, state)
                state = final_state(result)
                with np.load(out / record["file"]) as saved:
                    for key in saved.files:
                        if not np.array_equal(result[key], saved[key]):
                            raise AssertionError(
                                f"Restart not byte-identical: {record['file']} {key}"
                            )
                        checked += 1
                with np.load(out / record["precision_file"]) as saved:
                    if not np.array_equal(
                        result["Pi_history"],
                        np.broadcast_to(
                            saved["Pi_history"], result["Pi_history"].shape
                        ),
                    ):
                        raise AssertionError("Restart precision differs")
                if (
                    array_sha(result["eval_obs"])
                    != record["evaluation_observation_sha256"]
                ):
                    raise AssertionError("Restart evaluation stream differs")
            restart_checks.append(
                dict(
                    seeds=ss,
                    start_state=branch_start,
                    final_state=camp_T,
                    checked_arrays=checked,
                    byte_identical=True,
                )
            )
            dump(out / "restart_equivalence.json", restart_checks)
        # Alias unchanged data only after the full continuation equality passes.
        alias = dict(by_name["C10"], name="S_keep_private")
        metadata(alias, p["camp_seeds"], camp_T, branch_start)
        publish(
            "S_keep_private",
            [
                dict(r, condition="S_keep_private", shared_from="C10")
                for r in c10_records
            ],
            dict(summaries["C10"], reused_after_exact_restart_check=True),
        )
        run_condition(dict(by_name["C10"], name="S_enable_weights", alpha_m=0.02))
        dump(out / "manifest.json", manifest)
        dump(out / "summary.json", summaries)
        finish_source_check(out, source_hashes)
        dump(
            out / "completion.json",
            dict(
                status="complete",
                conditions=len(summaries),
                logical_chunks=len(manifest),
                unique_data_files=len({r["file"] for r in manifest}),
                elapsed_seconds=time.time() - started,
                protocol_sha256=sha(protocol_path),
            ),
        )
        print(
            "COMPLETE "
            + json.dumps(
                dict(conditions=len(summaries), seconds=time.time() - started)
            ),
            flush=True,
        )
    except Exception:
        dump(
            out / "failure.json",
            dict(
                traceback=traceback.format_exc(), elapsed_seconds=time.time() - started
            ),
        )
        raise
    return out
