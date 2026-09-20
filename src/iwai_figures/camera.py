"""The original camera figures: cycle, camp, topology, adaptation and static check.

This intentionally preserves the legacy experiment recipes, including differing
camp references/retention and the phase-specific PRNG resets. It is not the
matched Gaussian comparison implemented in gaussian.py.
"""

from __future__ import annotations
from pathlib import Path
import hashlib
import json
import time
import traceback
import numpy as np
from . import engine as rv
from . import world as wm

jax, jnp = rv.jax, rv.jnp
from .simulation import CONFIGS, sha, dump, begin_run, finish_source_check


def batch(cfg, W, A, attention, phis, seeds, engine, *, initial=None, draws=4):
    """Batch rv._step; evaluate independent full-observation predictive draws.

    The reported means put zero in the unobserved slot: P/O have different
    meanings, so MAP/mixture/gate readouts compare only shared C,G,M values.
    Evaluation keys are folded from, but never replace, the engine's RNG state.
    """
    jax, jnp, rv, _ = engine
    n = len(W)
    if initial is None:
        Pi, h, L = rv.init_uniform(cfg, n)
        initial = {
            "Pi": np.broadcast_to(Pi, (len(seeds),) + Pi.shape),
            "h": np.broadcast_to(h, (len(seeds),) + h.shape),
            "L": np.broadcast_to(L, (len(seeds),) + L.shape),
        }

    def one(seed, Pi, h, L):
        def step(carry, phi):
            state, _ = rv._step(carry, phi, cfg, W, A, attention, track_means=False)
            pn, hn, ln, _ = state
            means = jnp.linalg.solve(pn, hn[..., None])[..., 0]
            q = jax.nn.softmax(ln, axis=-1)
            # Never pool the explanatory coordinates across candidate meanings.
            obs_means = means[..., 1:]
            mixture = jnp.einsum("nk,nkd->nd", q, obs_means)
            map_means = jnp.take_along_axis(
                obs_means, jnp.argmax(ln, axis=-1)[:, None, None], axis=1
            )[:, 0, :]
            mu_map = jnp.concatenate((jnp.zeros((n, 1)), map_means), axis=-1)
            mu_mix = jnp.concatenate((jnp.zeros((n, 1)), mixture), axis=-1)
            mse = jnp.mean((mixture - phi[1:]) ** 2, axis=-1)
            cov = jnp.linalg.inv(pn)[..., 1:, 1:] + cfg.sigma_o**2 * jnp.eye(3)
            cf = jnp.linalg.cholesky(cov)
            key_eval = jax.random.fold_in(carry[-1], 314159)
            obs_eval = phi[1:] + cfg.sigma_o * jax.random.normal(
                key_eval, (n, draws, 3), dtype=pn.dtype
            )
            resid = obs_eval[:, :, None, :] - obs_means[:, None, :, :]
            # Solve independently for each candidate; draw axis precedes K.
            quadratic = jnp.einsum(
                "nrki,nkij,nrkj->nrk", resid, jnp.linalg.inv(cov), resid
            )
            logdet = 2 * jnp.log(jnp.diagonal(cf, axis1=-2, axis2=-1)).sum(-1)
            logp = -0.5 * (3 * jnp.log(2 * jnp.pi) + logdet[:, None, :] + quadratic)
            nll = -jax.scipy.special.logsumexp(
                jax.nn.log_softmax(ln)[:, None, :] + logp, axis=-1
            ).mean(-1)
            return state, (ln, mu_map, mu_mix, mse, nll)

        final, out = jax.lax.scan(step, (Pi, h, L, jax.random.PRNGKey(seed)), phis)
        return final[:3], out

    compiled = jax.jit(jax.vmap(one))
    final, out = compiled(
        jnp.asarray(seeds),
        jnp.asarray(initial["Pi"]),
        jnp.asarray(initial["h"]),
        jnp.asarray(initial["L"]),
    )
    jax.block_until_ready(out)
    L = np.asarray(out[0])
    result = {
        "L": L,
        "q": np.asarray(jax.nn.softmax(jnp.asarray(L), axis=-1)),
        "mu_map": np.asarray(out[1]),
        "mu_mixture": np.asarray(out[2]),
        "mse_obs": np.asarray(out[3]),
        "heldout_nll": np.asarray(out[4]),
        "final_Pi": np.asarray(final[0]),
        "final_h": np.asarray(final[1]),
        "final_L": np.asarray(final[2]),
    }
    if not all(np.isfinite(v).all() for v in result.values()):
        raise FloatingPointError("Nonfinite output")
    return result


def state(result):
    return {"Pi": result["final_Pi"], "h": result["final_h"], "L": result["final_L"]}


def concat(first, second):
    return {
        k: second[k]
        if k.startswith("final_")
        else np.concatenate((first[k], second[k]), axis=1)
        for k in first
    }


def crossings(q, threshold, start=0):
    hit = q > threshold
    hit[:, :start] = False
    event = hit.any(axis=1)
    t = np.where(event, hit.argmax(axis=1), -1)
    return {
        "times": t.tolist(),
        "events": int(event.sum()),
        "total": int(event.size),
        "censored": int((~event).sum()),
        "median_among_events": float(np.median(t[event])) if event.any() else None,
    }


def summarize(result, p, ids=None, shift=None):
    ids = np.arange(result["q"].shape[2]) if ids is None else np.asarray(ids)
    T = result["q"].shape[1]
    tail = slice(T - T // 4, T)
    answer = {
        "seeds": int(result["q"].shape[0]),
        "steps": T,
        "final_mse": float(result["mse_obs"][:, -1, ids].mean()),
        "tail_mse": float(result["mse_obs"][:, tail, ids].mean()),
        "tail_heldout_nll": float(result["heldout_nll"][:, tail, ids].mean()),
    }
    if result["q"].shape[-1] > 1:
        q = result["q"][..., 1][:, :, ids]
        mean = q.mean(-1)
        from scipy.special import xlogy

        entropy = -xlogy(result["q"], result["q"]).sum(-1)[:, :, ids].mean(-1)
        answer.update(
            final_mean_probability=float(mean[:, -1].mean()),
            group_first_crossings=crossings(mean, p["threshold"], shift or 0),
            individual_first_crossings=crossings(q, p["threshold"], shift or 0),
            peak_mean_individual_entropy=float(entropy.max(axis=1).mean()),
        )
        if shift is not None:
            answer["pre_shift_probability"] = float(mean[:, shift - 1].mean())
            answer["post_shift_peak_entropy"] = float(
                entropy[:, shift:].max(axis=1).mean()
            )
    return answer


def observable_baselines(world):
    """Marginalize P/O BEFORE constructing K1 priors; keep an unused dummy slot."""
    cov = np.asarray(world["covariance"])[..., 1:, 1:]
    precision = np.linalg.inv(cov)
    P = np.zeros((2, 4, 4))
    P[:, 0, 0] = 4.0
    P[:, 1:, 1:] = precision
    h = np.zeros((2, 4))
    h[:, 1:] = np.einsum("kij,kj->ki", precision, np.asarray(world["mu"])[:, 1:])
    return P, h


def assess(p, world, engine, outdir):
    jax, jnp, rv, wm = engine
    P, h, H = (jnp.asarray(world[k]) for k in ("Pi", "h", "H"))
    baseP, baseh = observable_baselines(world)
    summary = {}
    graphs = {}
    run_files = {}

    def config(**kwargs):
        values = dict(
            Pi0=P,
            h0=h,
            H=H,
            sigma_o=p["sigma_cycle"],
            omega=p["omega_cycle"],
            alpha_m=p["alpha_m"],
            rho_L=p["rho_L"],
        )
        values.update(kwargs)
        return rv.RivalConfig(**values)

    def path(T):
        return jnp.asarray(np.stack([wm.phi(t, p["shift"]) for t in range(T)]))

    def run(cfg, W, A, att, phis, seeds, initial=None):
        return batch(
            cfg,
            jnp.asarray(W),
            jnp.asarray(A),
            jnp.asarray(att),
            jnp.asarray(phis),
            seeds,
            engine,
            initial=initial,
            draws=p["heldout_samples_per_agent_step"],
        )

    def keep(name, result, ids=None, shift=None, details=None):
        file = outdir / (name + ".npz")
        np.savez_compressed(file, **result)
        row = summarize(result, p, ids, shift)
        if details:
            row.update(details)
        summary[name] = row
        run_files[name] = {
            "file": file.name,
            "sha256": sha(file),
            "arrays": {
                k: {
                    "shape": list(v.shape),
                    "dtype": str(v.dtype),
                    "sha256": hashlib.sha256(
                        np.ascontiguousarray(v).tobytes()
                    ).hexdigest(),
                }
                for k, v in result.items()
            },
        }
        dump(outdir / "summary.partial.json", summary)
        compact = {
            k: v
            for k, v in row.items()
            if k not in ("individual_first_crossings", "group_first_crossings")
        }
        if "group_first_crossings" in row:
            compact["group_crossings"] = {
                k: v for k, v in row["group_first_crossings"].items() if k != "times"
            }
        print(name + ": " + json.dumps(compact), flush=True)

    n, T, seeds = p["cycle_N"], p["cycle_T"], p["cycle_seeds"]
    A = np.ones((n, n))
    W = A / n
    att = np.ones((n, 3))
    print("CYCLE: full attention, same paired observations, no utility", flush=True)
    keep(
        "cycle",
        run(config(), W, A, att, path(T), seeds),
        shift=p["shift"],
        details={
            "truth": "mass-only reference mean change at 80",
            "crossing_window": "post-shift only, stored post-update indices",
        },
    )

    print(
        "NETWORK: original graphs; branch controls from identical pre-shift states",
        flush=True,
    )
    archive = np.load(CONFIGS / "graphs.npz")
    n, T, seeds, ts = p["network_N"], p["network_T"], p["network_seeds"], p["shift"]
    sources = np.asarray(p["informed_nodes"])
    restricted = np.setdiff1d(np.arange(n), sources)
    att0 = np.ones((n, 3))
    att1 = att0.copy()
    att1[np.ix_(restricted, [1, 2])] = p["restricted_diagnostic_weight"]
    for name in p["network_graphs"]:
        A = archive["A_" + name]
        As = A + np.eye(n)
        W = As / As.sum(1, keepdims=True)
        # Retain the original identity-indexed copies used by the producer.
        ids_all = np.arange(n)
        Wr = W[np.ix_(ids_all, ids_all)]
        Ar = As[np.ix_(ids_all, ids_all)]
        prehistory = run(config(), Wr, Ar, att0, path(T)[:ts], seeds)
        cfg = config(alpha_m=p["alpha_m"], fuse_nets=True)
        after = att1.copy()
        second = run(
            cfg,
            Wr,
            Ar,
            after,
            path(T)[ts:],
            [s + 1000 for s in seeds],
            state(prehistory),
        )
        result = concat(prehistory, second)
        graphs.update({"A_" + name: A, "W_" + name: W, "informed_" + name: sources})
        keep(
            "network_" + name,
            result,
            restricted,
            ts,
            {
                "gas_mass_attention": p["restricted_diagnostic_weight"],
                "intervention": "post-shift attention change; original phase-specific RNG reset",
                "crossing_window": "post-shift only, stored post-update indices",
            },
        )
    np.savez_compressed(outdir / "graphs.npz", **graphs)

    print(
        "SELECTIVE EVIDENCE: oxygen-reference world, observe G vs M; no imposed opposing signs",
        flush=True,
    )
    n, T, seeds = p["wall_N"], p["wall_T"], p["wall_seeds"]
    A = np.ones((n, n))
    W = A / n
    camps = (np.arange(n) >= n // 2).astype(int)
    campP, camph = baseP[camps, None], baseh[camps, None]
    attention = wm.wall_attention(n)
    truth = jnp.broadcast_to(jnp.asarray(wm.wall_truth()), (T, 4))

    def wall_details(result, alpha, condition):
        extra = {"condition": condition}
        tail = slice(-(result["q"].shape[1] // 4), None)
        for readout in ("mu_map", "mu_mixture"):
            difference = result[readout][:, :, : n // 2, 1:].mean(2) - result[readout][
                :, :, n // 2 :, 1:
            ].mean(2)
            distance = np.linalg.norm(difference, axis=-1)
            result["D_" + readout] = distance
            extra[readout] = {
                "final_mean_distance": float(distance[:, -1].mean()),
                "tail_mean_distance": float(distance[:, tail].mean()),
            }
        if result["q"].shape[-1] > 1:
            odds = result["L"][..., 1] - result["L"][..., 0]
            initial = np.zeros_like(odds[:, :1])
            previous = np.concatenate((initial, odds[:, :-1]), axis=1)
            source = (
                odds - (1 - alpha) * previous - alpha * previous.mean(-1, keepdims=True)
            )
            gap = odds[:, :, : n // 2].mean(-1) - odds[:, :, n // 2 :].mean(-1)
            result["log_odds_gap"] = gap
            result["mean_source"] = source.mean(-1)
            extra["model_odds"] = {
                "tail_gap_mean": float(gap[:, tail].mean()),
                "tail_global_source_mean": float(source[:, tail].mean()),
                "tail_source_by_camp": [
                    float(source[:, tail, : n // 2].mean()),
                    float(source[:, tail, n // 2 :].mean()),
                ],
                "final_oxygen_probability_by_camp": [
                    float(result["q"][:, -1, : n // 2, 1].mean()),
                    float(result["q"][:, -1, n // 2 :, 1].mean()),
                ],
            }
        return extra

    for cond in p["wall_conditions"]:
        opts = dict(
            sigma_o=p["sigma_wall"],
            omega=cond["omega"],
            social_nu=p["social_nu"] if cond["gate"] else None,
            gate_idx=(1, 2, 3),
            alpha_m=cond.get("alpha_m", p["alpha_m"]),
            fuse_nets=cond.get("fuse_nets", True),
        )
        initial = None
        if cond["K"] == 1:
            opts.update(
                Pi0=jnp.asarray(baseP.mean(0)[None]),
                h0=jnp.asarray(baseh.mean(0)[None]),
            )
            if cond["anchor"] == "camp":
                opts.update(Pi0_agent=jnp.asarray(campP), h0_agent=jnp.asarray(camph))
            initial = {
                "Pi": np.broadcast_to(campP, (len(seeds),) + campP.shape),
                "h": np.broadcast_to(camph, (len(seeds),) + camph.shape),
                "L": np.zeros((len(seeds), n, 1)),
            }
        result = run(config(**opts), W, A, attention, truth, seeds, initial)
        extra = wall_details(result, opts["alpha_m"], cond)
        extra["truth"] = (
            "oxygen reference [C,G,M]=[1,1,1]; gas-only evidence not designed to oppose mass evidence"
        )
        keep("wall_" + cond["name"], result, details=extra)
    print(
        "ADAPTATION: same prehistory; candidate-consistent post-shift signals",
        flush=True,
    )
    T, ts = p["adaptation_T"], p["shift"]
    cfg = config(sigma_o=p["adaptation_sigma"], omega=p["adaptation_omega"])
    prehistory = run(cfg, W, A, np.ones((n, 3)), path(T)[:ts], seeds)
    for name, cfg, att in [
        ("social", cfg, attention),
        (
            "isolated",
            config(
                sigma_o=p["adaptation_sigma"],
                omega=p["adaptation_omega"],
                alpha_m=0.0,
                fuse_nets=False,
            ),
            attention,
        ),
        (
            "reference",
            config(
                sigma_o=p["adaptation_sigma"],
                omega=p["adaptation_omega"],
                alpha_m=0.0,
                fuse_nets=False,
            ),
            np.ones((n, 3)),
        ),
    ]:
        result = concat(
            prehistory,
            run(
                cfg,
                W,
                A,
                att,
                path(T)[ts:],
                [s + 1000 for s in seeds],
                state(prehistory),
            ),
        )
        details = {
            "reference_type": "independent full-observation agent"
            if name == "reference"
            else "G-only or M-only after shift",
            "intervention": "post-shift only",
            "final_oxygen_probability_by_camp": [
                float(result["q"][:, -1, : n // 2, 1].mean()),
                float(result["q"][:, -1, n // 2 :, 1].mean()),
            ],
        }
        keep("adaptation_" + name, result, shift=ts, details=details)
    return summary, run_files


def run(output, protocol=None):
    """Reproduce every empirical figure from the first supplied camera source."""
    protocol_path = Path(protocol) if protocol is not None else CONFIGS / "camera.json"
    p = json.loads(protocol_path.read_text())
    started = time.perf_counter()
    out, source_hashes = begin_run(output, p, "camera")
    world = wm.build_world()
    dump(out / "world.json", world["metadata"])
    np.savez_compressed(
        out / "world.npz",
        **{
            k: np.asarray(v)
            for k, v in world.items()
            if isinstance(v, (np.ndarray, jax.Array))
        },
    )
    try:
        summary, manifest = assess(p, world, (jax, jnp, rv, wm), out)
        from .structural_check import run_check

        summary["structural_check"] = run_check(world, p, out)
        summary.update(
            status="complete",
            failed_runs=[],
            elapsed_seconds=time.perf_counter() - started,
        )
        dump(out / "summary.json", summary)
        dump(out / "results_manifest.json", manifest)
        finish_source_check(out, source_hashes)
        dump(
            out / "completion.json",
            dict(
                status="complete",
                conditions=len(manifest),
                structural_check=True,
                elapsed_seconds=summary["elapsed_seconds"],
                protocol_sha256=sha(protocol_path),
            ),
        )
        print(
            "COMPLETE "
            + json.dumps(
                dict(conditions=len(manifest), seconds=summary["elapsed_seconds"])
            ),
            flush=True,
        )
    except Exception:
        dump(
            out / "failure.json",
            dict(
                traceback=traceback.format_exc(),
                elapsed_seconds=time.perf_counter() - started,
            ),
        )
        raise
    return out
