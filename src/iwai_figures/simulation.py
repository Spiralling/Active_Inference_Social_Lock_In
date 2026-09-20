"""Checkpointed rollouts and lossless scientific output helpers.

All states are post-update. Evaluation uses folded keys without consuming the
training stream. The separate camera wrapper preserves its historical resets.
"""

from __future__ import annotations
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
import numpy as np
from . import engine as rv
from . import world as wm

jax, jnp = rv.jax, rv.jnp
from .engine import sample_o

STATE_KEYS = ("Pi", "h", "L", "key")
HISTORY_KEYS = (
    "L",
    "q",
    "ell",
    "obs",
    "mu_conditional",
    "mu_mixture",
    "mu_map",
    "mse_full",
    "nll_full",
)
CONFIGS = Path(__file__).resolve().parents[2] / "configs"


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


def initial_state(cfg, N, seeds):
    values = rv.init_uniform(cfg, N)
    result = {
        name: np.broadcast_to(np.asarray(value), (len(seeds),) + value.shape).copy()
        for name, value in zip(STATE_KEYS[:3], values)
    }
    result["key"] = np.stack(
        [np.asarray(jax.random.PRNGKey(int(seed))) for seed in seeds]
    )
    return result


def make_batch(cfg, W, A, attention, draws=4):
    """Create a compiled actual-engine batch with non-state-changing readouts."""
    if cfg.nu_m is not None or cfg.social_nu is not None or cfg.rho_L != 0.0:
        raise ValueError(
            "Instrumentation is specified only for ungated, no-score-decay runs"
        )
    N = W.shape[0]

    def one(Pi, h, L, key, phis):
        def step(carry, phi):
            Pi, h, L, key = carry
            pp, hh = rv.fuse_within_frame(Pi, h, W) if cfg.fuse_nets else (Pi, h)
            _, observation_key = jax.random.split(key)
            keys = jax.random.split(observation_key, N)
            obs = jax.vmap(lambda k: sample_o(cfg.H, phi, cfg.sigma_o, k))(keys)
            ell = jax.vmap(
                lambda p, h, o, w: rv.score_candidates(
                    p, h, cfg.H, o, cfg.sigma_o, w, cfg.jitter
                )
            )(pp, hh, obs, attention)
            state, _ = rv._step(carry, phi, cfg, W, A, attention, track_means=False)
            pn, hn, ln, _ = state
            q = jax.nn.softmax(ln, axis=-1)
            mu = jnp.linalg.solve(pn, hn[..., None])[..., 0][..., 1:]
            mixture = jnp.einsum("nk,nkd->nd", q, mu)
            selected = jnp.take_along_axis(
                mu, jnp.argmax(ln, axis=-1)[:, None, None], axis=1
            )[:, 0, :]
            mse = jnp.mean((mixture - phi[1:]) ** 2, axis=-1)
            cov = jnp.linalg.inv(pn)[..., 1:, 1:] + cfg.sigma_o**2 * jnp.eye(3)
            cf = jnp.linalg.cholesky(cov)
            eval_key = jax.random.fold_in(key, 314159)
            eval_obs = phi[1:] + cfg.sigma_o * jax.random.normal(
                eval_key, (N, draws, 3), dtype=pn.dtype
            )
            residual = eval_obs[:, :, None, :] - mu[:, None, :, :]
            quadratic = jnp.einsum(
                "nrki,nkij,nrkj->nrk", residual, jnp.linalg.inv(cov), residual
            )
            logdet = 2 * jnp.log(jnp.diagonal(cf, axis1=-2, axis2=-1)).sum(-1)
            logp = -0.5 * (3 * jnp.log(2 * jnp.pi) + logdet[:, None, :] + quadratic)
            nll = -jax.scipy.special.logsumexp(
                jax.nn.log_softmax(ln)[:, None, :] + logp, axis=-1
            ).mean(-1)
            return state, (
                ln,
                q,
                ell,
                obs,
                mu,
                mixture,
                selected,
                mse,
                nll,
                pn,
                eval_obs,
            )

        return jax.lax.scan(step, (Pi, h, L, key), phis)

    compiled = jax.jit(jax.vmap(one, in_axes=(0, 0, 0, 0, None)))

    def call(phis, seeds, initial=None):
        entering = initial_state(cfg, N, seeds) if initial is None else initial
        if set(STATE_KEYS) - set(entering):
            raise ValueError(
                "Restart requires exact Pi,h,L,key, including the random state"
            )
        final, out = compiled(
            *(jnp.asarray(entering[k]) for k in STATE_KEYS), jnp.asarray(phis)
        )
        jax.block_until_ready(out)
        result = {
            name: np.asarray(value)
            for name, value in zip(HISTORY_KEYS + ("Pi_history", "eval_obs"), out)
        }
        result.update({"initial_" + k: np.asarray(entering[k]) for k in STATE_KEYS})
        result.update(
            {"final_" + k: np.asarray(value) for k, value in zip(STATE_KEYS, final)}
        )
        if not all(np.isfinite(value).all() for value in result.values()):
            raise FloatingPointError(
                "Nonfinite output from actual engine or predictive evaluation"
            )
        if not np.allclose(result["q"].sum(-1), 1.0, rtol=0.0, atol=5e-15):
            raise AssertionError("Invalid candidate probability normalization")
        previous = np.concatenate(
            (result["initial_L"][:, None], result["L"][:, :-1]), axis=1
        )
        pooled = (1 - cfg.alpha_m) * previous + cfg.alpha_m * np.einsum(
            "ij,stjk->stik", np.asarray(W), previous
        )
        residual = result["L"] - pooled - result["ell"]
        if not np.allclose(result["L"], pooled + result["ell"], rtol=1e-11, atol=1e-8):
            raise AssertionError(
                f"Direct-score recurrence failed: {np.max(np.abs(residual))}"
            )
        return result

    return call


def batch(cfg, W, A, attention, phis, seeds, initial=None, draws=4):
    return make_batch(cfg, W, A, attention, draws)(phis, seeds, initial)


def final_state(result):
    return {k: result["final_" + k] for k in STATE_KEYS}


def begin_run(output, protocol, family):
    """Snapshot the local scientific sources and actual execution environment."""
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    dump(out / "protocol.json", protocol)
    names = ["engine.py", "world.py", "simulation.py", family + ".py"]
    if family == "camera":
        names.append("structural_check.py")
    here = Path(__file__).resolve().parent
    source = out / "source"
    source.mkdir()
    hashes = {}
    for name in names:
        shutil.copy2(here / name, source / name)
        hashes[name] = sha(here / name)
    dump(out / "source_hashes.json", hashes)
    dump(out / "engine_hashes.json", {"engine.py": sha(here / "engine.py")})
    dump(
        out / "environment.json",
        dict(
            python=sys.version,
            platform=platform.platform(),
            backend=jax.default_backend(),
            jax_enable_x64=jax.config.jax_enable_x64,
            jax_default_prng_impl=str(jax.config.jax_default_prng_impl),
            jax_devices=[str(d) for d in jax.devices()],
            engine_file=str(Path(rv.__file__).resolve()),
            world_sha256=sha(wm.__file__),
            runner_sha256=sha(here / (family + ".py")),
            packages={
                n: importlib.metadata.version(n)
                for n in ("jax", "jaxlib", "numpy", "scipy")
            },
        ),
    )
    return out, hashes


def finish_source_check(out, hashes):
    here = Path(__file__).resolve().parent
    if hashes != {name: sha(here / name) for name in hashes}:
        raise RuntimeError("Scientific source changed during the run")
