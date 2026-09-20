# Production runner and numerical output

`run.py` imports the versioned `paper/four_variable_experiments/engine/src`
package and the unchanged `paper/combustion_experiments/world.py`. It checks the
actual imported engine path. Every condition and round calls `rival._step`;
scores are evaluated again as readouts from the identical post-sharing,
pre-learning state and identical samples. No stored likelihood is replayed to
produce a trajectory. Float64 is enabled before constructing any references.

The ordinary entry point is `python paper/communication_experiments/run.py`.
It reads the adjacent frozen `protocol.json` and writes
`runs/confirmation-2026-09-20`; it refuses to reuse an existing output directory.
`--protocol PATH` and `--output PATH` allow an explicitly specified portable
reproduction without overwriting results. A different scientific protocol must
be identified as a separate experiment.

## Callable interface

- `make_setup(condition, N=None)` returns `(cfg,W,A,attention,phi,metadata)`.
  `condition` is a condition name or a settings dictionary. `phi` is the constant
  oxygen-reference signal; the network runner constructs the separate common
  prehistory. `N` overrides the population only for implementation tests.
- `make_batch(cfg,W,A,attention,draws=4)` returns a compiled callable
  `call(phis,seeds,initial=None)`. `phis` has shape `(rounds,4)` and contains the
  full storage signal, with the hidden padding unobserved.
- `batch(cfg,W,A,attention,phis,seeds,initial=None,draws=4)` is a convenience
  wrapper. A restart requires all four entries `Pi,h,L,key`.
- `initial_state(cfg,N,seeds)` initializes references and uniform model weights;
  `final_state(result)` extracts a restartable state from a batch result.
- `weighted_matrix(beta,N=16)` constructs the weighted block matrix directly.

## Manifest and time conventions

`manifest.json` is a list of logical chunk records. Each record contains:

| Field | Meaning |
|---|---|
| `condition` | Logical experiment arm, such as `N01` or `C10`. |
| `file`, `sha256` | Relative path and SHA256 of a lossless raw NPZ. |
| `seeds` | Seed identifiers in the file's first-axis order. |
| `start_step`, `steps` | Absolute zero-based first round and number of rounds. |
| `precision_file`, `precision_sha256` | Shared deterministic precision history and its SHA256. |
| `observation_sha256` | SHA256 of the contiguous float64 training-observation array. |
| `evaluation_observation_sha256` | SHA256 of the contiguous float64 held-out array before it is discarded from memory. |
| `deterministic_precision_seed_checks` | Number of seeds compared exactly before retaining one precision copy. |
| `array_shapes` | Stored raw-array shapes. |
| `shared_from`, when present | Source of physically reused observations/states. |

For array entry `j`, the observation and score belong to round
`t=start_step+j`; the probabilities and conditional readouts are state
`n=start_step+j+1`, after that observation and conditional memory. Do not add one
to the round when reconstructing a signal change: the first changed observation
has round `80` and produces state `81`. A branch at state `1500` changes the
update beginning with round `1500`.

Read records for a condition and seed batch in ascending `start_step`. The same
physical file can appear under multiple logical conditions. Count files by
unique path when validating storage; retain the repeated logical records when
assembling conditions. Network prehistory files are shared by all four N arms.
They include all 80 rounds generated once per seed. At their end the archived
natural engine key remains in `final_key`; the next chunk deliberately begins
with `PRNGKey(seed+1000)`, reproducing the legacy phase-two convention.

`S_keep_private` reuses all C10 records only after every unchanged continuation
from state 1500 has been rerun and found byte-identical through state 5000.
`S_enable_weights` shares the prefix through state 1500 and has independent
files for its changed communication rule thereafter. Its observations and
conditional states remain paired with C10. The two branch setup files identify
the intervention origin as `start_step=1500`, although their logical manifests
also contain the common prefix so a complete 5000-state trajectory can be read.

## Raw arrays

Let `S` be chunk seed count, `T` chunk rounds, `N` agents, `K=2` candidates,
`D=4` full state dimensions, and `M=3` observables. Candidate order is release,
oxygen. Observable order is C,G,M. Hidden coordinates P and O are never averaged
or differenced across candidates.

| Raw NPZ array | Shape | Definition |
|---|---|---|
| `L`, `q` | `(S,T,N,K)` | Post-update log weights and softmax probabilities. |
| `ell` | `(S,T,N,K)` | Direct post-sharing, pre-learning predictive scores. |
| `obs` | `(S,T,N,M)` | Actual raw noisy observation, before attention transformation. |
| `mu_conditional` | `(S,T,N,K,M)` | Each candidate's post-update observable mean. |
| `mu_mixture` | `(S,T,N,M)` | Probability-weighted observable mean. |
| `mu_map` | `(S,T,N,M)` | Observable mean of the candidate with maximal log weight. |
| `mse_full` | `(S,T,N)` | Mean squared mixture error against the noiseless signal over all 3 observables. |
| `nll_full` | `(S,T,N)` | Mixture negative log density, averaged over four independent full-observation replicates. |
| `initial_Pi`, `final_Pi` | `(S,N,K,D,D)` | Full precision at chunk boundaries. |
| `initial_h`, `final_h` | `(S,N,K,D)` | Full information vector at chunk boundaries. |
| `initial_L`, `final_L` | `(S,N,K)` | Model log weights at chunk boundaries. |
| `initial_key`, `final_key` | `(S,2)` | Exact uint32 JAX PRNG state. |

The companion precision NPZ contains `Pi_history` of shape `(T,N,K,D,D)`.
The runner verifies exact equality across all seeds in a batch and against the
saved first batch before omitting duplicate precision arrays. Positive-definite
precisions and this history determine every conditional covariance needed for
post-update evaluation. No covariance estimate across seeds is substituted.

The callable batch interface additionally returns `Pi_history` with its seed
axis `(S,T,N,K,D,D)` and `eval_obs` with shape `(S,T,N,4,M)` for tests and hashes.
These duplicate/reconstructible arrays are omitted from individual raw files.
All floating-point outputs remain float64. No lossy conversion or smoothing is
performed by storage.

## Exact random-stream reconstruction

Starting from each seed's `initial_key`, for each round execute the following
using the recorded JAX version, PRNG implementation, dtype, population, and
signal:

```python
evaluation_key = jax.random.fold_in(key, 314159)
evaluation_obs = signal[1:] + sigma_o * jax.random.normal(
    evaluation_key, (N, 4, 3), dtype=jnp.float64)
next_key, observation_key = jax.random.split(key)
agent_keys = jax.random.split(observation_key, N)
training_obs = jax.vmap(lambda k: sample_o(H, signal, sigma_o, k))(agent_keys)
key = next_key
```

Collect replicas in `(seed,time,agent,draw,observable)` order before checking the
evaluation hash. Evaluation keys are derived without replacing or advancing
the training state. The numerical NLL uses the legacy Cholesky log-determinant
arithmetic and log-sum-exp of the mixture; it is not an average of candidate
NLLs. Training attention leaves sampled noise unchanged and enters the working
score and information deposit as specified by the frozen engine.

## Execution records

The output includes the frozen protocol, source snapshot, engine-source hashes,
environment/backend/PRNG identity, condition setup files, and full restart
equivalence checks. `summary.json` and `endpoint_by_seed.json` are deliberately
small execution summaries of endpoints. The separate analysis script produces
the prespecified reporting windows, bootstrap intervals, event/censoring tables,
paired contrasts, and mechanism diagnostics. These execution summaries must not
be substituted for those analyses.

Numerical errors cause a recorded `failure.json` and exception. Seeds are never
silently replaced. `completion.json` is written only after the required arms,
restart verification, and unchanged-engine check have all completed.
