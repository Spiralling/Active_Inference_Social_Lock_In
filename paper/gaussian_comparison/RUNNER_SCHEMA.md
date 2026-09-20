# Matched Gaussian baseline output

The runner imports the archived rival engine through the existing communication
experiment module and calls `rv._step` at every update. It does not edit the
engine, original world, or existing candidate runs. Production is launched from
a frozen orx snapshot; run completion records the protocol SHA-256.

Four conditions are run: `G_plain_info`, `G_gated_info`, `G_plain_moment`, and
`G_gated_moment`. All use 16 agents, 5,000 updates, seeds 10000–10049, observation
noise standard deviation .5, anchored retention .3, and the same weighted
two-community matrix with cross-group mass .05. Gas agents 0–7 attend only to G;
mass agents 8–15 attend only to M. The fixed unobserved-padded signal is
`[0,1,1,1]`. All agents start from and retain the same Gaussian reference.

## Reference construction and gate

Both compression rules first marginalize the two original candidate references
to the **observable** coordinates C, G, M. The `info` reference averages those
marginals' precision matrices and information vectors with weights (1/2,1/2).
The `moment` reference matches the equal mixture's mean and covariance, including
between-component mean variation. Their hidden P and O coordinates are never
identified or pooled. An independent zero-mean dummy coordinate of variance .25
is added solely for the archived four-coordinate interface.

The gated arms compute the archived `reliability.social_gamma` on entering
conditional states, using observable indices (1,2,3) and nu=.01. They multiply
the **weighted** base matrix elementwise by gamma and divide by its row sums.
They do not replace base weights with adjacency. Positive self weights guarantee
strictly positive row sums, so no denominator epsilon is added. The effective
matrix is passed directly to `rv._step` with the engine's internal gate disabled.
Score instrumentation uses that exact same matrix. There are no observation or
model-support gates, utility terms, model-score decay, or agent-specific anchors.

## Chunk arrays

Every compressed NPZ covers five seeds and 250 consecutive updates. With S=5,
T=250, N=16, K=1, D=4, and V=3, its arrays are:

| Key | Shape | Meaning |
|---|---|---|
| `L`, `q`, `ell` | S,T,N,K | Updated accumulated score, candidate probability (identically one), current pre-deposit score |
| `obs` | S,T,N,V | Training observations before attention weighting |
| `mu_conditional` | S,T,N,K,V | Updated conditional observable means |
| `mu_mixture`, `mu_map` | S,T,N,V | Identical K=1 posterior predictive means |
| `mse_full`, `nll_full` | S,T,N | Full-observable prediction loss after updating |
| `gate_crossweight` | S,T,N | Total effective incoming weight from the opposite group for this update |
| `Pi_history` | S,T,N,K,D,D | Full updated precision for every seed, without seed deduplication |
| `initial_Pi`, `final_Pi` | S,N,K,D,D | Exact entering and exiting precision |
| `initial_h`, `final_h` | S,N,K,D | Exact information vectors |
| `initial_L`, `final_L` | S,N,K | Exact accumulated scores |
| `initial_key`, `final_key` | S,2 | Exact uint32 JAX random states |

All real-valued arrays are float64. `Pi_history` is retained separately for
every seed because the social gate makes precision depend on realized noise.
NPZ compression is lossless. Nothing is clipped, quantized, or downsampled.
The first time entry is state `start_step+1`; gate weights refer to its entering
state, immediately before that observation. The final state of one chunk is the
next chunk's initial state, including its random key.

MSE averages squared error over C, G, M relative to the noiseless fixed signal.
NLL averages Gaussian posterior predictive loss over four fresh full-observable
measurements per agent, generated from `fold_in(entering_key,314159)`. Evaluation
draws are not used for learning and do not advance the training stream. Their
SHA-256 is stored, and exact draws can be reconstructed from checkpoints; the
draws themselves are omitted from NPZ storage. Training RNG splitting matches
the candidate study and is independent of the model or gate.

Each manifest record identifies its condition, five seed identifiers, entering
state, number of updates, file SHA-256, all array shapes, and hashes of training
and held-out observations. `setup.json` records the resolved reference, marginal
candidate distributions, attention, weighted graph, gate, and time convention.
`environment.json`, `engine_hashes.json`, and `source/` record actual packages and
source provenance. `endpoint_by_seed.json` contains endpoint diagnostics only;
scientific claims use the separately specified full-window analysis.

All four arms reuse 50 paired random streams. There are 200 seed-condition
histories, **not 200 independent replications**. Each condition uses all 50
specified seeds without replacement. Failed production runs leave `failure.json`
and do not receive a completion marker.
