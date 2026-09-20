# Independent adversarial review of the clean scientific implementation

## Conclusion and scope

The separately reviewed implementation matches the scientific mechanisms needed by the two supplied figure sets. **No production-model error was found in the exercised, fixed experimental protocols.** Nine independent mathematical and invariance tests passed on the frozen source, in 16.644 seconds. The test log is `checks/independent_tests.log`; source and test hashes are recorded in `checks/independent_review.json`.

This review was performed by a separate agent from the implementation author. It inspected `engine.py`, `world.py`, `simulation.py`, `camera.py`, `communication.py`, `gaussian.py`, `structural_check.py` and `experiments.py`, and cross-checked the design against both manuscript archives and their original protocols. It also inspected the raw-data reduction paths. It did not edit production source, select outcome-favorable seeds, alter run settings or require a particular scientific result.

These checks complement, but do not replace, the separate archived-code equivalence tests, full experiment reruns and figure reproduction checks. This report makes no claim that its small mathematical examples constitute independent replications of the full scientific experiments. The source-hash record identifies exactly which implementation was tested.

## Independent mathematical checks

The main update oracle uses NumPy covariance-form conditioning, Kalman gains and SciPy multivariate-normal log densities. It does not call the production information deposit, predictive-score or retention routines to calculate expected results. Its comparison exercises heterogeneous observations, fractional attention, a zero-attention agent, model-weight pooling, enabled/disabled conditional exchange, and retention values zero, 0.3 and one. It checks pre-learning score timing, conditional posterior state, anchoring and the next random key.

The complete tests are in `tests/test_independent.py`:

| Check | Independent target |
|---|---|
| Reference covariance and observable marginal | Explicit structural transforms reproduce the Gaussian references; observable covariance is marginalized before inversion; the result differs from simply taking a principal precision block. |
| Hidden-coordinate invariance | Separate nonsingular affine changes to the hidden release and oxygen variables leave both Gaussian compression schemes unchanged on observables. |
| Full update and score timing | Covariance-form Bayesian conditioning and SciPy likelihoods agree with the information-form kernel, including pooled priors and post-observation retention. |
| Attention | Arbitrarily changing ignored observation channels leaves candidate scores unchanged; zero attention gives zero information and candidate-independent scores. |
| Weighted social gate | A hand-computed scalar disagreement metric reproduces the gate; equal beliefs recover original weighted influence; row sums are one; changing hidden padding does not affect observable gating. |
| Model-support intervention | Changing only model-sharing rate changes categorical choices while preserving conditional states, local scores, training observations, held-out observations and random keys exactly. SciPy independently verifies the posterior predictive mixture NLL. |
| Checkpoint and evaluation invariance | A nine-update run equals a four-plus-five exact checkpoint continuation. Changing the number of held-out draws leaves training unchanged. Gaussian and candidate arms receive identical training/evaluation streams. |
| Historical phase reset | The camera wrapper agrees with a separately constructed continuation that explicitly replaces the entering random keys by `PRNGKey(seed + 1000)`. |
| Structural discrimination controls | Fixed-model marginal matching, chance-level marginal and duplicate controls, label swapping, paired noise, and an independently evaluated cumulative likelihood agree. |

All nine passed on the first execution. There were no failed tests requiring a change of assertion tolerance or production code.

## Scientific details checked adversarially

### Distinct latent meanings and reference construction

The hidden coordinate is a storage position, not a shared empirical quantity: it means release `P` in one candidate and incorporated oxygen `O` in the other. Shared readouts use only `(C,G,M)`. The single-Gaussian baselines integrate hidden variables out before pooling. Information pooling averages marginal precision and information vectors. Moment matching includes both within-component covariance and covariance from differences between component means. An independent mean-zero dummy coordinate preserves the interface but is excluded from comparisons and gating.

### The two gated comparisons intentionally differ

The old camera camp experiment applies the archived gate to binary contact support and preserves its historical `1e-12` row-denominator addition. Retaining that tiny numerical mass deficit is intentional for faithful reproduction of the old figure.

The newer matched experiment applies `W * gamma` to the original **weighted** matrix and divides by its exact positive row sum. It does not discard the prescribed `beta = 0.05` cross-group weight, and it adds no normalization epsilon. The gate is calculated from entering observable means, before social pooling and before observing the next datum. Reciprocal precision diagonals in its distance are conditional variances, not marginal variances. The resulting diagonal distance is an approximation, not the full multivariate Mahalanobis distance.

### Learning, support sharing and evaluation are separate operations

The kernel exchanges model support and pools conditional beliefs, scores the incoming observation using those pooled beliefs, deposits observation information, and then applies reference retention. Candidate support is not forgotten in these protocols. Attention weights affect both the predictive score and the conditional evidence deposit. Held-out evaluation is full-observation posterior predictive density after the update, with measurement-noise variance included. It is not the local pre-learning score and is not fed back into learning.

The mixture NLL uses a log-sum-exp over component predictive densities and categorical log weights. It does not score only the selected component or treat the mixture as a single moment-matched Gaussian.

### Random streams and interventions

Every update consumes one split of the entering training key and one agent-key split, independent of candidate count. Evaluation derives a folded key without consuming or replacing the training state. The newer wrappers preserve complete `Pi,h,L,key` checkpoints. The older camera family deliberately starts a new phase at `seed + 1000`, reproducing its historical protocol.

The communication intervention branches every primary seed at fixed state 1,500. It reuses the private continuation only after a complete unchanged-branch restart comparison, and enables support sharing in the other branch without altering conditional states or observations. No outcome-dependent branch selection was found.

### Conditions and measurement semantics

The original camera camp plot uses unequal retention, group-specific Gaussian initial references, complete uniform communication and a MAP-selected readout. The newer matched plot uses common Gaussian references, matched retention, weak cross-group influence and mixture predictions. Keeping these recipes separate is scientifically necessary to reproduce the supplied figures.

The camp probability strips show agents nested within seeds; agents are not independent replications. The reductions use completed-state windows `451–600` and `3751–5000` as array slices `450:600` and `3750:5000`. They preserve the distinction between averaging instantaneous norms, taking a norm after time averaging, and taking a norm after both time and seed averaging. The last quantity remains descriptive without a percentile confidence interval.

The static discrimination check samples fresh independent structural states. The adaptive experiments use a prescribed signal plus sensor noise. The static check is not evidence that the adaptive experiments recover a causal graph.

## Limits of the scientific claims

Nothing in this review establishes that candidate models are necessary for every form of prediction disagreement. The matched Gaussian experiment itself contradicts that stronger claim under its continuing observations and reference anchoring. The private-weight candidate split depends on the specified reference uncertainty and restricted communication; the variance and support-sharing controls must remain visible in its interpretation.

Persistent finite-horizon opposition is not a proof of permanent fixation. Selecting a supplied candidate is not discovering it, uniquely identifying a causal structure, or obtaining a universal accuracy benefit. These are interpretation limits of the experiments, not implementation defects.
