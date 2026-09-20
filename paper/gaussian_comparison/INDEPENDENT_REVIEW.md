# Independent adversarial review: completed

The implementation, completed raw trajectories, statistical reductions, mathematical methods and result interpretation passed independent checks. No production implementation defect or unsupported numerical claim was found. The producer was frozen before outcomes and was not changed in response to this review.

## Verification completed

- Eleven implementation tests cover independent observable marginalization/Schur complements, information and moment compression, weighted-graph semantics, social gate limits, hidden-coordinate exclusion, direct NumPy updates and pre-deposit scoring, original-engine equivalence, paired observations/evaluation draws, exact restart, full predictive NLL and observable-coordinate permutation.
- The raw audit checks all 800 new chunks and independently re-reduces the 600 archived candidate chunks used in the figures. It verifies all 800 training/evaluation stream hash pairs and 1,600 entering/final RNG-state equalities against archived C10.
- All 5,320 chunk-boundary state comparisons passed. Every Gaussian update is checked algebraically from the entering state, and every retained precision passes Cholesky: 1,000,000 seed-update states, covering 16,000,000 agent precisions and pre-deposit scores.
- A prespecified first/last-chunk sample additionally runs a free-running sequential NumPy implementation for 2,000 updates. Independently regenerated held-out observations verify 160,000 per-agent NLL values. These sampled NLL checks are distinguished from the exhaustive state/score checks.
- All 23,600 seed-level scalar comparisons, 1,360 summary scalar comparisons and 1,128 paired-contrast scalar comparisons passed. The largest seed-metric difference was about 6.44e-15; summary and paired-interval differences were at most 1.34e-15.
- Independent plain-Gaussian stationary predictions were recorded before production. All 20 comparisons of expected signed camp differences and full-signal MSE with matching pointwise intervals are consistent. These are correlated diagnostics, not 20 independent confirmations.

Largest absolute state-equation discrepancies are 2.22e-15 for precision and 2.66e-15 for information. Pre-deposit score error is at most 5.33e-15. The accumulated log-score recurrence differs by at most 9.10e-13, and independently computed held-out NLL by at most 1.78e-15. These are floating-point rounding differences, with no altered scientific tolerance or parameters.

The archive engine hashes, world, parent instrumentation and pinned numerical package versions match the preceding study. The complete machine-readable evidence is in `INDEPENDENT_TESTS.json`, `INDEPENDENT_RAW_AUDIT.json`, `INDEPENDENT_SEED_METRICS.csv` and the retained logs. This audit does not assert correctness of every unused legacy engine function.

## Defects found in the checking/reporting machinery

No failed result was discarded. The following issues were repaired and recorded:

1. One independent unit-test assertion originally demanded an absolute row-sum error below 4e-16. A valid normalized row differed from one by 4.44e-16, or two double-precision ulp. The test now uses the standard 16-term summation bound gamma_16. All ten other tests had passed; the entire affected test passed on rerun. Initial and rerun logs and `INITIAL_TEST_HARNESS_DIAGNOSTIC.json` preserve this history. No producer code changed.
2. The independent vectorized raw auditor initially reused one einsum label for both the seed batch and matrix coordinates. It stopped on its first chunk before comparing the numerical updates. Distinct labels repaired the auditor; the complete rerun then passed. `initial_raw_audit.log` and `RAW_AUDIT_HARNESS_DIAGNOSTIC.json` preserve this audit-only error.
3. Before production, review identified that a naive percentile interval for the norm of an ensemble-mean vector can falsely appear to exclude zero when the true signed vector is zero. The analysis therefore reports that norm descriptively, with no such interval, while retaining intervals for signed components and the separately defined mean of seed-level norms. The protocol amendment and six analysis boundary tests document this correction.

## Interpretation signoff

`METHODS.md` and `RESULTS.md` were read in full and agree with the implemented equations and reported numbers. In particular:

- Plain Gaussians show systematic numerical prediction differences in this continually driven, reference-retaining system. The pre-run analytic calculation predicts them. Closed-system averaging consensus is not a theorem about this full update.
- The social gate decreases effective cross-group influence and modestly increases the difference. In these runs it worsens the reported predictive losses, so stronger division is not presented as an improvement.
- Gaussian prior compression matters substantially. The information pool's mean is not the arithmetic mean of candidate means. The moment sensitivity comparison and the limits of a K=1 versus K=2 comparison remain essential.
- Private versus shared candidate weights are visibly distinguished. Persistent private-weight opposition is not presented as opposition under exchange of model support.
- Shared candidates retain an instantaneous distance near 0.106, but essentially no ensemble directional difference; that residual is consistent with private noisy updates. Nonzero norms alone do not establish opposing camps.
- Candidate variance sensitivity, full-observable predictive costs, finite horizons, inherited gate choice, paired rather than independent new seeds, and reuse of archived candidate trajectories are disclosed.

This signoff supports the specified comparison and its bounded interpretation. It does not establish a general advantage of diversity, a universal ordering of mechanisms, permanent schisms or causal-structure discovery.
