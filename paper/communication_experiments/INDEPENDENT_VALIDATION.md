# Independent validation of the communication suite

The production experiment passed the independent checks. No change to the scientific settings or production update arithmetic was required after observing results.

## Before confirmation

All 11 tests in `independent_tests.py` passed. They checked independent reference compilation, weighted-matrix semantics, exact agreement of the instrumented wrapper with the unchanged engine, independent NumPy one-step equations and score timing, exact channel invariants, exact restart and branch behavior, fractional and zero attention, candidate relabelling, matched-gas behavior at zero cross-group influence, the model-score recurrence, and independent held-out mixture prediction. The held-out check uses SciPy Gaussian log densities and separate reconstruction of the recorded JAX random-key convention.

The first test harness compared a JAX normal draw transformed in NumPy with a fully compiled JAX observation. Eight of 48 values differed by at most 2.22e-16 because the affine arithmetic was compiled differently. Compiling the independent sample transform in JAX restored exact sample equality without changing production code or loosening tolerance. `INITIAL_TEST_HARNESS_DIAGNOSTIC.json` preserves that diagnostic; `INDEPENDENT_TESTS.json` and `independent_tests.log` record the passing suite.

`VALIDATION_UPDATE_STAGES.npz` retains a deterministic validation example with entering, post-sharing, post-learning and post-memory means and information parameters. Its seed 900000 is not a confirmation seed.

## Complete production-data audit

`audit.py` inspected all 1,960 logical manifest records, representing 1,670 unique raw-data files and 167 precision files. All SHA256 checks passed. Precision histories were finite and positive definite; all stored non-key numerical arrays were float64. Initial references, neutral model weights, seed keys and all checkpoint boundaries matched the specification. The phase-two network key reset was checked explicitly.

The 13 logical conditions contain 650 seed-condition histories, including reused branch histories and paired arms. These are not 650 independent replications: the independent units are 50 network seeds and 50 camp seeds, with comparisons paired within seed.

All raw observations and held-out-sample hashes match across paired conditions. All 680 matched condition/chunk pairs have exactly identical conditional states, local scores and precision histories where the theory requires that invariant. The fixed-time branch changes only model weights and readouts depending on them. Its unchanged continuation was additionally rerun by the production wrapper for every seed and every remaining round, with byte-identical output.

| Check over every logical chunk | Maximum absolute residual |
|---|---:|
| Candidate-probability normalization | 2.22e-16 |
| Direct log-score recurrence | 1.82e-12 |
| Log-odds source recurrence using the actual weighted matrix | 2.58e-12 |
| Conditional precision update | 3.56e-15 |
| Mixture mean from observable candidate means | 0 |
| MSE against the noiseless full signal | 4.45e-16 |

## Archived network regression

The existing `N00`, `N10` and `N11` arms reproduce the archived combustion experiment **exactly at the array-value level** for all 50 seeds and all 600 updates. Maximum differences are zero for all candidate scores accumulated in `L`, probabilities, selected and mixture observable means, MSE, held-out NLL, and all final precision/information/log-weight arrays. Each source NPZ hash also matches its original manifest.

This verifies a paired extension of the same archived comparison. It does not turn those three repeated arms into an independent confirmation on new samples. `N01` supplies the missing model-weight-only arm on the same paired samples.

## Independent numerical replay

A second calculation replayed 6,150 updates with directly coded NumPy equations rather than production fusion, scoring or update helpers. Chunk selection was fixed by position: the first and last chunk of every logical condition, the network change boundary and the sharing-intervention boundary. The first seed in each of 31 chunks was replayed. All five seeds' training and evaluation streams in each selected chunk were independently reconstructed from the initial keys, yielding exact observation arrays, exact held-out-array hashes and exact final keys.

Maximum independent-replay differences were 1.07e-14 for predictive scores, 1.28e-11 for accumulated log weights, 3.64e-13 for probabilities, 4.00e-15 for observable conditional means, and 6.44e-13 for held-out mixture NLL. These are numerical round-off differences between independent arithmetic paths. The complete per-field errors and coverage are in `INDEPENDENT_RAW_AUDIT.json`.

The four stationary score predictions were also recomputed independently without adaptive-engine imports, including stationary noise in the information vectors. They match the brief's rounded values within 4.71e-10. This calculation is separate from the trajectory audit and does not require finite-sample estimates to equal expectations exactly.

Finally, the raw audit independently sums 1,100 seed/horizon rows of group probabilities, predictive losses, local scores, and primary/secondary separation occupancies. Those values are stored in `INDEPENDENT_SEED_METRICS.csv`. All 1,100 rows and 21,200 shared scalar quantities agree with the separately implemented publication analysis to a maximum absolute difference of 1.78e-15. Primary and secondary opposition occupancies and endpoint indicators agree exactly. `INDEPENDENT_ANALYSIS_COMPARISON.json` records the comparison, reproducible with `compare_independent_analysis.py`.
