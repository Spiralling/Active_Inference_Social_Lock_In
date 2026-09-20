# Validation and reproducibility status

The required simulation suite completed successfully on the pinned local float64 CPU environment. No production condition failed, no seed was replaced or removed, and no scientific setting was changed after inspecting the outcomes. The frozen producer commit is `139225df25e641d387f4ea5e020e54203fe32c6b`, orx run `ccaba0e8-39ca-4993-a219-64ce4cd09de5`.

## Source and implementation

The supplied camera-ready `world.py`, `run.py` and `protocol.json` match the existing combustion experiment sources byte for byte. All **46** imported engine Python files match the supplied engine hash manifest. The actual import path is checked at runtime. The experiment wrapper adds independent communication switches, readouts and checkpointing; it does not replace the adaptive engine.

All **11** independent implementation tests pass. They cover reference compilation, weighted-matrix totals, exact agreement with the uninstrumented engine, pre-learning scoring, independent NumPy update equations, paired channel behavior, restarts, fractional/zero attention, candidate permutation, the matched-variance zero-cross-weight limit, exact log-odds recurrence, independent random streams, and a SciPy mixture-NLL calculation. A separate analysis test file checks event-boundary, censoring and pairing conventions.

During validation, one independent sampling check initially applied the affine observation transform in NumPy after generating JAX normals. It differed from fused XLA arithmetic by at most $2.22\times10^{-16}$. Compiling the independent affine transform under the same arithmetic convention made the samples bit-identical. No production code or scientific parameter was changed, and no tolerance was relaxed. The diagnostic and final passing logs are retained.

## Full raw-data audit

The independent audit checked **1,960 logical chunk records**, representing **1,670 unique raw chunks plus 167 deterministic precision chunks**. Every file hash passed. Every stored state/score/readout was finite float64; working/reference precision matrices were positive definite. Probability normalization, exact stream pairing, state continuity and score recurrences passed.

| Check | Scope | Maximum absolute discrepancy |
|---|---|---:|
| Three archived network arms | All 50 seeds × 600 updates, all six history readouts and three final state arrays | **0**, bit-identical arrays |
| Conditional-state/score/observation invariants | `00/01`, `10/11`, both intervention branches | **0**, exact |
| Unchanged state-1,500 restart | All 50 seeds, complete 3,500-update future, all recorded arrays/precision/evaluation streams | **0**, exact |
| Direct log-odds source recurrence | Every logical chunk and update | $2.57\times10^{-12}$ |
| Precision update equation | Every logical chunk and update | $3.55\times10^{-15}$ |
| Observable mixture reconstruction | Every logical chunk and update | **0** |
| Full-observable MSE recomputation | Every logical chunk and update | $4.44\times10^{-16}$ |

Precision is stored once per condition/time chunk only after checking exact equality across every seed and seed batch. Each raw chunk retains full entering/final information state and RNG key. Training observations are retained directly; held-out draws are exactly reconstructible from the keys and frozen generator, with per-chunk draw hashes recorded. The independent audit reconstructed exact sample hashes and final keys for its declared replay subset.

An independent NumPy/SciPy implementation replayed **6,150 updates** in **31 preselected chunks**, covering each condition's first and last chunks and the network/intervention boundaries. It did not call production scoring, pooling or update helpers. Maximum score discrepancy was $1.07\times10^{-14}$; accumulated log-weight discrepancy $1.27\times10^{-11}$; probability discrepancy $3.64\times10^{-13}$; held-out mixture-NLL discrepancy $6.44\times10^{-13}$. These are floating-point-scale differences, not discrepancies in the scientific update.

Full detail: [INDEPENDENT_RAW_AUDIT.json](INDEPENDENT_RAW_AUDIT.json), [INDEPENDENT_VALIDATION.md](INDEPENDENT_VALIDATION.md), and the runnable `audit.py`/`independent_tests.py` scripts.

## Statistical audit

Two independent reductions of raw data agree with the publication analysis:

- One checks all **1,100 seed/window rows**, with **21,200** shared endpoint/window values; maximum disagreement $1.78\times10^{-15}$.
- Another checks **17,600** seed values, **9,000** opposition/episode statistics, **1,800** pass flags, all **3,300** window bounds, **43,100** event flags and **172,400** event-time/censoring fields. It also checks all network/branch paired contrasts, bootstrap intervals and Wilson intervals. Occupancy and event fields agree exactly; the largest numeric discrepancy is $1.78\times10^{-15}$.

The seed is the replication unit. The 650 logical condition/seed histories include common prehistories and an aliased continuation; they are not 650 independent replications. The 50 paired network streams and 50 paired camp streams are reported separately. No censored event is treated as an observed event at the horizon.

Full detail: [ANALYSIS_CHECK.json](ANALYSIS_CHECK.json), [INDEPENDENT_ANALYSIS_COMPARISON.json](INDEPENDENT_ANALYSIS_COMPARISON.json), and their independent scripts.

## Stationary predictions

An independent full-agent affine calculation and discrete Lyapunov solution reproduce all four forecasts in the brief to its stated rounding (maximum $4.71\times10^{-10}$ nats per round). A separate convergent covariance series checks the Lyapunov solution. The calculation includes fluctuations caused by previous observation noise and distinguishes sampled attention-transformed noise from the working likelihood covariance.

All 16 measured comparisons (four conditions × two groups × two horizons) include the corresponding forecasts in their pointwise 95% bootstrap intervals. These are correlated comparisons, not 16 independent confirmations or simultaneous coverage guarantees.

## Provenance limits

The external independent-check ZIP and pilot support archive were unavailable. Their pilot samples are not claimed to have been reproduced. The independent stationary derivation supplied here provides a separate check of the numerical predictions. Network regressions do use the actual available prior production arrays and their recorded identities.

Exact numerical comparisons were made on the recorded macOS arm64 CPU environment with Python 3.12.9, JAX/jaxlib 0.9.0, NumPy 2.3.4 and SciPy 1.16.3. The full dependency versions are included. A different backend or software version may change floating-point last bits; mathematical agreement and the frozen source identities remain independently checkable.

Figure layout, embedded fonts, vector content, interval construction and fresh-extraction regeneration are recorded in the figure and package validation files delivered with the final ZIP. Numerical audits above are complete and separate from those presentation checks.
