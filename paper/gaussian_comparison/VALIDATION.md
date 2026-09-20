# Independent validation of the matched comparison

The complete four-condition production run and its scientific analysis passed independent adversarial checks. The producer is frozen at `69f054f7150abfebcc7914e28dd17db247d449ca`, orx run `84d916ef-d0b4-42b6-b245-9ad6855fd35e`. All 50 paired seeds completed all 5,000 updates in each new condition. There were no failed production conditions, removed seeds or scientific changes after inspecting outcomes.

## Independent implementation and analysis tests

Eleven independently authored tests cover observable marginalization, information and moment reference construction, the between-mean covariance term, weighted-graph preservation, independent gate calculations, the constant-gate limit, exclusion of the dummy coordinate, independent NumPy updates and pre-deposit scores, exact original-engine transitions, paired candidate observations, exact restarts, independent random streams and full-observation SciPy NLL, and observable coordinate permutation. Six further analysis tests check inclusive state windows, seed pairing, temporal and across-seed direction reversals, nonfinite inputs and observable-only comparisons.

One initial test used an excessively tight row-sum tolerance of 4e-16 and rejected a 4.44e-16 floating-point rounding residual. The assertion was corrected to the standard error bound for a 16-term sum, approximately 3.55e-15, and the complete affected test passed. The other ten had already passed and their producer inputs/code were unchanged. Both logs and the numerical diagnostic are retained. This tolerance remains far below an epsilon-induced weight deficit of 1e-12; no production tolerance or equation was modified.

## Exhaustive raw-data and equation checks

The adversarial audit checked all **800 new chunks** and **600 reused candidate chunks**, their hashes and complete coverage. Across every new chunk, training and held-out stream hashes and entering/final random keys match the parent C10 data exactly. Full precision histories are stored for each seed; the gated precision is not incorrectly treated as deterministic across seeds.

The independent NumPy implementation checked every one of **1,000,000 Gaussian seed/update states**, including **16,000,000 agent precision matrices and pre-deposit scores**. It verified update order, weighted gates, positive definiteness, information updates, score recurrences, observable predictions, full-signal MSE and checkpoint continuity. Maximum absolute discrepancies were approximately 2.22e-15 for precision, 2.66e-15 for information, 5.33e-15 for predictive scores and 9.09e-13 for accumulated-score recurrence.

Eight chunks selected before production were also replayed sequentially in an independent NumPy implementation, totaling **2,000 complete network updates**. Independent predictive-density calculations checked **160,000 held-out agent/update NLL values**, with maximum discrepancy 1.78e-15. These density checks sample the declared chunks; they are not claimed to independently recompute every held-out density in the full run.

The raw audit initially stopped because its own vectorized einsum reused an index symbol for two axes. That audit-only code was corrected and the entire audit restarted. The failed log and `RAW_AUDIT_HARNESS_DIAGNOSTIC.json` document the issue. Production code and arrays were unchanged.

## Independent statistical and mathematical checks

Independent raw reductions reproduce all **23,600 seed metric values**, **1,360 aggregate summary values**, and **1,128 paired-contrast values**, including bootstrap intervals. Pairing is by the same 50 seed identities across conditions, not by treating agents or timepoints as independent observations. Full details and maximum discrepancies are in `INDEPENDENT_RAW_AUDIT.json` and `INDEPENDENT_SEED_METRICS.csv`.

The plain-Gaussian stationary precision/information equations and Lyapunov covariance were derived independently before examining the new outcomes. All **20 signed-coordinate and MSE comparisons** include their predictions in the corresponding pointwise intervals. These comparisons are correlated diagnostics, not 20 independent confirmations. A further portable-data check examines the displayed curve calculations separately from the raw-data audit.

Before production, statistical review also identified that percentile intervals for the norm of an across-seed mean vector can misleadingly exclude zero at a true zero vector. That aggregate is now explicitly descriptive with no interval. Signed-coordinate intervals and the independent stationary equations provide the relevant direction checks. The original and clarified protocol hashes are retained in `AMENDMENTS.md`; no dynamical setting changed.

## Figures and portable handoff

The previous seven communication figures remain byte-identical. The two new figures use actual LaTeX Computer Modern text and mathematics, vector PDF/SVG exports and orx layout checks. Figure validation checks embedded fonts, physical width, vector content, bounds, source/data identities and the nine-page pack. The figure and final package check records define the exact scope of visual inspection and fresh-extraction reproduction.

All run-time comparisons used the recorded Python 3.12.9/JAX 0.9.0 float64 macOS arm64 CPU environment. Another backend may differ in final floating-point bits. Every scientific input, source identity, seed convention and raw chunk hash remains available for independent verification. No public repository or DOI was created and no manuscript was changed.

See `INDEPENDENT_REVIEW.md` for the adversarial interpretation review, `INDEPENDENT_DESIGN_AUDIT.md` for the pre-production mathematical rationale, and the JSON reports and preserved logs for executable evidence. Passing these checks supports the stated implementation and numerical results; it does not establish universal consensus or separation beyond the tested model and horizon.

## Fresh extraction of the delivered payload

A separate agent extracted the draft compact ZIP into a new directory and verified every payload hash. From those extracted sources it reran all 11 independent mathematical tests and six analysis boundary tests, validated the shipped figures, regenerated both new figures and the nine-page pack, and validated the outputs again. All six new PDF/SVG/PNG assets, the pack PDF and plotted curve CSV were byte-identical. All 21 previous figure assets were preserved. This verifies the supplied plotting reductions and engine/test imports without claiming that the entire 50-seed production suite was rerun from the ZIP.

The first portability harness rejected valid members because macOS exposes the temporary directory through both `/var` and `/private/var`. Resolving both paths corrected the containment check; the initial diagnostic is preserved. No package or scientific change was required.

The final archive adds this paragraph, the unchanged portability-check script, and its reports/logs to that tested payload, then regenerates the complete SHA256 manifest and verifies all final ZIP entries. `package_checks/PORTABLE_PAYLOAD_CHECK.json` records the reproduction scope; the external `PACKAGE_VALIDATION.json` binds the final ZIP hash and delivery commit without a circular self-reference.
