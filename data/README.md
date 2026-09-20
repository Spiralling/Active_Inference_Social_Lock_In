# Portable inputs for the supplied figures

These files contain unrounded numerical reductions of the existing runs, not new simulation results and not digitizations of the PDFs. `iwai_figures.reduce` regenerates them from the corresponding raw simulation outputs. Each NPZ is a namespace; each array is named inside it. `manifest.json` records file SHA-256 hashes and hashes of the source raw files or raw manifest. The renderer verifies the inputs before use.

- `camera/`: the historical revised-combustion assessment. Oxygen probabilities retain every agent, seed and update used in the figures. Historical camp-distance files contain the observable MAP-mean distance for each seed and update. Adaptation prediction errors retain the per-seed final-375-update means, exactly as plotted. The identifiability input retains probabilities only at the five displayed independent-sample counts. World matrices and the four exact graph adjacencies are included.
- `communication/`: the four communication-channel arms and their controls/interventions. Group probabilities retain all 600 or 5,000 completed states. The individual-probability strips retain all agents and all 50 seeds at the two reported endpoints. Prediction/score inputs retain the exact per-seed windows drawn in the figures. `protocol.json` contains the previously specified stationary score forecasts.
- `gaussian/`: the four matched Gaussian arms and the three reused candidate arms. All plotted distance, model-probability and realized communication-weight curves retain every completed update and all 50 paired seeds. Small arrays contain per-seed prediction errors and the signed, time-averaged camp difference needed to distinguish the three distance estimands. `summary.json` contains the exact bootstrap summaries consumed by the two Gaussian plots.

`common.npz` in the latter two families contains the same 2,000 whole-seed bootstrap index vectors (NumPy seed 20260920). The camera plots recreate their original 2,000 bootstrap draws using seed 20260919. Bootstrap bands are pointwise percentile intervals for seed means. Agent-level strip distributions and topology traces are descriptive displays, not additional independent replications.

Source provenance:

- Camera assessment: `assessment-2026-09-19` from the combustion experiment revision.
- Communication run: `ccaba0e8-39ca-4993-a219-64ce4cd09de5`, producer commit `139225df25e641d387f4ea5e020e54203fe32c6b`.
- Matched Gaussian extension: `84d916ef-d0b4-42b6-b245-9ad6855fd35e`, producer commit `69f054f7150abfebcc7914e28dd17db247d449ca`.

The Gaussian candidate curves reuse the communication runs and their observation streams. They are not an independent candidate-model replication. Arrays are never rounded, smoothed or filtered by their outcomes. Raw covariance histories, unplotted metrics and unrelated historical experiments are excluded from this compact figure code branch.
