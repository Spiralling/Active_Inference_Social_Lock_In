# Matched Gaussian comparison: results and interpretation

The requested comparison is implemented and all four new conditions completed. The updated figure pack contains the original seven communication figures, a new main Gaussian/gated-Gaussian/candidate comparison, and a sensitivity figure. The main finding is that **a single Gaussian already sustains a systematic prediction difference between these observation groups**. The social gate modestly enlarges it. Representing candidate theories additionally allows opposing explicit theoretical preferences, but those preferences disappear when model weights are shared.

The new comparison is more controlled than the older figure: all mechanisms use retention .3, the same base matrix with cross-group influence .05, the same world and observation groups, and identical noise streams. Gaussian arms begin with one common reference for all agents. The underlying engine and existing candidate trajectories are preserved.

## Completed work and provenance

Four new arms cross plain/gated conditional sharing with information-pooled/moment-matched observable references. Each runs 16 agents for 5,000 updates with the 50 existing paired camp streams, seeds 10000–10049. These are 200 new condition/seed histories, not 200 independent replications. Three archived candidate arms are reused for comparison and do not constitute new confirmations. There were no failed production conditions, removed seeds or outcome-driven parameter changes.

The frozen producer is `69f054f7150abfebcc7914e28dd17db247d449ca`, orx run `84d916ef-d0b4-42b6-b245-9ad6855fd35e`. It completed in 544.45 seconds and retained 800 lossless raw chunks. Scientific settings and windows were fixed before these new outcomes. The follow-up itself was selected after the earlier results, so this is not a preregistration claim. A pre-production statistical clarification is recorded in `AMENDMENTS.md`.

## Main quantitative comparison

The table averages completed states **3,751–5,000** within each seed, then averages 50 seeds. Instantaneous distance is the Euclidean distance between camp-average observable predictive means, including their sampling noise. The next column first time-averages the signed difference within each seed and then takes its norm. This better resolves a consistent direction of disagreement, although finite-window sampling noise remains. It is not the norm after averaging seeds. Predictions use the candidate mixture, not a selected candidate.

| Condition | Instantaneous distance | Distance after time averaging | Gas MSE | Mass MSE | Gas NLL | Mass NLL |
|---|---:|---:|---:|---:|---:|---:|
| Gaussian, information pool | 0.502249 | 0.495953 | 0.355519 | 0.159392 | 3.352806 | 3.091940 |
| Gated Gaussian, information pool | 0.526806 | 0.520758 | 0.368099 | 0.160097 | 3.370810 | 3.096131 |
| Gaussian, moment matched | 0.701686 | 0.696728 | 0.311040 | 0.044460 | 3.352886 | 2.991523 |
| Gated Gaussian, moment matched | 0.759441 | 0.754840 | 0.346573 | 0.044592 | 3.388043 | 2.997238 |
| Candidates, private weights | 1.946371 | 1.944098 | 1.278857 | 0.019634 | 4.832537 | 2.841225 |
| Candidates, shared weights | 0.105834 | 0.003850 | 0.019563 | 0.019634 | 2.841172 | 2.841225 |
| Candidates, matched gas variance | 0.111447 | 0.033630 | 0.020661 | 0.019634 | 2.854588 | 2.841225 |

MSE is against the noiseless full observable signal. NLL evaluates four fresh full observations per agent/update and includes sensor noise in the predictive density; lower is better. Group metrics remain separate. Confidence intervals and endpoints are retained in `analysis/summary.json`; every seed/window value is in `analysis/seed_metrics.csv`.

## 1. Gaussian disagreement is systematic here

The ordinary Gaussian's distance is not only a positive norm caused by noise. With the information-pooled reference, the late signed gas-minus-mass differences are approximately `(0.05751, 0.12890, -0.47544)` for `(C,G,M)`; all three pointwise seed intervals exclude zero. With the moment-matched reference, the major mass difference is about `-0.69642`, with interval `[-0.69747,-0.69537]`. These results agree with the independent stationary equations derived before the production run.

Continual selective observations supply different information, and retention .3 continually restores the reference. Pooling therefore operates inside a driven, anchored system. A consensus result for an isolated averaging operator does not imply that this whole system converges to identical predictions. Explicit alternative theories and a social gate are not necessary for every form of sustained numerical disagreement.

This does not contradict zero preservation of Gaussian precision pooling: preserving absent precision entries is a different claim from eliminating prediction differences under continued observations and anchoring.

## 2. The gate strengthens disagreement and reduces information exchange

The gate reweights the original weighted matrix using the inherited parameter .01. It does not replace that matrix with complete uniform pooling. Average late cross-group influence falls from .05 to **.006389/.006869** for gas/mass observers under information pooling, and **.007398/.008809** under moment matching. Exchange remains positive; there is no imposed zero-weight disconnection.

The following are gated-minus-plain effects formed **within each paired seed**. Brackets give pointwise 95% whole-seed bootstrap intervals (2,000 resamples).

| Reference | Change in distance | Change in gas MSE | Change in mass MSE |
|---|---:|---:|---:|
| Information pool | +0.024557 [+0.024518, +0.024596] | +0.012580 [+0.012548, +0.012611] | +0.000705 [+0.000687, +0.000723] |
| Moment matched | +0.057755 [+0.057656, +0.057855] | +0.035533 [+0.035465, +0.035602] | +0.000132 [+0.000102, +0.000164] |

The gate also raises held-out NLL for both groups under both references. Thus it has a measurable effect, but larger disagreement is accompanied by worse prediction in these tests. The numerical costs differ by observation group; a larger camp distance is not a benefit by itself. This is a conditional result for the inherited gate strength, not an optimized or universal assessment of all gating rules.

## 3. Reference construction changes the comparison substantially

Information pooling gives reference mean approximately **(.623448, .644510, .008680)**. Equal mixture moment matching gives **(1,1,0)**. The first is the mean of a product/logarithmic pool of correlated observable densities, not the arithmetic mean of their means. Both constructions integrate out the incompatible hidden P/O coordinates first. The moment covariance includes between-candidate mean variation.

Changing the plain Gaussian from information pooling to moment matching increases its mean distance by **.199437** [ .199019, .199850 ], a larger change than the gate effects in this comparison. Yet it *improves* full-signal MSE for both groups: changes are approximately −.044480 for gas and −.114932 for mass. Mass-group NLL also improves; gas-group NLL has a paired interval spanning zero. Consequently distance and prediction quality do not have a universal monotonic relationship.

The main plot uses the information-pool reference to retain continuity with the paper's precision-pooling construction. The sensitivity panel is essential. Matched retention and observation access do not make a single Gaussian's predictive density identical to the two-candidate mixture. This is a comparison of specified mechanisms and references, not an isolated causal estimate of candidate count.

## 4. Candidate choices and prediction distances tell different stories

The private-weight candidate arm retains the largest distance, about 1.946, and the gas and mass groups favor release and oxygen respectively. This repeats the existing C10 result on the same archived trajectories. It is sensitive to reference variance and is predictively costly for gas observers: their MSE is about 1.279, compared with .01956 when model support is shared. The matched gas-variance candidate control also largely removes the costly division.

With shared model weights, both groups favor oxygen. Its instantaneous distance still averages **.105834**, but distance after time averaging is only **.003850**. The norm after additionally averaging signed differences across seeds is **.0000441**, a descriptive value, and all three signed-coordinate pointwise intervals include zero. This residual instantaneous distance is therefore consistent with noisy private updates rather than a systematic difference of theoretical commitments. Finite-run intervals do not prove exact equality.

The new main figure keeps both private- and shared-weight candidate curves visible, displays candidate probabilities separately, and includes the variance control in predictive losses. Single-Gaussian probabilities are never relabeled as oxygen support. The selected-candidate/MAP distance appears only as a secondary diagnostic.

## Why the curves differ from the older three-model figure

The new figure preserves the comparison's visual purpose, not the earlier confounds or its expected curve shapes. Previously, the Gaussian groups began with different release/oxygen references and used retention 1 or .95, while the candidate arm used .3. Here all agents within each Gaussian arm start at one shared reference and all mechanisms use .3. A Gaussian gap therefore develops under selective observation rather than merely relaxing from an imposed initial division.

The main candidate arm now keeps model weights private and has restricted cross-group conditional influence .05; the shared-weight control makes that choice explicit. The primary distance also uses mixture predictions rather than MAP-selected candidate means. The supplement compares the readouts. These changes are documented experimental and measurement choices, not plotting adjustments made to restore a desired visual pattern.

## What to say in the paper

A defensible interpretation is:

> Under matched retention and information access, Gaussian agents retain different numerical predictions because selective observations and reference retention continually sustain a directional difference. A social disagreement gate reduces cross-group influence and enlarges that difference. Agents retaining competing candidate explanations can additionally maintain opposed theoretical preferences when model weights remain private. Sharing those weights removes the opposed preferences and substantially improves prediction for the gas-observing group in this world.

Keep the reference-compression and gas-variance qualifications beside that statement. Avoid the former stronger story that plain Gaussians necessarily agree, gating merely delays agreement, and candidate models uniquely permit persistent disagreement. The new data support distinctions among numerical predictions, explicit model choices and predictive accuracy; they do not establish permanent schisms, causal-graph discovery or a universal advantage of diversity.

See `METHODS.md` for the equations and complete pipeline, `INTEGRATION_GUIDE.md` for placement and wording, and `VALIDATION.md` for independent verification and its scope. The original seven figure assets and manuscript are unchanged.
