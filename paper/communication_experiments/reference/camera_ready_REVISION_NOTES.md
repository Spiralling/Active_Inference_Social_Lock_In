# Camera-ready revision notes

Date: 19 September 2026.

## 1. Scope and controlling versions

The starting manuscript was the latest figures-updated version,
`figure_revision/main.tex`, not the earlier prose-edit manuscript embedded in the
scientific handoff. The revised experiment account uses the completed assessment
in `IWAI_camera_ready_handoff_2026-09-19.zip`, source revision `a869664`.

The later complementary-evidence / belief-separation proposal, exploratory pilot,
and finite model-score-memory configuration are excluded. No such pilot numbers
or plots enter the paper. The author's later unsuccessful runs were not supplied
as a new dataset and are not described as an experiment in this manuscript.
Belief utility remains removed.

The completed assessment contains 29 adaptive conditions: four cycle conditions,
nine network conditions, eleven constant-world conditions, two continuations,
and three adaptation conditions. The separately recorded fixed-model
cross-variable-discrimination check is retained as Appendix F. The revision does
not run a new experiment or silently modify the source data.

## 2. Title, abstract, argument, and literature

The title is now **Paradigm Shifts: Theory Revision and Belief Sharing in
Epistemic Communities**. "Theory Revision" replaces the preceding
"Structure Learning" wording because the principal experiments compare supplied
candidate explanations rather than discover or uniquely identify a causal graph.
The authors and their order are unchanged. Submission metadata should be updated
to match the title and abstract.

The main empirical sequence is delayed theory revision, communication under
selective observation, variation across contact networks and source positions,
and the distinction between theory agreement and agreement about quantities.
The adaptation comparison is a principal result and includes predictive losses,
not just candidate probabilities.

The original claim of sustained opposing theories in the communicating-camp
experiment has been replaced by the result supported by the completed assessment:
both groups select oxygen while retaining different observable estimates. The
constant-world distance and candidate-choice figures are presented together.
Neither a nonzero distance nor a log-odds gap is used as evidence of opposite
candidate commitments.

The philosophical motivation retains the distinction between accommodation and
revision of an explanatory account, along with the role of scientific
communication. The literature discussion retains 34 cited references from the
preceding manuscript. Unused entries are removed from this version's bibliography;
no new bibliographic claims were introduced through outside research. The
bibliography has been checked for internal citation consistency, not exhaustively
reverified against publishers in this turn.

## 3. Exposition and terminology

The main account introduces general conditional probability distributions and
separate candidate probabilities before presenting the Gaussian implementation.
Terms including logarithmic pooling, precision, information vector, entropy,
mean squared error, predictive negative log likelihood, right-censoring, and
bootstrap intervals are defined in context. Captions identify plotted populations,
statistical summaries, interval construction, and timing.

Author-facing instructions, repeated editorial qualifications, unused structural
and utility machinery, and the unverified retrospective-reliability illustration
are omitted. Limitations relevant to the reported results remain in the design,
results, and discussion. Claims are confined to the measured horizons and tested
configurations. The discussion distinguishes local variational updating from
centralized conditioning on all agents' observations.

## 4. Mathematical and indexing audit

All main-text and appendix notation follows the same sequence:

1. State indexed by t enters round t; every agent reads that entering snapshot.
2. Bars indicate the state after communication.
3. The predictive score evaluates datum t before learning from it.
4. A plus sign indicates the learned state before conditional memory.
5. The stored state is indexed t+1 and includes datum t.

Plot labels "Simulation step" use t for that post-update result. The logarithmic
camp-distance plot uses completed updates t+1, as stated in its caption and the
notation appendix. Code agents 0,...,N-1 correspond to mathematical indices
1,...,N. The four candidate coordinates and three observation coordinates are
also mapped explicitly between code and mathematical numbering.

Working and reference distributions are distinguished by subscripts and the
superscript 0. Candidate-specific hidden quantities occupy separate state spaces;
comparisons and single-Gaussian controls use the common observable marginals.
The mathematical reference coefficients, intercepts, means, residual covariances,
and observable covariances are specified numerically in Appendix B.

The appendices provide the normalized distributional update, local variational
objectives, complete Gaussian natural-parameter update, predictive score,
attention-density Jacobian, precision bound, marginal reference construction,
tested social gate, evaluation metrics, experiment inventory, exact graph
specification, and log-odds identities. Model scores do not decay in these runs;
conditional retention acts on both precision and information vector after learning.

Predictive evaluation uses the mixture density, not the density of an averaged
Gaussian. Parameter distance uses each agent's selected candidate. Individual
entropy is computed before population averaging. Noncrossings are censored rather
than assigned an event at the horizon.

## 5. Figures and their provenance

The controlling assets are the PDFs in
`paper/combustion_technical_note/figures/` in the completed handoff. Earlier
standalone PDFs and obsolete results from prior manuscript versions are not used.

| Manuscript figure | Supplied PDF asset(s) | Placement |
| --- | --- | --- |
| 1 | fig01_general_representation.pdf | Main: two-level beliefs and communication |
| 2 | fig_model.pdf | Main: candidate reference dependencies |
| 3 | fig04_cycle.pdf | Main: delayed revision and entropy |
| 4 | fig_adaptation.pdf | Main: communication under selective observation |
| 5 | fig06_topology.pdf | Main: network and informed-source comparison |
| 6 | fig02_precision_pool.pdf | Appendix B: observable precision pooling |
| 7 | fig03_camp_distance.pdf + fig_camp_choices.pdf | Appendix D: distance alongside actual candidate choices |
| 8 | fig_identifiability.pdf | Appendix F: separate fixed-model discrimination |

All nine PDFs are byte-for-byte unchanged. Their original supplied checksums are
recorded in `checks/figure_source_hashes.json`. Figure 7 stacks two assets in
LaTeX, without editing the plotted data. Captions use the completed assessment's
95% whole-seed bootstrap definitions and correct 375-update adaptation tail.
The adaptation caption explains that the source legend's "Full information"
means independent full observation, not centralized pooling. The topology
caption identifies the source's "Cycle" label as the ring graph.

The source cycle plot's dual-axis composition is preserved. No curves were
reconstructed from images or regenerated without the missing raw trajectories.

## 6. Page limits and rendering

The final PDF has **25 pages**:

- Main text and figures: pages 1-11 (11 pages).
- References: pages 12-13 (2 pages).
- Appendices A-F: pages 14-25 (12 pages).

This fits the official IWAI 2026 limits of 12 main-text pages excluding references
and 12 appendix pages including additional references. The standard LNCS class
is retained; body font size, line spacing, and page geometry are not reduced.
The final conclusion ends on page 11 rather than spilling onto a nearly empty
page. All floats are included before the final appendix page-count marker.

Official instructions checked on 19 September 2026:
https://iwaiworkshop.github.io/ (Submissions / Camera-ready / Important dates).
The site lists the camera-ready deadline as 20 September 2026 and requests a
signed copyright form inside the source ZIP.

All 25 pages were rendered for visual inspection. Equations, tables, captions,
and appendix figures were inspected for overlaps, clipping, and legibility.

## 7. Verification and its limits

`checks/validate_manuscript.py` passed **48 checks** covering Gaussian reference
compilation, positive definiteness, observable marginalization, geometric pooling
and anchoring, predictive-score algebra, the attention Jacobian, the variational
identity, a precision bound, entropy decomposition, log-odds recurrences, exact
graph properties, all 29 CSV/JSON condition-summary correspondences, archived
crossing counts and medians, paired summary means, reported rounded loss tables,
selected discrimination percentages, figure hashes, labels, citations, page
limits, and text boxes within PDF bounds.

These are algebraic and transcription checks. They are not reruns of the
29-condition simulation suite or independent reconstructions of all bootstrap
intervals. The handoff omits the companion raw trajectory archive,
`IWAI_camera_ready_data_2026-09-19.zip`. Original parameter arrays, summary data,
figure assets, and selected source snapshots are preserved in `provenance/`.
The original handoff and omitted data archive remain necessary for complete
historical replication and figure regeneration.

The final compilation has no unresolved citations or cross-references, overfull
boxes, or duplicate hyperlink destinations. A standard LNCS/amsmath math-accent
warning and three underfull bibliography boxes remain; none produces clipped
text. A separate clean source-package build is checked before delivery, with
its result in `checks/clean_build.json`.

## 8. Remaining author inputs

The source still contains the original affiliation/email placeholders. These
must be replaced with confirmed author details, followed by recompilation.
The revised title and abstract should match submission metadata. No signed
copyright form was supplied, and none has been signed or fabricated. The authors
must add their completed form before uploading the source ZIP.
