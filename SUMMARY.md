# Collaborator summary: the experiments behind the two IWAI figure versions

Branch: **`collaboration/iwai-figure-experiments`**. It has not been pushed.

This branch is a clean, standalone extraction of the code needed for the figures in `IWAI2026_camera_ready_source.zip` and `IWAI2026_figure_layout_source.zip`. It covers **18 numbered figures, 19 asset appearances and 16 unique PDF assets**. Camera-ready Figure 7 combines two PDFs; the model diagram, probability/entropy cycle and topology plot are shared between versions.

This is an explanation and working codebase for a collaborator. It is **not the final paper's reproduction release**, and it does not reproduce controls or quantitative statements appearing only in the manuscript prose. The original working tree, manuscripts and frozen experiment branches have not been changed. Normal Git ancestry is preserved; the new branch's current tree contains the focused implementation described below.

## Reproduction result

The clean implementation was run with every original seed and setting required by the displayed figures. The full scientific comparison checked **43,923 arrays in 2,651 NPZ files: 1,444,739,973 array elements**, including saved states, random keys, observations, scores, probabilities, predictions, losses and precision histories. Every array matched its archived counterpart **byte-for-byte**, with identical shape and dtype and maximum numerical error zero. No tolerance fallback was needed.

The code then reduced those **fresh runs** to plotting inputs and regenerated all 16 PDFs. **Every PDF was byte-identical to its supplied archive target.** A separate comparison also checked complete pages at 216 dpi, extracted text and page dimensions. Rendering from the compact data supplied on this branch independently reproduced the same 16 PDFs.

An independent adversarial reviewer added and ran nine covariance-form mathematical and invariance tests. They passed without production-model changes. Six short cases drawn from archived production trajectories also passed, covering both communication and Gaussian arms, a late gated state, and the intervention restart. The source extraction had additionally passed 29 short old/new rollout and restart comparisons involving 525 arrays.

Verification records are in [`checks/verification.json`](checks/verification.json), [`checks/full_array_equivalence.json`](checks/full_array_equivalence.json), [`checks/fresh_run_figures.json`](checks/fresh_run_figures.json), and [`docs/INDEPENDENT_REVIEW.md`](docs/INDEPENDENT_REVIEW.md). The recorded environment was Python 3.12.9, JAX/JAXlib 0.9.0, NumPy 2.3.4, SciPy 1.16.3 and Matplotlib 3.10.7, using float64 on macOS arm64 CPU. Bit identity is an observed result in that environment, not a promise across every BLAS, operating system or TeX installation.

## Exactly what is in the branch

| File or directory | Purpose |
|---|---|
| `reproduce.py` | One command-line entry point: simulate, reduce, render and compare. |
| `requirements.txt` | Pinned numerical and plotting dependencies. No runtime orx or Equinox dependency. |
| `src/iwai_figures/engine.py` | The shared Gaussian information updates, predictive scores, conditional pooling, categorical support pooling, retention and social gate. |
| `src/iwai_figures/world.py` | The two small combustion reference models, observation matrix and selective-observation groups. |
| `src/iwai_figures/simulation.py` | Shared checkpointed rollouts, independent predictive evaluation, exact random-state handling and run records. |
| `src/iwai_figures/camera.py` | The 11 older adaptive conditions actually used in the camera-ready figures. |
| `src/iwai_figures/structural_check.py` | The separate fixed-model dependence-discrimination experiment. |
| `src/iwai_figures/communication.py` | Eleven network/camp/control conditions and the two paired intervention-continuation labels. |
| `src/iwai_figures/gaussian.py` | The four matched Gaussian conditions and the two reference-compression constructions. |
| `src/iwai_figures/experiments.py` | Small dispatcher for the three fixed experiment families. |
| `src/iwai_figures/reduce.py` | Raw histories to exactly the per-seed quantities drawn in the figures. |
| `src/iwai_figures/plotting/` | Four plot modules, shared topology drawing, the data loader, one vendored orx style and its LaTeX adapter, and the two required editable TikZ diagrams. The precision diagram's TikZ is generated from the model matrices. |
| `src/iwai_figures/validation.py` | Exact rendered-PDF comparisons against the supplied targets. |
| `configs/` | Three fixed protocols and the four archived topology matrices. |
| `data/` | About 64 MiB of compact, unsmoothed per-seed plotting inputs, checksummed and split into condition files. These are sufficient to redraw every figure without rerunning simulation. |
| `reference/figures/` | The 16 supplied PDFs, used only as comparison targets. |
| `reference/archives.json`, `reference/figure_inventory.json`, `reference/source_map.json` | Archive identities, figure numbering/captions and the exact source lineage. Original protocols are retained here for reference, separately from the clean runtime configs. |
| `reference/simulation_fixtures/` | Six small, preselected production-trajectory regression fixtures with their original chunk hashes. |
| `tests/` | Archived-trajectory checks and independent mathematical tests. |
| `scripts/compare_archived_runs.py` | Optional full raw-array comparison when the earlier raw directories are available. It is not needed for normal simulation or plotting. |
| `checks/` | Environment, completion records, actual execution logs and passed verification reports. |
| `README.md`, `SUMMARY.md`, `NOTICE.md`, `data/README.md`, `docs/INDEPENDENT_REVIEW.md` | Usage, this figure-by-figure explanation, source attribution, data schema and the independent review. |

Unrelated notebooks, prototype model families, historical plotting alternatives, old manuscript sources, whole handoff archives, graph-search machinery, utility terms, structure proposal/pruning and policy-selection code are absent from this tree. The raw reruns occupy about **8.5 GiB** and remain ignored under `runs/`; generated renderings remain ignored under `build/`. No single tracked plotting-data file approaches GitHub's 100 MiB file limit.

The three families are deliberately separate because their initialization, observation schedules and reporting conventions differ. Sharing their numerical engine does not make them one experiment:

| Family | Required scope |
|---|---|
| `camera` | One cycle, four topology conditions, three original camp recipes and three adaptation conditions; plus the separate known-model check. |
| `communication` | Network `N00/N10/N01/N11`; camp `C00/C10/C01/C11`; three one-factor controls; private/enabled support continuations at state 1500. |
| `gaussian` | Plain and gated Gaussians, each with an information-pooled or moment-matched reference. Candidate curves reuse the communication histories. |

## How to use it

Follow the dependency and TeX setup in [`README.md`](README.md). From the branch root:

```bash
# Redraw from the included compact data and compare against both paper archives.
python reproduce.py figures
python reproduce.py compare

# Run every required original experiment, then draw from those new results.
python reproduce.py simulate
python reproduce.py reduce
python reproduce.py figures --data build/recomputed_data --output build/recomputed_figures
python reproduce.py compare --figures build/recomputed_figures --report build/recomputed_comparison.json

# Mathematical and archived-trajectory regression tests.
python -m unittest discover -s tests -v
```

A family can be run separately with `simulate --family camera`, `communication`, or `gaussian`. Output directories must be new; completed results are not overwritten. All full runs use the declared seed lists and original horizons. The reductions retain all plotted timepoints and use the original bootstrap draws. They do not smooth curves, clip bands or select seeds by outcome. Compact data omit unused histories, not inconvenient outcomes.

The figures use real LaTeX Computer Modern/CM-Super fonts and the orx plotting style. Two supplied schematics and the calculated precision illustration use TikZ; the other diagrams and numerical plots use Matplotlib. Original layouts and typography are preserved rather than redesigned. PDF/SVG are vector outputs; PNG files are previews.

## Shared model and update pipeline

Each agent has a Gaussian parameter belief **within each supplied candidate** and a separate categorical probability **over candidates**. The candidates are the release and oxygen accounts. Their unobserved explanatory coordinates `P` and `O` have different meanings; all cross-candidate prediction comparisons use only the common observables `(C,G,M)`.

A reference candidate has linear structural equations `x = B x + b + epsilon`, with independent residual variances `s`. Writing `A = I-B`, the compiler forms reference precision and information vector

\[
\Pi_0=A^\top\operatorname{diag}(s^{-1})A,\qquad
h_0=A^\top\operatorname{diag}(s^{-1})b.
\]

The reference DAGs supply initial distributions and anchors. Updates act on Gaussian natural parameters without searching over DAGs or re-estimating their directed coefficients. Direct-observation deposits are diagonal, so shared zero precision entries are preserved. Most simulations observe an externally prescribed signal plus private sensor noise. The separate identifiability check, described below, instead samples from the supplied candidate distributions.

At each update the implemented order is:

1. **Communicate.** Conditional precision/information pairs are averaged between matching candidate identities when that channel is enabled. Separately, candidate log support is pooled as `L <- (1-alpha_m)L + alpha_m W L` when model-weight exchange is enabled. Conditional communication and support communication are distinct switches.
2. **Generate private observations and score them before learning.** Each agent receives an independent noisy observation of the same signal. Attention weights determine the observation channels used in its Gaussian predictive score and precision deposit. The candidate score uses the already communicated conditional belief but does not reuse the just-deposited datum.
3. **Deposit the observation.** With observation matrix `H`, observation `y` and fixed attention vector `a`, add `H.T diag(a) H / sigma_o^2` to precision and `H.T diag(a) y / sigma_o^2` to the information vector. Add the pre-deposit predictive log density to candidate support.
4. **Retain and anchor.** Replace the updated parameters by `Pi0 + omega (Pi_updated-Pi0)` and the corresponding information-vector expression. Thus `omega=1` retains all accumulated conditional information, while smaller values repeatedly restore the reference. Candidate log scores are not forgotten in these runs.
5. **Read out and evaluate.** Normalize log support with softmax, compute observable predictive means, and evaluate full-signal MSE and likelihood on independent held-out observations. Evaluation uses folded random keys without consuming or replacing the training random stream.

The newer gated Gaussian reweights the **existing weighted social matrix** by `gamma=(nu+1)/(nu+z^2)` and then divides each row by its exact positive sum. The disagreement `z^2` uses observable means and diagonal conditional-variance terms. This gates conditional communication, not new sensory evidence. The older gated camera baseline intentionally retains its original binary-contact normalization, including its `1e-12` denominator constant. Their numerics were preserved separately because silently unifying these formulas would change a reproduction.

The common implementation has no belief-utility contribution, sensory reliability learning, candidate proposal, candidate pruning, model-support gate or action policy. Group membership, attention, references and the signal change are prescribed by the experiment.

## The camp comparisons are different experiments

| Property | Camera-ready Figure 7 | Layout Figure 4 and related controls |
|---|---|---|
| World | Constant oxygen-pattern signal | Same constant signal |
| Groups | Eight gas-only, eight mass-only observers | Same division |
| Seeds/horizon | 20 seeds; 1,500 updates | 50 paired seeds; 5,000 updates |
| Cross-group influence | Uniform complete pooling: total 0.5 | Weighted pooling: total `beta=0.05` |
| Gaussian starting references | Different release/oxygen references for the two groups | One shared reference for all agents |
| Conditional retention | Plain 1; gated 0.95; candidates 0.3 | All primary arms 0.3 |
| Candidate model-weight sharing | Enabled at 0.02 | Private and shared arms compared explicitly |
| Primary distance readout | Highest-weight-candidate/MAP mean | Probability-weighted mixture mean |
| Observed candidate choices | Both groups eventually favour oxygen | Private support retains opposing choices; shared support removes them |

The matched Gaussian result does not show that Gaussian predictions must reach consensus. Selective observations and repeated reference anchoring sustain a directional prediction difference even without categorical alternatives. Gating increases that difference modestly while worsening prediction. The strong candidate split occurs with private model support and is sensitive to reference variance, retention and cross-group influence. Its existence is not automatically evidence of better learning.

The following sections explain every numbered figure in the two supplied versions. The PDF filenames refer to `reference/figures/` or the corresponding generated output directory.

## Figures in `IWAI2026_camera_ready_source.zip`

The camera-ready source has **eight numbered figures using nine PDF assets**. Figure 7 combines two assets. Its experiments predate the restricted-cross-group camp comparison in the figure-layout source. The filenames retain earlier numbering and therefore do not match the manuscript's figure numbers.

Unless specified otherwise, the simulation world supplies a fixed observable signal plus independent Gaussian observation noise. It does **not** generate a fresh sample from a candidate causal graph at each update. Agents receive supplied release and oxygen references; neither candidate is discovered during the simulation. Probabilities and entropies concern those two supplied candidates. Reference DAGs supply initial distributions and anchors; updates act on Gaussian natural parameters without searching over DAGs or re-estimating their directed coefficients. Direct-observation deposits are diagonal, so shared zero precision entries are preserved.

### Camera-ready Figure 1 — General representation and communication

**Asset:** `fig01_general_representation.pdf`.

**Setup:** A schematic introduces private observations, conditional beliefs within each supplied candidate, probabilities over candidates, and communication between agents. It shows two candidate graphs using generic quantities and expands one agent. It makes no Gaussian or combustion-specific assumption. Its editable source is TikZ, compiled by the clean plotting module using the supplied layout.

**Result:** This is a diagram, with no numerical experiment, seed set or statistical result.

**Interpretation:** It explains the two levels of belief that the experiments manipulate separately. The drawing does not imply candidate generation, an action-selection policy, belief utility, or autonomous formation of social groups.

### Camera-ready Figure 2 — Reference dependencies

**Asset:** `fig_model.pdf`, identical to layout Figure 6.

**Setup:** Release has `C → P → M` and `C → G`; oxygen has `C → O → G` and `O → M`. Here `C` is combustion extent, `G` gas uptake and `M` solid mass change. `P` and `O` are distinct, unobserved explanatory quantities. Every edge coefficient has magnitude 0.8; only `P → M` is negative. Residual variances are 0.25 for the hidden explanatory variable and 0.5 for each observable. Reference observable means are `(1,1,−1)` for release and `(1,1,1)` for oxygen. Intercepts enforce these means.

**Result:** A TikZ diagram of the references compiled by the world code; there are no seeds or simulated outcomes. Both candidates connect combustion to mass through their explanatory substance.

**Interpretation:** This is a schematic example, not calibrated chemistry. Only shared observable coordinates are compared between candidates. Hidden `P` and `O` coordinates are never averaged together. The drawn reference dependencies do not demonstrate causal identification from the later persistent-signal experiments.

### Camera-ready Figure 3 — Probability and entropy during revision

**Asset and condition:** `fig04_cycle.pdf`; `cycle`. Identical PDF to layout Figure 2.

**Setup:** Sixteen fully observing agents communicate on a complete graph for 240 updates, using seeds 0–49, observation-noise standard deviation 1, conditional retention `omega = 0.9`, and model-weight sharing `alpha_m = 0.02`. The signal changes from `(C,G,M)=(1,1,−1)` to `(1,1,1)` at generating round 80. Only mass changes. Green is the mean oxygen probability; grey is entropy calculated within each agent and then averaged. Bands are pointwise 95% percentile-bootstrap intervals over whole seed trajectories, using 2,000 resamples.

**Results:** All 50 population means cross oxygen probability 0.9. The median plotted crossing round is 204.5; final mean oxygen probability is 0.968865. The maximum of the mean individual-entropy curve is 0.546870 nats. The amber interval, rounds 100–188, marks post-change mean individual entropy above 0.35 nats.

**Interpretation:** Accumulated support delays revision, and revision passes through uncertainty over the supplied alternatives. The amber threshold is a descriptive display choice; it neither triggers an update change nor defines a validated crisis boundary. A single Gaussian lacks this categorical entropy but can still change its numerical predictions.

**Index convention:** These older plots label the generating round `t` while displaying its resulting state `t+1`. The changed observation at round 80 therefore produces completed state 81. This differs from the completed-state axes used by the newer camp figures.

### Camera-ready Figure 4 — Adaptation under selective observation

**Asset and conditions:** `fig_adaptation.pdf`; `adaptation_social`, `adaptation_isolated`, `adaptation_reference`.

**Setup:** Sixteen agents, seeds 0–19, 1,500 total updates, sensor standard deviation 1 and retention 0.9. All conditions branch from the same 80-update, fully observing and communicating prehistory. After the mass change, eight agents observe only gas and eight only mass. They either communicate or remain isolated. A third reference condition gives each agent all three observation channels independently. The upper panels show the first 500 rounds. The lower panels average the **last 375 updates**, first within each seed, then across seeds. MSE evaluates the mixture mean against the noiseless signal; NLL evaluates four independent full-observation draws per agent and update.

**Results:** Communication leads both groups to oxygen. Isolated gas observers end at mean oxygen probability 0.001444, while isolated mass observers approach one. Independent fully observing agents also approach one. Final-quarter performance is:

| Condition | Observable MSE | Predictive NLL |
|---|---:|---:|
| Networked selective observers | 0.009788 | 4.304792 |
| Isolated selective observers | 0.681840 | 4.935801 |
| Independent full observation | 0.035748 | 4.312920 |

**Interpretation:** Sharing allows useful mass evidence to reach gas observers. The full-observation reference is an independent observer, not a centralized oracle, and has a different total information budget. Its comparison cannot isolate communication while holding all accessible observations constant. An older narrative report mentions a 200-update plotting window; that sentence is stale. The actual supplied figure renderer and manuscript caption both use 375 updates.

### Camera-ready Figure 5 — Network topology and revision

**Asset and conditions:** `fig06_topology.pdf`; `network_cycle`, `network_wheel`, `network_community`, `network_complete`. Identical PDF to layout Figure 3.

**Setup:** Twenty-four agents, seeds 0–49, 600 updates, sensor standard deviation 1, retention 0.9 and model-weight sharing 0.02. After the 80-update fully observing prehistory, agents 0–3 remain fully observing; the remaining 20 use channel weights `(1,0.01,0.01)`. The signal changes only in mass. The graphs are a 24-node ring, a wheel, a two-community graph with sizes `[4,20]`, within-community probability 1, cross-community probability 0.06 and graph seed 0, and a complete graph. Communication uses row-normalized adjacency plus self-links. Black outlines mark the four fully observing nodes. Thin trajectories show all 20 restricted agents in seed 0; heavy curves and bands summarize 50 seed-level restricted-group means.

**Results:** Restricted-group crossings above oxygen probability 0.9 occur in 50/50 seeds for complete, wheel and community graphs, with median plotted rounds 248, 261 and 530. The ring has 0/50 group crossings by the end; its final mean oxygen probability is 0.734864. The community graph ends at approximately 0.9774.

**Interpretation:** These specified networks and source locations produce different finite-horizon revision times. Ring noncrossings are right-censored, not proof of permanent lock-in. Density and source location vary together with topology. The displayed second eigenvalue of the unnormalized Laplacian is descriptive, not an isolated explanatory variable. “Cycle” in the legend means the ring graph.

### Camera-ready Figure 6 — Observable precision pooling

**Asset:** `fig02_precision_pool.pdf`.

**Setup:** Integrate out the hidden variable in each reference, invert the resulting covariance on `(C,G,M)`, and average the two observable precision matrices equally. Information vectors are included in the actual pooled reference but omitted from the drawing. This is a closed-form calculation, with no seeds or simulation.

**Results:** Off-diagonal entries for `(C,G)`, `(C,M)` and `(G,M)` are respectively `(-1.6, +0.969697, 0)` for release and `(-0.780488, -0.780488, -0.390244)` for oxygen. The equal pool gives `(-1.190244, +0.094605, -0.195122)`; labels round these to three decimals.

**Interpretation:** Entries absent from both precisions remain absent under their entrywise average. The drawing shows undirected conditional associations, not directed regression arrows. Importantly, this shared-reference construction is **not** the group-specific reference used by the Gaussian curves in camera Figure 7. It is used by separate shared-reference controls and by the later matched Gaussian comparison.

### Camera-ready Figure 7 — Original camp distances and actual theory choices

**Assets and conditions:** The upper asset is `fig03_camp_distance.pdf`, using `wall_plain_original`, `wall_gated_original` and `wall_rivals_original`. The lower asset is `fig_camp_choices.pdf`, reusing `wall_rivals_original`.

**Setup:** Sixteen agents, seeds 0–19, 1,500 updates, constant signal `(1,1,1)` from the outset, sensor standard deviation 0.5, eight gas-only and eight mass-only observers. The graph is complete with uniform influence `W_ij=1/16`. Plain Gaussian, gated Gaussian and two-candidate configurations use retention 1, 0.95 and 0.3 respectively. Gaussian groups start at different release/oxygen observable references and retain group-specific anchors. Each candidate agent has both references, with model-weight exchange 0.02. The gated arm uses the inherited social mean-disagreement gate. Upper-panel distance uses observable means of each agent's **highest-weight candidate**, with logarithmic axes. The lower panels show actual oxygen probabilities over the first 150 and all 1,500 rounds.

**Results:** Endpoint distances are approximately 0.000256 for plain Gaussians, 0.055324 for gated Gaussians and 0.097304 for candidates. The candidate distance averaged over the final quarter is approximately 0.1083. Nevertheless, both candidate groups end at oxygen probability one, and no seed meets the simultaneous opposed-group criterion. All 20 population means cross 0.9, with median plotted round 16.

**Interpretation:** A residual prediction distance does not establish opposing theories. Retention and reference construction differ between the three configurations, so this plot is not an isolated test of representation count. The lower asset supplies the essential theory-choice check. This experiment differs from the newer restricted-exchange, private-weight camp comparison; the different curve shapes cannot be explained as a cosmetic plotting change.

### Camera-ready Figure 8 — Discrimination from cross-variable dependence

**Asset and condition:** `fig_identifiability.pdf`; `structural_check`.

**Setup:** This separate check uses fresh iid samples from the candidate DAGs, with means set to zero and observable marginal variances matched at `(0.5,0.8648,0.8648)`. Release's gas residual variance changes from 0.5 to 0.5448 for this check. A fixed known-model classifier receives nested prefixes of 1, 5, 20, 100 or 500 observations at measurement-noise standard deviations 0, 0.5, 1 or 2. There are 200 seeds per generating candidate. Noise settings share underlying draws. Parameters do not adapt and agents do not communicate.

**Results:** After 500 samples the joint classifier is correct in 100%, 100%, 100% and 97% of the balanced cases at the four noise levels. At noise standard deviation 1, accuracy is 83.75% after 20 observations and 99.75% after 100. Product-of-marginals and duplicated-candidate controls remain at chance, 50%.

**Interpretation:** Dependence distinguishes this supplied pair after marginal matching. The result does not demonstrate unique causal identification, graph discovery, or structural recovery by the adaptive social model. Covariance signs and path coefficients also differ. Observed 100% accuracy is not a population guarantee. Bootstrap resampling retains both source-candidate sequences for each seed index; noise levels and nested sample counts are not independent replications.

## Figures in `IWAI2026_figure_layout_source.zip`

The layout source has **ten numbered figures and ten PDF assets**. Figures 2, 3 and 6 reuse the earlier cycle, topology and reference-model PDFs exactly. The new communication and matched Gaussian studies use 50 paired camp streams, seeds 10000–10049, and distinguish numerical disagreement from opposing candidate choices. Reusing a stream or an archived candidate arm is not an additional independent replication.

### Layout Figure 1 — Restored graphical abstract

**Asset:** `fig_graphical_abstract.pdf`.

**Setup:** The earlier observations–agent–social-graph illustration is restored without modifying its artwork. It was generated as `fig01_model.pdf` by the publication schematic renderer, using Matplotlib rather than TikZ. The candidate graphs use schematic labels `A,B,C`; they are not the combustion references.

**Result:** No numerical experiment. The supplied asset is byte-identical to the repository's earlier `paper/publication_figures/figures/fig01_model.pdf`.

**Interpretation:** It gives a compact overview of private data, candidate beliefs, model probabilities and communication. This is a different drawing from camera Figure 1, not a different model. It should not be mistaken for an experiment-specific contact graph.

### Layout Figure 2 — Fully observing revision cycle

**Asset and condition:** `fig04_cycle.pdf`; `cycle`. This is the exact camera Figure 3 asset.

**Setup:** Sixteen fully observing agents on a complete graph; 240 updates; seeds 0–49; sensor standard deviation 1; retention 0.9; model sharing 0.02. Mass changes from −1 to +1 at generating round 80 while combustion and gas remain +1. The figure plots oxygen probability and mean individual entropy, with whole-seed 95% bootstrap bands.

**Results:** All 50 population means cross 0.9; median plotted crossing round 204.5; final mean oxygen probability 0.968865. The amber interval covers rounds 100–188, where post-change mean individual entropy exceeds 0.35 nats.

**Interpretation:** This is delayed revision with intermediate uncertainty, not a simulated historical revolution or an endogenous change in the world. The uncertainty threshold is descriptive. Its round axis remains the older convention described under camera Figure 3.

### Layout Figure 3 — Network-dependent revision

**Asset and conditions:** `fig06_topology.pdf`; the four `network_*` topology conditions listed under camera Figure 5. The PDF is unchanged.

**Setup:** Twenty-four agents, seeds 0–49, 600 updates, four fully observing agents at nodes 0–3 and twenty restricted agents with weights `(1,0.01,0.01)` after round 80. Sensor standard deviation 1, retention 0.9 and model sharing 0.02. Ring, wheel, two-community and complete graphs use the archived adjacencies and source positions.

**Results:** Complete, wheel and community graphs have 50/50 restricted-group crossings above 0.9, at median plotted rounds 248, 261 and 530. The ring has no group crossing by the horizon and ends at mean oxygen probability 0.734864.

**Interpretation:** Finite-horizon adoption depends on the specified communication network and source placement. The ring is censored; graph density and source placement prevent treating the plotted Laplacian eigenvalue as an isolated causal factor. Thin lines are individual agents from seed 0, not a confidence band.

### Layout Figure 4 — Matched Gaussian, gated Gaussian and candidate comparison

**Asset and conditions:** `fig_matched_camp_models.pdf`; `G_plain_info`, `G_gated_info`, and archived candidate arms `C10`, `C11`, `C10_gasvar_matched`.

**Setup:** All arms use 16 agents, seeds 10000–10049, 5,000 updates, the constant signal `(1,1,1)`, sensor standard deviation 0.5, retention 0.3 and total cross-group conditional influence `beta = 0.05`. Eight agents observe gas only and eight mass only. Gaussian arms initialize every agent at one common reference obtained by equally pooling the observable marginal precisions and information vectors. The gated arm reweights the same weighted matrix by the inherited disagreement gate with `nu = 0.01`, then normalizes each row. Candidate `C10` keeps model weights private; `C11` shares them with rate 0.02. The variance control matches release's gas reference variance. Panel (a) uses **mixture** prediction distance, panel (b) candidate probabilities, and panel (c) full-observable MSE over states 3,751–5,000.

**Results:** Final-quarter averages are:

| Condition | Instantaneous camp distance | Gas MSE | Mass MSE |
|---|---:|---:|---:|
| Plain Gaussian | 0.502249 | 0.355519 | 0.159392 |
| Gated Gaussian | 0.526806 | 0.368099 | 0.160097 |
| Candidates, private weights | 1.946371 | 1.278857 | 0.019634 |
| Candidates, shared weights | 0.105834 | 0.019563 | 0.019634 |
| Candidates, matched gas variance | 0.111447 | 0.020661 | 0.019634 |

**Interpretation:** Continual selective observations and reference retention sustain a directional Gaussian prediction difference even without a gate. Gating modestly enlarges it while worsening both groups' prediction. The private candidate arm additionally retains opposing theory choices, with a large gas-group predictive cost; sharing model support or matching gas variance removes most of this costly division. Matched retention and access do not make a Gaussian prior identical to a candidate mixture. The candidate histories are reused, not independent confirmations.

**Difference from camera Figure 7:** The earlier Gaussian arms have different initial group references and retention rates; their contact matrix is uniformly complete. This comparison uses one shared Gaussian reference, matched retention, weak cross-group influence and an explicit private/shared candidate-weight contrast. It also changes the primary distance from MAP-selected to mixture predictions. Those are scientific design and measurement differences, not cosmetic changes.

### Layout Figure 5 — Communication channels after a changed signal

**Asset and conditions:** `fig_network_channels.pdf`; `N00`, `N10`, `N01`, `N11`, meaning neither channel, conditional exchange only, model-weight exchange only, and both.

**Setup:** Twenty-four agents, seeds 0–49, 600 updates and a common 80-update fully observing prehistory on the complete graph. Four agents remain fully observing; twenty switch to weights `(1,0.01,0.01)`. The mass signal changes to +1; sensor standard deviation is 1 and retention 0.9. Enabled model sharing has rate 0.02. Conditions share the prehistory and subsequent observation streams. Performance panels average states 451–600. State 80 is the shared state before the first changed observation, which produces state 81.

**Results:** Restricted-group final-quarter means are:

| Communication | Oxygen probability | MSE | NLL |
|---|---:|---:|---:|
| Neither | 0.579145 | 0.325758 | 4.665943 |
| Conditional only | 0.102894 | 0.287230 | 4.615481 |
| Model weights only | 0.923908 | 0.030871 | 4.422448 |
| Both | 1.000000 | 0.004791 | 4.307251 |

Model-weight-only and both-channel conditions cross restricted-group probability 0.9 in all 50 seeds, at median completed states 496 and 249. Neither and conditional-only conditions have no group crossing by the horizon.

**Interpretation:** Model-support communication transmits adoption; adding conditional exchange accelerates it and improves restricted-group prediction. Conditional exchange alone lowers oxygen preference relative to isolation while improving late-window prediction. Preference and prediction are different quantities, and effects are not uniformly beneficial for fully observing agents. Three arms reuse the earlier network seed design; the weight-only arm completes the channel comparison.

### Layout Figure 6 — Reference dependencies

**Asset:** `fig_model.pdf`, identical to camera Figure 2.

**Setup:** Release uses `C → P → M` and `C → G`; oxygen uses `C → O → G` and `O → M`. Edge magnitudes are 0.8 with a negative release-to-mass edge. Hidden residual variance is 0.25 and observed residual variances are 0.5; observable reference means differ only in the sign of mass. The TikZ diagram shows reference structure, not a numerical run.

**Result:** Both candidates mediate combustion's relation to mass through a different hidden substance. No simulation or seed set belongs to this figure.

**Interpretation:** The example motivates the model without claiming historical or chemical realism. Incompatible hidden meanings are kept separate; comparisons use `(C,G,M)`. The actual experiments learn full Gaussian parameters rather than discovering these supplied DAGs.

### Layout Figure 7 — Theory choices across communication conditions

**Asset and conditions:** `fig_camp_channels.pdf`; `C00`, `C10`, `C01`, `C11`.

**Setup:** Sixteen agents, 50 paired seeds 10000–10049, 5,000 updates, constant signal `(1,1,1)`, sensor standard deviation 0.5, retention 0.3, and cross-group conditional weight 0.05. Eight agents observe gas only, eight mass only. The four arms independently enable conditional exchange and model-weight exchange, with model rate 0.02 when enabled. Curves show group mean probabilities; strips show individual gas-observer probabilities at states 1,500 and 5,000. Each strip contains 400 individuals clustered within 50 independent seeds.

**Results:** Late gas-group oxygen probabilities are approximately `1.85e−9` with neither channel, `6.21e−6` with conditional exchange only, and one in either model-weight-sharing arm. Mass observers strongly favor oxygen. The primary opposed-group criterion passes in 50/50 seeds for neither and conditional-only exchange, and 0/50 for weight-only and both-channel exchange, at both reporting horizons. The stricter criterion requiring all eight individuals per group passes conditional-only in 47/50 seeds at 1,500 and 50/50 at 5,000.

**Interpretation:** Opposing choices can persist over this horizon with positive conditional communication when model support stays private. Communication does not create the division: isolation also divides. The result is not opposition under unrestricted exchange of both kinds of information, and finite-window occupancy does not prove permanent fixation. The 400 individual points are not 400 independent replications.

### Layout Figure 8 — Reference variance and memory diagnostics

**Asset and conditions:** `fig_reference_diagnostics.pdf`; `C10`, `C10_beta50`, `C10_omega90`, `C10_gasvar_matched`.

**Setup:** The same 50 camp streams and 5,000-update constant-world experiment, with conditional exchange and private model weights. Controls change only total cross-group influence from 0.05 to 0.5, retention from 0.3 to 0.9, or release's gas residual variance from 0.5 to 0.5448. The variance control matches the candidates' marginal gas variance while preserving their reference means and regression coefficients. Panel (a) averages local oxygen-minus-release predictive scores over states 3,751–5,000; grey ticks show stationary expectations calculated before confirmation. Panel (b) plots **endpoint** oxygen probabilities, not late-window averages.

**Results:** Measured gas scores are −0.003018703 for the primary condition, +0.013823549 for stronger exchange, +0.001135666 for higher retention, and +0.001086645 after variance matching. Corresponding analytical expectations are −0.003022079, +0.013773431, +0.001130619 and +0.001078491 nats per agent per update. All three controls remove the primary opposition outcome. Their late-window gas probabilities are approximately 1, 0.938449 and 0.982782; these are supporting summaries, **not** the endpoint values displayed in panel (b).

**Interpretation:** A small persistent score advantage from reference uncertainty can accumulate into confident commitment. Changing reference variance reverses that advantage. Stronger conditional influence or weaker reference anchoring also changes its sign. The split is consequently not an isolated effect of graph wiring or an unavoidable consequence of retaining alternatives. The variance control changes the full reference distribution even though means and coefficients remain fixed.

### Layout Figure 9 — Matched-reference and readout sensitivity

**Asset and conditions:** `fig_matched_camp_diagnostics.pdf`; all four Gaussian arms (`G_plain_info`, `G_gated_info`, `G_plain_moment`, `G_gated_moment`) and candidate arms `C10`, `C11`.

**Setup:** The same 50 paired camp streams, 16 agents, 5,000 updates, retention 0.3, sensor standard deviation 0.5 and base cross-group weight 0.05. The alternative Gaussian reference matches the equal candidate mixture's mean and covariance, including between-candidate mean variation. Both Gaussian constructions first integrate out hidden variables. The four panels compare reference constructions, realized cross-group influence, MAP versus mixture distances, and three different orders of averaging signed group differences and taking their norm.

**Results:** Plain/gated final-quarter distances are 0.502249/0.526806 with information pooling and 0.701686/0.759441 with moment matching. The gated effective cross-group weights average 0.006389/0.006869 for gas/mass groups under information pooling and 0.007398/0.008809 under moment matching, versus the base 0.05. Shared candidate weights leave instantaneous distance 0.105834, but the mean norm after time averaging within each seed is 0.003850. After also averaging signed differences across seeds, the descriptive norm is approximately 0.0000441.

**Interpretation:** Reference compression changes the comparison more than the gate increment here. Information pooling gives reference mean approximately `(0.623448,0.644510,0.008680)`, whereas moment matching gives `(1,1,0)`. Moment matching increases separation while improving MSE, so separation is not a monotonic measure of predictive quality. Shared-candidate residual instantaneous distance is consistent with noisy private updates rather than a sustained directional theoretical split. The across-seed norm has no inferential interval; signed-coordinate intervals supply the directional check.

### Layout Figure 10 — Enable model-weight sharing after a split forms

**Asset and conditions:** `fig_weight_sharing_intervention.pdf`; `S_keep_private`, `S_enable_weights`, branching from every `C10` seed.

**Setup:** All 50 primary conditional-only runs branch at exactly completed state 1,500. One continuation keeps weights private; the other enables model-weight sharing at rate 0.02. Conditional beliefs, references, local predictive scores, observations and random states are identical between branches. The full continuation runs to state 5,000; the zoom shows the first 100 updates. Zero on the figure axis is the shared pre-intervention state.

**Results:** All seeds initially meet the group-average opposition criterion. Enabling weight sharing removes opposition in all 50: the first 20-state nonopposing interval begins at state 1,503 and becomes detectable at 1,522. Both groups first exceed mean oxygen probability 0.9 at median state 1,508; a 20-state interval becomes detectable at median state 1,527. The private branch retains opposition. Gas-group late MSE changes from 1.278857 to 0.019563, a paired reduction of 1.259294 with 95% bootstrap interval [1.259026,1.259576]. Its NLL falls from 4.832537 to 2.841172. Mass-group late metrics are unchanged to recorded precision.

**Interpretation:** This fixed-time intervention isolates the effect of exchanging accumulated model support. Gas observers' local scores still favor release, but transmitted support changes their mixture and its predictions. The experiment shows no autonomous bistability or confidence-driven change in attention/trust. Loss of the opposition threshold and subsequent strong oxygen adoption are different events; their detection times also differ from their first qualifying states.
