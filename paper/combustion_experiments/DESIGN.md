# Combustion-mediated explanations: fixed revision

This assessment implements the graph revision agreed with the user. It is a
child of the completed four-variable assessment, not a replacement for that
run's evidence. No parameters will be selected to obtain a preferred outcome.

## Scientific change

The shared variables are combustion extent C, net gas uptake G, and solid mass
change M. Each supplied explanation has one unobserved variable: hypothetical
phlogiston released P, or oxygen incorporated O. Oxygen availability before a
reaction is not what O denotes. C indexes reaction extent, not ignition.

The release graph is C → P → M, plus C → G. The oxygen graph is C → O,
O → G, O → M. All edge coefficients have magnitude 0.8; only P → M is
negative. Both models connect combustion to mass change through a substance.
These are schematic accounts, not calibrated historical chemistry.

Storage order is (z,C,G,M); z means P or O within its own candidate. The
topological order is (C,z,G,M). General matrix compilation is exact in this
storage order, and the accepted inference engine remains unchanged. The
explanatory variable has residual variance 0.25; each observable has residual
variance 0.5. Joint reference means are (0,1,1,-1) and (0,1,1,1). Intercepts
are computed as (I-B)mu. Values are standardized; zero in the centred P/O
coordinate is not absence of the substance.

With independent residuals, Cov(G,M | true C) is zero in the release prior
and 0.16 in the oxygen prior. Noisy measurement of C does not imply the same
conditional-independence statement for measured variables. No causal graph
discovery or identification against every possible DAG is claimed.

## Main experiments retained

The same 29 conditions, seeds, observation noise, memory settings, attention
weights and communication graphs as the previous assessment are rerun. Cycle
and network conditions use 50 seeds; selective-evidence and adaptation
conditions use 20. The exact protocol is in protocol.json.

The prescribed persistent signal changes only M at step 80, from -1 to +1;
C=G=+1 throughout. The selective-evidence experiment uses the complete oxygen
reference (C,G,M)=(1,1,1), with one camp observing G and the other M. Its legacy
file prefix is wall_, but it no longer imposes opposing gas/mass mean evidence.
Gas-only evidence can still distinguish the candidates weakly through their
different predictive variances. A persistent split is an empirical question.

Controls include no world change, no forgetting, one Gaussian, matched memory,
shared anchors, no model-score sharing, no communication, alternative source
placement, zero direct gas/mass attention, and full attention. The long
continuation and source-off contraction check are retained. First-crossing
statistics for a changing world start at step 80 and preserve censoring.

Primary readouts are candidate probability, mean individual entropy, camp
distance in shared observable coordinates, observable MSE, and proper
held-out mixture negative log density. P/O estimates are never averaged across
candidate meanings. Sharing interventions branch from identical prehistories.

## Separate structural-discrimination check

Main experiments estimate a persistent state from noisy repeated observations.
They do not sample a new latent world each step and cannot isolate structural
identification. A separate fixed-candidate Bayesian calculation therefore
samples independent draws from each DAG, with all reference means zero and
matched individual observable variances. Raising only the release model's G
residual variance from 0.5 to 0.5448 matches observable variances at
(0.5,0.8648,0.8648) without changing either graph. Remaining distinctions are
cross-variable relationships.

The prespecified grid uses observation noise standard deviations 0, 0.5, 1, 2;
200 seeds per generating candidate; and prefixes of 1, 5, 20, 100, 500 draws.
It saves samples, accumulated scores, posterior trajectories and recovery
confusion matrices. Product-of-marginals and duplicated-candidate controls
must remain uninformative. The positive result, if obtained, means supplied
Gaussian distributions can be distinguished when their relationships are
observed. It is not a test of the adaptive social engine or general causal
identification. No outcome threshold is used to tune the models.

## Provenance and scope

The fixed orx command still names four_variable_experiments/run.py. On this
child, its no-argument entrypoint explicitly dispatches to this package. The
old suite can still be run with its explicit --protocol and --output arguments;
its source functions, world, protocol and existing figures remain intact.
The new suite can also be reproduced directly with this package's run.py.
All scientific data come from a committed isolated orx snapshot.

There is no belief utility, active experiment selection, candidate generation,
pruning, or retrospective evidence reweighting. Original utility/prototype
figures are outside this suite. Candidate priors differ in mean and covariance
in the main experiments; only the separate structural check matches means and
individual variances. The accepted manuscript is not edited by this run.
