# Independent adversarial design audit

This audit was written before examining production trajectories. The numerical predictions in `STATIONARY_PREDICTIONS.json` were derived independently in NumPy/SciPy; that file records the pre-production UTC time and protocol SHA-256. The implementation tests use validation seeds 900100–900109, outside the scientific paired streams.

## What this comparison can identify

Plain and gated Gaussian arms differ only in the social weighting rule within a fixed reference construction. Their paired differences isolate that gate within this setup. Changing the reference construction is a prespecified sensitivity comparison. The comparison with two candidates is a comparison of complete representational/update systems, not a single-parameter ablation: a single Gaussian cannot preserve a two-component distribution or its separate model-weight dynamics. Matching retention, communication graph and observations does not remove that representational distinction.

The common Gaussian reference must be constructed on the observable marginal distributions, after integrating out each theory's distinct hidden explanatory variable. Using the observable block of a joint precision instead would condition on the unobserved variable; pooling the full joint Gaussian references would identify phlogiston and oxygen. Neither operation is used here. An independent dummy coordinate provides the four-state engine interface but is excluded from observations, gate distances and figures.

Information pooling and moment matching imply materially different references. Equal information pooling gives common observable mean approximately (0.623448, 0.644510, 0.008680), whereas moment matching gives (1, 1, 0). Information pooling averages precision and information, not means; different correlation structures shift all coordinates. Moment matching must include the between-component mass-mean variance of 1. These are substantive modeling choices, not interchangeable numerical implementations.

## Gate semantics

Let W be the original weighted communication matrix. Each row gives total weight 0.95 to its own eight-agent group and 0.05 to the other group, including the self weight in the former. The gated matrix is

    W^g_ij = W_ij gamma_ij / sum_l W_il gamma_il,
    gamma_ij = (nu+1)/(nu+z_ij^2),
    z_ij^2 = sum_v (mu_iv-mu_jv)^2 / (1/Pi_i,vv+1/Pi_j,vv),

with v running over C,G,M and nu=0.01. All inputs are entering states, before observation and updating. The reciprocals of precision diagonals are conditional variances; they are not marginal posterior variances. The archived gate uses this convention, and it should be named correctly.

Replacing W by binary support would silently turn the initial cross-group weight from 0.05 to 0.5. Adding an epsilon to the row denominator would introduce an avoidable sub-stochastic mass deficit. The wrapper avoids both. Positive self weight guarantees a positive denominator. At equal observable means and in the infinite-nu limit, the gate recovers the weighted W. The finite Student-t gate remains positive: reduced communication is not a disconnected graph. Gates do not consume randomness or alter the sensory observation stream.

## Independent stationary prediction for the plain arms

Write Q_i,t and b_i,t for the three-dimensional observable precision and information vector. With shared reference Q_0,b_0, retention omega, observations y_i,t of the fixed signal x=(1,1,1), diagonal binary attention D_i, and sigma=0.5,

    Q_i,t+1 = (1-omega) Q_0 + omega [sum_j W_ij Q_j,t + D_i/sigma^2],
    b_i,t+1 = (1-omega) b_0 + omega [sum_j W_ij b_j,t + D_i y_i,t/sigma^2].

At stationarity the stacked expectations solve linear systems with I-omega W. Q is deterministic. For b, let A=omega(W tensor I_3). Its innovation covariance is blockdiag(omega^2 D_i/sigma^2), since attention is binary and observation noises are independent. The stationary covariance V_b solves V_b=A V_b A^T+Q_noise. Multiplying by blockdiag(Q_i^-1) on both sides gives the covariance of posterior means. The camp-centroid difference uses coefficients +1/8 for gas agents and -1/8 for mass agents.

This produces the following predictions before simulation:

| Reference | Norm of expected centroid difference | RMS instantaneous centroid difference | Gas MSE | Mass MSE |
|---|---:|---:|---:|---:|
| Information pool | 0.495893 | 0.506370 | 0.355559 | 0.159391 |
| Moment match | 0.696658 | 0.708473 | 0.311082 | 0.044432 |

MSE averages coordinate-wise squared deviation from the noiseless full signal, including the stochastic posterior-mean variance. RMS centroid difference is sqrt(E||difference||^2). It is not E||difference||; these must not be compared as if they were equal. The norm of the expected signed difference is also distinct from the expected norm of a finite-window average. The full signed predictions and calculation residuals are retained in the JSON.

Thus a persistent numerical prediction difference in the plain Gaussian is expected. Distinct ongoing observations and reference retention act continuously; this is not the closed-system averaging process for which a consensus contraction theorem applies. No ordering of the three mechanisms is assumed, and a large instantaneous norm alone would not establish a systematic gap: noisy finite groups can have nonzero norm even when expected signed differences vanish. Report the time-averaged signed differences and their uncertainty as well.

## Interpretation boundaries

- A K=1 probability is identically one and is not evidence of commitment to oxygen or release. Only K=2 model weights have that interpretation.
- Candidate probabilities must retain the archive's release=0, oxygen=1 labels.
- Candidate predictive distance uses posterior mixture means as the primary readout. Selecting the most likely candidate is a secondary, potentially discontinuous readout.
- All observable comparisons exclude hidden P/O coordinates. Their meanings are different.
- Persistent opposition in the private-weight candidate condition does not demonstrate opposition under exchange of model support; the shared-weight control must remain visible.
- Fifty paired seeds support uncertainty across repeated noise streams. The 16 agents and 5,000 rounds within a seed are dependent observations, not extra replications.
- The Gaussian arms are a paired extension of an existing experiment, not an independent confirmation on new seed streams.
- The signal is imposed and constant, with sensor noise. These data do not identify a unique causal graph or demonstrate discovery of new structures.
- Predictive accuracy and persistent disagreement are separate outcomes. Neither greater distance nor a gate-induced reduction of communication is automatically desirable.
