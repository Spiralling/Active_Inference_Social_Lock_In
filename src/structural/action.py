"""The action layer: pricing the three structural moves on one ledger.

``src/structural/bmr.py`` supplies the *scorers* (the only place an inverse
appears); ``belief.border`` / ``linalg.savage_dickey`` supply the *moves*. This
module turns them into a **decision**: at each step an agent chooses among

  * ``sample``  -- ordinary inference, leave the structure alone (score 0);
  * ``reduce``  -- prune an edge the data no longer support (closed-form ``ΔF``,
                   Savage-Dickey -- cheap, no new inversion);
  * ``expand``  -- *wake* a dormant hub: propose a new common-cause node, wired to
                   the commitments where a residual correlation localizes, and pay
                   the one real inversion to score it.

Three selection problems, one ledger (Friston et al. 2023, "Supervised structure
learning": every structural move is scored by ``ΔF + ΔG``, with ``ΔG`` the expected
information gain used as a log model prior -- so *exploration is identified with
expansion*, no bonus added). The cost asymmetry the paper turns on (cheap reduction
within a fixed support, expensive expansion of the support) is not bolted on: it is
the prior/likelihood line of Eq. (5). Reduction reads ``ΔF`` off the *existing*
posterior; expansion borders a node and re-evaluates the log-evidence.

The proposal mechanism is the paper's Eq. (2) read backwards. Marginalizing a hidden
hub induces, among its neighbours, a coupling where the joint had a zero (the Schur
fill-in ``-Π_ab Π_bb^{-1} Π_ba``). So an *un*explained partial correlation among
commitments -- one reduction cannot prune, sitting in the residual after BMR has done
all it can -- is the fingerprint of a hub not yet drawn. ``propose_hub`` eigendecomposes
that residual block: the leading eigenvalue is the height of the stubborn floor (the
expansion *trigger*) and the leading eigenvector is the hub's coupling pattern (a single
common cause induces a rank-1 fill-in). Reduction lowers ``F`` within a support;
expansion lowers the floor of the support; the trigger to expand is a high, localized
floor.

"Wake a dormant hub" is the reservoir reading of expansion: a node bordered with
``couplings = 0`` is structurally inert (``test_reserved_slot_disconnected_is_a_noop``)
until the data wire it in, so expanding and reducing are two directions of one
closed-form move on one precision cube -- with the genuinely-unconceived (outside the
cube) expansion left as the explicitly-bracketed Knightian frontier.

See ``notes/`` and ``notebooks/37_model_expansion_action.ipynb``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import jax
import jax.numpy as jnp

from src.structural import belief, bmr, linalg
from src.structural.belief import GaussianBeliefNet
from src.structural.dual_field import PrecisionUtilityNet


# ----------------------------------------------------------------------
# 1. The inverse-Schur proposal: where does a hub want to be drawn?
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class HubProposal:
    """A candidate hub read off the residual coupling block (inverse-Schur).

    ``strength``    : ``|λ_max(R)|`` -- the height of the residual floor, the
                      EXPANSION TRIGGER. High => a stubborn correlation the current
                      structure cannot account for.
    ``pattern``     : (k,) unit leading eigenvector over the neighbour block -- the
                      hub's coupling *pattern* (sign and relative weight per neighbour).
    ``neighbours``  : the node names the pattern is over (the proposed hub's children).
    ``rank1_ratio`` : ``|λ_max| / Σ|λ_i|`` in [1/k, 1] -- how LOCALIZED the floor is.
                      Near 1 => the residual is rank-1 => a *single* common cause
                      explains it (the clean case for one hub); near 1/k => diffuse.
    """

    strength: float
    pattern: jax.Array
    neighbours: tuple[str, ...]
    rank1_ratio: float


def _zero_diag_sym(R: jax.Array) -> jax.Array:
    """Symmetrize and zero the diagonal -- keep only the off-diagonal couplings."""
    R = R - jnp.diag(jnp.diagonal(R))
    return 0.5 * (R + R.T)


def residual_block(net: GaussianBeliefNet,
                   neighbours: tuple[str, ...]) -> jax.Array:
    """The residual coupling block ``R`` among ``neighbours`` -- the off-diagonal of
    their conditional precision (the partial correlations, holding the rest of the net
    fixed). A non-zero entry is an induced correlation that either a direct edge or a
    *hidden common cause* must explain; after BMR has pruned the edges the data do not
    support, what remains here is the fingerprint of a hub not yet drawn (Eq. 2 read
    backwards).

    Reads the submatrix of ``Π`` directly (the conditional precision is the submatrix in
    information form) -- no inverse, so it is well-defined even on the improper hub prior,
    exactly as ``efe.py`` reads ``Π[v,v]``. Returns the symmetric (k, k) block with the
    diagonal zeroed (only the couplings carry the residual).

    NOTE: in this linear-Gaussian model ``Π``'s off-diagonal is set by the observation
    operator (``J = HᵀH/σ²``), not by the data, so a NODE-WISE agent's residual is NOT here
    -- it is in the means / prediction errors. Use ``residual_from_errors`` + the ``residual=``
    argument of ``propose_hub`` for that case (see ``world_net.py``)."""
    idx = net.indices(neighbours)
    return _zero_diag_sym(net.Pi[jnp.ix_(idx, idx)])


def residual_from_errors(errors: jax.Array) -> jax.Array:
    """The residual block read from PREDICTION ERRORS rather than ``Π`` -- the footprint a
    hidden cause leaves when the observation operator keeps ``Π`` diagonal.

    ``errors`` (T, k): per-step prediction errors ``o_t - prediction_t`` on the ``k`` candidate
    nodes. Returns the mean UNCENTERED second moment ``(1/T) Σ e_t e_tᵀ`` with the diagonal
    zeroed. Uncentered on purpose: a hidden cause with a non-zero mean shifts the driven nodes
    COHERENTLY, so ``E[e_i]E[e_j]`` (the shift product) is part of the signal, not noise to remove
    -- its leading eigenvector is the cause's direction. Feed this to ``propose_hub(net, nbrs,
    residual=...)``."""
    E = jnp.asarray(errors)
    return _zero_diag_sym((E.T @ E) / E.shape[0])


def propose_hubs_topk(net: GaussianBeliefNet, neighbours: tuple[str, ...],
                      residual: jax.Array | None = None, k: int = 1
                      ) -> tuple[HubProposal, ...]:
    """The top-``k`` eigenpairs of the residual block as COMPETING hub candidates.

    The limitations section names the single pre-allocated slot with its rank-one residual
    reading as the model's narrow waist: "a real discovery process entertains many structures
    at once". This is the searched version: each eigenpair of the residual is the best
    ``m``-th orthogonal common-cause explanation, returned in descending ``|lambda|`` order
    so they compete on the same ledger (``expansion_score`` each, accept the winner -- or
    every positive one). ``k = 1`` reproduces ``propose_hub`` exactly. The couplings remain
    ESTIMATED from the residual (eigenvalue and eigenvector), not selected from a fixed set.
    """
    R = _zero_diag_sym(residual) if residual is not None else residual_block(net, neighbours)
    evals, evecs = jnp.linalg.eigh(R)
    absvals = jnp.abs(evals)
    order = jnp.argsort(-absvals)
    total = float(absvals.sum()) + 1e-12
    out = []
    for m in range(min(int(k), int(R.shape[0]))):
        top = int(order[m])
        s = float(absvals[top])
        out.append(HubProposal(strength=s, pattern=evecs[:, top],
                               neighbours=tuple(neighbours), rank1_ratio=s / total))
    return tuple(out)


def coupling_candidates(residual: jax.Array, neighbours: tuple[str, ...],
                        top_m: int = 3) -> tuple[tuple[str, str, float], ...]:
    """Candidate NEW COUPLINGS among existing commitments, read off the residual block --
    the second structure family the limitations name beyond a single hub.

    Returns the ``top_m`` largest-|magnitude| off-diagonal entries of the residual as
    ``(node_a, node_b, magnitude)`` triples (the magnitude is the residual entry itself --
    estimated from the data's error structure, not drawn from a fixed candidate set). Score
    each with :func:`coupling_score` on the same ledger as the hub candidates."""
    R = np.asarray(_zero_diag_sym(residual))
    k = R.shape[0]
    pairs = [(abs(R[i, j]), i, j) for i in range(k) for j in range(i + 1, k)]
    pairs.sort(reverse=True)
    return tuple((neighbours[i], neighbours[j], float(R[i, j]))
                 for (_, i, j) in pairs[: int(top_m)])


def _pd_capped_edge(Pi: jax.Array, a: int, b: int, s: float,
                    fraction: float = 0.7) -> float:
    """Cap an off-diagonal precision edit so the 2x2 principal minor stays PD with margin
    (followed by an exact eigenvalue check in :func:`coupling_score`)."""
    bound = float(jnp.sqrt(Pi[a, a] * Pi[b, b]))
    lim = fraction * bound - float(Pi[a, b]) * np.sign(s) if bound > 0 else 0.0
    return float(np.clip(s, -abs(lim), abs(lim)))


def coupling_score(net_post: GaussianBeliefNet, net_prior: GaussianBeliefNet,
                   edge: tuple[str, str], magnitude: float,
                   accept_eps: float = 1e-4) -> MoveScore:
    """Score ADDING a coupling between two existing nodes -- expansion within the support.

    The same model-log-Bayes-factor construction as ``expansion_score``, for the edit that
    writes a residual-estimated off-diagonal precision entry into both the posterior and the
    prior (the augmented model must carry the same wiring in both, so the d-dim normalizers
    cancel inside each bracket):

        delta_F = [logZ(post+e) - logZ(prior+e)] - [logZ(post) - logZ(prior)].

    ``delta_G = 0``: no new latent is introduced, so there is no epistemic-gain subsidy --
    couplings among observed commitments must pay in evidence alone (the prior/likelihood
    line of the ledger, same standing as reduction). The magnitude is PD-capped (2x2 minor
    bound, then halved against an exact eigenvalue check) so the edit never makes either
    net improper; a zero residual entry wires nothing and is correctly declined."""
    a, b = net_post.names.index(edge[0]), net_post.names.index(edge[1])
    s = _pd_capped_edge(net_post.Pi, a, b, float(magnitude))
    s = _pd_capped_edge(net_prior.Pi, a, b, s)

    def _edited(net, s_val):
        Pi = net.Pi.at[a, b].add(s_val).at[b, a].add(s_val)
        return GaussianBeliefNet(Pi=Pi, h=net.h, names=net.names)

    for _ in range(8):                            # exact PD guard (cheap: d <= ~12)
        if s == 0.0:
            break
        ok = all(float(jnp.linalg.eigvalsh(_edited(n, s).Pi)[0]) > 1e-9
                 for n in (net_post, net_prior))
        if ok:
            break
        s *= 0.5
    else:
        s = 0.0

    if s == 0.0:
        return MoveScore(move="couple", delta_F=0.0, delta_G=0.0, delta_U=0.0,
                         score=0.0, accept=False,
                         detail={"edge": edge, "magnitude": 0.0})
    post_e, prior_e = _edited(net_post, s), _edited(net_prior, s)
    delta_F = float(
        (bmr.log_evidence(post_e) - bmr.log_evidence(prior_e))
        - (bmr.log_evidence(net_post) - bmr.log_evidence(net_prior))
    )
    return MoveScore(move="couple", delta_F=delta_F, delta_G=0.0, delta_U=0.0,
                     score=delta_F, accept=bool(delta_F > accept_eps),
                     detail={"edge": edge, "magnitude": s})


def propose_hub(net: GaussianBeliefNet, neighbours: tuple[str, ...],
                residual: jax.Array | None = None) -> HubProposal:
    """Eigendecompose the residual block ``R`` and read off the dominant hub.

    A single hidden hub ``b`` coupled to the neighbours with vector ``c`` induces, on
    marginalizing, the rank-1 fill-in ``-c cᵀ / Π_bb`` (Eq. 2). So the leading eigenpair
    of ``R`` *is* the best single-hub explanation of the residual: ``|λ_max|`` is the
    floor height (the trigger), and the eigenvector is the coupling pattern ``c ∝ v``.
    The keep-or-prune verdict on the *magnitude* is then left to ``expansion_score``
    (the data decide via ``ΔF``); the eigenvector only fixes the *direction*.

    ``residual=None`` reads the block from ``Π`` (``residual_block`` -- right on the relational
    substrate where the data deposit off-diagonal Fisher). Pass an explicit ``(k, k)`` block (e.g.
    from ``residual_from_errors``) for the node-wise case where the footprint is in the errors."""
    R = _zero_diag_sym(residual) if residual is not None else residual_block(net, neighbours)
    evals, evecs = jnp.linalg.eigh(R)             # symmetric => real spectrum
    absvals = jnp.abs(evals)
    top = int(jnp.argmax(absvals))
    strength = float(absvals[top])
    total = float(absvals.sum()) + 1e-12
    return HubProposal(strength=strength,
                       pattern=evecs[:, top],
                       neighbours=tuple(neighbours),
                       rank1_ratio=strength / total)


# ----------------------------------------------------------------------
# 2. The wake move: border the dormant hub at the proposed couplings.
# ----------------------------------------------------------------------

def hub_couplings(net: GaussianBeliefNet, proposal: HubProposal,
                  coupling_scale: float) -> jax.Array:
    """The full-basis (d,) coupling vector for the woken hub: ``coupling_scale * pattern``
    on the proposed neighbours, zero elsewhere. ``coupling_scale`` is the amount of edge
    precision the wake commits (see ``pd_safe_scale`` for the default)."""
    c = jnp.zeros((net.dim,))
    idx = net.indices(proposal.neighbours)
    return c.at[idx].set(coupling_scale * proposal.pattern)


def _pattern_dir(net: GaussianBeliefNet, proposal: HubProposal) -> jax.Array:
    """The unit coupling pattern embedded on the full (d,) basis (zero off-neighbours)."""
    return hub_couplings(net, proposal, 1.0)


def pd_safe_scale(nets: tuple[GaussianBeliefNet, ...], proposal: HubProposal,
                  hub_self_prec: float, fraction: float = 0.7) -> float:
    """The largest coupling scale that keeps every bordered net positive-definite, times a
    safety ``fraction``. Bordering a hub at couplings ``c = s·v`` with self-precision ``p``
    is PD iff the Schur scalar ``S = p - cᵀ Π⁻¹ c = p - s²(vᵀ Π⁻¹ v) > 0``. So the PD bound
    on ``s`` is ``√(p / max_net(vᵀ Π⁻¹ v))`` and we wire at ``fraction`` of it (``S`` keeps a
    ``1 - fraction²`` margin). ``vᵀ Π⁻¹ v`` is the one genuine ``solve`` the expansion pays --
    the "real inversion" the paper says reduction is spared. Robust to the improper prior
    via a clipped projection (a non-PD incumbent contributes no positive bound)."""
    bound_sq = float("inf")
    for net in nets:
        v = _pattern_dir(net, proposal)
        proj = float(v @ jnp.linalg.solve(net.Pi, v))
        if proj > 1e-9:
            bound_sq = min(bound_sq, hub_self_prec / proj)
    if not jnp.isfinite(bound_sq):
        return 0.0
    return float(fraction * jnp.sqrt(bound_sq))


def default_coupling_scale(nets: tuple[GaussianBeliefNet, ...], proposal: HubProposal,
                           hub_self_prec: float, fraction: float = 0.7) -> float:
    """The default wake magnitude: the inverse-Schur reproduction scale ``√(strength·p)``
    (so a hub reproduces the residual fill-in ``c cᵀ/p ≈ strength·v vᵀ``), CAPPED by the
    PD-safe bound. A *zero* residual floor (``strength = 0``) therefore wires nothing -- the
    hub stays inert and earns no evidence or epistemic gain, so a no-residual control is
    correctly declined; a residual taller than the PD bound allows is wired as strongly as
    the joint stays proper."""
    want = float(jnp.sqrt(proposal.strength * hub_self_prec))
    return min(want, pd_safe_scale(nets, proposal, hub_self_prec, fraction))


def wake_hub(net: GaussianBeliefNet, proposal: HubProposal, hub_name: str,
             hub_self_prec: float, coupling_scale: float | None = None,
             h_new: float = 0.0) -> GaussianBeliefNet:
    """EXPAND: border ``net`` with the woken hub -- a new latent common cause wired to
    ``proposal.neighbours`` at the inverse-Schur pattern. ``coupling_scale=None`` ties the
    committed edge precision to ``√strength`` (the rank-1 magnitude). The hub is latent
    (``h_new=0`` by default: no direct observation). Returns the (d+1)-node net.

    This is the one move that grows the matrix -- the "real inversion" cost lands when the
    bordered net's log-evidence is evaluated in ``expansion_score``. A latent hub among
    already-observed nodes stays prior-class (Savage-Dickey-scorable); a hub that opened a
    new *observable* channel would enlarge the likelihood -- the same Eq. (5) line.

    ``coupling_scale=None`` uses ``default_coupling_scale`` (residual-proportional, PD-capped)
    so a hub wired past the PD bound never makes the joint improper, and a zero floor wires
    nothing."""
    if coupling_scale is None:
        coupling_scale = default_coupling_scale((net,), proposal, hub_self_prec)
    c = hub_couplings(net, proposal, coupling_scale)
    return belief.border(net, hub_name, Pi_diag=hub_self_prec, couplings=c, h_new=h_new)


# ----------------------------------------------------------------------
# 3. Scoring the moves on one ledger.
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class MoveScore:
    """The score of one candidate structural move.

    ``delta_F`` : evidence term (accuracy - complexity), the closed-form log Bayes factor.
                  For ``reduce`` it is the Savage-Dickey ``ΔF`` of pruning the edge
                  (``> 0`` => prune). For ``expand`` it is ``-ΔF_prune`` of the woken hub
                  (``> 0`` => the data HOLD the new coupling => wake) -- expansion scored
                  as reduction run backwards.
    ``delta_G`` : epistemic value -- the expected information gain about the new latent
                  (0 for ``reduce`` / ``sample``). The novelty term that pays for the
                  inversion (Friston 2023 / Millidge 2021).
    ``delta_U`` : conviction-field value change of the move (0 unless a tilt is supplied).
    ``score``   : ``delta_F + delta_G + tilt * delta_U`` -- the unified ledger.
    ``accept``  : ``score > 0``.
    """

    move: str
    delta_F: float
    delta_G: float
    delta_U: float
    score: float
    accept: bool
    detail: dict = field(default_factory=dict)


def _epistemic_gain(net_post: GaussianBeliefNet, proposal: HubProposal,
                    coupling_scale: float, free_prec: float) -> float:
    """Robust expected information gain about the woken latent (the ``ΔG`` novelty term).

    The blessed form is ``½(ln Π_post,new - ln Π_prior,new)`` over the candidate node's
    marginal. The data reach the latent only through its couplings to the (informed)
    neighbours, so the information that flows in is ``I_hub = Σ_k c_k² Π_post[nb_k, nb_k]``
    -- each coupling weighted by that neighbour's *conditional confidence* (its diagonal,
    read like ``efe.predictive_variance`` so it is well-defined on the improper prior).
    Against the free prior ``free_prec`` (α_e → 0),

        ΔG = ½ ln( (free_prec + I_hub) / free_prec )  ≥ 0,

    growing with both the coupling strength and the neighbours' certainty -- positive
    exactly when the residual loads on the proposed hub. No inverse."""
    idx = net_post.indices(proposal.neighbours)
    nb_prec = jnp.diagonal(net_post.Pi)[idx]                     # (k,) conditional conf.
    c = coupling_scale * proposal.pattern                        # (k,)
    info = float(jnp.sum(c ** 2 * nb_prec))
    return 0.5 * float(jnp.log((free_prec + info) / free_prec))


def _conviction_delta(post_full: GaussianBeliefNet, dormant: GaussianBeliefNet,
                      u_full: jax.Array, alpha: float) -> float:
    """The conviction-field value change of waking the hub (Eqs. 6-8): the value the
    community attaches to where the woken model lands versus where the dormant model
    sits. ``U = T u`` propagated by ``dual_field.PrecisionUtilityNet`` on each net's own
    couplings; the score is ``U · μ`` (``effective_utility · mean``). A hub that pulls
    belief toward the rival reading carries ``ΔU < 0`` under a core that values the
    incumbent -- which, scaled by a large tilt ``λ``, can VETO an evidence-favoured wake
    (``delta_F + delta_G > 0`` yet ``score < 0``). That veto is the lock-in: deciding a
    knob is missing but conviction will not let you build it."""
    pun_full = PrecisionUtilityNet(names=post_full.names, Pi=post_full.Pi,
                                   h=post_full.h, u=u_full, alpha=alpha)
    pun_dorm = PrecisionUtilityNet(names=dormant.names, Pi=dormant.Pi,
                                   h=dormant.h, u=u_full, alpha=alpha)
    return float(pun_full.score_state() - pun_dorm.score_state())


def expansion_score(net_post: GaussianBeliefNet, net_prior: GaussianBeliefNet,
                    proposal: HubProposal, hub_name: str,
                    hub_self_prec: float = 2.0, coupling_scale: float | None = None,
                    free_prec: float = 1e-2,
                    u_full: jax.Array | None = None, tilt: float = 0.0,
                    alpha: float = 0.5, accept_eps: float = 1e-4) -> MoveScore:
    """Score the WAKE of ``proposal`` as a hub named ``hub_name`` -- expansion as reduction
    run backwards.

    Two (d+1)-node models, the parent and the augmented one carrying the woken hub:
      * ``post_full``  = ``net_post`` bordered with the hub WIRED at the proposal couplings;
      * ``prior_full`` = ``net_prior`` bordered with the same wiring (the augmented parent).

    The expansion evidence is the **model log Bayes factor** (Friston et al. 2023,
    ``ln p(o|m') - ln p(o|m)``), each model scored as its own within-model evidence gain
    so the ``d`` vs ``d+1`` normalizers cancel inside each bracket:

        delta_F = [logZ(post_full) - logZ(prior_full)] - [logZ(net_post) - logZ(net_prior)].

    Algebraically this is ``½[(cᵀμ_post)²/S_post - (cᵀμ_prior)²/S_prior - ln(S_post/S_prior)]``
    (``S = Π_hub - cᵀ Π⁻¹ c`` the Schur scalar): the woken hub earns evidence exactly when
    the data shift the observed mean ALONG its coupling pattern ``c`` -- the residual *loads*
    on the hub. ``> 0`` => wake; ``≤ 0`` => the residual does not project onto this hub.
    The closed-form Savage-Dickey verdict on pruning the hub back to dormancy is kept in
    ``detail['dF_prune']`` (expansion run backwards: ``dF_prune < 0`` ⇔ data hold the hub).
    Add the epistemic gain ``delta_G`` and the conviction term; accept iff
    ``score = delta_F + delta_G + tilt·delta_U > 0``. ``coupling_scale=None`` picks the
    residual-proportional PD-capped scale over BOTH nets so neither bordered model goes
    improper and a zero floor wires nothing."""
    if coupling_scale is None:
        coupling_scale = default_coupling_scale((net_post, net_prior), proposal, hub_self_prec)

    post_full = wake_hub(net_post, proposal, hub_name, hub_self_prec, coupling_scale)
    prior_full = wake_hub(net_prior, proposal, hub_name, hub_self_prec, coupling_scale)
    zero = jnp.zeros((net_prior.dim,))
    dormant = belief.border(net_prior, hub_name, Pi_diag=free_prec,
                            couplings=zero, h_new=0.0)

    delta_F = float(
        (bmr.log_evidence(post_full) - bmr.log_evidence(prior_full))
        - (bmr.log_evidence(net_post) - bmr.log_evidence(net_prior))
    )
    dF_prune = float(bmr.bmr_prune(post_full, prior_full, dormant)["delta_F"])
    delta_G = _epistemic_gain(post_full, proposal, coupling_scale, free_prec)

    delta_U = 0.0
    if u_full is not None and tilt != 0.0:
        post_dorm_data = belief.border(net_post, hub_name, Pi_diag=free_prec,
                                       couplings=zero, h_new=0.0)
        delta_U = _conviction_delta(post_full, post_dorm_data, u_full, alpha)

    score = delta_F + delta_G + tilt * delta_U
    return MoveScore(move="expand", delta_F=delta_F, delta_G=delta_G, delta_U=delta_U,
                     score=score, accept=bool(score > accept_eps),
                     detail={"dF_prune": dF_prune, "coupling_scale": coupling_scale,
                             "strength": proposal.strength,
                             "rank1_ratio": proposal.rank1_ratio})


def reduction_score(net_post: GaussianBeliefNet, net_prior: GaussianBeliefNet,
                    edge: tuple[str, str], accept_eps: float = 1e-4) -> MoveScore:
    """Score PRUNING ``edge`` by closed-form BMR (Savage-Dickey) -- the cheap half, no new
    inversion. ``delta_F = ΔF`` of pinning the coupling to zero: ``> 0`` => the data are
    content without it (prune); ``< 0`` => the data hold it (keep)."""
    reduced_prior = bmr.prune_edge_prior(net_prior, edge=edge)
    dF = float(bmr.bmr_prune(net_post, net_prior, reduced_prior)["delta_F"])
    return MoveScore(move="reduce", delta_F=dF, delta_G=0.0, delta_U=0.0,
                     score=dF, accept=bool(dF > accept_eps), detail={"edge": edge})


# ----------------------------------------------------------------------
# 4. Arbitration: the three selection problems on one ledger.
# ----------------------------------------------------------------------

def select_action(net_post: GaussianBeliefNet, net_prior: GaussianBeliefNet,
                  neighbours: tuple[str, ...], hub_name: str,
                  prune_edges: tuple[tuple[str, str], ...] = (),
                  *, hub_self_prec: float = 2.0, coupling_scale: float | None = None,
                  free_prec: float = 1e-2, trigger: float = 0.0,
                  u_full: jax.Array | None = None, tilt: float = 0.0,
                  alpha: float = 0.5, residual: jax.Array | None = None) -> dict:
    """Choose among ``{sample, reduce, expand}`` on one ledger.

    Builds the inverse-Schur ``proposal`` over ``neighbours``, scores the best expansion
    (the wake) and the best reduction (over ``prune_edges``), and takes the arg-max move:

      * EXPAND if the wake is accepted (``score > 0``), its residual floor clears
        ``trigger``, AND its score beats the best pruning ``ΔF``;
      * else REDUCE if the best edge has ``ΔF > 0``;
      * else SAMPLE.

    The conviction term enters only the expand branch (``u_full``/``tilt``): a high tilt
    with ``ΔU < 0`` is the γ-veto that keeps a paradigm captured even when the evidence
    favours expansion. Returns ``{action, proposal, expand: MoveScore, reduce: MoveScore|None}``.
    """
    proposal = propose_hub(net_post, neighbours, residual=residual)
    expand = expansion_score(net_post, net_prior, proposal, hub_name,
                             hub_self_prec=hub_self_prec, coupling_scale=coupling_scale,
                             free_prec=free_prec, u_full=u_full, tilt=tilt, alpha=alpha)

    reduce_best: MoveScore | None = None
    for e in prune_edges:
        s = reduction_score(net_post, net_prior, e)
        if reduce_best is None or s.delta_F > reduce_best.delta_F:
            reduce_best = s

    can_expand = expand.accept and (proposal.strength >= trigger)
    reduce_dF = reduce_best.delta_F if reduce_best is not None else float("-inf")

    if can_expand and expand.score >= max(reduce_dF, 0.0):
        action = "expand"
    elif reduce_best is not None and reduce_best.accept:
        action = "reduce"
    else:
        action = "sample"

    return {"action": action, "proposal": proposal,
            "expand": expand, "reduce": reduce_best}
