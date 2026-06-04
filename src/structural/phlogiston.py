"""The ten-node phlogiston-vs-oxygen scenario.

The Gaussian structural sibling of ``src/pomdp/gen_model.py``: named constants
and topology builders, no dynamics. Two candidate paradigms over the same ten
theoretical commitments, differing only in their *prior topology* (the locked
design decision — shared likelihood, so BMR stays closed-form):

  * PHLOGISTON: a dense hub (``phlogiston``, node 0) on which the combustion /
    calcination / mass commitments all depend. The hub is the hidden principle
    that makes the surface phenomena co-vary (a common cause). Its prior gets
    the anomaly node ``calx_heavier_than_metal`` *wrong*: phlogiston escapes on
    combustion, so the calx should be lighter (prior mean negative).

  * OXYGEN: no hub. The Lavoisian "direct mass law" couples mass-change, gas
    consumption and calx-weight directly. Built as the Schur complement of the
    phlogiston net over the hub (so it inherits the carry-over edges the hub
    leaves behind — note Appendix A->B) PLUS a small independent gravimetric
    prior that installs the *correct* mass law (calx heavier; mean positive).
    Lives on the same ten-node basis with ``phlogiston`` held at a vague prior
    (the locked shared-basis decision).

The world (``src/structural/world.py``) is the truth: the calx really is
heavier. Early *phlogiston-regime* experiments probe the correlated combustion
commitments, where the hub's common-cause prior predicts the joint pattern
better than oxygen's independent prior — so phlogiston leads on evidence. Late
*oxygen-regime* gravimetric experiments load on the anomaly node, where
phlogiston's prior is wrong and oxygen's is right — so the evidence drifts
re-order and the upper envelope crosses. That crossing is the paradigm shift.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from src.structural.belief import GaussianBeliefNet, combine, vague_prior
from src.structural.bmr import schur_marginalize
from src.structural import precision as P
from src.structural.bayesnet import LinearGaussianBN, relational_operator
from src.structural.dual_field import PrecisionUtilityNet
from src.structural.dual_field import conviction_field as _conviction_field
from src.config import NetworkConfig


# ----------------------------------------------------------------------
# The shared node vocabulary (the fusion basis). Order IS the matrix basis.
# ----------------------------------------------------------------------

NODE_NAMES: tuple[str, ...] = (
    "phlogiston",                 # 0  the hidden hub principle
    "combustion_releases",        # 1  combustion = release of phlogiston
    "calcination_releases",       # 2  metal -> calx releases phlogiston
    "metal_is_calx_plus_phlog",   # 3  composition commitment
    "mass_change_sign",           # 4  does calcination gain or lose mass
    "gas_consumed",               # 5  is a gas consumed in combustion
    "air_has_capacity",           # 6  air's finite phlogiston capacity
    "reduction_with_charcoal",    # 7  charcoal restores the metal
    "respiration_like_combustion",# 8  breathing = slow combustion
    "calx_heavier_than_metal",    # 9  the anomaly (Lavoisier's weighing)
)

HUB = "phlogiston"
# commitments the hub couples to (the common-cause neighbours)
HUB_NEIGHBOURS: tuple[str, ...] = (
    "combustion_releases", "calcination_releases", "metal_is_calx_plus_phlog",
    "mass_change_sign", "gas_consumed", "air_has_capacity",
    "reduction_with_charcoal", "respiration_like_combustion",
)
# the Lavoisian direct mass law couples these (gravimetric block)
MASS_LAW: tuple[str, ...] = ("mass_change_sign", "gas_consumed",
                             "calx_heavier_than_metal")
# commitments the paradigms disagree on (probed by the oxygen/gravimetric regime)
DISAGREEMENT_NODES: tuple[str, ...] = ("mass_change_sign", "gas_consumed",
                                       "calx_heavier_than_metal")


@dataclass(frozen=True)
class StructuralConfig:
    """Configuration for the structural phlogiston experiment.

    Mirrors the frozen-dataclass style of ``src/config.py`` / ``PomdpConfig``.
    The coupling/mean knobs are tuned so the evidence envelope crosses; see
    ``notebooks/18_structural_phlogiston.ipynb`` for the demonstration.
    """

    node_names: tuple[str, ...] = NODE_NAMES

    # --- prior topology knobs ---
    base_prec: float = 1.0           # diagonal self-precision of every tracked node
    hub_coupling: float = 0.8        # phlogiston hub off-diagonal coupling
    hub_self_prec: float = 2.0       # phlogiston hub's own self-precision
    mass_coupling: float = 0.9       # oxygen direct-mass-law coupling
    mass_prec: float = 2.5           # gravimetric prior precision on the mass-law block
    kappa_vague: float = 1e-3        # vague-prior precision for untracked nodes

    # --- prior means (the empirical content the paradigms disagree on) ---
    mu_agree: float = 1.0            # both priors agree the combustion phenomena occur
    mu_phlog_mass: float = -1.0      # phlogiston: calx lighter (mass escapes) -- WRONG
    mu_oxy_mass: float = 1.0         # oxygen: calx heavier (gas absorbed)    -- RIGHT

    # --- world / data-generating process ---
    sigma_o: float = 1.0             # observation noise std
    observation_operator: str = "node"  # "node" (DEFAULT): H_observable, one row per
    #   measured node => DIAGONAL Fisher => the prior's edges never move (the legacy
    #   substrate, so nb18-29 + the test suite stay byte-identical). "relational":
    #   gravimetric_H, which adds a mass-balance row (calx - mass_change - gas) reading a
    #   COMBINATION of nodes, so the deposit carries OFF-DIAGONAL Fisher and the belt edge
    #   weights LEARN over the rollout -- structure learning, the precondition for the
    #   belt-first/core-last staircase. Host-side static branch (frozen cfg), scan-safe.

    # --- biased experiment selection (theory-laden observation) ---
    experiment_bias: float = 0.0     # 0 = unbiased (measure everything equally);
    #   in (0, 1] an agent that currently *holds* phlogiston down-weights the
    #   disagreement (gravimetric) experiments, so a committed bloc stops
    #   gathering the evidence that would refute it. bias=1 => an agent holding
    #   phlogiston never runs the refuting experiment at all (permanent lock-in).
    bias_sharpness: float = 8.0      # steepness of the held-paradigm gate at oxy=1/2;
    #   large => a hard "which paradigm do I hold" switch, small => graded.

    # --- derived evidential precision (the precision-field dual; see precision.py) ---
    precision_mode: str = "heuristic"   # "heuristic": the attention_weights sigmoid gate
    #   (the DEFAULT, so all existing tests + notebooks 18/19 are byte-identical);
    #   "derived": rho_k derived once from the incumbent prior net's core coupling;
    #   "derived_live": rho_k from each agent's live (fused) covariance (adaptive);
    #   "efe": expected-free-energy sampling (src/structural/efe.py) -- the "derived"
    #   pragmatic baseline PLUS two prediction-error drives (see epistemic_weight /
    #   social_weight). With both weights 0 it is byte-identical to "derived".

    # --- expected-free-energy attention drives (precision_mode="efe"; see efe.py) ---
    epistemic_weight: float = 0.0    # beta_e: agent<->world drive (own predictive variance
    #   1/Pi[v,v]); look where YOU are uncertain. Self-extinguishes as evidence sharpens v
    #   => self-sealing is emergent. 0 => off (the back-compat anchor).
    social_weight: float = 0.0       # beta_s: agent<->neighbour drive (trust- and confidence-
    #   weighted disagreement sum_j W_ij Pi_j[v,v] (mu_j[v]-mu_i[v])^2 -- the structural
    #   surprisal_matrix). Independent of i's own certainty, so it can re-open a channel a
    #   self-sealed agent abandoned: the term that lets a trusted peer break lock-in. 0 => off.
    core_governance: float = 0.0     # g >= 0: 0 => rho = rho_max everywhere (paradigm-
    #   neutral). The phase-transition knob; lock-in basin opens ~O(100-1000) over a
    #   finite horizon (no forgetting => ratchet). The sensory-edge analogue of prec_scale.
    rho_max: float = 1.0             # ceiling evidential gain; 1.0 so g=0 == plain deposit.
    evidential_cost_kind: str = "carryover"    # "carryover" (default): the Schur brace on
    #   the core Pi[c,v]^2/Pi[v,v] -- the dual of lambda_v, robust to the improper hub
    #   prior. "fe"/"normalized": Sherman-Morrison covariance forms (need a proper PD net).
    core_node: str = HUB             # which node is the entrenched hidden core whose
    #   stiffness governs the precision field (default the phlogiston hub).
    anomaly_coupling: float | None = None   # coupling that binds the anomaly node
    #   (calx_heavier_than_metal) into the core web *for the precision derivation only*
    #   (paradigm_field; the dynamics prior phlogiston_prior is untouched). The phlogiston
    #   prior asserts a *mean* for calx-weight but leaves it structurally unbound; a
    #   commitment with a mean and no coupling is inert, so the paradigm cannot govern the
    #   gain on the very channel that would refute it. None => bind at hub_coupling (the
    #   "weakly attached" anomaly, realized); 0.0 => leave it unbound (the rival then leaks
    #   through the ungoverned anomaly channel -- self-sealing is incomplete).

    # --- conviction field (the SECOND field, U = T u; see dual_field.py) ---
    conviction_tilt: float = 0.0     # lambda: the value-tilt strength in the motivated
    #   posterior q_lambda(s) prop p(s|o) e^{lambda U(s)}. For a linear utility U(s)=U.s on
    #   a Gaussian this is one shift of the potential, h <- h + lambda U, applied each step.
    #   0 (DEFAULT) => no tilt => pure-evidence update, byte-identical to before. The
    #   belief-utility knob is JUST a balance: pick lambda ~ balanced_lambda(cfg) so neither
    #   the evidence/conservatism field nor the conviction field dominates (nothing more).
    conviction_toward: str = "phlogiston"   # which paradigm the intrinsic utility u favours
    #   ("phlogiston" | "oxygen" | "neutral"); see conviction_u. The conviction field U = T u
    #   propagates this along the net's couplings (the SAME operator T that gives carry-over).
    conviction_alpha: float = 0.5    # propagation strength of the conviction field (alpha in
    #   the (I - alpha W) u_eff = u solve of dual_field.effective_utility).

    # --- social sharing channel (what a peer transmits across the trust graph) ---
    sharing_mode: str = "posterior"  # "posterior" (DEFAULT): peers transmit their WHOLE
    #   belief net and the receiver fuses by precision addition -- a conclusion propagates,
    #   prior bias and all (the original dynamics, byte-identical). "observation": peers
    #   transmit only their RAW per-step observations (Fisher deposits); each agent keeps its
    #   own prior and pools neighbours' data, so the signal propagates but the bias does not.
    #   The axis the paper keeps explicit: "sharing conclusions propagates bias along with
    #   signal."

    # --- regime schedule ---
    regime_schedule: str = "step"    # "step" | "reversal" | "ramp"
    t_shift: int = 40                # phlogiston regime -> oxygen regime
    t_reverse: int | None = None     # for "reversal"
    ramp_rate: float = 0.01          # "ramp" (triangular): pole-to-pole in 1/ramp_rate
    #   steps. The disagreement-node truth ramps phlogiston-pole -> oxygen-pole -> back,
    #   the reversible drive for the rate-resolved hysteresis test (E1). Loop area
    #   A_hys(r,g) vanishing as r->0 => monostable (tracking lag); a residual loop => a
    #   genuine fold (bistable). Set n_steps = 2/ramp_rate for one full triangle.
    n_steps: int = 120               # long enough for the (lagged) crossing

    # --- social layer ---
    network: NetworkConfig = field(default_factory=NetworkConfig)
    n_agents: int = 60
    seed: int = 0

    def __post_init__(self):
        if self.regime_schedule == "reversal" and self.t_reverse is None:
            raise ValueError("reversal schedule requires t_reverse")


# ----------------------------------------------------------------------
# phi_true — the world's actual commitment values.
# ----------------------------------------------------------------------

def phi_true_at(cfg: StructuralConfig, t: int) -> jnp.ndarray:
    """The true commitment vector at step ``t`` -- the *world*, which changes
    regime (not the paradigms, which differ only by prior).

    Agreement nodes always hold ``mu_agree`` (the combustion phenomena occur).
    The mass-law nodes hold the regime-dependent truth: in the phlogiston regime
    (``t < t_shift``) crude experiments read the calx as *lighter*
    (``mu_phlog_mass``, consistent with escaping phlogiston -- phlogiston was a
    good fit to pre-Lavoisier data); in the oxygen regime precise gravimetry
    reveals the calx is *heavier* (``mu_oxy_mass``). The hidden ``phlogiston``
    node has no observable value (sits at 0). This is the matrix analogue of
    ``src/world.theta_schedule``: the world's revealed behaviour flips, which is
    what re-orders the candidate paradigms' evidence drifts.
    """
    idx = _idx(cfg)
    phi = jnp.full((len(cfg.node_names),), cfg.mu_agree)
    phi = phi.at[idx[HUB]].set(0.0)

    if cfg.regime_schedule == "ramp":
        # triangular drive: phlog pole (frac 0) -> oxy pole (frac 1) -> back.
        frac = ramp_frac(cfg, t)
        mass_val = cfg.mu_phlog_mass + frac * (cfg.mu_oxy_mass - cfg.mu_phlog_mass)
    else:
        if cfg.regime_schedule == "reversal":
            in_oxy = (cfg.t_shift <= t < cfg.t_reverse)
        else:  # step
            in_oxy = t >= cfg.t_shift
        mass_val = cfg.mu_oxy_mass if in_oxy else cfg.mu_phlog_mass
    for n in DISAGREEMENT_NODES:
        phi = phi.at[idx[n]].set(mass_val)
    return phi


def ramp_frac(cfg: StructuralConfig, t: int) -> float:
    """The triangular evidence coordinate at step ``t`` for ``regime_schedule='ramp'``:
    0 at the phlogiston pole, 1 at the oxygen pole, ramping up over ``1/ramp_rate``
    steps then back down (host-side float; ``phi_true_at`` is built host-side). This is
    the abscissa of the hysteresis loop -- the (frac, m_t) curve whose enclosed area is
    ``A_hys`` (E1)."""
    half = 1.0 / cfg.ramp_rate                 # steps per pole-to-pole leg
    cyc = t % (2.0 * half)
    return cyc / half if cyc <= half else 2.0 - cyc / half


# ----------------------------------------------------------------------
# Prior topologies (the two candidate paradigms).
# ----------------------------------------------------------------------

def _idx(cfg: StructuralConfig) -> dict[str, int]:
    return {n: i for i, n in enumerate(cfg.node_names)}


def phlogiston_prior(cfg: StructuralConfig) -> GaussianBeliefNet:
    """The phlogiston paradigm's prior belief net: a dense hub on ``phlogiston``
    coupling to every combustion/calcination/mass commitment, with the anomaly
    node weakly attached and given the *wrong* prior mean (calx lighter)."""
    idx = _idx(cfg)
    d = len(cfg.node_names)
    Pi = cfg.base_prec * jnp.eye(d)
    Pi = Pi.at[idx[HUB], idx[HUB]].set(cfg.hub_self_prec)
    for n in HUB_NEIGHBOURS:
        Pi = Pi.at[idx[HUB], idx[n]].set(cfg.hub_coupling)
        Pi = Pi.at[idx[n], idx[HUB]].set(cfg.hub_coupling)

    # prior mean: agreement nodes +mu_agree; mass nodes get the WRONG sign.
    mu = jnp.full((d,), cfg.mu_agree)
    mu = mu.at[idx[HUB]].set(0.0)
    for n in DISAGREEMENT_NODES:
        mu = mu.at[idx[n]].set(cfg.mu_phlog_mass)
    h = Pi @ mu
    return GaussianBeliefNet(Pi=Pi, h=h, names=cfg.node_names)


def oxygen_prior(cfg: StructuralConfig) -> GaussianBeliefNet:
    """The oxygen paradigm's prior: the Schur complement of the phlogiston net
    over the hub (inheriting the carry-over mass-law edges -- note Appendix
    A->B) PLUS a small independent gravimetric prior installing the correct mass
    law (calx heavier). Re-embedded on the full ten-node basis with
    ``phlogiston`` held at a vague prior.
    """
    idx = _idx(cfg)
    # 1. Schur survivor of the phlogiston net (9 nodes; carries the fill-in).
    surv = schur_marginalize(phlogiston_prior(cfg), (HUB,))   # names = all but HUB

    # 2. Independent gravimetric prior on the mass-law block, mean = +mu_oxy_mass.
    grav_base = vague_prior(surv.names, kappa=0.0)   # zero off the mass-law block
    mass_idx = surv.indices(MASS_LAW)
    k = len(MASS_LAW)
    # dense mass-law block: mass_prec on the diagonal, mass_coupling off it.
    block = jnp.full((k, k), cfg.mass_coupling).at[jnp.diag_indices(k)].set(cfg.mass_prec)
    Pi_g = grav_base.Pi.at[jnp.ix_(mass_idx, mass_idx)].set(block)
    mu_g = jnp.zeros((len(surv.names),)).at[mass_idx].set(cfg.mu_oxy_mass)
    grav = GaussianBeliefNet(Pi=Pi_g, h=Pi_g @ mu_g, names=surv.names)

    oxy_surv = combine(surv, grav)     # 9-node oxygen prior

    # 3. Re-embed onto the full basis: phlogiston node gets a vague prior.
    d = len(cfg.node_names)
    surv_idx = jnp.asarray([idx[n] for n in oxy_surv.names])
    Pi = (cfg.kappa_vague * jnp.eye(d)).at[jnp.ix_(surv_idx, surv_idx)].set(oxy_surv.Pi)
    h = jnp.zeros((d,)).at[surv_idx].set(oxy_surv.h)
    return GaussianBeliefNet(Pi=Pi, h=h, names=cfg.node_names)


# ----------------------------------------------------------------------
# Observation operator.
# ----------------------------------------------------------------------

def H_observable(cfg: StructuralConfig) -> jnp.ndarray:
    """Measure every observable commitment (all nodes except the hidden hub):
    one row per node, reading that commitment directly. Used as the per-step
    measurement operator -- the mass-law rows carry the regime-flipping signal
    of ``phi_true_at`` that drives the evidence crossing, while the
    agreement-node rows confirm the (stable) combustion phenomena for both
    paradigms.
    """
    idx = _idx(cfg)
    d = len(cfg.node_names)
    rows = [jnp.zeros((d,)).at[idx[n]].set(1.0)
            for n in cfg.node_names if n != HUB]
    return jnp.stack(rows, axis=0)      # (d-1, d)


def measured_nodes(cfg: StructuralConfig) -> tuple[str, ...]:
    """The commitments actually measured each round -- every node except the
    hidden hub, in the row order of ``H_observable``."""
    return tuple(n for n in cfg.node_names if n != HUB)


def disagreement_row_mask(cfg: StructuralConfig) -> jnp.ndarray:
    """A (m,) 0/1 mask over the rows of the *active* observation operator
    (``observation_rows(cfg)``): 1 where the row is a disagreement (mass-law)
    channel -- the diagnostic experiments a committed paradigm is tempted to skip.
    Used to build the per-agent attention weights for biased experiment selection.

    A row counts as disagreement if its label is a ``DISAGREEMENT_NODE`` (a direct
    read of a mass-law commitment) OR it is a relational row (a combination, not a
    bare node -- the gravimetric mass-balance row, which loads on the anomaly). In
    ``observation_operator='node'`` the rows are exactly ``measured_nodes`` so the
    second clause never fires and the mask is byte-identical to the legacy one."""
    rows = observation_rows(cfg)
    node_set = set(cfg.node_names)
    return jnp.asarray(
        [1.0 if (r in DISAGREEMENT_NODES or r not in node_set) else 0.0
         for r in rows])


def attention_weights(oxy_index: jnp.ndarray, cfg: StructuralConfig
                      ) -> jnp.ndarray:
    """Per-agent experiment-selection weights over the ``H_observable`` rows.

    ``oxy_index`` : (N,) each agent's current position on the phlogiston(0) ->
    oxygen(1) axis. Returns (N, m) weights. Agreement experiments always run
    (weight 1); a *disagreement* experiment runs with weight
    ``1 - bias * gate(oxy)`` where ``gate`` is a smooth switch ~1 while the agent
    *holds phlogiston* (oxy < 1/2) and ~0 once it holds oxygen. So a committed
    phlogistonist self-censors the refuting experiment across its whole basin --
    gating on the held paradigm (not linearly on oxy) is what makes the lock-in a
    robust basin rather than a knife-edge at oxy=0. Agents with different models
    thereby gather different information about the same world.
    """
    mask = disagreement_row_mask(cfg)                      # (m,)
    bias = cfg.experiment_bias
    # held-paradigm gate: sigmoid centred at the oxy=1/2 separatrix.
    gate = 1.0 / (1.0 + jnp.exp(-cfg.bias_sharpness * (0.5 - oxy_index)))   # (N,)
    skip = bias * gate[:, None] * mask[None, :]            # (N, m)
    return jnp.clip(1.0 - skip, 0.0, 1.0)


def candidate_priors(cfg: StructuralConfig) -> dict[str, GaussianBeliefNet]:
    """The two candidate paradigms keyed by name, for the running-evidence race."""
    return {"phlogiston": phlogiston_prior(cfg), "oxygen": oxygen_prior(cfg)}


# ----------------------------------------------------------------------
# The phlogiston paradigm as a WELL-DEFINED Bayes net (explicit CPDs).
#
# The same scenario as ``phlogiston_prior`` but built the right way: a directed
# common-cause DAG whose every node carries a named conditional distribution
# p(node | parents), instead of hand-poking entries of a joint precision matrix.
# See ``src/structural/bayesnet.py`` and ``notebooks/29_*``.
# ----------------------------------------------------------------------

def phlogiston_bn(cfg: StructuralConfig, conviction: float = 1.0
                  ) -> LinearGaussianBN:
    """The phlogiston paradigm as a Linear-Gaussian Bayes net (CPDs).

    Structure (a DAG in ``NODE_NAMES`` order, which is already parents-before-
    children):

      * the hidden hub ``phlogiston`` is the *root common cause*: it is a parent of
        every combustion / calcination / mass commitment (``HUB_NEIGHBOURS``), with
        CPD weight ``cfg.hub_coupling`` -- this is what makes the surface phenomena
        co-vary, expressed as genuine conditionals rather than a dense ``Pi`` block.
      * a *belt mass-balance*: the anomaly ``calx_heavier_than_metal`` is a child of
        the mass-law nodes ``mass_change_sign`` and ``gas_consumed`` (weight
        ``cfg.mass_coupling``) -- the edge a gravimetric (relational) experiment can
        actually move.

    Prior means reproduce the paradigm's empirical content exactly: the intercepts
    are set ``b = (I - B) mu*`` so the marginal means equal ``mu*`` (agreement nodes
    ``mu_agree``; the disagreement / anomaly nodes the *wrong* ``mu_phlog_mass`` ==
    "calx lighter"). ``conviction`` sets the inverse residual variance (stiffness):
    larger => tighter CPDs => a more entrenched paradigm (the lock-in knob, the CPD
    analogue of ``hub_self_prec`` / ``core_governance``).
    """
    names = cfg.node_names
    d = len(names)
    idx = _idx(cfg)

    B = jnp.zeros((d, d))
    for n in HUB_NEIGHBOURS:                       # hub -> each neighbour
        B = B.at[idx[n], idx[HUB]].set(cfg.hub_coupling)
    anom = idx["calx_heavier_than_metal"]
    for n in ("mass_change_sign", "gas_consumed"):  # belt mass-balance edges
        B = B.at[anom, idx[n]].set(cfg.mass_coupling)

    target = jnp.full((d,), cfg.mu_agree)
    target = target.at[idx[HUB]].set(0.0)          # hidden hub centred at 0
    for n in DISAGREEMENT_NODES:
        target = target.at[idx[n]].set(cfg.mu_phlog_mass)
    b = (jnp.eye(d) - B) @ target                  # => marginal means == target

    s = jnp.full((d,), 1.0 / conviction)
    s = s.at[idx[HUB]].set(1.0 / (conviction * cfg.hub_self_prec))
    return LinearGaussianBN(B=B, b=b, s=s, names=names)


def gravimetric_H(cfg: StructuralConfig) -> jnp.ndarray:
    """The relational observation operator for the corrected scenario: rows that
    read *combinations* of nodes, so the data deposit off-diagonal Fisher
    information and the belt edges learn over time.

      * a mass-balance row ``calx_heavier_than_metal - mass_change_sign -
        gas_consumed`` -- the gravimetric experiment that couples the anomaly to the
        mass law (this is the row that *moves the structure*);
      * direct rows on the stable agreement phenomena (the combustion commitments),
        which both paradigms share.

    Contrast ``H_observable`` (one row per node => diagonal Fisher => frozen edges).

    The operator is a strict superset of the direct read: a direct row on every
    observable node (these carry the regime *level* signal of ``phi_true_at`` -- so
    a node's marginal mean flips over time) PLUS one relational *mass-balance* row
    ``calx_heavier_than_metal - mass_change_sign - gas_consumed`` (this carries the
    *coupling* signal -- so the belt edge weights move over time). The level story
    and the structure story are both visible in one rollout.
    """
    meas = measured_nodes(cfg)
    rels = [{n: 1.0} for n in meas]                       # direct reads (levels)
    rels.append({"calx_heavier_than_metal": 1.0,          # mass-balance (coupling)
                 "mass_change_sign": -1.0, "gas_consumed": -1.0})
    return relational_operator(cfg.node_names, rels)


def gravimetric_rows(cfg: StructuralConfig) -> tuple[str, ...]:
    """Human-readable labels for the rows of ``gravimetric_H`` (row order)."""
    return measured_nodes(cfg) + ("mass_balance(calx-mass-gas)",)


def observation_operator(cfg: StructuralConfig) -> jnp.ndarray:
    """The per-step measurement operator ``H`` selected by ``cfg.observation_operator``:

      * ``"node"`` (default) => ``H_observable`` -- one row per measured node, so the
        Fisher deposit ``H^T H`` is DIAGONAL and the prior's off-diagonal couplings
        (the edges) never move. The legacy substrate.
      * ``"relational"`` => ``gravimetric_H`` -- adds a mass-balance row reading a
        combination of nodes, so the deposit carries OFF-DIAGONAL Fisher and the belt
        edges LEARN over the rollout (structure learning).

    Host-side static branch on the frozen-cfg string -- safe inside ``jax.lax.scan``.
    This is the single point ``step._transition`` calls to pick the substrate, so the
    whole population loop inherits structure learning by flipping one cfg flag."""
    if cfg.observation_operator == "relational":
        return gravimetric_H(cfg)
    if cfg.observation_operator == "node":
        return H_observable(cfg)
    raise ValueError(
        "observation_operator must be 'node' or 'relational', got "
        f"{cfg.observation_operator!r}")


def observation_rows(cfg: StructuralConfig) -> tuple[str, ...]:
    """Row labels of ``observation_operator(cfg)`` in row order -- the common
    length/order anchor for every per-row vector (the ``rho_k`` evidential-precision
    weights and the disagreement mask), so the node and relational operators stay
    consistent and the existing rollouts work unchanged in either mode."""
    if cfg.observation_operator == "relational":
        return gravimetric_rows(cfg)
    return measured_nodes(cfg)


# ----------------------------------------------------------------------
# Derived evidential precision (the dual of carry-over; see precision.py).
# ----------------------------------------------------------------------

def core_index(cfg: StructuralConfig) -> int:
    """Basis index of the entrenched hidden core node (default the phlogiston hub)
    -- the node whose stiffness governs the sensory precision field."""
    return cfg.node_names.index(cfg.core_node)


def paradigm_field(cfg: StructuralConfig,
                   net: GaussianBeliefNet | None = None) -> GaussianBeliefNet:
    """The paradigm's full internal precision field, used to DERIVE evidential precision.

    Starts from ``net`` (default ``phlogiston_prior(cfg)``) and binds the anomaly node
    ``calx_heavier_than_metal`` into the core web at ``cfg.anomaly_coupling`` (default
    ``cfg.hub_coupling``). This realises the prior's stated-but-unbuilt "anomaly weakly
    attached": the paradigm asserts a mean for calx-weight, so the commitment must be
    coupled to the hub, else it is structurally inert and the core cannot govern the gain
    on the channel that would refute it. Used ONLY by ``derived_channel_precision`` -- the
    dynamics prior ``phlogiston_prior`` and the oxygen prior built from its Schur complement
    are left untouched, so notebooks 18/19 and the existing tests are byte-identical.
    ``anomaly_coupling = 0.0`` returns the bare prior (the anomaly stays unbound, so the
    rival leaks through that ungoverned channel -- self-sealing is incomplete).
    """
    if net is None:
        net = phlogiston_prior(cfg)
    ac = cfg.hub_coupling if cfg.anomaly_coupling is None else cfg.anomaly_coupling
    c = net.index(cfg.core_node)
    a = net.index("calx_heavier_than_metal")
    Pi = net.Pi.at[c, a].set(ac).at[a, c].set(ac)
    return GaussianBeliefNet(Pi=Pi, h=net.h, names=net.names)


def derived_channel_precision(cfg: StructuralConfig,
                              net: GaussianBeliefNet | None = None) -> jnp.ndarray:
    """Prior-derived evidential precision ``rho_k`` over the ``H_observable`` rows for the
    incumbent paradigm, over the rows of the *active* observation operator. ``net``
    defaults to the paradigm field ``paradigm_field(cfg)`` (the incumbent prior with its
    anomaly bound) -- so ``rho`` is a fixed ``(m,)`` vector computed once from the core
    coupling (no per-agent re-inversion, no rho<->Sigma feedback). With
    ``core_governance = 0`` this is all ``rho_max`` (== the unbiased deposit when
    ``rho_max = 1``).

    ``observation_operator='node'`` (default): a thin wrapper over
    ``precision.channel_precision`` over ``measured_nodes`` (byte-identical to before).
    ``'relational'``: ``precision.channel_precision_H`` over the rows of ``gravimetric_H``
    -- the relational cost ``(H[k].Pi[:,c])^2/(H[k].Pi.H[k])`` reduces to the node form on
    the direct rows, so the only new entry is the mass-balance row's gain (the channel the
    core silences as ``core_governance`` rises -- now the channel that *moves structure*).
    """
    if net is None:
        net = paradigm_field(cfg)
    if cfg.observation_operator == "relational":
        return P.channel_precision_H(net, cfg.core_node, gravimetric_H(cfg),
                                     cfg.core_governance, cfg.rho_max)
    return P.channel_precision(net, cfg.core_node, measured_nodes(cfg),
                               cfg.core_governance, cfg.rho_max,
                               cfg.evidential_cost_kind)


# ----------------------------------------------------------------------
# The conviction field U = T u (the SECOND field; see dual_field.py).
#
# The same propagation operator T = (I - alpha W)^{-1} that gives carry-over kappa = T 1
# carries a second, linearly independent source: the intrinsic utility u, giving the
# conviction field U = T u. A value-tilted (motivated) posterior is q_lambda prop
# p(s|o) e^{lambda U(s)}; for a linear utility on a Gaussian it is one shift of the
# potential, h <- h + lambda U. The "belief-utility vs precision" knob is just the balance
# lambda at which neither field dominates -- balanced_lambda below.
# ----------------------------------------------------------------------

def conviction_u(cfg: StructuralConfig, toward: str | None = None) -> jnp.ndarray:
    """The intrinsic utility vector ``u`` (d,) -- the value the community attaches to
    commitments, BEFORE propagation. It values the disagreement (mass-law) commitments
    toward one paradigm's reading:

      * ``"phlogiston"`` (the entrenched community *wants* the calx lighter -- negative u
        on the mass-law nodes, matching ``mu_phlog_mass < 0``);
      * ``"oxygen"`` the reverse (+u);
      * ``"neutral"`` all zero (then U = 0 and there is no tilt to balance).

    ``toward`` defaults to ``cfg.conviction_toward``. The conviction field ``U = T u``
    (``conviction_field``) then propagates this along the net's couplings."""
    toward = cfg.conviction_toward if toward is None else toward
    d = len(cfg.node_names)
    idx = _idx(cfg)
    u = jnp.zeros((d,))
    if toward == "neutral":
        return u
    sign = 1.0 if toward == "oxygen" else -1.0
    for n in DISAGREEMENT_NODES:
        u = u.at[idx[n]].set(sign)
    return u


def conviction_field(Pi: jnp.ndarray, h: jnp.ndarray, names: tuple[str, ...],
                     u: jnp.ndarray, alpha: float = 0.5) -> jnp.ndarray:
    """The conviction field ``U = T u`` for a stack of belief nets -- a thin re-export of the
    now-general :func:`dual_field.conviction_field` (its natural home: it just wraps
    ``PrecisionUtilityNet.effective_utility``). Kept here under the original name so existing
    callers (``step._transition`` and the test suite) are byte-identical. ``Pi`` (N, d, d),
    ``h`` (N, d), ``u`` ``(d,)`` (broadcast) or ``(N, d)`` (per-agent). Returns ``(N, d)``."""
    return _conviction_field(Pi, h, names, u, alpha)


def balanced_lambda(cfg: StructuralConfig, toward: str | None = None) -> float:
    """The tilt ``lambda`` at which the conviction field's contribution to the potential
    matches a typical single-step evidence deposit -- so neither the evidence/conservatism
    field nor the conviction field dominates the update. This is *all* the "belief-utility
    ~ precision" balance asks for -- a reference scale, not a tuned optimum. Closed-form norm
    ratio (no optimiser), on the **contested (disagreement) subspace** where the two fields
    actually compete:

        lambda* = || j_typical[dis] || / || U[dis] ||,

    with ``U = T u`` the conviction field on the incumbent prior and ``j_typical = H^T (H
    phi_ref) / sigma^2`` a representative one-step Fisher potential (``phi_ref`` the oxygen
    pole), both restricted to the disagreement nodes.

    NOTE (reported, not tuned away): the *dynamical* 50/50 crossover sits BELOW ``lambda*``
    (empirically ~0.5 lambda*), because the conviction is added coherently every step while
    the noisy evidence partly cancels -- the value field punches above its per-step norm.
    ``lambda*`` is the principled scale; nb33 locates the crossover relative to it.
    ``lambda << lambda*`` evidence-dominated; ``lambda >> lambda*`` conviction-dominated
    (motivated reasoning that locks the belief by value alone)."""
    net = phlogiston_prior(cfg)
    u = conviction_u(cfg, toward)
    pun = PrecisionUtilityNet(names=cfg.node_names, Pi=net.Pi, h=net.h, u=u,
                              alpha=cfg.conviction_alpha)
    U = pun.effective_utility()
    H = observation_operator(cfg)
    phi_ref = phi_true_at(cfg, cfg.n_steps - 1)
    j = H.T @ (H @ phi_ref) / (cfg.sigma_o ** 2)
    # restrict to the contested (disagreement) subspace -- where the two fields compete.
    dis = jnp.asarray([cfg.node_names.index(n) for n in DISAGREEMENT_NODES])
    return float(jnp.linalg.norm(j[dis]) / (jnp.linalg.norm(U[dis]) + 1e-12))
