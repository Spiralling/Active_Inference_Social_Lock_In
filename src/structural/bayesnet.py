"""A well-defined Bayes net: the Linear-Gaussian Bayesian network (LGBN).

This is the object the structural model *should* have had from the start. The
legacy ``GaussianBeliefNet`` (``belief.py``) carries only the joint precision
matrix ``Pi`` and treats "off-diagonal ``Pi[i,j]`` = edge". That is the
*undirected* Gaussian-MRF reading; it has no conditional distributions, no
per-node belief you can name, and -- because the phlogiston observation operator
is one row per node -- the Fisher deposit ``H^T H`` is diagonal, so data never
touch the off-diagonals and the "structure" is frozen in the prior forever.

A *Bayes net* is a DAG ``G = (V, E)`` whose joint factorises into explicit
conditional probability distributions (CPDs), one per node:

    p(x) = prod_v  p(x_v | x_{pa(v)}),

and for the linear-Gaussian family each CPD is

    x_v | x_{pa(v)} ~ N( b_v + sum_{u in pa(v)} B_vu x_u ,  s_v ),       (CPD_v)

with intercept ``b_v``, parent weights ``B_vu`` and residual variance ``s_v``.
This module stores exactly those CPD parameters and nothing else, so:

  * every node has a genuine, inspectable conditional (``LinearGaussianBN.cpd``);
  * the joint is *derived* from the CPDs, not the other way round
    (``to_info`` -> the precision form the dynamics consume);
  * a relational observation (an ``H`` row reading a *combination* of nodes, e.g.
    a mass-balance ``calx - metal - gas``) deposits genuine *off-diagonal* Fisher
    information, so re-reading the CPDs after an update (``from_info``) shows the
    edge weights *move* -- the Bayes net actually learns its structure over time.

The CPD<->precision bridge is exact (it is a bijection given the node order):

    A := I - B   (unit lower-triangular when nodes are in parents-before-children
                  topological order),   D := diag(s),
    Pi = A^T D^{-1} A ,        h = A^T D^{-1} b .                        (compile)

Going back, ``from_info`` factors a posterior ``Pi = U D^{-1} U^T`` with ``U``
unit upper-triangular (a UDU^T / reversed-Cholesky factorisation) and reads off
``A = U^T``, ``B = I - A``, ``s = D``, ``b`` by a triangular solve. Round-tripping
is identity to machine precision (see ``tests/test_bayesnet.py``).

The heavy linear algebra is delegated to ``src/structural/linalg.py`` (the same
kernel ``belief``/``bmr``/``world`` use), so there is one implementation of the
Gaussian math; this module only adds the directed-CPD structure on top.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from src.structural import linalg


# ----------------------------------------------------------------------
# The object: a linear-Gaussian Bayes net stored as its CPD parameters.
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class LinearGaussianBN:
    """A Bayes net over commitments ``x in R^d`` as explicit per-node CPDs.

    ``B``     : (d, d) parent-weight matrix. ``B[v, u]`` is the weight of parent
                ``u`` in node ``v``'s CPD (``0`` => ``u`` is not a parent of
                ``v``). The diagonal is ``0``. For a valid DAG ``B`` must be
                acyclic; the canonical storage order is *parents before children*
                (topological), which makes ``B`` strictly lower-triangular and the
                CPD<->precision maps below pure triangular algebra.
    ``b``     : (d,) per-node CPD intercepts (the conditional mean offset).
    ``s``     : (d,) per-node residual variances (the CPD noise ``s_v > 0``).
    ``names`` : length-d node vocabulary; the order IS the row/column order of
                ``B`` and the basis of the compiled precision form.

    Frozen / immutable: every operation returns a fresh ``LinearGaussianBN``.
    ``names`` is static metadata (never enters array math), exactly as in
    ``GaussianBeliefNet`` -- a plain dataclass, not an ``eqx.Module``, because the
    structural algebra runs on small (d ~ 10) matrices.
    """

    B: jax.Array
    b: jax.Array
    s: jax.Array
    names: tuple[str, ...]

    # -- basic accessors ------------------------------------------------

    @property
    def dim(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        return self.names.index(name)

    def parents(self, name: str) -> tuple[str, ...]:
        """Names of a node's parents (the nonzero entries of its ``B`` row)."""
        v = self.index(name)
        row = self.B[v]
        return tuple(self.names[u] for u in range(self.dim)
                     if u != v and float(row[u]) != 0.0)

    def cpd(self, name: str) -> dict:
        """The explicit CPD of one node:

            x_v | pa(v) ~ N( intercept + sum_u weight_u * x_u ,  variance ).

        Returns ``{"node", "parents", "weights", "intercept", "variance"}`` -- the
        well-defined conditional distribution the legacy joint-only form could not
        name. This is what "the conditional distributions are now built in" means.
        """
        v = self.index(name)
        pa = self.parents(name)
        return {
            "node": name,
            "parents": pa,
            "weights": {u: float(self.B[v, self.index(u)]) for u in pa},
            "intercept": float(self.b[v]),
            "variance": float(self.s[v]),
        }

    # -- the CPD <-> precision bridge (exact bijection) -----------------

    def to_info(self):
        """Compile the CPDs into the joint precision form ``(Pi, h)``.

            A = I - B ,   Pi = A^T diag(1/s) A ,   h = A^T diag(1/s) b .

        Returns a ``belief.GaussianBeliefNet`` so the existing dynamics
        (``step.fuse``, ``world.add_fisher``, ``bmr.*``) consume it unchanged --
        the LGBN is the *definition*, the precision form is its compiled view.
        """
        from src.structural.belief import GaussianBeliefNet
        Pi, h = to_info(self.B, self.b, self.s)
        return GaussianBeliefNet(Pi=Pi, h=h, names=self.names)

    @classmethod
    def from_info(cls, net) -> "LinearGaussianBN":
        """Recover the CPDs from any joint Gaussian ``(Pi, h)`` (requires ``Pi``
        positive-definite). Inverse of ``to_info``: factor ``Pi = U diag(1/s) U^T``
        with ``U`` unit upper-triangular (reversed Cholesky), then ``B = I - U^T``.

        ``net`` is a ``belief.GaussianBeliefNet`` (or anything with ``Pi``/``h``/
        ``names``). This is how an *evolving posterior* is re-read as a Bayes net so
        the per-node CPDs / edge weights can be watched changing over time.
        """
        B, b, s = from_info(net.Pi, net.h)
        return cls(B=B, b=b, s=s, names=tuple(net.names))

    # -- per-node beliefs (well-defined marginals & conditionals) -------

    def joint(self):
        """The joint ``(mean, cov)`` of all nodes (numpy-friendly read-out)."""
        gbn = self.to_info()
        return gbn_mean(gbn), gbn_cov(gbn)

    def sample(self, key: jax.Array, n: int = 1) -> jax.Array:
        """Ancestral samples from the generative DAG -- the direction the joint-only
        ``GaussianBeliefNet`` cannot provide. The structural equation ``x = B x + b + e``
        with ``e ~ N(0, diag(s))`` rearranges to

            A x = b + e ,    A = I - B  (unit lower-triangular in topological order),

        so ``x = A^{-1}(b + sqrt(s) * z)``, ``z ~ N(0, I)``. The marginal mean ``A^{-1} b``
        and covariance ``A^{-1} diag(s) A^{-T}`` equal ``joint()`` exactly (the same ``A``,
        ``s`` ``to_info`` compiles), so this draws from precisely the joint the precision
        form encodes. Returns ``(n, d)``.

        This is what lets an LGBN act as the *true world* a population samples from: when a
        node is HIDDEN (dropped from the returned observation, see ``world_net.sample_world``)
        its couplings carry a correlation into the data that no agent's menu represents -- a
        residual that *emerges* over time rather than being planted."""
        d = self.dim
        A = jnp.eye(d) - self.B
        z = jax.random.normal(key, (d, n))
        rhs = self.b[:, None] + jnp.sqrt(self.s)[:, None] * z       # (d, n)
        x = jnp.linalg.solve(A, rhs)                                # (d, n)
        return x.T                                                  # (n, d)

    def marginal(self, name: str) -> tuple[float, float]:
        """The marginal belief ``(mean, variance)`` at one node -- the per-node
        belief the joint-only form could only get at via an explicit Schur
        complement. Derived from the joint covariance diagonal."""
        mu, cov = self.joint()
        i = self.index(name)
        return float(mu[i]), float(cov[i, i])

    def conditional(self, name: str, evidence: dict[str, float]) -> tuple[float, float]:
        """The conditional belief ``p(x_node | evidence)`` as ``(mean, variance)``,
        where ``evidence`` fixes some other nodes to values. Exact Gaussian
        conditioning in information form: for target ``a`` and observed ``b=x_b``,

            precision_a = Pi_aa ,   mean_a = Pi_aa^{-1} ( h_a - Pi_ab x_b ).
        """
        gbn = self.to_info()
        a = self.index(name)
        b_idx = jnp.asarray([self.index(n) for n in evidence], dtype=jnp.int32)
        x_b = jnp.asarray([evidence[n] for n in evidence])
        Pi_aa = gbn.Pi[a, a]
        Pi_ab = gbn.Pi[a, b_idx]
        h_a = gbn.h[a]
        mean = (h_a - Pi_ab @ x_b) / Pi_aa
        return float(mean), float(1.0 / Pi_aa)

    # -- learning: deposit an observation (relational => edges move) ----

    def observe(self, H: jax.Array, o: jax.Array, sigma_o: float
                ) -> "LinearGaussianBN":
        """Fold one linear-Gaussian observation ``o = H x + eps`` into the net and
        return the updated CPDs.

        ``H`` is (m, d); a row that reads a *single* node is the legacy (diagonal)
        deposit, but a *relational* row (e.g. ``+1`` on ``calx``, ``-1`` on
        ``metal``, ``-1`` on ``gas`` -- a mass balance) deposits off-diagonal Fisher
        information ``H^T H``, which moves the parent weights ``B``: the structure
        is *learned*, not frozen. Implemented as: compile -> ``linalg.add_information``
        -> ``from_info``.
        """
        gbn = self.to_info()
        J, j = linalg.fisher_deposit(H, o, sigma_o)
        Pi, h = linalg.add_information(gbn.Pi, gbn.h, J, j)
        from src.structural.belief import GaussianBeliefNet
        return LinearGaussianBN.from_info(
            GaussianBeliefNet(Pi=Pi, h=h, names=self.names))

    # -- structure edit + Bayesian Model Reduction ---------------------

    def prune_edge(self, child: str, parent: str) -> "LinearGaussianBN":
        """Return a copy with the directed edge ``parent -> child`` removed
        (``B[child, parent] = 0``). A genuine *structural* edit at the CPD level --
        the canonical Bayes-net move BMR scores."""
        v, u = self.index(child), self.index(parent)
        B = self.B.at[v, u].set(0.0)
        return LinearGaussianBN(B=B, b=self.b, s=self.s, names=self.names)

    def log_evidence(self) -> jax.Array:
        """Closed-form Gaussian log-evidence of the compiled joint (``bmr``)."""
        return linalg.log_evidence(*to_info(self.B, self.b, self.s))

    def bmr_prune_edge(self, child: str, parent: str,
                       likelihood: tuple[jax.Array, jax.Array]) -> dict:
        """The closed-form log Bayes factor ``Delta F`` for pruning ``parent ->
        child``, given a shared ``likelihood`` deposit ``(J, j)`` (e.g. the summed
        Fisher information of the data seen so far).

        This is Bayesian Model Reduction at the *edge* level (Friston/Penny
        post-hoc identity): same likelihood, sharper prior (the edge pinned out).
        ``Delta F > 0`` => the data are content without the edge (prune it);
        ``< 0`` => the data hold the coupling, keep it. Delegates the four-term
        log-evidence difference to ``linalg.savage_dickey``.
        """
        J, j = likelihood
        Pi0, h0 = to_info(self.B, self.b, self.s)
        reduced = self.prune_edge(child, parent)
        Pi0r, h0r = to_info(reduced.B, reduced.b, reduced.s)
        Pi_post, h_post = linalg.add_information(Pi0, h0, J, j)
        delta_F, _, _ = linalg.savage_dickey(Pi_post, h_post, Pi0, h0, Pi0r, h0r)
        return {"delta_F": delta_F, "favour_prune": bool(delta_F > 0),
                "reduced": reduced}


# ----------------------------------------------------------------------
# The exact CPD <-> precision algebra (raw arrays; jit/vmap friendly).
# ----------------------------------------------------------------------

def to_info(B: jax.Array, b: jax.Array, s: jax.Array
            ) -> tuple[jax.Array, jax.Array]:
    """Compile CPD parameters into the joint information form.

        A = I - B ,    Pi = A^T diag(1/s) A ,    h = A^T diag(1/s) b .

    Valid for any ``B`` with ``(I - B)`` invertible (always true for a DAG, where
    ``B`` is nilpotent). ``s`` must be strictly positive.
    """
    d = B.shape[-1]
    A = jnp.eye(d) - B
    Dinv = 1.0 / s
    AtD = A.T * Dinv[None, :]          # A^T @ diag(Dinv)
    Pi = AtD @ A
    h = AtD @ b
    return Pi, h


def _uduT(Pi: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Factor a PD matrix as ``Pi = U diag(dinv) U^T`` with ``U`` unit
    upper-triangular. Done by Cholesky on the *reversed* matrix (reversing a
    unit-lower factor's rows and columns yields a unit-upper one), so it reuses the
    stable library Cholesky rather than a hand-rolled UDU^T sweep."""
    d = Pi.shape[-1]
    r = jnp.arange(d - 1, -1, -1)
    Pr = Pi[jnp.ix_(r, r)]
    Lr = jnp.linalg.cholesky(Pr)              # lower: Pr = Lr Lr^T
    sdiag = jnp.diagonal(Lr)
    L_unit = Lr / sdiag[None, :]              # unit lower
    dinv = (sdiag ** 2)[r]                    # reverse back
    U = L_unit[jnp.ix_(r, r)]                 # reverse -> unit upper
    return U, dinv


def from_info(Pi: jax.Array, h: jax.Array
              ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Recover CPD parameters ``(B, b, s)`` from a PD joint ``(Pi, h)`` -- the exact
    inverse of ``to_info`` (parents-before-children order, so ``B`` is strictly
    lower-triangular).

        Pi = U diag(1/s) U^T   (unit upper U)  =>  A = U^T,  B = I - A,
        s  = 1 / dinv,         b = solve(A^T, diag(s) @ (A^T D^{-1})^{-1} h) ...

    concretely ``b`` is the triangular solve ``b = (U diag(dinv))^{-1}-image`` of
    ``h``: since ``h = U diag(dinv) b``, solve ``U y = h`` (U unit upper) then
    ``b = y / dinv``.
    """
    U, dinv = _uduT(Pi)
    d = Pi.shape[-1]
    A = U.T                                   # unit lower
    B = jnp.eye(d) - A
    s = 1.0 / dinv
    y = jnp.linalg.solve(U, h)                # U unit upper-triangular
    b = y / dinv
    return B, b, s


# ----------------------------------------------------------------------
# Small helpers shared with the rest of the package.
# ----------------------------------------------------------------------

def gbn_mean(gbn) -> jax.Array:
    return linalg.info_mean(gbn.Pi, gbn.h)


def gbn_cov(gbn) -> jax.Array:
    return linalg.info_cov(gbn.Pi)


def from_edges(names: tuple[str, ...],
               edges: dict[tuple[str, str], float],
               intercepts: dict[str, float] | None = None,
               variances: dict[str, float] | float = 1.0
               ) -> LinearGaussianBN:
    """Author a Bayes net from an explicit edge list -- the readable constructor.

    ``edges``      : ``{(parent, child): weight}`` -- the DAG and its CPD weights.
    ``intercepts`` : ``{node: b_v}`` (default all 0).
    ``variances``  : ``{node: s_v}`` or a single float (default 1.0).

    ``names`` must already be in parents-before-children (topological) order so the
    CPD<->precision algebra stays triangular; raises if an edge points backwards.
    """
    d = len(names)
    idx = {n: i for i, n in enumerate(names)}
    B = jnp.zeros((d, d))
    for (parent, child), w in edges.items():
        u, v = idx[parent], idx[child]
        if u >= v:
            raise ValueError(
                f"edge {parent!r} -> {child!r} is not parents-before-children; "
                "reorder `names` topologically (roots first).")
        B = B.at[v, u].set(w)
    if intercepts is None:
        b = jnp.zeros((d,))
    else:
        b = jnp.asarray([float(intercepts.get(n, 0.0)) for n in names])
    if isinstance(variances, dict):
        s = jnp.asarray([float(variances.get(n, 1.0)) for n in names])
    else:
        s = jnp.full((d,), float(variances))
    return LinearGaussianBN(B=B, b=b, s=s, names=tuple(names))


def relational_operator(names: tuple[str, ...],
                        relations: list[dict[str, float]]) -> jax.Array:
    """Build an observation operator ``H`` (m, d) whose rows read *combinations* of
    nodes -- the device that lets data deposit off-diagonal Fisher information and
    so move the net's edges.

    Each relation is ``{node: coefficient}``; e.g. a mass-balance experiment that
    weighs ``calx`` against ``metal + gas`` is ``{"calx": 1, "metal": -1,
    "gas": -1}``. A relation on a single node recovers the legacy direct read.
    """
    d = len(names)
    idx = {n: i for i, n in enumerate(names)}
    rows = []
    for rel in relations:
        row = jnp.zeros((d,))
        for n, c in rel.items():
            row = row.at[idx[n]].set(float(c))
        rows.append(row)
    return jnp.stack(rows, axis=0)
