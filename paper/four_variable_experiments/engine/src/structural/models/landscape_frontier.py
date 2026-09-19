"""The landscape environment WITH its endogenous observation frontier (paper section 5).

``landscape_environment.py`` implements the deep world, the partition lattice, and gated
*refinement* -- but its communities observe the full fine vector every step, so the paper's one
novel ingredient (channels with prerequisites; the world legible only through earned channels) is
absent. This module adds it, ground up:

* **Channels.** One observation channel per dendrogram node ``k``: ``y_k = mean(x[members(k)]) +
  sigma_c eps``. A community's *unlocked* set is ``F(cut) = cut ∪ ancestors(cut) ∪ children(cut)``
  -- you can read every aggregate you hold, every coarser aggregate above it, and the one-level
  *probe* into each held block (prereq(child channel) = {parent held}). Splitting a block is
  exactly the move that unlocks its grandchildren's channels: the tech tree of the paper.
* **Frontier-honest statistics.** Each community accumulates forgetful (omega-decayed) first/second
  moments of its channel readings *masked to the unlocked set*. A newly unlocked channel starts
  from zero evidence and must warm up before it can license anything -- structure unlocks channels,
  channels supply Fisher, Fisher supports expansion (the loop of Fig. 1).
* **A bounded learner: budgeted split / swap from channel evidence only.** A community holds at
  most ``budget`` blocks -- finite resolution to allocate. The split signal for a held block is
  the absolute residual of its two *probe* channels (the variance a single common factor cannot
  carry, after subtracting the known readout noise). Under budget it splits the best frontier
  block; at budget it *swaps* -- merging its least-interesting resolved pair -- only when the
  challenger beats the worst incumbent by ``swap_margin``. Selection is by RANK, so no fragile
  scale calibration; lock-in is hysteresis in the rank competition. With ``gate="off"`` (the
  ablation) all channels are visible and the split signal may look arbitrarily deep (max pair
  residual over the whole subtree) -- full observability, the world as an open book.
* **Population.** ``n_communities`` communities on an SBM social graph (inter-block density ->
  lambda2). Seeds are per-community *initial cuts* (uniform, or biased descents into distinct
  subtrees) and/or *value priors* over distinct subtrees with selectivity ``value_beta`` and
  conviction tilt ``value_lambda`` -- the landscape analogues of the big sweep's seed / attention
  beta / tilt axes. A neighbour's split proposal lets a community descend toward that region iff
  its own (unlocked) evidence there clears the floor: sharing advances the unlock operator on the
  union of held structures.
* **World modes.** ``static`` (fixed points freeze; Prop. 1), ``drifting_band`` (the sweeping band
  of landscape_environment), ``regime_change`` (the band jumps at ``t_change`` to the least-valued
  region and stays -- the landscape analogue of the big sweep's changing-epochs lock-in test).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import numpy as np

from src.structural import graphs
from src.structural.bayesnet import LinearGaussianBN


# ----------------------------------------------------------------------
# Deep world (self-contained: same construction as landscape_environment)
# ----------------------------------------------------------------------

def build_deep_world(widths, seed, *, min_parents=2, max_parents=4, coupling_lo=0.20,
                     coupling_hi=0.55, mean_scale=1.0, resid_var=0.5):
    """A multi-level random DAG: each node draws a few parents from shallower levels (mostly the
    level above, some cross-scale), random signed weights. Level (topological) order => ``B``
    strictly lower-triangular and ``Pi* = (I-B)^T diag(1/s) (I-B)`` is PD."""
    rng = np.random.default_rng(seed)
    names = []
    for l, w in enumerate(widths):
        names += [f"L{l}_{i}" for i in range(w)]
    d = len(names)
    start = np.concatenate([[0], np.cumsum(widths)])
    B = np.zeros((d, d), dtype=np.float64)
    for l in range(1, len(widths)):
        for i in range(widths[l]):
            child = start[l] + i
            cand = []
            for pl in range(max(0, l - 3), l):
                pref = 1.0 if pl == l - 1 else 0.35
                cand += [(p, pref) for p in range(start[pl], start[pl + 1])]
            k = min(int(rng.integers(min_parents, max_parents + 1)), len(cand))
            probs = np.array([w for _, w in cand], dtype=np.float64)
            probs /= probs.sum()
            chosen = rng.choice(len(cand), size=k, replace=False, p=probs)
            for c in chosen:
                pidx = cand[int(c)][0]
                w = float(rng.choice([-1.0, 1.0]) * rng.uniform(coupling_lo, coupling_hi))
                B[child, pidx] = w
    b = rng.normal(size=d) * mean_scale
    s = np.full(d, resid_var, dtype=np.float64)
    return LinearGaussianBN(B=np.asarray(B), b=np.asarray(b), s=np.asarray(s), names=tuple(names))


def world_moments(world):
    info = world.to_info()
    Pi = np.asarray(info.Pi, dtype=np.float64)
    Sigma = np.linalg.inv(Pi)
    mu = np.asarray(world.joint()[0], dtype=np.float64)
    return Pi, Sigma, mu


# ----------------------------------------------------------------------
# The multiscale partition lattice (dendrogram) and cut moves
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class DNode:
    id: int
    members: tuple[int, ...]
    children: tuple[int, int] | None
    parent: int | None
    depth: int


@dataclass(frozen=True)
class Dendrogram:
    nodes: tuple[DNode, ...]
    root: int
    n_fine: int

    def node(self, nid: int) -> DNode:
        return self.nodes[nid]

    def atoms(self) -> tuple[int, ...]:
        return tuple(n.id for n in self.nodes if n.children is None)


def _fiedler_order(W):
    deg = W.sum(axis=1)
    L = np.diag(deg) - W
    _, vecs = np.linalg.eigh(L)
    f = vecs[:, 1] if W.shape[0] > 1 else np.zeros(W.shape[0])
    return np.argsort(f, kind="stable")


def build_dendrogram(Sigma, min_atom):
    """Recursive spectral (Fiedler) bisection of the correlation graph -> a binary tree of nested
    fine-node sets (the renormalization hierarchy / partition lattice)."""
    d = Sigma.shape[0]
    dinv = 1.0 / np.sqrt(np.clip(np.diag(Sigma), 1e-12, None))
    corr = np.abs(Sigma * np.outer(dinv, dinv))
    np.fill_diagonal(corr, 0.0)
    nodes = []

    def build(members, parent, depth):
        nid = len(nodes)
        nodes.append(DNode(id=nid, members=tuple(int(m) for m in members),
                           children=None, parent=parent, depth=depth))
        if len(members) > min_atom:
            order = _fiedler_order(corr[np.ix_(members, members)])
            half = len(members) // 2
            left = build(np.sort(members[order[:half]]), nid, depth + 1)
            right = build(np.sort(members[order[half:]]), nid, depth + 1)
            nodes[nid] = replace(nodes[nid], children=(left, right))
        return nid

    root = build(np.arange(d), None, 0)
    return Dendrogram(nodes=tuple(nodes), root=root, n_fine=d)


def leaf_order(dendro):
    """Fine-node indices in dendrogram DFS order (nearby indices share a block)."""
    order = []

    def dfs(nid):
        node = dendro.node(nid)
        if node.children is None:
            order.extend(node.members)
        else:
            dfs(node.children[0]); dfs(node.children[1])

    dfs(dendro.root)
    return np.array(order, dtype=np.int64)


Cut = frozenset


def _assignment(cut, dendro):
    assign = np.full(dendro.n_fine, -1, dtype=np.int64)
    for si, nid in enumerate(sorted(cut)):
        for m in dendro.node(nid).members:
            assign[m] = si
    return assign


def aggregation_matrix(cut, dendro):
    S = np.zeros((len(cut), dendro.n_fine), dtype=np.float64)
    for si, nid in enumerate(sorted(cut)):
        members = dendro.node(nid).members
        S[si, list(members)] = 1.0 / len(members)
    return S


def frontier(cut, dendro):
    """Splittable super-nodes: cut members with children."""
    return tuple(nid for nid in sorted(cut) if dendro.node(nid).children is not None)


def mergeable(cut, dendro):
    """Parents whose BOTH children are currently in the cut -- the coarsening move."""
    members = set(cut)
    return tuple(n.id for n in dendro.nodes
                 if n.children is not None and n.children[0] in members
                 and n.children[1] in members)


def split(cut, nid, dendro):
    left, right = dendro.node(nid).children
    return (cut - {nid}) | {left, right}


def merge(cut, nid, dendro):
    left, right = dendro.node(nid).children
    return (cut - {left, right}) | {nid}


def cut_block_covering(cut, nid, dendro):
    members = set(dendro.node(nid).members)
    for cm in cut:
        if members.issubset(set(dendro.node(cm).members)):
            return cm
    return None


def uniform_cut(dendro, depth):
    """Descend EVERY branch ``depth`` levels -- the common coarse start."""
    cut = {dendro.root}
    for _ in range(depth):
        nxt = set()
        for nid in cut:
            node = dendro.node(nid)
            nxt.update(node.children if node.children is not None else (nid,))
        cut = nxt
    return frozenset(cut)


def partition_distance(cutA, cutB, dendro):
    """Co-membership (Rand-style) distance in [0,1]: fraction of fine-node pairs grouped together
    in one cut but separated in the other."""
    a = _assignment(cutA, dendro); b = _assignment(cutB, dendro)
    iu = np.triu_indices(dendro.n_fine, 1)
    return float(np.mean((a[iu[0]] == a[iu[1]]) != (b[iu[0]] == b[iu[1]])))


def reconstruction_error(cut, Sigma, dendro):
    S = aggregation_matrix(cut, dendro)
    Cc = S @ Sigma @ S.T
    assign = _assignment(cut, dendro)
    Sigma_hat = Cc[np.ix_(assign, assign)]
    return float(np.linalg.norm(Sigma_hat - Sigma) / (np.linalg.norm(Sigma) + 1e-12))


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class FrontierConfig:
    # --- deep world (forwarded to build_deep_world via LandscapeEnvConfig) ---
    widths: tuple[int, ...] = (6, 14, 30, 70)
    min_atom: int = 4
    sigma_o: float = 0.25            # fine-level observation noise
    sigma_c: float = 0.20            # per-channel readout noise
    # --- population / seeds ---
    n_communities: int = 2
    init_depth: int = 1              # uniform part of the initial cut
    init_mode: str = "uniform"       # "uniform" | "subtree_biased" (descend extra in own subtree)
    init_bias_depth: int = 3         # extra descent inside the community's own subtree
    value_depth: int = 2             # depth of the subtrees communities value (4 at depth 2)
    value_beta: float = 1.0          # value selectivity hi/lo (1.0 => uniform values)
    value_lambda: float = 0.0        # conviction tilt on split priority
    # --- frontier ---
    gate: str = "on"                 # "on" (probe channels only) | "off" (full visibility)
    # --- deep pockets (the world's buried structure; what there is to discover) ---
    n_pockets: int = 4
    pocket_depth: int = 4            # how deep the pockets are buried
    pocket_amp: float = 2.0          # idiosyncratic std injected inside a pocket
    # --- dynamics: a bounded learner with a resolution budget ---
    n_steps: int = 80
    omega: float = 0.9               # forgetting (memory ~ 1/(1-omega) steps)
    budget: int = 16                 # max cut size: finite resolution to allocate (the bound)
    trigger: float = 0.03            # absolute priority floor: below this, nothing worth a split
    swap_margin: float = 1.3         # challenger must beat the worst incumbent by this factor
    evidence_min: float = 5.0        # min effective pair count before a channel pair counts
    warmup: int = 6
    # --- world mode ---
    world_mode: str = "static"       # "static" | "drifting_band" | "regime_change" | "pocket_appear"
    band_frac: float = 0.20
    band_amp: float = 2.2
    band_period: int = 30            # drifting_band only
    t_change: int = 30               # regime_change: when the band jumps
    band_pos_a: float = 0.18         # regime_change: pre-change band centre (leaf-order fraction)
    regime_target: str = "coldest"   # "coldest" (least-valued subtree) | "random" (any subtree)
    # --- social graph ---
    inter: float = 0.0
    seed: int = 0


# ----------------------------------------------------------------------
# Channels: one per dendrogram node; unlock = held + one-level probe
# ----------------------------------------------------------------------

def channel_matrix(dendro: Dendrogram) -> np.ndarray:
    """(n_nodes, n_fine) row-normalized membership: ``y = C x`` reads every node's aggregate."""
    C = np.zeros((len(dendro.nodes), dendro.n_fine), dtype=np.float64)
    for node in dendro.nodes:
        C[node.id, list(node.members)] = 1.0 / len(node.members)
    return C


def unlocked_mask(cut: Cut, dendro: Dendrogram) -> np.ndarray:
    """Boolean (n_nodes,): F(cut) = cut ∪ ancestors(cut) ∪ children(cut). Monotone in refinement:
    splitting a block keeps every previously unlocked channel and adds the new probes."""
    mask = np.zeros(len(dendro.nodes), dtype=bool)
    for nid in cut:
        mask[nid] = True
        node = dendro.node(nid)
        if node.children is not None:
            mask[list(node.children)] = True
        p = node.parent
        while p is not None and not mask[p]:
            mask[p] = True
            p = dendro.node(p).parent
    return mask


def _subtree_internal(dendro: Dendrogram, nid: int) -> list[int]:
    """All internal (splittable) nodes of the subtree rooted at ``nid``, including ``nid``."""
    out, stack = [], [nid]
    while stack:
        n = dendro.node(stack.pop())
        if n.children is not None:
            out.append(n.id)
            stack.extend(n.children)
    return out


def depth_nodes(dendro: Dendrogram, depth: int) -> list[int]:
    """Nodes at exactly ``depth`` (or the leaves above it), in DFS (leaf-order) sequence."""
    out: list[int] = []

    def dfs(nid: int, d: int) -> None:
        node = dendro.node(nid)
        if d == depth or node.children is None:
            out.append(nid)
            return
        dfs(node.children[0], d + 1)
        dfs(node.children[1], d + 1)

    dfs(dendro.root, 0)
    return out


# ----------------------------------------------------------------------
# Community state: cut + forgetful masked channel statistics
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class FrontierCommunity:
    name: str
    cut: Cut
    u: np.ndarray                    # (n_fine,) value prior
    Z: np.ndarray                    # (n_nodes,) forgetful per-channel counts (since unlock)
    s1: np.ndarray                   # (n_nodes,) forgetful sum of y
    Z2: np.ndarray                   # (n_nodes, n_nodes) forgetful pair counts
    s2: np.ndarray                   # (n_nodes, n_nodes) forgetful sum of y y^T
    moves: tuple[tuple[int, int, int], ...]   # (step, type[0=split,1=merge], node)


def _pair_residual(comm: FrontierCommunity, left: int, right: int,
                   evidence_min: float, sigma_c: float) -> float | None:
    """ABSOLUTE residual of the (left, right) probe pair: the joint variance a single common
    factor cannot carry (tr - lambda_max of the 2x2 covariance), after subtracting the KNOWN
    readout noise sigma_c^2 from the diagonal (the community knows its instrument). Absolute --
    not fractional -- because the deep world is self-similar in correlation; what distinguishes
    a region worth refining is the *amount* of unexplained variance its aggregates hide.
    ``None`` until the pair has accumulated ``evidence_min`` effective observations since
    unlock -- the Fisher gate of the frontier."""
    if comm.Z2[left, right] < evidence_min:
        return None
    zl, zr, zlr = comm.Z[left], comm.Z[right], comm.Z2[left, right]
    ml, mr = comm.s1[left] / zl, comm.s1[right] / zr
    a = max(comm.s2[left, left] / comm.Z2[left, left] - ml * ml - sigma_c ** 2, 1e-12)
    b = max(comm.s2[right, right] / comm.Z2[right, right] - mr * mr - sigma_c ** 2, 1e-12)
    c = comm.s2[left, right] / zlr - ml * mr
    tr = a + b
    lam_max = 0.5 * tr + np.sqrt(0.25 * (a - b) ** 2 + c * c)
    return float(max(tr - lam_max, 0.0))


# ----------------------------------------------------------------------
# The environment
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class FrontierStepRecord:
    t: int
    cut_sizes: np.ndarray            # (n_c,)
    recon_error: np.ndarray          # (n_c,)
    mean_pdist: float                # mean pairwise partition distance
    unlocked_frac: np.ndarray        # (n_c,)
    union_unlocked_frac: float
    n_split: np.ndarray
    n_merge: np.ndarray
    band_mask: np.ndarray            # (n_fine,) bool
    cold_block_size: np.ndarray      # (n_c,) mean block size over the cold region (regime mode)


class FrontierEnvironment:
    def __init__(self, config: FrontierConfig | None = None):
        self.cfg = FrontierConfig() if config is None else config
        self.world = build_deep_world(self.cfg.widths, self.cfg.seed)
        self.Pi_star, self.Sigma_star, self.mu_star = world_moments(self.world)
        self.dendro = build_dendrogram(self.Sigma_star, self.cfg.min_atom)
        self.order = leaf_order(self.dendro)
        self.C = channel_matrix(self.dendro)
        self.n_nodes = len(self.dendro.nodes)
        n = self.cfg.n_communities
        self.graph = graphs.community([1] * n, intra=1.0, inter=self.cfg.inter,
                                      seed=self.cfg.seed) if n > 1 else graphs.complete(1)
        # community subtrees (round-robin over the depth-`value_depth` nodes, leaf order)
        self.subtrees = depth_nodes(self.dendro, self.cfg.value_depth)
        self.comm_subtree = [self.subtrees[c % len(self.subtrees)] for c in range(n)]
        # deep pockets: one per depth-2 subtree (round robin), buried at pocket_depth -- the
        # world's discoverable structure. Detectable only ~1 level above (aggregation hides them).
        rng = np.random.default_rng(self.cfg.seed + 7919)
        hosts = depth_nodes(self.dendro, 2)
        self.pocket_nodes = []
        for p in range(self.cfg.n_pockets):
            cand = [nid for nid in _subtree_internal(self.dendro, hosts[p % len(hosts)])
                    if self.dendro.node(nid).depth >= self.cfg.pocket_depth]
            if not cand:   # shallow subtree: take its deepest internal node
                sub = _subtree_internal(self.dendro, hosts[p % len(hosts)])
                cand = [max(sub, key=lambda nid: self.dendro.node(nid).depth)]
            self.pocket_nodes.append(int(rng.choice(cand)))
        self.pocket_members = np.array(
            sorted({m for nid in self.pocket_nodes for m in self.dendro.node(nid).members}),
            dtype=np.int64)

    @property
    def lambda2(self) -> float:
        return graphs.algebraic_connectivity(self.graph)

    # -- world modes ------------------------------------------------------------------------
    def _leaf_positions(self) -> np.ndarray:
        pos = np.empty(self.dendro.n_fine, dtype=np.int64)
        pos[self.order] = np.arange(self.dendro.n_fine)
        return pos

    def band_center(self, t: int) -> float | None:
        cfg, n = self.cfg, self.dendro.n_fine
        if cfg.world_mode in ("static", "pocket_appear"):
            return None
        if cfg.world_mode == "drifting_band":
            phase = (t % cfg.band_period) / cfg.band_period
            tri = 2.0 * phase if phase < 0.5 else 2.0 * (1.0 - phase)
            return tri * (n - 1)
        # regime_change: parked at band_pos_a, then jumps to the least-valued region forever
        return (cfg.band_pos_a if t < cfg.t_change else self._cold_band_pos) * (n - 1)

    def band_mask(self, t: int) -> np.ndarray:
        c = self.band_center(t)
        mask = np.zeros(self.dendro.n_fine, dtype=bool)
        if c is None:
            return mask
        w = self.cfg.band_frac * self.dendro.n_fine
        in_band = np.abs(np.arange(self.dendro.n_fine) - c) <= w / 2.0
        mask[self.order[in_band]] = True
        return mask

    def _set_cold_band(self, comms: tuple[FrontierCommunity, ...], run_seed: int) -> None:
        """Regime-change target: the subtree the population values LEAST ("coldest"), or a
        uniformly random subtree ("random") -- new structure appears wherever it pleases."""
        pos = self._leaf_positions()
        if self.cfg.regime_target == "random":
            rng = np.random.default_rng(1_000_003 * (run_seed + 1) + self.cfg.seed)
            target = int(rng.choice(self.subtrees))
        else:
            total_u = np.sum([c.u for c in comms], axis=0)
            scores = [(float(np.mean(total_u[list(self.dendro.node(nid).members)])), nid)
                      for nid in self.subtrees]
            target = min(scores)[1]
        members = list(self.dendro.node(target).members)
        self._cold_band_pos = float(np.mean(pos[members])) / (self.dendro.n_fine - 1)
        self._cold_members = np.array(members, dtype=np.int64)
        self._cold_target = target

    def _set_new_pocket(self, run_seed: int) -> None:
        """pocket_appear target: at t_change a NEW pocket of buried (coarse-invisible) structure
        appears at a random deep node that is not already a pocket."""
        rng = np.random.default_rng(1_000_003 * (run_seed + 1) + self.cfg.seed)
        cand = [n.id for n in self.dendro.nodes
                if n.children is not None and n.depth >= self.cfg.pocket_depth
                and n.id not in self.pocket_nodes]
        target = int(rng.choice(cand))
        self._cold_target = target
        self._cold_members = np.array(self.dendro.node(target).members, dtype=np.int64)

    # -- sampling through the channel bank ----------------------------------------------------
    def _sample_channels(self, t: int, key: jax.Array, n_comm: int) -> np.ndarray:
        """One world draw, shared across communities; per-community channel readout noise.
        Returns (n_comm, n_nodes)."""
        keys = jax.random.split(key, 3 + n_comm)
        x = np.asarray(self.world.sample(keys[0], 1)[0], dtype=np.float64)
        noise = np.asarray(jax.random.normal(keys[1], (self.dendro.n_fine,)), dtype=np.float64)
        x = x + self.cfg.sigma_o * noise
        mask = self.band_mask(t)
        x[mask] += self.cfg.band_amp * noise[mask]
        pk = np.asarray(jax.random.normal(keys[2], (len(self.pocket_members),)), dtype=np.float64)
        x[self.pocket_members] += self.cfg.pocket_amp * pk
        if self.cfg.world_mode == "pocket_appear" and t >= self.cfg.t_change:
            new = np.asarray(jax.random.normal(keys[2], (len(self._cold_members) + 1,)),
                             dtype=np.float64)[1:]      # offset draw, independent of pk pattern
            x[self._cold_members] += self.cfg.pocket_amp * new
        base = self.C @ x
        out = np.empty((n_comm, self.n_nodes))
        for ci in range(n_comm):
            eps = np.asarray(jax.random.normal(keys[3 + ci], (self.n_nodes,)), dtype=np.float64)
            out[ci] = base + self.cfg.sigma_c * eps
        return out

    # -- lifecycle ----------------------------------------------------------------------------
    def _initial_cut(self, ci: int) -> Cut:
        cut = uniform_cut(self.dendro, self.cfg.init_depth)
        if self.cfg.init_mode != "subtree_biased":
            return cut
        # descend extra levels toward and into this community's own subtree (a biased coarse seed)
        own_members = set(self.dendro.node(self.comm_subtree[ci]).members)
        for _ in range(self.cfg.init_bias_depth):
            nxt = set(cut)
            for nid in cut:
                node = self.dendro.node(nid)
                blk = set(node.members)
                if node.children is not None and (blk <= own_members or own_members <= blk):
                    nxt.discard(nid)
                    nxt.update(node.children)
            cut = frozenset(nxt)
        return cut

    def reset(self, seed: int | None = None) -> tuple[tuple[FrontierCommunity, ...], jax.Array]:
        cfg = self.cfg
        key = jax.random.PRNGKey(cfg.seed if seed is None else seed)
        lo = 1.0 / cfg.value_beta
        comms = []
        for ci in range(cfg.n_communities):
            u = np.full(self.dendro.n_fine, lo, dtype=np.float64)
            u[list(self.dendro.node(self.comm_subtree[ci]).members)] = 1.0
            comms.append(FrontierCommunity(
                name=f"C{ci}", cut=self._initial_cut(ci), u=u,
                Z=np.zeros(self.n_nodes), s1=np.zeros(self.n_nodes),
                Z2=np.zeros((self.n_nodes, self.n_nodes)),
                s2=np.zeros((self.n_nodes, self.n_nodes)), moves=tuple()))
        comms = tuple(comms)
        run_seed = cfg.seed if seed is None else seed
        if cfg.world_mode == "regime_change":
            self._set_cold_band(comms, run_seed)
        elif cfg.world_mode == "pocket_appear":
            self._set_new_pocket(run_seed)
        return comms, key

    def _mask(self, cut: Cut) -> np.ndarray:
        if self.cfg.gate == "off":
            return np.ones(self.n_nodes, dtype=bool)
        return unlocked_mask(cut, self.dendro)

    def _accumulate(self, comm: FrontierCommunity, y: np.ndarray) -> FrontierCommunity:
        """Forgetful sufficient statistics over the UNLOCKED channels only."""
        w = self.cfg.omega
        m = self._mask(comm.cut).astype(np.float64)
        ym = y * m
        return replace(comm, Z=w * comm.Z + m, s1=w * comm.s1 + ym,
                       Z2=w * comm.Z2 + np.outer(m, m), s2=w * comm.s2 + np.outer(ym, ym))

    @property
    def _evidence_min(self) -> float:
        """The forgetful count asymptotes at 1/(1-omega); the Fisher gate must sit below it or
        evidence never 'counts'. Floor = min(cfg.evidence_min, 60% of the asymptote)."""
        return min(self.cfg.evidence_min, 0.6 / (1.0 - self.cfg.omega + 1e-9))

    def _split_priority(self, comm: FrontierCommunity, nid: int) -> float | None:
        """Gate on: residual of the block's two probe channels. Gate off: the deepest pair
        residual anywhere in the subtree (full observability sees buried structure)."""
        cfg = self.cfg
        node = self.dendro.node(nid)
        candidates = ([nid] if cfg.gate == "on" else _subtree_internal(self.dendro, nid))
        best = None
        for cand in candidates:
            l, r = self.dendro.node(cand).children
            resid = _pair_residual(comm, l, r, self._evidence_min, cfg.sigma_c)
            if resid is not None and (best is None or resid > best):
                best = resid
        if best is None:
            return None
        value = float(comm.u[list(node.members)].mean())
        return best * (1.0 + cfg.value_lambda * value)

    def _keep_priority(self, comm: FrontierCommunity, nid: int) -> float | None:
        """How much keeping the pair under mergeable parent ``nid`` resolved is worth: its pair
        residual x the same conviction tilt -- you defend resolution where you value."""
        l, r = self.dendro.node(nid).children
        resid = _pair_residual(comm, l, r, self._evidence_min, self.cfg.sigma_c)
        if resid is None:
            return None
        value = float(comm.u[list(self.dendro.node(nid).members)].mean())
        return resid * (1.0 + self.cfg.value_lambda * value)

    def _try_expand(self, comm: FrontierCommunity, nid: int, pri: float,
                    t: int) -> tuple[FrontierCommunity, int, int]:
        """Split ``nid`` if the budget allows; otherwise swap it in against the community's worst
        incumbent pair iff it wins by ``swap_margin``. Returns (state, n_split, n_merge)."""
        cfg = self.cfg
        if len(comm.cut) < cfg.budget:
            return (replace(comm, cut=split(comm.cut, nid, self.dendro),
                            moves=comm.moves + ((t, 0, nid),)), 1, 0)
        worst_m, worst_keep = -1, np.inf
        for cand in mergeable(comm.cut, self.dendro):
            if cand == nid or cand == self.dendro.node(nid).parent:
                continue   # merging the challenger's own parent would undo the swap
            keep = self._keep_priority(comm, cand)
            if keep is not None and keep < worst_keep:
                worst_m, worst_keep = cand, keep
        if worst_m >= 0 and pri > cfg.swap_margin * worst_keep:
            cut = merge(comm.cut, worst_m, self.dendro)
            cut = split(cut, nid, self.dendro)
            return (replace(comm, cut=cut,
                            moves=comm.moves + ((t, 1, worst_m), (t, 0, nid))), 1, 1)
        return comm, 0, 0

    def _refine(self, comm: FrontierCommunity, t: int) -> tuple[FrontierCommunity, int, int]:
        """One expansion attempt (best frontier block over the trigger; budgeted, possibly a swap)
        plus one free merge of a pair that has gone quiet (hysteresis floor at trigger/2)."""
        cfg = self.cfg
        best_nid, best_pri = -1, cfg.trigger
        for nid in frontier(comm.cut, self.dendro):
            pri = self._split_priority(comm, nid)
            if pri is not None and pri > best_pri:
                best_nid, best_pri = nid, pri
        n_split = n_merge = 0
        if best_nid >= 0:
            comm, n_split, n_merge = self._try_expand(comm, best_nid, best_pri, t)
        # free merge: release budget from a pair with nothing left to say
        quiet_m, quiet_keep = -1, 0.5 * cfg.trigger
        for cand in mergeable(comm.cut, self.dendro):
            keep = self._keep_priority(comm, cand)
            if keep is not None and keep < quiet_keep:
                quiet_m, quiet_keep = cand, keep
        if quiet_m >= 0:
            comm = replace(comm, cut=merge(comm.cut, quiet_m, self.dendro),
                           moves=comm.moves + ((t, 1, quiet_m),))
            n_merge += 1
        return comm, n_split, n_merge

    def step(self, comms: tuple[FrontierCommunity, ...], key: jax.Array,
             t: int) -> tuple[tuple[FrontierCommunity, ...], jax.Array, FrontierStepRecord]:
        cfg = self.cfg
        key, sub = jax.random.split(key)
        ys = self._sample_channels(t, sub, len(comms))
        comms = list(comms)
        n_split = np.zeros(len(comms), dtype=np.int64)
        n_merge = np.zeros(len(comms), dtype=np.int64)
        local_splits = [-1] * len(comms)

        for ci in range(len(comms)):
            comm = self._accumulate(comms[ci], ys[ci])
            if t >= cfg.warmup:
                comm, ns, nm = self._refine(comm, t)
                n_split[ci], n_merge[ci] = ns, nm
                local_splits[ci] = next((nid for (tt, ty, nid) in reversed(comm.moves)
                                         if tt == t and ty == 0), -1)
            comms[ci] = comm

        # social sharing: a data-supported split proposal travels the social graph. The receiver
        # descends ONE step toward the reported region, acting on the better of its own evidence
        # and the sender's reported priority (testimony) -- the budget/swap rule still applies.
        # This is how connected communities advance Phi on the union of their held structures.
        if len(comms) > 1:
            A = np.asarray(self.graph.A)
            for sender, nid in enumerate(local_splits):
                if nid < 0:
                    continue
                pri_s = self._split_priority(comms[sender], nid)  # probe stats survive the split
                for recv in range(len(comms)):
                    if recv == sender or A[sender, recv] <= 0:
                        continue
                    comm = comms[recv]
                    cm = cut_block_covering(comm.cut, nid, self.dendro)
                    if cm is None or self.dendro.node(cm).children is None:
                        continue
                    pri_own = self._split_priority(comm, cm)
                    pri = max((p for p in (pri_own, pri_s) if p is not None), default=None)
                    if pri is not None and pri > cfg.trigger:
                        comm, ns, nm = self._try_expand(comm, cm, pri, t)
                        comms[recv] = comm
                        n_split[recv] += ns
                        n_merge[recv] += nm

        masks = np.stack([unlocked_mask(c.cut, self.dendro) for c in comms])
        pd = ([partition_distance(a.cut, b.cut, self.dendro)
               for i, a in enumerate(comms) for b in comms[i + 1:]] or [0.0])
        cold = np.zeros(len(comms))
        if cfg.world_mode in ("regime_change", "pocket_appear"):
            for ci, c in enumerate(comms):
                cold[ci] = self._region_block_size(c.cut, self._cold_members)
        rec = FrontierStepRecord(
            t=t,
            cut_sizes=np.array([len(c.cut) for c in comms], dtype=np.int64),
            recon_error=np.array([reconstruction_error(c.cut, self.Sigma_star, self.dendro)
                                  for c in comms]),
            mean_pdist=float(np.mean(pd)),
            unlocked_frac=masks.mean(axis=1),
            union_unlocked_frac=float(masks.any(axis=0).mean()),
            n_split=n_split, n_merge=n_merge,
            band_mask=self.band_mask(t),
            cold_block_size=cold)
        return tuple(comms), key, rec

    def _region_block_size(self, cut: Cut, members: np.ndarray) -> float:
        """Mean held-block size over a fine-node region (large = unresolved)."""
        assign = _assignment(cut, self.dendro)
        blocks, counts = np.unique(assign, return_counts=True)
        size_of = dict(zip(blocks, counts))
        return float(np.mean([size_of[assign[m]] for m in members]))

    def run(self, seed: int | None = None) -> dict:
        comms, key = self.reset(seed)
        recs: list[FrontierStepRecord] = []
        for t in range(self.cfg.n_steps):
            comms, key, rec = self.step(comms, key, t)
            recs.append(rec)
        return self._to_arrays(recs, comms)

    # -- output -------------------------------------------------------------------------------
    def _to_arrays(self, recs: list[FrontierStepRecord],
                   comms: tuple[FrontierCommunity, ...]) -> dict:
        n_t, n_c = len(recs), len(comms)
        out = {
            "cut_size_tc": np.array([r.cut_sizes for r in recs]),
            "recon_error_tc": np.array([r.recon_error for r in recs]),
            "mean_pdist_t": np.array([r.mean_pdist for r in recs]),
            "unlocked_frac_tc": np.array([r.unlocked_frac for r in recs]),
            "union_unlocked_frac_t": np.array([r.union_unlocked_frac for r in recs]),
            "n_split_tc": np.array([r.n_split for r in recs]),
            "n_merge_tc": np.array([r.n_merge for r in recs]),
            "band_mask_tf": np.array([r.band_mask for r in recs]),
            "final_assignment_cf": np.stack([_assignment(c.cut, self.dendro) for c in comms]),
            "final_cuts": [sorted(c.cut) for c in comms],
            "moves": [list(map(list, c.moves)) for c in comms],
            "leaf_order": self.order,
            "n_fine": np.array(self.dendro.n_fine),
            "lambda2": np.array(self.lambda2),
            "comm_subtree": np.array(self.comm_subtree),
        }
        masks = np.stack([unlocked_mask(c.cut, self.dendro) for c in comms])
        out["final_unlocked_cn"] = masks
        out["never_unlocked_frac"] = np.array(1.0 - masks.any(axis=0).mean())
        out["final_pdist"] = np.array(recs[-1].mean_pdist)
        warm = self.cfg.warmup
        out["mean_pdist"] = np.array(float(out["mean_pdist_t"][warm:].mean()))
        if self.cfg.world_mode in ("regime_change", "pocket_appear"):
            # resolution of the post-change target region: mean block size over its members
            # (large = unresolved = locked in / unconceived).
            out["cold_block_size_tc"] = np.array([r.cold_block_size for r in recs])
            out["cold_region_block_size_c"] = np.array(
                [self._region_block_size(c.cut, self._cold_members) for c in comms])
            out["cold_members"] = self._cold_members
        return out
