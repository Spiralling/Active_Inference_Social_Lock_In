"""A *dynamic* deep landscape environment learned by coarse-graining refinement.

Runnable counterpart of ``paper/landscape_environment.tex``. The world is deep AND **non-stationary**:
a localized band of extra structure sweeps across the landscape over time, so *where* fine resolution
is worth spending keeps moving. Communities hold genuinely different coarse-grainings (cuts of a
multiscale partition lattice), **forget** old evidence, and continually **split** (refine where the
band now is) and **merge** (coarsen where it has left) -- so the cut never freezes. There is no
refinement budget; the cut size breathes with the world.

Pieces, ground up:

* **Deep world.** ``build_deep_world`` -- a multi-level random DAG (~120 nodes), irregular multi-scale
  ``Σ*``; this fixes the *base* correlation and the partition lattice.
* **Drift.** ``band_intensity`` -- a window (in dendrogram-leaf order) of extra idiosyncratic variance
  whose centre sweeps back and forth. A block overlapping the band carries variance a single
  super-node cannot summarise -> it asks to be split; once the band leaves, the block can be merged.
* **Partition lattice.** ``build_dendrogram`` -- recursive Fiedler bisection of the correlation graph.
  A community's model is a **cut** (antichain) with a real coarse net ``Π_c = (S Σ* Sᵀ)⁻¹``.
* **Forgetting + split/merge.** Each step a community accumulates *decayed* sufficient statistics
  (memory ``~1/(1-ω)`` steps), splits the frontier block of highest ``residual × value`` over a
  trigger, and merges back any sibling pair whose combined residual has fallen below a (lower)
  hysteresis trigger. The value prior gates *where* a community invests resolution.
* **Pluralism, emergent and persistent.** Both communities start from the *same* coarse cut; A values
  the left half, B the right. As the band sweeps, A refines its half, B refines its -- a persistent,
  oscillating structural divergence. A social graph (SBM) lets neighbours share refinements, which
  pulls connected communities to track together and collapses the divergence.

We reuse the repo's belief algebra and graph library; the cosmology hub-wake is not used.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import numpy as np

from src.structural import graphs
from src.structural.bayesnet import LinearGaussianBN
from src.structural.belief import GaussianBeliefNet


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class LandscapeEnvConfig:
    # --- deep world ---
    widths: tuple[int, ...] = (6, 14, 30, 70)   # nodes per level (deep core -> belt); ~120 total
    min_parents: int = 2
    max_parents: int = 4
    coupling_lo: float = 0.20
    coupling_hi: float = 0.55
    mean_scale: float = 1.0
    resid_var: float = 0.5
    # --- partition lattice ---
    min_atom: int = 4
    init_depth: int = 2                          # initial cut: uniform descent (SAME for both)
    # --- drift (the dynamism) ---
    n_steps: int = 70
    band_frac: float = 0.22                      # band width as a fraction of the nodes
    band_amp: float = 2.2                        # extra idiosyncratic std injected inside the band
    band_period: int = 30                        # steps for one back-and-forth sweep
    sigma_o: float = 0.25
    # --- forgetting + refinement ---
    omega: float = 0.82                          # forgetting factor (memory ~ 1/(1-ω) steps)
    trigger: float = 0.55                        # min residual×value PRIORITY to split
    merge_resid: float = 0.18                    # merge a sibling pair when its residual drops below
    value_lambda: float = 6.0                    # conviction tilt on split priority
    value_hi: float = 1.0
    value_lo: float = 0.15
    warmup: int = 4                              # let the forgetful estimate fill before acting
    # --- social graph ---
    intra: float = 1.0
    inter: float = 0.0
    seed: int = 0


# ----------------------------------------------------------------------
# 1. The deep world
# ----------------------------------------------------------------------

def build_deep_world(cfg: LandscapeEnvConfig) -> LinearGaussianBN:
    """A multi-level random DAG: each node draws a few parents from shallower levels (mostly the level
    above, some cross-scale), random signed weights. Level (topological) order => ``B`` strictly
    lower-triangular and ``Π* = (I-B)ᵀ diag(1/s) (I-B)`` is PD."""
    rng = np.random.default_rng(cfg.seed)
    widths = cfg.widths
    names: list[str] = []
    for l, w in enumerate(widths):
        names += [f"L{l}_{i}" for i in range(w)]
    d = len(names)
    start = np.concatenate([[0], np.cumsum(widths)])
    B = np.zeros((d, d), dtype=np.float64)
    for l in range(1, len(widths)):
        for i in range(widths[l]):
            child = start[l] + i
            cand: list[tuple[int, float]] = []
            for pl in range(max(0, l - 3), l):
                pref = 1.0 if pl == l - 1 else 0.35
                cand += [(p, pref) for p in range(start[pl], start[pl + 1])]
            k = min(int(rng.integers(cfg.min_parents, cfg.max_parents + 1)), len(cand))
            probs = np.array([w for _, w in cand], dtype=np.float64)
            probs /= probs.sum()
            chosen = rng.choice(len(cand), size=k, replace=False, p=probs)
            for c in chosen:
                p = cand[int(c)][0]
                w = float(rng.choice([-1.0, 1.0]) * rng.uniform(cfg.coupling_lo, cfg.coupling_hi))
                B[child, p] = w
    b = rng.normal(size=d) * cfg.mean_scale
    s = np.full(d, cfg.resid_var, dtype=np.float64)
    return LinearGaussianBN(B=np.asarray(B), b=np.asarray(b), s=np.asarray(s), names=tuple(names))


def world_moments(world: LinearGaussianBN) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    info = world.to_info()
    Pi = np.asarray(info.Pi, dtype=np.float64)
    Sigma = np.linalg.inv(Pi)
    mu = np.asarray(world.joint()[0], dtype=np.float64)
    return Pi, Sigma, mu


# ----------------------------------------------------------------------
# 2. The multiscale partition lattice (dendrogram)
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


def _fiedler_order(W: np.ndarray) -> np.ndarray:
    deg = W.sum(axis=1)
    L = np.diag(deg) - W
    _, vecs = np.linalg.eigh(L)
    f = vecs[:, 1] if W.shape[0] > 1 else np.zeros(W.shape[0])
    return np.argsort(f, kind="stable")


def build_dendrogram(Sigma: np.ndarray, min_atom: int) -> Dendrogram:
    """Recursive spectral (Fiedler) bisection of the correlation graph -> a balanced binary tree of
    nested fine-node sets (the renormalization hierarchy / partition lattice)."""
    d = Sigma.shape[0]
    dinv = 1.0 / np.sqrt(np.clip(np.diag(Sigma), 1e-12, None))
    corr = np.abs(Sigma * np.outer(dinv, dinv))
    np.fill_diagonal(corr, 0.0)
    nodes: list[DNode] = []

    def build(members: np.ndarray, parent: int | None, depth: int) -> int:
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


def leaf_order(dendro: Dendrogram) -> np.ndarray:
    """Fine-node indices in dendrogram DFS order (nearby indices share a block)."""
    order: list[int] = []

    def dfs(nid: int) -> None:
        node = dendro.node(nid)
        if node.children is None:
            order.extend(node.members)
        else:
            dfs(node.children[0]); dfs(node.children[1])

    dfs(dendro.root)
    return np.array(order, dtype=np.int64)


# ----------------------------------------------------------------------
# 3. Cuts, the coarse net, and the split / merge moves
# ----------------------------------------------------------------------

Cut = frozenset


def _assignment(cut: Cut, dendro: Dendrogram) -> np.ndarray:
    assign = np.full(dendro.n_fine, -1, dtype=np.int64)
    for si, nid in enumerate(sorted(cut)):
        for m in dendro.node(nid).members:
            assign[m] = si
    return assign


def aggregation_matrix(cut: Cut, dendro: Dendrogram) -> np.ndarray:
    S = np.zeros((len(cut), dendro.n_fine), dtype=np.float64)
    for si, nid in enumerate(sorted(cut)):
        members = dendro.node(nid).members
        S[si, list(members)] = 1.0 / len(members)
    return S


def coarse_net(cut: Cut, Sigma: np.ndarray, mu: np.ndarray, dendro: Dendrogram) -> GaussianBeliefNet:
    """The community's belief on its super-nodes: ``Π_c = (S Σ* Sᵀ)⁻¹``. Dimension = #super-nodes."""
    S = aggregation_matrix(cut, dendro)
    Pi_c = np.linalg.inv(S @ Sigma @ S.T)
    return GaussianBeliefNet(Pi=np.asarray(Pi_c), h=np.asarray(Pi_c @ (S @ mu)),
                             names=tuple(f"S{nid}" for nid in sorted(cut)))


def frontier(cut: Cut, dendro: Dendrogram) -> tuple[int, ...]:
    """Splittable super-nodes: cut members with children (reachable because you split their parent)."""
    return tuple(nid for nid in sorted(cut) if dendro.node(nid).children is not None)


def mergeable(cut: Cut, dendro: Dendrogram) -> tuple[int, ...]:
    """Parents whose BOTH children are currently in the cut -- the reverse (coarsening) move."""
    members = set(cut)
    out = []
    for n in dendro.nodes:
        if n.children is not None and n.children[0] in members and n.children[1] in members:
            out.append(n.id)
    return tuple(out)


def split(cut: Cut, nid: int, dendro: Dendrogram) -> Cut:
    left, right = dendro.node(nid).children
    return (cut - {nid}) | {left, right}


def merge(cut: Cut, nid: int, dendro: Dendrogram) -> Cut:
    left, right = dendro.node(nid).children
    return (cut - {left, right}) | {nid}


def cut_block_covering(cut: Cut, nid: int, dendro: Dendrogram) -> int | None:
    members = set(dendro.node(nid).members)
    for cm in cut:
        if members.issubset(set(dendro.node(cm).members)):
            return cm
    return None


def uniform_cut(dendro: Dendrogram, depth: int) -> Cut:
    """Descend EVERY branch ``depth`` levels -- the common coarse start for both communities."""
    cut = {dendro.root}
    for _ in range(depth):
        nxt = set()
        for nid in cut:
            node = dendro.node(nid)
            nxt.update(node.children if node.children is not None else (nid,))
        cut = nxt
    return frozenset(cut)


def value_prior(dendro: Dendrogram, prefer: str, hi: float, lo: float) -> np.ndarray:
    u = np.full(dendro.n_fine, lo, dtype=np.float64)
    left, right = dendro.node(dendro.root).children
    target = left if prefer == "left" else right
    for m in dendro.node(target).members:
        u[m] = hi
    return u


# ----------------------------------------------------------------------
# 4. Structural metrics
# ----------------------------------------------------------------------

def partition_distance(cutA: Cut, cutB: Cut, dendro: Dendrogram) -> float:
    """Co-membership (Rand-style) distance in [0,1]: fraction of fine-node pairs grouped together in
    one cut but separated in the other."""
    a = _assignment(cutA, dendro); b = _assignment(cutB, dendro)
    iu = np.triu_indices(dendro.n_fine, 1)
    return float(np.mean((a[iu[0]] == a[iu[1]]) != (b[iu[0]] == b[iu[1]])))


def reconstruction_error(cut: Cut, Sigma: np.ndarray, dendro: Dendrogram) -> float:
    S = aggregation_matrix(cut, dendro)
    Cc = S @ Sigma @ S.T
    assign = _assignment(cut, dendro)
    Sigma_hat = Cc[np.ix_(assign, assign)]
    return float(np.linalg.norm(Sigma_hat - Sigma) / (np.linalg.norm(Sigma) + 1e-12))


# ----------------------------------------------------------------------
# 5. Community state + the dynamic environment
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class CommunityState:
    name: str
    cut: Cut
    u: np.ndarray
    Z: float                 # forgetful effective count
    s1: np.ndarray           # forgetful sum of x
    s2: np.ndarray           # forgetful sum of x xᵀ
    moves: tuple[tuple[int, int, int], ...]   # (step, type[0=split,1=merge], node)


@dataclass(frozen=True)
class EnvState:
    t: int
    communities: tuple[CommunityState, ...]
    key: jax.Array


@dataclass(frozen=True)
class StepRecord:
    t: int
    cut_sizes: np.ndarray
    recon_error: np.ndarray
    assignments: np.ndarray      # (n_c, n_fine)
    band_intensity: np.ndarray   # (n_fine,)
    partition_distance: float
    n_split: np.ndarray          # (n_c,)
    n_merge: np.ndarray          # (n_c,)


class LandscapeEnvironment:
    def __init__(self, config: LandscapeEnvConfig | None = None):
        self.cfg = LandscapeEnvConfig() if config is None else config
        self.world = build_deep_world(self.cfg)
        self.Pi_star, self.Sigma_star, self.mu_star = world_moments(self.world)
        self.dendro = build_dendrogram(self.Sigma_star, self.cfg.min_atom)
        self.order = leaf_order(self.dendro)                 # fine index per leaf-position
        self.graph = graphs.community([1, 1], intra=self.cfg.intra, inter=self.cfg.inter,
                                      seed=self.cfg.seed)

    @property
    def lambda2(self) -> float:
        return graphs.algebraic_connectivity(self.graph)

    # -- the drift: a band of extra idiosyncratic variance sweeping the leaf order ---------------
    def band_center(self, t: int) -> float:
        phase = (t % self.cfg.band_period) / self.cfg.band_period
        tri = 2.0 * phase if phase < 0.5 else 2.0 * (1.0 - phase)   # 0->1->0 triangle
        return tri * (self.dendro.n_fine - 1)

    def band_mask(self, t: int) -> np.ndarray:
        """Boolean (n_fine,) over fine indices: which nodes carry the extra band variance at step t."""
        w = self.cfg.band_frac * self.dendro.n_fine
        c = self.band_center(t)
        pos = np.arange(self.dendro.n_fine)
        in_band_positions = np.abs(pos - c) <= w / 2.0
        mask = np.zeros(self.dendro.n_fine, dtype=bool)
        mask[self.order[in_band_positions]] = True
        return mask

    def _sample(self, t: int, key: jax.Array) -> np.ndarray:
        k1, k2 = jax.random.split(key)
        x = np.asarray(self.world.sample(k1, 1)[0], dtype=np.float64)      # deep base draw
        noise = np.asarray(jax.random.normal(k2, (self.dendro.n_fine,)), dtype=np.float64)
        x = x + self.cfg.sigma_o * noise
        mask = self.band_mask(t)
        x[mask] += self.cfg.band_amp * noise[mask]                        # extra idiosyncratic structure
        return x

    @staticmethod
    def _residual_of(cov: np.ndarray, members: tuple[int, ...]) -> float:
        """FRACTION of block variance a single (rank-one) super-node summary cannot carry."""
        block = cov[np.ix_(members, members)]
        if block.shape[0] <= 1:
            return 0.0
        tr = float(block.trace())
        if tr <= 1e-12:
            return 0.0
        return float((tr - np.linalg.eigvalsh(block).max()) / tr)

    def _cov(self, comm: CommunityState) -> np.ndarray:
        mean = comm.s1 / comm.Z
        return comm.s2 / comm.Z - np.outer(mean, mean)

    # -- lifecycle ------------------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> EnvState:
        key = jax.random.PRNGKey(self.cfg.seed if seed is None else seed)
        d = self.dendro.n_fine
        start = uniform_cut(self.dendro, self.cfg.init_depth)             # SAME coarse start for both
        comms = tuple(
            CommunityState(name=name, cut=start, u=value_prior(self.dendro, prefer,
                                                               self.cfg.value_hi, self.cfg.value_lo),
                           Z=0.0, s1=np.zeros(d), s2=np.zeros((d, d)), moves=tuple())
            for name, prefer in (("A", "left"), ("B", "right")))
        return EnvState(t=0, communities=comms, key=key)

    def _forget_accumulate(self, comm: CommunityState, x: np.ndarray) -> CommunityState:
        w = self.cfg.omega
        return replace(comm, Z=w * comm.Z + 1.0, s1=w * comm.s1 + x, s2=w * comm.s2 + np.outer(x, x))

    def _refine(self, comm: CommunityState, t: int) -> tuple[CommunityState, int, int]:
        """One split (highest residual×value over trigger) and one merge (a sibling pair whose residual
        has dropped below the hysteresis floor). Returns updated state + (#split, #merge) this step."""
        cfg = self.cfg
        cov = self._cov(comm)
        n_split = n_merge = 0

        # SPLIT: invest resolution where the band now is, on the community's valued side
        best_nid, best_pri = -1, cfg.trigger
        for nid in frontier(comm.cut, self.dendro):
            members = self.dendro.node(nid).members
            resid = self._residual_of(cov, members)
            value = float(comm.u[list(members)].mean())
            pri = resid * (1.0 + cfg.value_lambda * value)
            if pri > best_pri:
                best_nid, best_pri = nid, pri
        if best_nid >= 0:
            comm = replace(comm, cut=split(comm.cut, best_nid, self.dendro),
                           moves=comm.moves + ((t, 0, best_nid),))
            n_split = 1

        # MERGE: coarsen a sibling pair the band has left (combined residual below the floor)
        best_m, best_resid = -1, cfg.merge_resid
        for nid in mergeable(comm.cut, self.dendro):
            resid = self._residual_of(cov, self.dendro.node(nid).members)
            if resid < best_resid:
                best_m, best_resid = nid, resid
        if best_m >= 0 and best_m != best_nid:
            comm = replace(comm, cut=merge(comm.cut, best_m, self.dendro),
                           moves=comm.moves + ((t, 1, best_m),))
            n_merge = 1
        return comm, n_split, n_merge

    def step(self, state: EnvState) -> tuple[EnvState, StepRecord]:
        cfg = self.cfg
        keys = jax.random.split(state.key, len(state.communities) + 1)
        comms = list(state.communities)
        n_split = np.zeros(len(comms), dtype=np.int64)
        n_merge = np.zeros(len(comms), dtype=np.int64)
        local_splits = [-1] * len(comms)

        # 1. observe the drifting world + 2. local split/merge
        for ci, k in enumerate(keys[1:]):
            comm = self._forget_accumulate(comms[ci], self._sample(state.t, k))
            if state.t >= cfg.warmup:
                comm, ns, nm = self._refine(comm, state.t)
                n_split[ci], n_merge[ci] = ns, nm
                local_splits[ci] = next((nid for (tt, ty, nid) in reversed(comm.moves)
                                         if tt == state.t and ty == 0), -1)
            comms[ci] = comm

        # 3. social sharing: a peer's split makes a connected neighbour descend ONE step toward that
        #    region -- but only if its OWN residual there clears the (relaxed) social floor.
        for sender in range(len(comms)):
            nid = local_splits[sender]
            if nid < 0:
                continue
            for recv in range(len(comms)):
                if recv == sender or self.graph.A[sender, recv] <= 0:
                    continue
                comm = comms[recv]
                cm = cut_block_covering(comm.cut, nid, self.dendro)
                if cm is None or self.dendro.node(cm).children is None:
                    continue
                resid = self._residual_of(self._cov(comm), self.dendro.node(cm).members)
                if resid >= cfg.merge_resid:        # reachable structure the receiver also sees
                    comms[recv] = replace(comm, cut=split(comm.cut, cm, self.dendro),
                                          moves=comm.moves + ((state.t, 0, cm),))
                    n_split[recv] += 1

        assigns = np.stack([_assignment(c.cut, self.dendro) for c in comms], axis=0)
        rec = StepRecord(
            t=state.t,
            cut_sizes=np.array([len(c.cut) for c in comms], dtype=np.int64),
            recon_error=np.array([reconstruction_error(c.cut, self.Sigma_star, self.dendro)
                                  for c in comms]),
            assignments=assigns,
            band_intensity=self.band_mask(state.t).astype(np.float64),
            partition_distance=(partition_distance(comms[0].cut, comms[1].cut, self.dendro)
                                if len(comms) == 2 else 0.0),
            n_split=n_split, n_merge=n_merge)
        return EnvState(t=state.t + 1, communities=tuple(comms), key=keys[0]), rec

    def run(self, seed: int | None = None, n_steps: int | None = None) -> dict:
        steps = self.cfg.n_steps if n_steps is None else int(n_steps)
        state = self.reset(seed)
        recs: list[StepRecord] = []
        for _ in range(steps):
            state, rec = self.step(state)
            recs.append(rec)
        return trajectory_to_arrays(recs, state, self)


def trajectory_to_arrays(records: list[StepRecord], state: EnvState,
                         env: LandscapeEnvironment) -> dict:
    n_t, n_c, n_fine = len(records), len(state.communities), env.dendro.n_fine
    cut_size = np.zeros((n_t, n_c), dtype=np.int64)
    recon = np.zeros((n_t, n_c))
    pdist = np.zeros(n_t)
    assign = np.zeros((n_t, n_c, n_fine), dtype=np.int64)
    band = np.zeros((n_t, n_fine))
    n_split = np.zeros((n_t, n_c), dtype=np.int64)
    n_merge = np.zeros((n_t, n_c), dtype=np.int64)
    for ti, rec in enumerate(records):
        cut_size[ti] = rec.cut_sizes
        recon[ti] = rec.recon_error
        pdist[ti] = rec.partition_distance
        assign[ti] = rec.assignments
        band[ti] = rec.band_intensity
        n_split[ti] = rec.n_split
        n_merge[ti] = rec.n_merge
    warm = env.cfg.warmup
    return {
        "cut_size_tc": cut_size,
        "recon_error_tc": recon,
        "partition_distance_t": pdist,
        "assignment_tcf": assign,                       # (n_t, n_c, n_fine) super-node id per node
        "band_intensity_tf": band,                      # (n_t, n_fine) the moving structure
        "n_split_tc": n_split, "n_merge_tc": n_merge,
        "leaf_order": env.order,
        "community_names": np.array([c.name for c in state.communities]),
        "n_fine": np.array(n_fine), "n_atoms": np.array(len(env.dendro.atoms())),
        "lambda2": np.array(env.lambda2),
        "mean_partition_distance": np.array(float(pdist[warm:].mean()) if n_t > warm else 0.0),
    }


def run_connectivity_sweep(inters: tuple[float, ...], *, seed: int = 0,
                           base: LandscapeEnvConfig | None = None) -> dict:
    cfg0 = LandscapeEnvConfig(seed=seed) if base is None else base
    lam, dist = [], []
    for inter in inters:
        tele = LandscapeEnvironment(replace(cfg0, inter=float(inter), seed=seed)).run(seed=seed)
        lam.append(float(tele["lambda2"]))
        dist.append(float(tele["mean_distance"]))
    return {"inters": np.asarray(inters, dtype=np.float64),
            "lambda2": np.asarray(lam, dtype=np.float64),
            "mean_distance": np.asarray(dist, dtype=np.float64)}
