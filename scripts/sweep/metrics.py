"""The fixed metric battery -- computed identically for every run, on the host, post-hoc.

Reuses existing readouts (closest_theory, residual_disagreement, settling_time, time_to_half) and
adds the structural glue: per-community wiring distance, structure-vs-truth match, theory
diversity, and a precision-health/divergence check (the last catches the forgetting=1.0 blow-up
automatically).

``compute_metrics(r, ctx) -> (scalars, traj)``:
  r    -- the dict returned by run_simulation (snap_Pi (S,N,d,d), snap_h (S,N,d), m_t, epoch_t...).
  ctx  -- a dict with community_idx, edges_ij, theory_mus, true_couplings, n_communities, world_mode.
  scalars -- flat dict of floats/ints/bools (one tidy row).
  traj    -- a few (S,) trajectories saved to the per-run npz (not full snap_Pi).
"""

from __future__ import annotations

import numpy as np

from src.structural.scenarios import closest_theory
from src.structural import shells, observables as obs

HEALTH_MAXPI = 1.0e3        # |Pi| above this => flagged diverged (the forgetting=1.0 pathology)


def _means(Pi, h):
    """Posterior means for a stack of nets: (N,d,d),(N,d) -> (N,d). Guards singular Pi."""
    try:
        return np.linalg.solve(Pi, h[..., None])[..., 0]
    except np.linalg.LinAlgError:
        return np.stack([np.linalg.lstsq(Pi[i], h[i], rcond=None)[0] for i in range(Pi.shape[0])])


def _struct_vec(Pi_mean, edges_ij):
    return np.array([Pi_mean[i, j] for (i, j) in edges_ij])


def _reldist(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / (0.5 * (na + nb) + 1e-9))


def _comm_struct_vecs(Pi_snap, community_idx, edges_ij):
    return [_struct_vec(Pi_snap[ci].mean(0), edges_ij) for ci in community_idx]


def _pairwise_comm_dist(vecs):
    if len(vecs) < 2:
        return 0.0
    ds = [_reldist(vecs[a], vecs[b])
          for a in range(len(vecs)) for b in range(a + 1, len(vecs))]
    return float(np.mean(ds))


def _comm_favored_theory(Pi_snap, h_snap, community_idx, theory_mus):
    """The theory each community's mean belief sits closest to -> (n_comm,) int."""
    out = []
    for ci in community_idx:
        m = _means(Pi_snap[ci], h_snap[ci]).mean(0)           # (d,)
        out.append(int(closest_theory(m[None, :], theory_mus)[0]))
    return np.array(out)


def compute_metrics(r, ctx):
    snap_Pi = np.asarray(r["snap_Pi"])                         # (S,N,d,d)
    snap_h = np.asarray(r["snap_h"])                           # (S,N,d)
    epoch_t = np.asarray(r["epoch_t"])
    community_idx = ctx["community_idx"]
    edges_ij = ctx["edges_ij"]
    theory_mus = ctx["theory_mus"]                             # (E,d)
    true_couplings = ctx["true_couplings"]                     # (E,n_edges)
    S = snap_Pi.shape[0]

    # ---- health (catches divergence) ----
    finite = bool(np.isfinite(snap_Pi).all() and np.isfinite(snap_h).all())
    max_pi = float(np.abs(snap_Pi).max()) if finite else float("inf")
    diverged = (not finite) or (max_pi > HEALTH_MAXPI)

    # ---- structural divergence between communities (the headline metric) ----
    struct_dist_t = np.array([_pairwise_comm_dist(_comm_struct_vecs(snap_Pi[s], community_idx, edges_ij))
                              for s in range(S)])
    final_struct_dist = float(struct_dist_t[-1])

    # ---- theory tracking / diversity / lock-in ----
    comm_div_t, agent_div_t = [], []
    for s in range(S):
        fav = _comm_favored_theory(snap_Pi[s], snap_h[s], community_idx, theory_mus)
        comm_div_t.append(len(np.unique(fav)))
        means_all = _means(snap_Pi[s], snap_h[s])
        agent_div_t.append(len(np.unique(closest_theory(means_all, theory_mus))))
    comm_div_t = np.array(comm_div_t); agent_div_t = np.array(agent_div_t)

    fav_final = _comm_favored_theory(snap_Pi[-1], snap_h[-1], community_idx, theory_mus)
    final_epoch = int(epoch_t[-1])
    n_tracking = int(np.sum(fav_final == final_epoch))          # communities on the current-true theory
    n_comm = len(community_idx)
    lockin_frac = float(1.0 - n_tracking / n_comm)              # fraction NOT tracking current truth

    # ---- convergence to the true structure (current epoch) ----
    true_now = true_couplings[final_epoch]                      # (n_edges,)
    comm_vecs = _comm_struct_vecs(snap_Pi[-1], community_idx, edges_ij)
    final_truth_dist = float(np.mean([_reldist(v, true_now) for v in comm_vecs]))

    # ---- dynamics + means-level order parameter (already on r) ----
    m_t = np.asarray(r["m_t"])
    res_dis = np.asarray(shells.residual_disagreement(snap_Pi))  # (S,)
    try:
        t_settle = int(obs.settling_time(struct_dist_t, struct_dist_t[-1], 0.05, 3))
    except Exception:
        t_settle = S
    try:
        t_half = int(obs.time_to_half(struct_dist_t / (struct_dist_t.max() + 1e-9), 0.5))
    except Exception:
        t_half = S

    scalars = dict(
        lambda2=float(r.get("lambda2", 0.0)),
        max_pi=max_pi, diverged=bool(diverged), finite=finite,
        final_struct_dist=final_struct_dist,
        final_comm_diversity=int(comm_div_t[-1]),
        final_agent_diversity=int(agent_div_t[-1]),
        n_tracking=n_tracking, lockin_frac=lockin_frac, final_epoch=final_epoch,
        favored_theories="".join(str(x) for x in fav_final),
        final_truth_dist=final_truth_dist,
        final_residual_disagreement=float(res_dis[-1]),
        m_final=float(m_t[-1]),
        settle_step=t_settle, half_step=t_half,
    )
    traj = dict(
        struct_dist_t=struct_dist_t.astype(np.float32),
        comm_diversity_t=comm_div_t.astype(np.int16),
        m_t=m_t.astype(np.float32),
        residual_disagreement_t=res_dis.astype(np.float32),
        snap_t=np.asarray(r["snap_t"]),
    )
    return scalars, traj
