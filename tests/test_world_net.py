"""Tests for the latent generative world (endogenous residuals).

Layer: structural (THE PAPER OBJECT) · Guards: src/structural/bayesnet.py
(LinearGaussianBN.sample), src/structural/world_net.py (true world + recovery),
src/structural/action.py (residual_from_errors).

The claims pinned:
  * LinearGaussianBN.sample draws from exactly the joint the CPDs encode (mean + cov);
  * the true latent world's observations carry the hidden cause's coherent shift + correlation
    among the driven nodes, and BOTH vanish at latent_coupling=0 (the negative control);
  * residual_from_errors localizes a coherent prediction-error shift on the driven nodes;
  * recovery scores the learned net against the true net's driven-node marginal.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from src.structural import action, world_net, phlogiston as ph
from src.structural.bayesnet import LinearGaussianBN
from src.structural.phlogiston import StructuralConfig
from src.structural.world_net import LatentWorldConfig, UNCONCEIVED_DRIVES


def test_lgbn_sample_matches_joint():
    d = 4
    B = np.zeros((d, d)); B[1, 0] = B[2, 0] = B[3, 0] = 0.8   # a,b,c <- root z
    bn = LinearGaussianBN(B=jnp.asarray(B), b=jnp.asarray([1.0, 0.2, 0.2, 0.2]),
                          s=jnp.asarray([1.0, 0.5, 0.5, 0.5]), names=("z", "a", "b", "c"))
    mu, cov = bn.joint()
    X = np.asarray(bn.sample(jax.random.PRNGKey(0), 100_000))
    assert np.allclose(X.mean(0), np.asarray(mu), atol=2e-2)
    assert np.allclose(np.cov(X.T), np.asarray(cov), atol=3e-2)
    # the common cause induces positive cross-correlation among children with zero own coupling
    assert np.cov(X.T)[1, 2] > 0.3


def test_world_carries_latent_footprint_and_null_at_zero():
    cfg = StructuralConfig()
    drives = UNCONCEIVED_DRIVES
    di = [cfg.node_names.index(n) for n in drives]
    meas = ph.measured_nodes(cfg)
    mi = [meas.index(n) for n in drives]                      # drive positions in the obs vector

    def sample_cov_and_mean(coupling):
        lw = LatentWorldConfig(latent_coupling=coupling, latent_mean=1.5)
        world = world_net.true_phlogiston_world(cfg, t=0, lw=lw)
        key = jax.random.PRNGKey(0)
        O = np.stack([np.asarray(world_net.sample_world(world, cfg, k))
                      for k in jax.random.split(key, 4000)])   # (n, m)
        sub = O[:, mi]
        return np.cov(sub.T), sub.mean(0)

    cov1, mean1 = sample_cov_and_mean(1.0)
    cov0, mean0 = sample_cov_and_mean(0.0)
    iu = np.triu_indices(len(drives), 1)
    # WITH the latent: positive cross-correlation among the driven nodes + a coherent shift.
    assert cov1[0, 1] > 0.3 and cov1[0, 2] > 0.3
    assert mean1.mean() > mean0.mean() + 0.5                  # the unconceived shift
    # NULL: latent-only world => no induced cross-correlation among the driven nodes at coupling 0.
    assert np.abs(cov0[iu]).max() < 0.1


def test_residual_from_errors_localizes_coherent_shift():
    # coherent shift on nodes 0,1,2; node 3 untouched -> leading eigvec concentrates on 0,1,2.
    rng = np.random.default_rng(0)
    shift = np.array([1.5, 1.5, 1.5, 0.0])
    errors = shift[None, :] + 0.2 * rng.normal(size=(500, 4))
    R = action.residual_from_errors(jnp.asarray(errors))
    prop = action.propose_hub(None, ("a", "b", "c", "d"), residual=R)
    v = np.abs(np.asarray(prop.pattern))
    assert prop.strength > 0.0
    assert v[0] > 0.4 and v[1] > 0.4 and v[2] > 0.4 and v[3] < 0.15


def test_recovery_zero_when_no_latent():
    cfg = StructuralConfig()
    drives = UNCONCEIVED_DRIVES
    world0 = world_net.true_phlogiston_world(cfg, t=0, lw=LatentWorldConfig(latent_coupling=0.0))
    # the true world's own observed marginal recovers itself perfectly.
    learned = world0.to_info()
    sc = world_net.recovery_scores(learned, world0, drives)
    assert sc["precision_frobenius"] < 1e-4

    # a latent world: a diagonal (node-wise) learned net does NOT hold the induced coupling.
    worldL = world_net.true_phlogiston_world(cfg, t=0, lw=LatentWorldConfig(latent_coupling=1.2))
    diag_pi = jnp.diag(jnp.diagonal(worldL.to_info().Pi)[:len(cfg.node_names)])
    from src.structural.belief import GaussianBeliefNet
    diag_learned = GaussianBeliefNet(Pi=diag_pi, h=jnp.zeros(len(cfg.node_names)),
                                     names=cfg.node_names)
    sc_gap = world_net.recovery_scores(diag_learned, worldL, drives)
    assert sc_gap["precision_frobenius"] > sc["precision_frobenius"]   # gap to recover
