"""Check the combustion-mediated candidate menu against independent Gaussian math.

Run in a fresh interpreter to select the frozen publication source explicitly:
    .venv-reproduction/bin/python -m unittest discover -s tests -p test_combustion_experiments.py -v
"""

import importlib.util
import os
from pathlib import Path
import sys
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import multivariate_normal


# Set precision before importing the recovered engine: it computes LOG_2PI at
# import time, so a later switch would retain a float32 normalization constant.
jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parents[1]
RECOVERED = ROOT / "paper/four_variable_experiments/engine"
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "paper/combustion_experiments/.matplotlib-cache"))
sys.path.insert(0, str(RECOVERED))
SPEC = importlib.util.spec_from_file_location(
    "combustion_world", ROOT / "paper/combustion_experiments/world.py")
WORLD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = WORLD
SPEC.loader.exec_module(WORLD)
RUN_SPEC = importlib.util.spec_from_file_location(
    "combustion_run", ROOT / "paper/combustion_experiments/run.py")
RUN = importlib.util.module_from_spec(RUN_SPEC)
RUN_SPEC.loader.exec_module(RUN)

from src.structural import rival as rv
from src.structural.world import sample_o


class CombustionExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = WORLD.build_world()

    def test_both_explanatory_entities_have_the_proposed_signed_edges(self):
        w = self.world
        B = np.asarray(w["B"])
        expected = ({(0, 1): .8, (2, 1): .8, (3, 0): -.8},
                    {(0, 1): .8, (2, 0): .8, (3, 0): .8})
        for k in range(2):
            actual = {tuple(idx): B[k][tuple(idx)] for idx in np.argwhere(B[k] != 0)}
            self.assertEqual(actual, expected[k])
            # Storage (z,C,G,M) is intentionally not topological.
            order = [1, 0, 2, 3]
            topological = B[k][np.ix_(order, order)]
            assert_allclose(np.triu(topological), 0, rtol=0, atol=0)
            assert_allclose(np.linalg.matrix_power(B[k], 4), 0, rtol=0, atol=0)
        self.assertEqual(w["candidate_node_names"][0][0], "P")
        self.assertEqual(w["candidate_node_names"][1][0], "O")
        assert_allclose(w["s"], [[.25, .5, .5, .5]] * 2, atol=0, rtol=0)

    def test_compiled_gaussians_have_intended_joint_moments_and_are_positive_definite(self):
        w = self.world
        for k in range(2):
            B, b, s = (np.asarray(w[name])[k] for name in ("B", "b", "s"))
            P, h = np.asarray(w["Pi"])[k], np.asarray(w["h"])[k]
            A = np.eye(4) - B
            # Independent ancestral moment derivation, without to_info/joint.
            T = np.linalg.solve(A, np.eye(4))
            covariance = T @ np.diag(s) @ T.T
            assert_allclose(np.linalg.solve(P, h), w["mu"][k], atol=2e-14)
            assert_allclose(T @ b, w["mu"][k], atol=2e-14)
            assert_allclose(np.linalg.inv(P), covariance, atol=2e-14)
            self.assertGreater(np.linalg.eigvalsh(P).min(), 0)
            self.assertGreater(np.linalg.eigvalsh(covariance).min(), 0)
        # Shared residual variances do not make observable covariances equal.
        self.assertGreater(np.linalg.norm(np.asarray(w["covariance"])[0, 1:, 1:]
                                         - np.asarray(w["covariance"])[1, 1:, 1:]), .1)

    def test_gas_mass_dependence_has_the_proposed_conditional_meaning(self):
        # For Gaussian variables, this Schur complement is Cov((G,M) | C).
        # It concerns the initial latent state; noisy C observations do not
        # generally preserve the same exact conditional independence.
        cov = np.asarray(self.world["covariance"])
        residual = []
        for k in range(2):
            gm = cov[k][np.ix_([2, 3], [2, 3])]
            gc = cov[k][np.ix_([2, 3], [1])]
            residual.append(gm - gc @ gc.T / cov[k, 1, 1])
        self.assertAlmostEqual(residual[0][0, 1], 0., places=14)
        self.assertAlmostEqual(residual[1][0, 1], .8 * .8 * .25, places=14)
        # Shared G means do not imply identical G predictive distributions.
        assert_allclose(self.world["mu"][:, 2], [1., 1.], rtol=0, atol=0)
        assert_allclose(cov[:, 2, 2], [.82, .8648], rtol=0, atol=2e-14)

    def test_non_topological_storage_compiles_equivalently_after_permutation(self):
        from src.structural.bayesnet import LinearGaussianBN
        order = np.array([1, 0, 2, 3])
        w = self.world
        for k in range(2):
            B = np.asarray(w["B"])[k]
            b = np.asarray(w["b"])[k]
            s = np.asarray(w["s"])[k]
            bn = LinearGaussianBN(
                B=jnp.asarray(B[np.ix_(order, order)]),
                b=jnp.asarray(b[order]), s=jnp.asarray(s[order]),
                names=tuple(w["candidate_node_names"][k][i] for i in order))
            net = bn.to_info()
            assert_allclose(net.Pi, np.asarray(w["Pi"])[k][np.ix_(order, order)],
                            rtol=0, atol=2e-14)
            assert_allclose(net.h, np.asarray(w["h"])[k][order], rtol=0, atol=2e-14)

    def test_observations_and_cross_frame_readouts_exclude_the_model_specific_core(self):
        w = self.world
        H = np.asarray(w["H"])
        assert_allclose(H, np.eye(4)[1:], atol=0, rtol=0)
        self.assertEqual(w["gate_idx"], (1, 2, 3))
        self.assertEqual(w["observable_idx"], (1, 2, 3))
        assert_allclose(H @ WORLD.phi(0), [1, 1, -1], atol=0, rtol=0)
        assert_allclose(H @ WORLD.phi(80), [1, 1, 1], atol=0, rtol=0)
        assert_allclose(H @ WORLD.wall_truth(), [1, 1, 1], atol=0, rtol=0)
        self.assertEqual(WORLD.DIAGNOSTIC_ROWS, (1, 2))
        # The zero-preservation mechanism is explicit for this sensor map.
        J = H.T @ np.diag([.3, 1., .01]) @ H
        assert_allclose(J, np.diag(np.diag(J)), atol=0, rtol=0)

    def test_predictive_score_matches_independent_multivariate_normal_density(self):
        w = self.world
        H = np.asarray(w["H"])
        observation = np.array([.7, -.3, 1.2])
        for sigma in (.5, 1.0):
            for attention in (np.ones(3), np.array([.4, 0., .01]), np.zeros(3)):
                sqrtw = np.sqrt(attention)
                weighted_H = sqrtw[:, None] * H
                for k in range(2):
                    P, h = np.asarray(w["Pi"])[k], np.asarray(w["h"])[k]
                    mu = np.linalg.solve(P, h)
                    cov = sigma**2 * np.eye(3) + weighted_H @ np.linalg.inv(P) @ weighted_H.T
                    expected = multivariate_normal.logpdf(sqrtw * observation,
                                                         weighted_H @ mu, cov)
                    actual = float(rv.log_predictive_weighted(
                        jnp.asarray(P), jnp.asarray(h), w["H"],
                        jnp.asarray(observation), sigma, jnp.asarray(attention)))
                    self.assertAlmostEqual(actual, expected, places=12)
        scores = rv.score_candidates(w["Pi"], w["h"], w["H"],
                                      jnp.asarray(observation), 1., jnp.zeros(3))
        self.assertEqual(float(scores[0]), float(scores[1]))

    def test_uniform_initial_weights_do_not_seed_a_preferred_candidate(self):
        w = self.world
        cfg = rv.RivalConfig(Pi0=w["Pi"], h0=w["h"], H=w["H"], sigma_o=1.)
        P, h, L = rv.init_uniform(cfg, 6)
        assert_allclose(P, np.broadcast_to(w["Pi"], (6, 2, 4, 4)), atol=0, rtol=0)
        assert_allclose(h, np.broadcast_to(w["h"], (6, 2, 4)), atol=0, rtol=0)
        assert_allclose(L, np.zeros((6, 2)), atol=0, rtol=0)
        assert_allclose(rv.model_posterior(L), .5, atol=0, rtol=0)

    def test_single_candidate_engine_equals_independent_fuse_observe_forget_recursion(self):
        w = self.world
        n, steps, sigma, omega = 3, 7, .7, .9
        # Use the observable marginal as the single-frame baseline. It avoids
        # averaging two model-specific explanatory quantities into a false core.
        covariance = np.asarray(w["covariance"])[0, 1:, 1:]
        P0 = np.linalg.inv(covariance)
        h0 = P0 @ np.asarray(w["mu"])[0, 1:]
        H = np.eye(3)
        cfg = rv.RivalConfig(Pi0=jnp.asarray(P0[None]), h0=jnp.asarray(h0[None]),
                             H=jnp.asarray(H), sigma_o=sigma, omega=omega,
                             alpha_m=.41)
        W = np.array([[.6, .3, .1], [.2, .5, .3], [.1, .3, .6]])
        attention = np.array([[1., .1, 0.], [.2, .9, .3], [0., .4, 1.]])
        P, h, L = rv.init_uniform(cfg, n)
        carry = (P, h, L, jax.random.PRNGKey(13))
        expected_P = np.broadcast_to(P0, (n, 3, 3)).copy()
        expected_h = np.broadcast_to(h0, (n, 3)).copy()
        expected_key = jax.random.PRNGKey(13)
        for t in range(steps):
            phi = jnp.asarray(WORLD.phi(t, t_shift=3)[1:])
            expected_P = np.einsum("ij,jab->iab", W, expected_P)
            expected_h = W @ expected_h
            expected_key, subkey = jax.random.split(expected_key)
            keys = jax.random.split(subkey, n)
            for i in range(n):
                observation = np.asarray(sample_o(jnp.asarray(H), phi, sigma, keys[i]))
                J = H.T @ np.diag(attention[i]) @ H / sigma**2
                j = H.T @ (attention[i] * observation) / sigma**2
                expected_P[i] += J
                expected_h[i] += j
            expected_P = P0 + omega * (expected_P - P0)
            expected_h = h0 + omega * (expected_h - h0)
            carry, readout = rv._step(carry, phi, cfg, jnp.asarray(W), jnp.ones((n, n)),
                                      jnp.asarray(attention), track_means=True)
            assert_allclose(carry[0][:, 0], expected_P, atol=2e-13, rtol=2e-13)
            assert_allclose(carry[1][:, 0], expected_h, atol=2e-13, rtol=2e-13)
            assert_allclose(rv.model_posterior(carry[2]), 1., atol=0, rtol=0)
            assert_allclose(readout[1], np.linalg.solve(expected_P, expected_h[..., None])[..., 0],
                            atol=2e-13, rtol=2e-13)
            assert_allclose(readout[2], 0., atol=0, rtol=0)

    def test_wall_attention_has_distinct_evidence_without_observing_the_hidden_core(self):
        attention = WORLD.wall_attention(4)
        assert_allclose(attention, [[0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]],
                        atol=0, rtol=0)
        for n in (1, 3, 0, -2, 4.5):
            with self.assertRaises(ValueError):
                WORLD.wall_attention(n)

    def test_first_crossings_distinguish_time_zero_from_censoring(self):
        q = np.array([[.95, .2, .2, .2], [.2, .9, .9, .2]])
        actual = RUN.crossings(q, .9)
        self.assertEqual(actual["times"], [0, -1])
        self.assertEqual(actual["events"], 1)
        self.assertEqual(actual["censored"], 1)
        self.assertEqual(actual["median_among_events"], 0.)

    def test_first_crossings_exclude_preintervention_events_for_each_agent(self):
        # Each seed/agent is independently censored. Values at the strict
        # threshold are not successes, and input probabilities stay untouched.
        q = np.array([[[.99, .1], [.99, .1], [.1, .9], [.1, .91]],
                      [[.1, .99], [.1, .99], [.95, .1], [.96, .1]]])
        original = q.copy()
        actual = RUN.crossings(q, .9, start=2)
        self.assertEqual(actual["times"], [[-1, 3], [2, -1]])
        self.assertEqual(actual["events"], 2)
        self.assertEqual(actual["total"], 4)
        self.assertEqual(actual["censored"], 2)
        self.assertEqual(actual["median_among_events"], 2.5)
        assert_allclose(q, original, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
