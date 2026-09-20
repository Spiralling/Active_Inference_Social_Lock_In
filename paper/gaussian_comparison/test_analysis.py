#!/usr/bin/env python3
"""Boundary cases for nonlinear distances, paired units and inclusive windows."""
import json
import unittest
import numpy as np
from analyze import bootstrap_indices, estimate, seed_window_metrics, vector_norm_estimate


class DistanceDefinitionTests(unittest.TestCase):
    def test_temporal_direction_reversal_is_not_persistent_separation(self):
        difference = np.zeros((2, 4, 3))
        difference[:, :, 0] = [1, -1, 1, -1]
        metrics = seed_window_metrics({'mixture_difference': difference,
                                       'mixture_distance': np.linalg.norm(difference, axis=-1)}, 1, 4)
        np.testing.assert_array_equal(metrics['mixture_distance_tail'], np.ones(2))
        np.testing.assert_array_equal(metrics['mixture_distance_of_time_means_tail'], np.zeros(2))

    def test_across_seed_direction_reversal_is_not_ensemble_direction(self):
        difference = np.zeros((2, 4, 3))
        difference[0, :, 0] = 2
        difference[1, :, 0] = -2
        metrics = seed_window_metrics({'mixture_difference': difference}, 1, 4)
        np.testing.assert_array_equal(metrics['mixture_distance_of_time_means_tail'], [2, 2])
        result = vector_norm_estimate(difference.mean(1), bootstrap_indices(2, 20, 1))
        self.assertEqual(result['mean'], 0.)
        self.assertIsNone(result['lo'])
        self.assertIsNone(result['hi'])

    def test_reporting_windows_use_inclusive_completed_states(self):
        values = np.tile(np.arange(1, 5001), (2, 1))
        for first, last in [(1126, 1500), (3751, 5000)]:
            metrics = seed_window_metrics({'mse_A': values}, first, last)
            np.testing.assert_array_equal(metrics['mse_A_tail'], [(first + last) / 2] * 2)
            np.testing.assert_array_equal(metrics['mse_A_endpoint'], [last] * 2)

    def test_pairing_is_applied_before_resampling(self):
        idx = bootstrap_indices(50, 2000, 20260920)
        x = np.linspace(-100, 100, 50)
        result = estimate((x + 3) - x, idx)
        self.assertEqual(result['mean'], 3.)
        self.assertEqual(result['lo'], 3.)
        self.assertEqual(result['hi'], 3.)

    def test_nonfinite_or_incomplete_seed_data_fail(self):
        idx = bootstrap_indices(50, 20, 1)
        with self.assertRaises(ValueError):
            estimate(np.arange(49), idx)
        with self.assertRaises(ValueError):
            estimate(np.r_[np.arange(49), np.nan], idx)

    def test_public_distance_excludes_candidate_specific_coordinates(self):
        # Input is explicitly three common observable coordinates. A four-vector
        # that accidentally includes the P/O coordinate must be rejected.
        idx = bootstrap_indices(50, 20, 1)
        with self.assertRaises(AssertionError):
            vector_norm_estimate(np.zeros((50, 4)), idx)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DistanceDefinitionTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({'status': 'passed' if result.wasSuccessful() else 'failed', 'tests': result.testsRun}))
    raise SystemExit(not result.wasSuccessful())
