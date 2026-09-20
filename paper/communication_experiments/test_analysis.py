"""Boundary tests for analysis events and whole-seed pairing, not model tests."""
import unittest
import numpy as np
from analyze import first_event, intervals, bootstrap_indices, estimate, wilson


class AnalysisBoundaries(unittest.TestCase):
    def test_sustained_start_detection_and_right_censor(self):
        a = np.r_[np.zeros(1500, bool), np.ones(19, bool)]
        self.assertIsNone(first_event(a, 1501, 20)["event_state"])
        b = np.r_[a, True]
        event = first_event(b, 1501, 20)
        self.assertEqual((event["event_state"], event["event_round"], event["detection_state"]), (1501, 1500, 1520))

    def test_no_crossing_is_not_horizon(self):
        event = first_event(np.zeros(600, bool), 81)
        self.assertFalse(event["event_observed"])
        self.assertIsNone(event["event_state"])
        self.assertEqual(event["censor_state"], 600)

    def test_search_does_not_inherit_prebranch_run(self):
        event = first_event(np.ones(1519, bool), 1501, 20)
        self.assertFalse(event["event_observed"])

    def test_intervals_retain_return_and_final_censor(self):
        iv = intervals([False, True, True, False, True], first_state=1126)
        self.assertEqual([(v["start_state"], v["end_state"], v["length"]) for v in iv], [(1127, 1128, 2), (1130, 1130, 1)])
        self.assertFalse(iv[0]["right_censored"])
        self.assertTrue(iv[1]["right_censored"])

    def test_paired_constant_difference_has_zero_interval_width(self):
        x = np.arange(50.)**2
        y = x + 3
        ci = estimate(y-x, bootstrap_indices(50))
        self.assertEqual((ci["mean"], ci["lo"], ci["hi"]), (3., 3., 3.))

    def test_wilson_does_not_assert_certainty_at_fifty_successes(self):
        ci = wilson(50, 50)
        self.assertGreater(ci["lo"], .92)
        self.assertLess(ci["lo"], 1)
        self.assertAlmostEqual(ci["hi"], 1)


if __name__ == "__main__":
    unittest.main()
