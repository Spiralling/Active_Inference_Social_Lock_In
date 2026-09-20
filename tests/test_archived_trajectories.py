"""Short regressions against real pre-existing production chunks.

Fixtures preserve entering states and random keys and were selected before the
new implementation's results: first seed, twelve rounds of each named chunk.
The tolerances accommodate harmless batched-linear-algebra roundoff; observation
and entering-key comparisons remain exact. No expected curve ordering is tested.
"""
from pathlib import Path
import hashlib
import json
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from iwai_figures import communication, gaussian, simulation


class ArchivedTrajectories(unittest.TestCase):
    def test_preselected_production_chunks(self):
        fixtures = ROOT / "reference" / "simulation_fixtures"
        records = json.loads((fixtures / "manifest.json").read_text())
        for record in records:
            with self.subTest(condition=record["condition"]):
                path = fixtures / record["file"]
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record["sha256"])
                with np.load(path, allow_pickle=False) as saved:
                    expected = {key: saved[key] for key in saved.files}
                module = gaussian if record["family"] == "gaussian" else communication
                cfg, W, A, attention, phi, metadata = module.make_setup(record["condition"])
                initial = {key: expected["initial_" + key] for key in simulation.STATE_KEYS}
                phis = np.broadcast_to(np.asarray(phi), (record["steps"], len(phi))).copy()
                if record["family"] == "gaussian":
                    execute = gaussian.make_batch(cfg, W, A, attention, gate_nu=metadata["gate_nu"])
                else:
                    execute = simulation.make_batch(cfg, W, A, attention)
                actual = execute(phis, [record["seed"]], initial=initial)
                for key, value in expected.items():
                    if key == "obs" or key.endswith("_key"):
                        np.testing.assert_array_equal(actual[key], value, err_msg=record["condition"] + ": " + key)
                    else:
                        np.testing.assert_allclose(actual[key], value, rtol=5e-12, atol=1e-12,
                                                   err_msg=record["condition"] + ": " + key)


if __name__ == "__main__":
    unittest.main(verbosity=2)
