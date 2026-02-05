
import unittest
import sys
import os
from unittest.mock import MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.main import get_gate_thresholds

class TestGateProfiles(unittest.TestCase):
    def test_get_gate_thresholds_dev(self):
        t = get_gate_thresholds("dev")
        self.assertEqual(t["sample_size_total"], 80)
        self.assertEqual(t["mean_ece"], 0.12)
        self.assertEqual(t["mean_brier"], 0.26)

    def test_get_gate_thresholds_freeze(self):
        t = get_gate_thresholds("freeze")
        self.assertEqual(t["sample_size_total"], 300)
        self.assertEqual(t["mean_ece"], 0.08)
        self.assertEqual(t["mean_brier"], 0.20)

    def test_get_gate_thresholds_unknown(self):
        with self.assertRaises(ValueError):
            get_gate_thresholds("unknown_profile")

    def test_dev_looser_than_freeze(self):
        dev = get_gate_thresholds("dev")
        freeze = get_gate_thresholds("freeze")
        self.assertLess(dev["sample_size_total"], freeze["sample_size_total"])
        self.assertGreater(dev["mean_ece"], freeze["mean_ece"])
        self.assertGreater(dev["mean_brier"], freeze["mean_brier"])
            
if __name__ == '__main__':
    unittest.main()
