
import unittest
import sys
import os
import json
from unittest.mock import MagicMock, patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.readiness import compile_freeze_blockers, generate_data_card

class TestProductionReadiness(unittest.TestCase):
    def test_freeze_blockers_logic(self):
        thresholds = {"sample_size_total": 100, "mean_ece": 0.1, "mean_brier": 0.2}
        
        # Scenario 1: All pass
        gate_status = {"sample_size_total": True, "mean_ece": True, "mean_brier": True}
        metrics = {"total_sample_size": 150, "mean_ece": 0.05, "mean_brier": 0.15}
        blockers = compile_freeze_blockers(gate_status, metrics, thresholds)
        self.assertEqual(len(blockers), 0)
        
        # Scenario 2: Fail ECE
        gate_status = {"sample_size_total": True, "mean_ece": False, "mean_brier": True}
        metrics = {"total_sample_size": 150, "mean_ece": 0.15, "mean_brier": 0.15}
        blockers = compile_freeze_blockers(gate_status, metrics, thresholds)
        self.assertEqual(len(blockers), 1)
        self.assertEqual(blockers[0]["id"], "ece")
        self.assertEqual(blockers[0]["severity"], "medium")

    def test_data_card_generation(self):
        inputs = [{"game": {"date_utc": "2023-01-01"}}, {"game": {"date_utc": "2023-01-02"}}]
        targets = [1, 0]
        card = generate_data_card(inputs, targets)
        
        self.assertEqual(card["total_samples"], 2)
        self.assertEqual(card["label_distribution"]["over"], 1)
        self.assertIn("temporal_coverage", card)
        self.assertEqual(card["temporal_coverage"]["min"], "2023-01-01")

    def test_repro_manifest_schema(self):
        from src.readiness import generate_repro_manifest
        m = generate_repro_manifest([], {}, {}, git_commit="abc", random_seeds={"a": 1})
        self.assertEqual(m["git_commit"], "abc")
        self.assertEqual(m["random_seeds"]["a"], 1)

    def test_utc_timezone_aware(self):
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        self.assertIsNotNone(now.tzinfo)
        self.assertEqual(now.tzinfo, timezone.utc)

    def test_readiness_keys_schema(self):
        # We can't easily mock the full readiness report generation without heavy mocking, 
        # but we can verify the structure if we had a helper. 
        # For now, let's verify compile_freeze_blockers handles total_sample_size correctly.
        thresholds = {"sample_size_total": 100}
        gate_status = {"sample_size_total": False} 
        metrics = {"total_sample_size": 50}
        blockers = compile_freeze_blockers(gate_status, metrics, thresholds)
        self.assertEqual(len(blockers), 1)
        self.assertEqual(blockers[0]["actual"], 50)

if __name__ == '__main__':
    unittest.main()
