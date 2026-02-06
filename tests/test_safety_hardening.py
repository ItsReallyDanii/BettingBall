"""
Tests for V1 safety hardening layer (src/safety.py).

Covers:
- test_confidence_grades
- test_prototype_only_threshold_trigger
- test_leakage_block_release
- test_all_factors_have_risk
- test_missing_data_flags_emitted
"""
import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.safety import (
    grade_confidence,
    classify_factor_risk,
    run_leakage_checks,
    run_generalization_gate,
    build_prediction_output,
    GRADE_MAP,
    RISK_MAP,
    TARGET_LEAK_COLUMNS,
)


class TestConfidenceGrades(unittest.TestCase):
    """Deterministic confidence grading from grade_map."""

    def test_grade_A(self):
        self.assertEqual(grade_confidence(0.70), "A")
        self.assertEqual(grade_confidence(0.85), "A")
        self.assertEqual(grade_confidence(1.0), "A")

    def test_grade_B(self):
        self.assertEqual(grade_confidence(0.60), "B")
        self.assertEqual(grade_confidence(0.69), "B")

    def test_grade_C(self):
        self.assertEqual(grade_confidence(0.55), "C")
        self.assertEqual(grade_confidence(0.59), "C")

    def test_grade_D(self):
        self.assertEqual(grade_confidence(0.50), "D")
        self.assertEqual(grade_confidence(0.54), "D")

    def test_grade_F(self):
        self.assertEqual(grade_confidence(0.49), "F")
        self.assertEqual(grade_confidence(0.0), "F")

    def test_boundary_values(self):
        """Exact boundary values map to the higher grade."""
        self.assertEqual(grade_confidence(GRADE_MAP["A_min"]), "A")
        self.assertEqual(grade_confidence(GRADE_MAP["B_min"]), "B")
        self.assertEqual(grade_confidence(GRADE_MAP["C_min"]), "C")
        self.assertEqual(grade_confidence(GRADE_MAP["D_min"]), "D")

    def test_deterministic(self):
        """Same input always produces same output."""
        for _ in range(10):
            self.assertEqual(grade_confidence(0.63), "B")


class TestPrototypeOnlyThresholdTrigger(unittest.TestCase):
    """Generalization gate forces prototype_only when below thresholds."""

    def test_all_below(self):
        result = run_generalization_gate(n_games=50, n_teams_covered=5, day_span=10)
        self.assertEqual(result["status"], "prototype_only")
        self.assertEqual(len(result["threshold_failures"]), 3)
        self.assertIn("production-ready edge", result["forbidden_claims"])
        self.assertIn("stable market alpha", result["forbidden_claims"])

    def test_games_below(self):
        result = run_generalization_gate(n_games=500, n_teams_covered=25, day_span=90)
        self.assertEqual(result["status"], "prototype_only")
        self.assertIn("min_games_required", result["threshold_failures"])

    def test_teams_below(self):
        result = run_generalization_gate(n_games=1500, n_teams_covered=10, day_span=90)
        self.assertEqual(result["status"], "prototype_only")
        self.assertIn("min_teams_covered", result["threshold_failures"])

    def test_days_below(self):
        result = run_generalization_gate(n_games=1500, n_teams_covered=25, day_span=30)
        self.assertEqual(result["status"], "prototype_only")
        self.assertIn("min_days_span", result["threshold_failures"])

    def test_all_above(self):
        result = run_generalization_gate(n_games=1500, n_teams_covered=25, day_span=90)
        self.assertEqual(result["status"], "production")
        self.assertEqual(len(result["threshold_failures"]), 0)
        self.assertEqual(result["forbidden_claims"], [])

    def test_prototype_status_in_output_contract(self):
        """build_prediction_output reflects prototype_only from gate."""
        gate = run_generalization_gate(n_games=50, n_teams_covered=5, day_span=10)
        output = build_prediction_output(
            hypothesis_id="H1",
            market_type="player_prop",
            prediction="over",
            confidence=0.65,
            reasoning_short="Test reasoning",
            factors=[{"name": "boxscore", "value": 1.0, "source": "local"}],
            generalization_gate=gate,
        )
        self.assertEqual(output["status"], "prototype_only")


class TestLeakageBlockRelease(unittest.TestCase):
    """Leakage gate blocks release claims on failure."""

    def test_all_pass(self):
        result = run_leakage_checks(
            feature_columns=["pace", "rest_days"],
            train_timestamps=["2023-01-01", "2023-02-01"],
            test_timestamps=["2023-03-01", "2023-04-01"],
        )
        self.assertTrue(result["all_passed"])
        self.assertEqual(result["release_status"], "ok")

    def test_target_leak_columns_detected(self):
        result = run_leakage_checks(
            feature_columns=["pace", "final_score", "rest_days"],
            train_timestamps=["2023-01-01"],
            test_timestamps=["2023-06-01"],
        )
        self.assertFalse(result["all_passed"])
        self.assertEqual(result["release_status"], "blocked_leakage_risk")
        col_check = next(c for c in result["checks"] if c["name"] == "no_target_leak_columns")
        self.assertFalse(col_check["passed"])

    def test_temporal_violation(self):
        result = run_leakage_checks(
            feature_columns=["pace"],
            train_timestamps=["2023-01-01", "2023-06-01"],
            test_timestamps=["2023-03-01"],
        )
        self.assertFalse(result["all_passed"])
        self.assertEqual(result["release_status"], "blocked_leakage_risk")

    def test_scaler_leak(self):
        result = run_leakage_checks(
            feature_columns=["pace"],
            train_timestamps=["2023-01-01"],
            test_timestamps=["2023-06-01"],
            scaler_fit_on_train_only=False,
        )
        self.assertFalse(result["all_passed"])
        self.assertEqual(result["release_status"], "blocked_leakage_risk")

    def test_calibration_leak(self):
        result = run_leakage_checks(
            feature_columns=["pace"],
            train_timestamps=["2023-01-01"],
            test_timestamps=["2023-06-01"],
            calibration_fit_on_validation_only=False,
        )
        self.assertFalse(result["all_passed"])
        self.assertEqual(result["release_status"], "blocked_leakage_risk")

    def test_blocked_release_in_output_contract(self):
        """build_prediction_output reflects blocked_leakage_risk."""
        leak_report = run_leakage_checks(
            feature_columns=["final_score"],
            train_timestamps=["2023-01-01"],
            test_timestamps=["2023-06-01"],
        )
        output = build_prediction_output(
            hypothesis_id="H2",
            market_type="spread",
            prediction="cover",
            confidence=0.72,
            reasoning_short="Test",
            factors=[{"name": "closing_odds", "value": 0.5, "source": "local"}],
            leakage_report=leak_report,
        )
        self.assertEqual(output["release_status"], "blocked_leakage_risk")


class TestAllFactorsHaveRisk(unittest.TestCase):
    """Every factor in prediction output must include a risk tag."""

    def test_factors_with_explicit_risk(self):
        factors = [
            {"name": "boxscore", "value": 1.0, "risk": "low", "source": "local"},
            {"name": "sentiment", "value": 0.5, "risk": "high", "source": "manual"},
        ]
        output = build_prediction_output(
            hypothesis_id="H3",
            market_type="team_total",
            prediction="over",
            confidence=0.60,
            reasoning_short="Explicit risk tags",
            factors=factors,
        )
        for f in output["factors"]:
            self.assertIn("risk", f)
            self.assertIn(f["risk"], ("low", "medium", "high"))

    def test_factors_without_risk_get_auto_tagged(self):
        factors = [
            {"name": "boxscore", "value": 1.0, "source": "local"},
            {"name": "news", "value": 0.8, "source": "feed"},
            {"name": "unknown_factor", "value": 0.3, "source": "derived"},
        ]
        output = build_prediction_output(
            hypothesis_id="H4",
            market_type="player_prop",
            prediction="over",
            confidence=0.55,
            reasoning_short="Auto risk tagging",
            factors=factors,
        )
        for f in output["factors"]:
            self.assertIn("risk", f)
            self.assertTrue(len(f["risk"]) > 0)

        # Check specific assignments
        by_name = {f["name"]: f for f in output["factors"]}
        self.assertEqual(by_name["boxscore"]["risk"], "low")
        self.assertEqual(by_name["news"]["risk"], "high")
        self.assertEqual(by_name["unknown_factor"]["risk"], "medium")  # default

    def test_empty_risk_string_replaced(self):
        factors = [{"name": "home_away", "value": 1.0, "risk": "", "source": "local"}]
        output = build_prediction_output(
            hypothesis_id="H5",
            market_type="spread",
            prediction="cover",
            confidence=0.65,
            reasoning_short="Empty risk replaced",
            factors=factors,
        )
        self.assertEqual(output["factors"][0]["risk"], "low")


class TestMissingDataFlagsEmitted(unittest.TestCase):
    """Unknown/missing inputs produce missing_data_flags + assumptions."""

    def test_missing_flags_present(self):
        output = build_prediction_output(
            hypothesis_id="H6",
            market_type="player_prop",
            prediction="over",
            confidence=0.52,
            reasoning_short="Missing inputs test",
            factors=[{"name": "rest_days", "value": None, "source": "local"}],
            missing_data_flags=["rest_days_unavailable", "injury_status_unknown"],
            assumptions=["rest_days assumed 2 (league avg)", "player assumed healthy"],
        )
        self.assertEqual(len(output["missing_data_flags"]), 2)
        self.assertIn("rest_days_unavailable", output["missing_data_flags"])
        self.assertEqual(len(output["assumptions"]), 2)
        self.assertIn("rest_days assumed 2 (league avg)", output["assumptions"])

    def test_no_missing_flags_yields_empty_lists(self):
        output = build_prediction_output(
            hypothesis_id="H7",
            market_type="spread",
            prediction="cover",
            confidence=0.70,
            reasoning_short="No missing data",
            factors=[{"name": "closing_odds", "value": 0.55, "source": "local"}],
        )
        self.assertIsInstance(output["missing_data_flags"], list)
        self.assertEqual(len(output["missing_data_flags"]), 0)
        self.assertIsInstance(output["assumptions"], list)
        self.assertEqual(len(output["assumptions"]), 0)

    def test_reasoning_short_truncated_at_140(self):
        long_reason = "A" * 200
        output = build_prediction_output(
            hypothesis_id="H8",
            market_type="spread",
            prediction="cover",
            confidence=0.60,
            reasoning_short=long_reason,
            factors=[{"name": "boxscore", "value": 1.0, "source": "local"}],
        )
        self.assertLessEqual(len(output["reasoning_short"]), 140)

    def test_output_contract_keys(self):
        """All required keys present in output."""
        output = build_prediction_output(
            hypothesis_id="H9",
            market_type="player_prop",
            prediction="over",
            confidence=0.65,
            reasoning_short="Contract test",
            factors=[{"name": "boxscore", "value": 1.0, "source": "local"}],
        )
        required_keys = {
            "hypothesis_id", "market_type", "prediction", "confidence",
            "confidence_grade", "reasoning_short", "factors", "risk_summary",
            "assumptions", "missing_data_flags", "status", "release_status",
        }
        self.assertEqual(required_keys, set(output.keys()))

    def test_risk_summary_structure(self):
        output = build_prediction_output(
            hypothesis_id="H10",
            market_type="spread",
            prediction="cover",
            confidence=0.60,
            reasoning_short="Risk summary test",
            factors=[
                {"name": "sentiment", "value": 0.5, "source": "manual"},
                {"name": "news", "value": 0.3, "source": "feed"},
            ],
        )
        rs = output["risk_summary"]
        self.assertIn("high_risk_factor_count", rs)
        self.assertIn("high_risk_factor_names", rs)
        self.assertIn("decision_flag", rs)
        self.assertEqual(rs["high_risk_factor_count"], 2)


if __name__ == "__main__":
    unittest.main()
