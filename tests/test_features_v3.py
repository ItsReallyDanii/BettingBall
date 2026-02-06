"""
Tests for features_v3 module.

Covers:
1. No-leakage for rolling features (past-only windows)
2. Missing optional columns does not crash; emits flags
3. Each new feature outputs deterministic values on fixed input
4. Risk tags present for all new factors
5. Prediction output includes missing_data_flags + assumptions when applicable
6. Feature availability report correctness
7. Sparse data run completes with flags/assumptions
8. Enriched data run computes all 6 feature families
"""
import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.features_v3 import (
    injury_impact_score,
    home_away_form_delta,
    pace_delta,
    matchup_edge_score,
    usage_trend_5g,
    minutes_trend_5g,
    shooting_eff_trend_5g,
    extract_v3_features,
    v3_features_to_vector,
    feature_availability_report,
    DataQualityTracker,
    V3_FEATURE_COLUMNS,
    V3_RISK_MAP,
    V3_RISK_WHEN_MISSING,
    classify_v3_factor_risk,
    _ols_slope,
    QUALITY_OK,
    QUALITY_MISSING,
    QUALITY_PARTIAL,
    QUALITY_ASSUMED,
    MISSING_INJURY_STATUS,
    MISSING_INJURY_MINUTES_LOST,
    MISSING_USAGE_LAST_5,
    MISSING_MINUTES_LAST_5,
    MISSING_EFG_LAST_5,
)

from src.safety import (
    build_prediction_output,
    classify_factor_risk,
    RISK_MAP,
)


# ---------------------------------------------------------------------------
# 1. No-leakage for rolling features (past-only windows)
# ---------------------------------------------------------------------------

class TestNoLeakageRollingFeatures(unittest.TestCase):
    """Rolling features must use past-only windows (oldest-first)."""

    def test_ols_slope_uses_sequential_index(self):
        """OLS slope: x = 0,1,2... (game index), strictly past-only order."""
        # Increasing trend
        values = [10.0, 12.0, 14.0, 16.0, 18.0]
        slope = _ols_slope(values)
        self.assertAlmostEqual(slope, 2.0, places=6)

    def test_ols_slope_flat(self):
        values = [5.0, 5.0, 5.0, 5.0, 5.0]
        slope = _ols_slope(values)
        self.assertAlmostEqual(slope, 0.0, places=6)

    def test_ols_slope_decreasing(self):
        values = [20.0, 18.0, 16.0, 14.0, 12.0]
        slope = _ols_slope(values)
        self.assertAlmostEqual(slope, -2.0, places=6)

    def test_ols_slope_single_value(self):
        slope = _ols_slope([5.0])
        self.assertEqual(slope, 0.0)

    def test_ols_slope_empty(self):
        slope = _ols_slope([])
        self.assertEqual(slope, 0.0)

    def test_usage_trend_uses_last_5_only(self):
        """If more than 5 values, only last 5 are used."""
        long_list = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22]
        val, quality = usage_trend_5g(usage_rates=long_list)
        # Should use last 5: [0.14, 0.16, 0.18, 0.20, 0.22]
        expected_slope = _ols_slope([0.14, 0.16, 0.18, 0.20, 0.22])
        self.assertAlmostEqual(val, expected_slope, places=6)
        self.assertEqual(quality, QUALITY_OK)

    def test_minutes_trend_uses_last_5_only(self):
        long_list = [30.0, 32.0, 34.0, 36.0, 38.0, 40.0, 42.0]
        val, quality = minutes_trend_5g(minutes_list=long_list)
        expected_slope = _ols_slope([34.0, 36.0, 38.0, 40.0, 42.0])
        self.assertAlmostEqual(val, expected_slope, places=6)
        self.assertEqual(quality, QUALITY_OK)


# ---------------------------------------------------------------------------
# 2. Missing optional columns does not crash; emits flags
# ---------------------------------------------------------------------------

class TestMissingColumnsNoCrash(unittest.TestCase):
    """Missing optional data must not crash; must emit flags."""

    def test_injury_all_none(self):
        tracker = DataQualityTracker()
        val, quality = injury_impact_score(None, None, None, tracker)
        self.assertIsInstance(val, float)
        self.assertEqual(quality, QUALITY_MISSING)
        self.assertIn(MISSING_INJURY_STATUS, tracker.missing_data_flags)
        self.assertIn(MISSING_INJURY_MINUTES_LOST, tracker.missing_data_flags)

    def test_home_away_all_none(self):
        tracker = DataQualityTracker()
        val, quality = home_away_form_delta(None, None, None, tracker)
        self.assertIsInstance(val, float)
        self.assertEqual(quality, QUALITY_MISSING)

    def test_pace_all_none(self):
        tracker = DataQualityTracker()
        val, quality = pace_delta(None, None, tracker)
        self.assertIsInstance(val, float)
        self.assertEqual(quality, QUALITY_MISSING)

    def test_matchup_all_none(self):
        tracker = DataQualityTracker()
        val, quality = matchup_edge_score(None, None, None, tracker)
        self.assertIsInstance(val, float)
        self.assertEqual(quality, QUALITY_MISSING)

    def test_usage_all_none(self):
        tracker = DataQualityTracker()
        val, quality = usage_trend_5g(None, None, tracker)
        self.assertIsInstance(val, float)
        self.assertEqual(quality, QUALITY_MISSING)
        self.assertIn(MISSING_USAGE_LAST_5, tracker.missing_data_flags)

    def test_minutes_all_none(self):
        tracker = DataQualityTracker()
        val, quality = minutes_trend_5g(None, None, tracker)
        self.assertIsInstance(val, float)
        self.assertEqual(quality, QUALITY_MISSING)
        self.assertIn(MISSING_MINUTES_LAST_5, tracker.missing_data_flags)

    def test_shooting_all_none(self):
        tracker = DataQualityTracker()
        val, quality = shooting_eff_trend_5g(None, None, None, tracker)
        self.assertIsInstance(val, float)
        self.assertEqual(quality, QUALITY_MISSING)
        self.assertIn(MISSING_EFG_LAST_5, tracker.missing_data_flags)

    def test_extract_v3_empty_record(self):
        """Completely empty joined record should not crash."""
        tracker = DataQualityTracker()
        features = extract_v3_features({}, tracker)
        self.assertEqual(len(features), len(V3_FEATURE_COLUMNS))
        for col in V3_FEATURE_COLUMNS:
            self.assertIn(col, features)
            self.assertIsInstance(features[col], float)
        self.assertTrue(len(tracker.missing_data_flags) > 0)
        self.assertTrue(len(tracker.assumptions) > 0)


# ---------------------------------------------------------------------------
# 3. Deterministic values on fixed input
# ---------------------------------------------------------------------------

class TestDeterministicFeatures(unittest.TestCase):
    """Each feature function must produce identical output on identical input."""

    def test_injury_deterministic(self):
        for _ in range(10):
            val, q = injury_impact_score("questionable", 12.0, 30.0)
            self.assertAlmostEqual(val, 0.15 * 0.5 + (12.0/48.0) * 0.3 + (1.0 - 30.0/36.0) * 0.2, places=6)
            self.assertEqual(q, QUALITY_OK)

    def test_home_away_deterministic(self):
        for _ in range(10):
            val, q = home_away_form_delta(True, 7.0, 3.0)
            self.assertAlmostEqual(val, (7.0/10.0 - 3.0/10.0), places=6)
            self.assertEqual(q, QUALITY_OK)

    def test_pace_delta_deterministic(self):
        for _ in range(10):
            val, q = pace_delta(105.0, 95.0)
            self.assertAlmostEqual(val, 10.0, places=6)
            self.assertEqual(q, QUALITY_OK)

    def test_matchup_deterministic(self):
        for _ in range(10):
            val, q = matchup_edge_score(2.5, 0.7, 105.0)
            # h2h_norm = (2.5 - 1.5) / 1.5 = 0.6667
            # def_idx_norm = (0.7 - 0.5) / 0.5 = 0.4
            # def_rating_adv = (110 - 105) / 20 = 0.25
            expected = 0.4 * (2.5-1.5)/1.5 + 0.3 * (0.7-0.5)/0.5 + 0.3 * (110.0-105.0)/20.0
            self.assertAlmostEqual(val, expected, places=6)
            self.assertEqual(q, QUALITY_OK)

    def test_usage_trend_deterministic(self):
        rates = [0.18, 0.20, 0.22, 0.24, 0.26]
        for _ in range(10):
            val, q = usage_trend_5g(usage_rates=rates)
            self.assertAlmostEqual(val, 0.02, places=6)
            self.assertEqual(q, QUALITY_OK)

    def test_minutes_trend_deterministic(self):
        mins = [30.0, 32.0, 34.0, 36.0, 38.0]
        for _ in range(10):
            val, q = minutes_trend_5g(minutes_list=mins)
            self.assertAlmostEqual(val, 2.0, places=6)
            self.assertEqual(q, QUALITY_OK)

    def test_shooting_eff_deterministic(self):
        efgs = [0.48, 0.50, 0.52, 0.54, 0.56]
        for _ in range(10):
            val, q = shooting_eff_trend_5g(efg_list=efgs)
            self.assertAlmostEqual(val, 0.02, places=6)
            self.assertEqual(q, QUALITY_OK)


# ---------------------------------------------------------------------------
# 4. Risk tags present for all new factors
# ---------------------------------------------------------------------------

class TestRiskTagsV3(unittest.TestCase):
    """Every v3 factor must have a risk classification."""

    def test_all_v3_features_have_risk(self):
        for col in V3_FEATURE_COLUMNS:
            risk = classify_v3_factor_risk(col, QUALITY_OK)
            self.assertIn(risk, ("low", "medium", "high"), f"Missing risk for {col}")

    def test_all_v3_features_have_missing_risk(self):
        for col in V3_FEATURE_COLUMNS:
            risk = classify_v3_factor_risk(col, QUALITY_MISSING)
            self.assertIn(risk, ("low", "medium", "high"), f"Missing risk for {col}")

    def test_injury_high_when_missing(self):
        risk = classify_v3_factor_risk("injury_impact_score", QUALITY_MISSING)
        self.assertEqual(risk, "high")

    def test_injury_medium_when_ok(self):
        risk = classify_v3_factor_risk("injury_impact_score", QUALITY_OK)
        self.assertEqual(risk, "medium")

    def test_v3_factors_in_safety_risk_map(self):
        """V3 factors are registered in safety.py RISK_MAP."""
        all_factors = []
        for level, names in RISK_MAP.items():
            all_factors.extend(names)
        for col in V3_FEATURE_COLUMNS:
            self.assertIn(col, all_factors, f"{col} not in safety RISK_MAP")

    def test_classify_factor_risk_for_v3(self):
        """safety.classify_factor_risk works for v3 factors."""
        for col in V3_FEATURE_COLUMNS:
            risk = classify_factor_risk(col)
            self.assertIn(risk, ("low", "medium", "high"))


# ---------------------------------------------------------------------------
# 5. Prediction output includes missing_data_flags + assumptions
# ---------------------------------------------------------------------------

class TestPredictionOutputFlags(unittest.TestCase):
    """build_prediction_output propagates flags from v3 tracker."""

    def test_flags_from_sparse_data(self):
        tracker = DataQualityTracker()
        extract_v3_features({}, tracker)
        quality = tracker.to_dict()

        output = build_prediction_output(
            hypothesis_id="V3_TEST_1",
            market_type="player_prop",
            prediction="over",
            confidence=0.60,
            reasoning_short="Sparse data test",
            factors=[{"name": "injury_impact_score", "value": 0.0}],
            missing_data_flags=quality["missing_data_flags"],
            assumptions=quality["assumptions"],
        )

        self.assertIsInstance(output["missing_data_flags"], list)
        self.assertTrue(len(output["missing_data_flags"]) > 0)
        self.assertIsInstance(output["assumptions"], list)
        self.assertTrue(len(output["assumptions"]) > 0)

    def test_no_flags_from_enriched_data(self):
        enriched = _make_enriched_record()
        tracker = DataQualityTracker()
        extract_v3_features(enriched, tracker)
        quality = tracker.to_dict()

        output = build_prediction_output(
            hypothesis_id="V3_TEST_2",
            market_type="player_prop",
            prediction="over",
            confidence=0.70,
            reasoning_short="Enriched data test",
            factors=[{"name": "boxscore", "value": 1.0}],
            missing_data_flags=quality["missing_data_flags"],
            assumptions=quality["assumptions"],
        )

        self.assertEqual(len(output["missing_data_flags"]), 0)
        self.assertEqual(len(output["assumptions"]), 0)

    def test_v3_factors_get_risk_in_output(self):
        """V3 factor names get auto-tagged by build_prediction_output."""
        factors = [{"name": col, "value": 0.5} for col in V3_FEATURE_COLUMNS]
        output = build_prediction_output(
            hypothesis_id="V3_TEST_3",
            market_type="player_prop",
            prediction="over",
            confidence=0.65,
            reasoning_short="V3 risk tag test",
            factors=factors,
        )
        for f in output["factors"]:
            self.assertIn("risk", f)
            self.assertIn(f["risk"], ("low", "medium", "high"))


# ---------------------------------------------------------------------------
# 6. Feature availability report
# ---------------------------------------------------------------------------

class TestFeatureAvailabilityReport(unittest.TestCase):
    """Feature availability report correctness."""

    def test_sparse_record_missing_groups(self):
        report = feature_availability_report({})
        self.assertIsInstance(report["available_features"], list)
        self.assertIsInstance(report["missing_feature_groups"], list)
        self.assertTrue(len(report["missing_feature_groups"]) > 0)
        self.assertIsInstance(report["coverage_pct"], float)

    def test_enriched_record_full_coverage(self):
        enriched = _make_enriched_record()
        report = feature_availability_report(enriched)
        self.assertEqual(len(report["missing_feature_groups"]), 0)
        self.assertEqual(len(report["available_features"]), len(V3_FEATURE_COLUMNS))
        self.assertAlmostEqual(report["coverage_pct"], 1.0, places=4)

    def test_partial_record(self):
        """Record with some v3 data but not all families."""
        partial = {
            "game": {"is_home_for_subject_team": "True", "head_to_head_last_3": 2.0,
                      "defender_matchup_index": 0.6},
            "player": {"injury_status": "healthy", "injury_minutes_lost": 0},
            "team": {"pace": 102.0},
            "opponent_team": {"pace": 98.0, "def_rating": 108.0},
        }
        report = feature_availability_report(partial)
        # Injuries, pace, matchup should be available
        self.assertIn("injury_impact_score", report["available_features"])
        self.assertIn("pace_delta", report["available_features"])
        # Usage/minutes/shooting should be missing
        self.assertIn("usage_trends", report["missing_feature_groups"])
        self.assertIn("minutes_trends", report["missing_feature_groups"])


# ---------------------------------------------------------------------------
# 7. Sparse data run (end-to-end)
# ---------------------------------------------------------------------------

class TestSparseDataRun(unittest.TestCase):
    """Sparse data: completes + emits flags/assumptions."""

    def test_sparse_extract_completes(self):
        tracker = DataQualityTracker()
        features = extract_v3_features({}, tracker)
        vec = v3_features_to_vector(features)
        self.assertEqual(len(vec), len(V3_FEATURE_COLUMNS))
        self.assertTrue(all(isinstance(v, float) for v in vec))
        self.assertTrue(len(tracker.missing_data_flags) > 0)

    def test_sparse_extract_assumptions_are_deterministic_strings(self):
        tracker = DataQualityTracker()
        extract_v3_features({}, tracker)
        for a in tracker.assumptions:
            self.assertIsInstance(a, str)
            self.assertTrue(len(a) > 0)


# ---------------------------------------------------------------------------
# 8. Enriched data run
# ---------------------------------------------------------------------------

class TestEnrichedDataRun(unittest.TestCase):
    """Enriched data: computes all 6 feature families."""

    def test_enriched_extract_all_features(self):
        enriched = _make_enriched_record()
        tracker = DataQualityTracker()
        features = extract_v3_features(enriched, tracker)

        self.assertEqual(len(features), len(V3_FEATURE_COLUMNS))
        self.assertEqual(len(tracker.missing_data_flags), 0)
        self.assertEqual(len(tracker.assumptions), 0)

        # Verify non-default values computed
        self.assertNotAlmostEqual(features["injury_impact_score"], 0.0, places=2)
        self.assertNotAlmostEqual(features["pace_delta"], 0.0, places=2)

    def test_enriched_vector_matches_columns(self):
        enriched = _make_enriched_record()
        features = extract_v3_features(enriched)
        vec = v3_features_to_vector(features)
        self.assertEqual(len(vec), len(V3_FEATURE_COLUMNS))
        for i, col in enumerate(V3_FEATURE_COLUMNS):
            self.assertAlmostEqual(vec[i], features[col], places=6)


# ---------------------------------------------------------------------------
# 9. DataQualityTracker
# ---------------------------------------------------------------------------

class TestDataQualityTracker(unittest.TestCase):

    def test_deduplication(self):
        t = DataQualityTracker()
        t.flag_missing(MISSING_INJURY_STATUS)
        t.flag_missing(MISSING_INJURY_STATUS)
        self.assertEqual(len(t.missing_data_flags), 1)
        self.assertEqual(len(t.assumptions), 1)

    def test_to_dict(self):
        t = DataQualityTracker()
        t.flag_missing(MISSING_USAGE_LAST_5)
        d = t.to_dict()
        self.assertIn("missing_data_flags", d)
        self.assertIn("assumptions", d)
        self.assertEqual(len(d["missing_data_flags"]), 1)

    def test_empty_tracker(self):
        t = DataQualityTracker()
        d = t.to_dict()
        self.assertEqual(d["missing_data_flags"], [])
        self.assertEqual(d["assumptions"], [])


# ---------------------------------------------------------------------------
# 10. Edge cases for individual features
# ---------------------------------------------------------------------------

class TestFeatureEdgeCases(unittest.TestCase):

    def test_injury_out_status(self):
        val, q = injury_impact_score("out", 48.0, 0.0)
        # base=1.0, minutes_penalty=1.0, cap_penalty=0.0 (cap_val=0 → skipped)
        self.assertAlmostEqual(val, 0.5 * 1.0 + 0.3 * 1.0 + 0.2 * 0.0, places=6)
        self.assertAlmostEqual(val, 0.8, places=6)

    def test_injury_healthy(self):
        val, q = injury_impact_score("healthy", 0.0, 36.0)
        self.assertAlmostEqual(val, 0.0, places=6)

    def test_home_away_away_advantage(self):
        val, q = home_away_form_delta(False, 3.0, 7.0)
        # away team benefits: delta = 0.3-0.7 = -0.4; result = -(-0.4) = 0.4
        self.assertAlmostEqual(val, 0.4, places=6)

    def test_pace_delta_clipping(self):
        val, q = pace_delta(150.0, 80.0)
        self.assertEqual(val, 30.0)  # clipped

    def test_matchup_neutral(self):
        val, q = matchup_edge_score(1.5, 0.5, 110.0)
        self.assertAlmostEqual(val, 0.0, places=6)

    def test_usage_scalar_only(self):
        val, q = usage_trend_5g(usage_rates=None, usage_rate_last_5=0.25)
        self.assertEqual(val, 0.0)
        self.assertEqual(q, QUALITY_ASSUMED)

    def test_shooting_ts_fallback(self):
        val, q = shooting_eff_trend_5g(efg_list=None, efg_last_5=None, true_shooting_last_5=0.55)
        self.assertEqual(val, 0.0)
        self.assertEqual(q, QUALITY_ASSUMED)

    def test_v3_features_to_vector_default(self):
        vec = v3_features_to_vector({})
        self.assertEqual(len(vec), len(V3_FEATURE_COLUMNS))
        self.assertTrue(all(v == 0.0 for v in vec))


# ---------------------------------------------------------------------------
# Helper: make enriched record
# ---------------------------------------------------------------------------

def _make_enriched_record():
    """Create a fully-populated joined record with all v3 optional data."""
    return {
        "game": {
            "is_home_for_subject_team": "True",
            "head_to_head_last_3": 2.0,
            "defender_matchup_index": 0.6,
        },
        "player": {
            "injury_status": "questionable",
            "injury_minutes_lost": 10.0,
            "probable_minutes_cap": 28.0,
            "usage_rate_last_5": 0.25,
            "usage_rates_last_5": [0.22, 0.24, 0.25, 0.26, 0.28],
            "minutes_last_5": 32.0,
            "minutes_list_last_5": [30.0, 31.0, 32.0, 33.0, 34.0],
            "efg_last_5": 0.52,
            "efg_list_last_5": [0.48, 0.50, 0.52, 0.54, 0.56],
            "true_shooting_last_5": 0.55,
        },
        "team": {
            "pace": 103.0,
            "home_wins_last_10": 7.0,
            "away_wins_last_10": 4.0,
        },
        "opponent_team": {
            "pace": 97.0,
            "def_rating": 108.0,
        },
    }


if __name__ == "__main__":
    unittest.main()
