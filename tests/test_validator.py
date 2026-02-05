"""
Comprehensive tests for src/validator.py - BettingValidator V1.

Covers:
    - Happy path (valid payloads)
    - Missing fields
    - Wrong types
    - Out-of-range values
    - Contradictory / inconsistent inputs
    - Edge / boundary values
"""

import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.validator import (
    BettingValidator,
    ValidationError,
    american_odds_to_implied_prob,
    VALID_EVENT_TYPES,
    VALID_MARKETS,
    VALID_CONFIDENCE_GRADES,
    VALID_HORIZONS,
    MAX_STAKE,
    MAX_EDGE_DIVERGENCE,
)


def _valid_payload(**overrides):
    """Return a fully valid payload, with optional field overrides."""
    base = {
        "event_type": "player_points_over",
        "threshold": 24.5,
        "probability": 0.62,
        "odds": -110,
        "stake": 50.0,
        "confidence": "B",
        "market": "player_prop",
        "horizon": "pregame",
    }
    base.update(overrides)
    return base


class TestHappyPath(unittest.TestCase):
    """Valid payloads must pass with zero errors."""

    def test_minimal_valid_payload(self):
        result = BettingValidator.validate(_valid_payload())
        self.assertTrue(result["valid"])
        self.assertEqual(result["errors"], [])

    def test_valid_without_market(self):
        """market is optional - payload is valid without it."""
        p = _valid_payload()
        del p["market"]
        result = BettingValidator.validate(p)
        self.assertTrue(result["valid"])

    def test_valid_without_horizon_gets_default(self):
        """horizon defaults to 'pregame' when absent."""
        p = _valid_payload()
        del p["horizon"]
        result = BettingValidator.validate(p)
        self.assertTrue(result["valid"])
        self.assertEqual(result["payload"]["horizon"], "pregame")

    def test_valid_spread_cover(self):
        result = BettingValidator.validate(_valid_payload(
            event_type="spread_cover",
            market="spread",
            threshold=-3.5,  # negative spread is valid for spread_cover
        ))
        self.assertTrue(result["valid"])

    def test_valid_team_total(self):
        result = BettingValidator.validate(_valid_payload(
            event_type="team_total_over",
            market="team_total",
            threshold=110.5,
        ))
        self.assertTrue(result["valid"])

    def test_valid_positive_odds(self):
        result = BettingValidator.validate(_valid_payload(odds=150))
        self.assertTrue(result["valid"])

    def test_valid_live_horizon(self):
        result = BettingValidator.validate(_valid_payload(horizon="live"))
        self.assertTrue(result["valid"])

    def test_all_confidence_grades(self):
        for grade in VALID_CONFIDENCE_GRADES:
            result = BettingValidator.validate(_valid_payload(confidence=grade))
            self.assertTrue(result["valid"], f"Grade {grade} should be valid")

    def test_all_event_types(self):
        compat = {
            "player_points_over": "player_prop",
            "player_assists_over": "player_prop",
            "player_rebounds_over": "player_prop",
            "team_total_over": "team_total",
            "spread_cover": "spread",
        }
        for et, mkt in compat.items():
            result = BettingValidator.validate(_valid_payload(
                event_type=et, market=mkt, threshold=10.5
            ))
            self.assertTrue(result["valid"], f"{et}/{mkt} should be valid")


class TestMissingFields(unittest.TestCase):
    """Each required field, when absent, must produce a MISSING_FIELD error."""

    def test_missing_event_type(self):
        p = _valid_payload()
        del p["event_type"]
        result = BettingValidator.validate(p)
        self.assertFalse(result["valid"])
        codes = {e["code"] for e in result["errors"]}
        self.assertIn("MISSING_FIELD", codes)
        fields = {e["field"] for e in result["errors"] if e["code"] == "MISSING_FIELD"}
        self.assertIn("event_type", fields)

    def test_missing_threshold(self):
        p = _valid_payload()
        del p["threshold"]
        result = BettingValidator.validate(p)
        self.assertFalse(result["valid"])
        fields = {e["field"] for e in result["errors"] if e["code"] == "MISSING_FIELD"}
        self.assertIn("threshold", fields)

    def test_missing_probability(self):
        p = _valid_payload()
        del p["probability"]
        result = BettingValidator.validate(p)
        self.assertFalse(result["valid"])
        fields = {e["field"] for e in result["errors"] if e["code"] == "MISSING_FIELD"}
        self.assertIn("probability", fields)

    def test_missing_odds(self):
        p = _valid_payload()
        del p["odds"]
        result = BettingValidator.validate(p)
        self.assertFalse(result["valid"])
        fields = {e["field"] for e in result["errors"] if e["code"] == "MISSING_FIELD"}
        self.assertIn("odds", fields)

    def test_missing_stake(self):
        p = _valid_payload()
        del p["stake"]
        result = BettingValidator.validate(p)
        self.assertFalse(result["valid"])
        fields = {e["field"] for e in result["errors"] if e["code"] == "MISSING_FIELD"}
        self.assertIn("stake", fields)

    def test_missing_confidence(self):
        p = _valid_payload()
        del p["confidence"]
        result = BettingValidator.validate(p)
        self.assertFalse(result["valid"])
        fields = {e["field"] for e in result["errors"] if e["code"] == "MISSING_FIELD"}
        self.assertIn("confidence", fields)

    def test_missing_all_required(self):
        result = BettingValidator.validate({})
        self.assertFalse(result["valid"])
        missing_fields = {e["field"] for e in result["errors"] if e["code"] == "MISSING_FIELD"}
        for f in ("event_type", "threshold", "probability", "odds", "stake", "confidence"):
            self.assertIn(f, missing_fields)

    def test_empty_payload_still_gets_horizon_default(self):
        result = BettingValidator.validate({})
        self.assertEqual(result["payload"].get("horizon"), "pregame")


class TestWrongTypes(unittest.TestCase):
    """Fields with wrong types must produce WRONG_TYPE errors."""

    def test_event_type_not_string(self):
        result = BettingValidator.validate(_valid_payload(event_type=123))
        self.assertFalse(result["valid"])
        self.assertTrue(any(
            e["code"] == "WRONG_TYPE" and e["field"] == "event_type"
            for e in result["errors"]
        ))

    def test_confidence_not_string(self):
        result = BettingValidator.validate(_valid_payload(confidence=2))
        self.assertFalse(result["valid"])
        self.assertTrue(any(
            e["code"] == "WRONG_TYPE" and e["field"] == "confidence"
            for e in result["errors"]
        ))

    def test_probability_not_number(self):
        result = BettingValidator.validate(_valid_payload(probability="high"))
        self.assertFalse(result["valid"])
        self.assertTrue(any(
            e["code"] == "WRONG_TYPE" and e["field"] == "probability"
            for e in result["errors"]
        ))

    def test_odds_not_number(self):
        result = BettingValidator.validate(_valid_payload(odds="even"))
        self.assertFalse(result["valid"])
        self.assertTrue(any(
            e["code"] == "WRONG_TYPE" and e["field"] == "odds"
            for e in result["errors"]
        ))

    def test_stake_not_number(self):
        result = BettingValidator.validate(_valid_payload(stake="fifty"))
        self.assertFalse(result["valid"])
        self.assertTrue(any(
            e["code"] == "WRONG_TYPE" and e["field"] == "stake"
            for e in result["errors"]
        ))

    def test_threshold_not_number(self):
        result = BettingValidator.validate(_valid_payload(threshold=None))
        self.assertFalse(result["valid"])
        self.assertTrue(any(
            e["code"] == "WRONG_TYPE" and e["field"] == "threshold"
            for e in result["errors"]
        ))

    def test_boolean_rejected_as_number(self):
        """bool is subclass of int in Python; we must reject it for numeric fields."""
        result = BettingValidator.validate(_valid_payload(probability=True))
        self.assertFalse(result["valid"])
        self.assertTrue(any(
            e["code"] == "WRONG_TYPE" and e["field"] == "probability"
            for e in result["errors"]
        ))

    def test_payload_not_dict(self):
        result = BettingValidator.validate("not a dict")
        self.assertFalse(result["valid"])
        self.assertEqual(result["errors"][0]["code"], "INVALID_TYPE")

    def test_payload_none(self):
        result = BettingValidator.validate(None)
        self.assertFalse(result["valid"])
        self.assertEqual(result["errors"][0]["code"], "INVALID_TYPE")

    def test_payload_list(self):
        result = BettingValidator.validate([1, 2, 3])
        self.assertFalse(result["valid"])
        self.assertEqual(result["errors"][0]["code"], "INVALID_TYPE")


class TestOutOfRange(unittest.TestCase):
    """Numeric values outside accepted bounds must produce OUT_OF_RANGE errors."""

    def test_probability_negative(self):
        result = BettingValidator.validate(_valid_payload(probability=-0.1))
        self.assertFalse(result["valid"])
        self.assertTrue(any(
            e["code"] == "OUT_OF_RANGE" and e["field"] == "probability"
            for e in result["errors"]
        ))

    def test_probability_above_one(self):
        result = BettingValidator.validate(_valid_payload(probability=1.01))
        self.assertFalse(result["valid"])
        self.assertTrue(any(
            e["code"] == "OUT_OF_RANGE" and e["field"] == "probability"
            for e in result["errors"]
        ))

    def test_odds_in_dead_zone(self):
        """American odds between -100 and 100 exclusive are invalid."""
        for odds_val in [-99, -50, 0, 50, 99]:
            result = BettingValidator.validate(_valid_payload(odds=odds_val))
            self.assertFalse(result["valid"], f"odds={odds_val} should be invalid")
            self.assertTrue(any(
                e["code"] == "OUT_OF_RANGE" and e["field"] == "odds"
                for e in result["errors"]
            ), f"odds={odds_val} should produce OUT_OF_RANGE")

    def test_stake_zero(self):
        result = BettingValidator.validate(_valid_payload(stake=0))
        self.assertFalse(result["valid"])
        self.assertTrue(any(
            e["code"] == "OUT_OF_RANGE" and e["field"] == "stake"
            for e in result["errors"]
        ))

    def test_stake_negative(self):
        result = BettingValidator.validate(_valid_payload(stake=-10))
        self.assertFalse(result["valid"])
        self.assertTrue(any(
            e["code"] == "OUT_OF_RANGE" and e["field"] == "stake"
            for e in result["errors"]
        ))

    def test_stake_exceeds_max(self):
        result = BettingValidator.validate(_valid_payload(stake=MAX_STAKE + 1))
        self.assertFalse(result["valid"])
        self.assertTrue(any(
            e["code"] == "OUT_OF_RANGE" and e["field"] == "stake"
            for e in result["errors"]
        ))

    def test_threshold_zero_for_prop(self):
        result = BettingValidator.validate(_valid_payload(
            event_type="player_points_over", threshold=0
        ))
        self.assertFalse(result["valid"])
        self.assertTrue(any(
            e["code"] == "OUT_OF_RANGE" and e["field"] == "threshold"
            for e in result["errors"]
        ))

    def test_threshold_negative_for_prop(self):
        result = BettingValidator.validate(_valid_payload(
            event_type="team_total_over", market="team_total", threshold=-5
        ))
        self.assertFalse(result["valid"])
        self.assertTrue(any(
            e["code"] == "OUT_OF_RANGE" and e["field"] == "threshold"
            for e in result["errors"]
        ))


class TestInvalidValues(unittest.TestCase):
    """Enum fields with unrecognized values must produce INVALID_VALUE errors."""

    def test_invalid_event_type(self):
        result = BettingValidator.validate(_valid_payload(event_type="goals_over"))
        self.assertFalse(result["valid"])
        self.assertTrue(any(
            e["code"] == "INVALID_VALUE" and e["field"] == "event_type"
            for e in result["errors"]
        ))

    def test_invalid_confidence(self):
        result = BettingValidator.validate(_valid_payload(confidence="E"))
        self.assertFalse(result["valid"])
        self.assertTrue(any(
            e["code"] == "INVALID_VALUE" and e["field"] == "confidence"
            for e in result["errors"]
        ))

    def test_invalid_market(self):
        result = BettingValidator.validate(_valid_payload(market="moneyline"))
        self.assertFalse(result["valid"])
        self.assertTrue(any(
            e["code"] == "INVALID_VALUE" and e["field"] == "market"
            for e in result["errors"]
        ))

    def test_invalid_horizon(self):
        result = BettingValidator.validate(_valid_payload(horizon="halftime"))
        self.assertFalse(result["valid"])
        self.assertTrue(any(
            e["code"] == "INVALID_VALUE" and e["field"] == "horizon"
            for e in result["errors"]
        ))


class TestContradictoryInputs(unittest.TestCase):
    """Cross-field consistency violations must be caught."""

    def test_market_event_type_mismatch_prop_on_spread(self):
        result = BettingValidator.validate(_valid_payload(
            event_type="player_points_over", market="spread"
        ))
        self.assertFalse(result["valid"])
        self.assertTrue(any(
            e["code"] == "MARKET_TYPE_MISMATCH"
            for e in result["errors"]
        ))

    def test_market_event_type_mismatch_spread_on_prop(self):
        result = BettingValidator.validate(_valid_payload(
            event_type="spread_cover", market="player_prop"
        ))
        self.assertFalse(result["valid"])
        self.assertTrue(any(
            e["code"] == "MARKET_TYPE_MISMATCH"
            for e in result["errors"]
        ))

    def test_market_event_type_mismatch_team_total_on_prop(self):
        result = BettingValidator.validate(_valid_payload(
            event_type="team_total_over", market="player_prop"
        ))
        self.assertFalse(result["valid"])
        self.assertTrue(any(
            e["code"] == "MARKET_TYPE_MISMATCH"
            for e in result["errors"]
        ))

    def test_edge_divergence_too_high(self):
        """Model probability far from implied odds probability should trigger error."""
        # odds=-110 implies ~0.524 probability
        # probability=0.05 gives divergence ~0.47 (below 0.60 threshold)
        # Let's use extreme values: probability=0.99, odds=500 (implied=0.167)
        # divergence = |0.99 - 0.167| = 0.823 > 0.60
        result = BettingValidator.validate(_valid_payload(
            probability=0.99, odds=500
        ))
        self.assertFalse(result["valid"])
        self.assertTrue(any(
            e["code"] == "EDGE_DIVERGENCE"
            for e in result["errors"]
        ))

    def test_edge_divergence_within_bounds(self):
        """Moderate divergence should be accepted."""
        # odds=-150 implies 0.600; probability=0.62 divergence=0.02 -> OK
        result = BettingValidator.validate(_valid_payload(
            probability=0.62, odds=-150
        ))
        self.assertTrue(result["valid"])

    def test_no_consistency_check_when_odds_in_dead_zone(self):
        """If odds are invalid, we should still get OUT_OF_RANGE but not
        a spurious EDGE_DIVERGENCE error (implied prob is None)."""
        result = BettingValidator.validate(_valid_payload(odds=50))
        codes = {e["code"] for e in result["errors"]}
        self.assertIn("OUT_OF_RANGE", codes)
        self.assertNotIn("EDGE_DIVERGENCE", codes)


class TestBoundaryValues(unittest.TestCase):
    """Exact boundary values must pass or fail deterministically."""

    def test_probability_zero(self):
        # probability=0.0 is valid (model says event won't happen)
        # odds=-110 implies ~0.524, divergence 0.524 < 0.60
        result = BettingValidator.validate(_valid_payload(probability=0.0, odds=-110))
        prob_errors = [e for e in result["errors"]
                       if e["field"] == "probability" and e["code"] == "OUT_OF_RANGE"]
        self.assertEqual(len(prob_errors), 0, "probability=0.0 should be in range")

    def test_probability_one(self):
        # probability=1.0 is valid
        # odds=-500 implies 0.833, divergence = |1.0 - 0.833| = 0.167 < 0.60
        result = BettingValidator.validate(_valid_payload(probability=1.0, odds=-500))
        prob_errors = [e for e in result["errors"]
                       if e["field"] == "probability" and e["code"] == "OUT_OF_RANGE"]
        self.assertEqual(len(prob_errors), 0, "probability=1.0 should be in range")

    def test_odds_exactly_minus_100(self):
        result = BettingValidator.validate(_valid_payload(odds=-100))
        odds_errors = [e for e in result["errors"]
                       if e["field"] == "odds" and e["code"] == "OUT_OF_RANGE"]
        self.assertEqual(len(odds_errors), 0, "odds=-100 should be valid")

    def test_odds_exactly_plus_100(self):
        result = BettingValidator.validate(_valid_payload(odds=100))
        odds_errors = [e for e in result["errors"]
                       if e["field"] == "odds" and e["code"] == "OUT_OF_RANGE"]
        self.assertEqual(len(odds_errors), 0, "odds=+100 should be valid")

    def test_stake_exactly_max(self):
        result = BettingValidator.validate(_valid_payload(stake=MAX_STAKE))
        stake_errors = [e for e in result["errors"]
                        if e["field"] == "stake" and e["code"] == "OUT_OF_RANGE"]
        self.assertEqual(len(stake_errors), 0, f"stake={MAX_STAKE} should be valid")

    def test_stake_tiny_positive(self):
        result = BettingValidator.validate(_valid_payload(stake=0.01))
        stake_errors = [e for e in result["errors"]
                        if e["field"] == "stake" and e["code"] == "OUT_OF_RANGE"]
        self.assertEqual(len(stake_errors), 0, "stake=0.01 should be valid")

    def test_threshold_tiny_positive_for_prop(self):
        result = BettingValidator.validate(_valid_payload(threshold=0.5))
        threshold_errors = [e for e in result["errors"]
                            if e["field"] == "threshold" and e["code"] == "OUT_OF_RANGE"]
        self.assertEqual(len(threshold_errors), 0, "threshold=0.5 should be valid for prop")

    def test_threshold_negative_allowed_for_spread(self):
        """spread_cover allows negative thresholds (e.g., -3.5 point spread)."""
        result = BettingValidator.validate(_valid_payload(
            event_type="spread_cover", market="spread", threshold=-7.0
        ))
        threshold_errors = [e for e in result["errors"]
                            if e["field"] == "threshold" and e["code"] == "OUT_OF_RANGE"]
        self.assertEqual(len(threshold_errors), 0,
                         "Negative threshold should be valid for spread_cover")


class TestAmericanOddsConversion(unittest.TestCase):
    """Unit tests for the odds -> implied probability helper."""

    def test_minus_110(self):
        p = american_odds_to_implied_prob(-110)
        self.assertAlmostEqual(p, 110 / 210, places=4)

    def test_plus_200(self):
        p = american_odds_to_implied_prob(200)
        self.assertAlmostEqual(p, 100 / 300, places=4)

    def test_minus_100(self):
        p = american_odds_to_implied_prob(-100)
        self.assertAlmostEqual(p, 0.5, places=4)

    def test_plus_100(self):
        p = american_odds_to_implied_prob(100)
        self.assertAlmostEqual(p, 0.5, places=4)

    def test_dead_zone_returns_none(self):
        self.assertIsNone(american_odds_to_implied_prob(0))
        self.assertIsNone(american_odds_to_implied_prob(50))
        self.assertIsNone(american_odds_to_implied_prob(-50))

    def test_large_favorite(self):
        p = american_odds_to_implied_prob(-1000)
        self.assertAlmostEqual(p, 1000 / 1100, places=4)

    def test_large_underdog(self):
        p = american_odds_to_implied_prob(1000)
        self.assertAlmostEqual(p, 100 / 1100, places=4)


class TestErrorStructure(unittest.TestCase):
    """Verify error objects have the required machine-readable shape."""

    def test_error_dict_keys(self):
        result = BettingValidator.validate({})
        for err in result["errors"]:
            self.assertIn("code", err)
            self.assertIn("field", err)
            self.assertIn("message", err)

    def test_error_meta_present_on_type_error(self):
        result = BettingValidator.validate(_valid_payload(probability="bad"))
        type_errors = [e for e in result["errors"] if e["code"] == "WRONG_TYPE"]
        self.assertTrue(len(type_errors) > 0)
        self.assertIn("meta", type_errors[0])
        self.assertEqual(type_errors[0]["meta"]["expected"], "number")

    def test_result_shape(self):
        result = BettingValidator.validate(_valid_payload())
        self.assertIn("valid", result)
        self.assertIn("errors", result)
        self.assertIn("payload", result)
        self.assertIsInstance(result["valid"], bool)
        self.assertIsInstance(result["errors"], list)

    def test_no_short_circuit(self):
        """Validator must report ALL errors, not stop at the first."""
        p = {
            "event_type": 123,        # wrong type
            "threshold": "bad",       # wrong type
            "probability": 5.0,       # out of range
            # missing: odds, stake, confidence
        }
        result = BettingValidator.validate(p)
        self.assertFalse(result["valid"])
        # Should have at least: 3 missing + 2 wrong type + 1 out-of-range = 6
        self.assertGreaterEqual(len(result["errors"]), 5)


class TestValidationErrorClass(unittest.TestCase):
    """Direct tests on the ValidationError dataclass."""

    def test_to_dict(self):
        e = ValidationError("TEST", "field", "msg", {"k": "v"})
        d = e.to_dict()
        self.assertEqual(d["code"], "TEST")
        self.assertEqual(d["field"], "field")
        self.assertEqual(d["message"], "msg")
        self.assertEqual(d["meta"]["k"], "v")

    def test_to_dict_no_meta(self):
        e = ValidationError("TEST", "field", "msg")
        d = e.to_dict()
        self.assertNotIn("meta", d)

    def test_repr(self):
        e = ValidationError("C", "f", "m")
        self.assertIn("C", repr(e))
        self.assertIn("f", repr(e))


if __name__ == '__main__':
    unittest.main()
