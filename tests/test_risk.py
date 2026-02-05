import pytest
from src.risk import recommend, CONFIDENCE_RANK

def test_recommend_bet():
    """Test BET recommendation with sufficient edge and EV."""
    result = recommend(
        model_prob=0.65,
        implied_prob=0.60,
        confidence_grade="B",
        expected_value_unit=0.05
    )
    assert result["recommendation"] == "BET"
    assert result["stake_units"] == 1.0

def test_recommend_lean():
    """Test LEAN recommendation with moderate edge."""
    result = recommend(
        model_prob=0.55,
        implied_prob=0.52,
        confidence_grade="B",
        expected_value_unit=0.01
    )
    assert result["recommendation"] == "LEAN"
    assert result["stake_units"] == 0.5

def test_recommend_pass():
    """Test PASS recommendation with insufficient edge."""
    result = recommend(
        model_prob=0.51,
        implied_prob=0.50,
        confidence_grade="B",
        expected_value_unit=0.002
    )
    assert result["recommendation"] == "PASS"
    assert result["stake_units"] == 0.0

def test_confidence_gate_blocks_bet():
    """Test that low confidence blocks BET even with good edge."""
    result = recommend(
        model_prob=0.65,
        implied_prob=0.60,
        confidence_grade="C",  # Below B threshold
        expected_value_unit=0.05
    )
    assert result["recommendation"] != "BET"

def test_confidence_ranking():
    """Test confidence ranking order."""
    assert CONFIDENCE_RANK["A"] > CONFIDENCE_RANK["B"]
    assert CONFIDENCE_RANK["B"] > CONFIDENCE_RANK["C"]
    assert CONFIDENCE_RANK["C"] > CONFIDENCE_RANK["D"]
    assert CONFIDENCE_RANK["D"] > CONFIDENCE_RANK["F"]

def test_custom_config():
    """Test custom config overrides."""
    custom = {
        "min_edge_bet": 0.10,
        "min_ev_bet": 0.05
    }
    result = recommend(
        model_prob=0.65,
        implied_prob=0.60,
        confidence_grade="B",
        expected_value_unit=0.03,
        config=custom
    )
    # Should not BET with custom higher thresholds
    assert result["recommendation"] != "BET"
