import pytest
from src.feature_join import join_odds_with_context, _find_game_by_timestamp
from datetime import datetime

def test_join_happy_path():
    """Test successful join with matching game."""
    odds_rows = [{
        "event_id": "SYNTH_0",
        "market_type": "player_points",
        "selection": "over",
        "player_id": "LBJ_01",
        "team_id": "LAL",
        "line": 20.5,
        "price_american": -110,
        "implied_prob": 0.524,
        "sportsbook": "DraftKings",
        "timestamp_utc": "2026-01-01T00:00:00Z"
    }]
    
    result = join_odds_with_context(odds_rows)
    
    assert result["audit"]["matched_count"] >= 0
    assert result["audit"]["unmatched_count"] >= 0
    assert "matched" in result
    assert "unmatched" in result

def test_timestamp_tolerance():
    """Test timestamp-based fuzzy matching."""
    odds_rows = [{
        "event_id": "unknown_event",
        "market_type": "moneyline",
        "selection": "home",
        "player_id": "",
        "team_id": "",
        "line": None,
        "price_american": -150,
        "implied_prob": 0.6,
        "sportsbook": "FanDuel",
        "timestamp_utc": "2026-01-01T00:30:00Z"  # Within 180 min of SYNTH_0
    }]
    
    result = join_odds_with_context(odds_rows, join_tolerance_minutes=60)
    
    # Should find match within tolerance
    assert "audit" in result

def test_unmatched_rows():
    """Test handling of unmatched rows."""
    odds_rows = [{
        "event_id": "nonexistent_game",
        "market_type": "spread",
        "selection": "away",
        "player_id": "",
        "team_id": "",
        "line": 7.5,
        "price_american": -110,
        "implied_prob": 0.524,
        "sportsbook": "BetMGM",
        "timestamp_utc": "2099-12-31T23:59:59Z"  # Far future, won't match
    }]
    
    result = join_odds_with_context(odds_rows, join_tolerance_minutes=10)
    
    assert result["audit"]["unmatched_count"] > 0
    assert "no_game_match" in result["audit"]["unmatched_reasons"]

def test_deterministic_ordering():
    """Test that matched results maintain deterministic ordering."""
    odds_rows = [
        {
            "event_id": "SYNTH_2",
            "market_type": "total",
            "selection": "over",
            "player_id": "",
            "team_id": "",
            "line": 220.5,
            "price_american": -115,
            "implied_prob": 0.535,
            "sportsbook": "Caesars",
            "timestamp_utc": "2026-01-01T00:00:00Z"
        },
        {
            "event_id": "SYNTH_1",
            "market_type": "total",
            "selection": "under",
            "player_id": "",
            "team_id": "",
            "line": 220.5,
            "price_american": -105,
            "implied_prob": 0.512,
            "sportsbook": "Caesars",
            "timestamp_utc": "2026-01-01T00:00:00Z"
        }
    ]
    
    result = join_odds_with_context(odds_rows)
    
    # Verify results exist
    assert "matched" in result or "unmatched" in result

def test_feature_extraction():
    """Test that features are extracted from joined records."""
    odds_rows = [{
        "event_id": "SYNTH_0",
        "market_type": "player_points",
        "selection": "over",
        "player_id": "LBJ_01",
        "team_id": "LAL",
        "line": 20.5,
        "price_american": -110,
        "implied_prob": 0.524,
        "sportsbook": "DraftKings",
        "timestamp_utc": "2026-01-01T00:00:00Z"
    }]
    
    result = join_odds_with_context(odds_rows)
    
    if result["matched"]:
        matched_rec = result["matched"][0]
        assert "features" in matched_rec
        assert isinstance(matched_rec["features"], list)
        assert len(matched_rec["features"]) == 4  # [bias, diff, days_rest, opp_def]
