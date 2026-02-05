import pytest
from src.odds_ingest import american_to_implied_prob, parse_odds_csv, parse_odds_json

def test_american_to_implied_prob_positive():
    """Test conversion of positive American odds."""
    assert abs(american_to_implied_prob(100) - 0.5) < 0.001
    assert abs(american_to_implied_prob(200) - 0.333333) < 0.001
    assert abs(american_to_implied_prob(150) - 0.4) < 0.001

def test_american_to_implied_prob_negative():
    """Test conversion of negative American odds."""
    assert abs(american_to_implied_prob(-100) - 0.5) < 0.001
    assert abs(american_to_implied_prob(-200) - 0.666667) < 0.001
    assert abs(american_to_implied_prob(-150) - 0.6) < 0.001

def test_parse_odds_csv():
    """Test CSV parsing."""
    records = parse_odds_csv("data/odds/sample_lines.csv")
    assert len(records) > 0
    assert "event_id" in records[0]
    assert "implied_prob" in records[0]
    assert "price_american" in records[0]

def test_parse_odds_json():
    """Test JSON parsing."""
    records = parse_odds_json("data/odds/sample_lines.json")
    assert len(records) > 0
    assert "event_id" in records[0]
    assert "implied_prob" in records[0]
    assert "price_american" in records[0]

def test_malformed_row_rejection():
    """Test that malformed rows are rejected."""
    import tempfile
    import os
    
    # Create temp CSV with missing required field
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        f.write("event_id,market_type\n")
        f.write("evt_001,moneyline\n")
        temp_path = f.name
    
    try:
        with pytest.raises(ValueError, match="Missing required fields"):
            parse_odds_csv(temp_path)
    finally:
        os.unlink(temp_path)

def test_zero_odds_rejection():
    """Test that zero American odds are rejected."""
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        f.write("event_id,market_type,selection,price_american,sportsbook\n")
        f.write("evt_001,moneyline,home,0,DraftKings\n")
        temp_path = f.name
    
    try:
        with pytest.raises(ValueError, match="price_american cannot be 0"):
            parse_odds_csv(temp_path)
    finally:
        os.unlink(temp_path)
