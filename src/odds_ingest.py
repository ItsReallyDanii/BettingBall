import csv
import json
import os
from typing import List, Dict, Any
from datetime import datetime, timezone

REQUIRED_FIELDS = ["event_id", "market_type", "selection", "price_american", "sportsbook"]

def american_to_implied_prob(price_american: int) -> float:
    """Convert American odds to implied probability."""
    if price_american > 0:
        return 100 / (price_american + 100)
    else:
        return abs(price_american) / (abs(price_american) + 100)

def parse_odds_csv(path: str) -> List[Dict[str, Any]]:
    """Parse odds from CSV file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Odds file not found: {path}")
    
    records = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            try:
                record = _normalize_odds_record(row, idx)
                records.append(record)
            except ValueError as e:
                raise ValueError(f"Row {idx}: {e}")
    
    return records

def parse_odds_json(path: str) -> List[Dict[str, Any]]:
    """Parse odds from JSON file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Odds file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("JSON must contain a list of odds records")
    
    records = []
    for idx, row in enumerate(data, start=1):
        try:
            record = _normalize_odds_record(row, idx)
            records.append(record)
        except ValueError as e:
            raise ValueError(f"Record {idx}: {e}")
    
    return records

def _normalize_odds_record(row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Normalize a single odds record with strict validation."""
    # Check required fields
    missing = [f for f in REQUIRED_FIELDS if f not in row or not row[f]]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")
    
    # Parse price_american
    try:
        price_american = int(row["price_american"])
    except (ValueError, TypeError):
        raise ValueError(f"Invalid price_american: {row.get('price_american')}")
    
    if price_american == 0:
        raise ValueError("price_american cannot be 0")
    
    # Calculate implied probability
    implied_prob = american_to_implied_prob(price_american)
    
    # Build normalized record
    record = {
        "event_id": str(row["event_id"]),
        "market_type": str(row["market_type"]),
        "selection": str(row["selection"]),
        "player_id": str(row.get("player_id", "")),
        "team_id": str(row.get("team_id", "")),
        "line": float(row["line"]) if row.get("line") else None,
        "price_american": price_american,
        "implied_prob": round(implied_prob, 6),
        "sportsbook": str(row["sportsbook"]),
        "timestamp_utc": row.get("timestamp_utc", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    }
    
    return record
