"""
Dataset builder with leak-safe temporal splits.
"""
import os
import csv
import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple

def build_dataset(
    games_csv: str = "data/raw/games.csv",
    players_csv: str = "data/raw/players.csv",
    teams_csv: str = "data/raw/teams.csv",
    targets_csv: str = "data/raw/targets.csv",
    output_dir: str = "outputs/training"
) -> Dict[str, Any]:
    """
    Build training dataset from local CSVs.
    
    Returns:
        Metadata dict with schema stats
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    games = _load_csv(games_csv)
    players = _load_csv(players_csv)
    teams = _load_csv(teams_csv)
    targets = _load_csv(targets_csv)
    
    # Index lookups
    players_by_id = {p["player_id"]: p for p in players}
    teams_by_id = {t["team_id"]: t for t in teams}
    
    # Build joined dataset
    dataset = []
    for target in targets:
        game_id = target.get("game_id")
        player_id = target.get("player_id")
        
        # Find matching game
        game = next((g for g in games if g.get("game_id") == game_id), None)
        if not game:
            continue
        
        player = players_by_id.get(player_id, {})
        
        # Get team and opponent
        home_team_id = game.get("home_team_id")
        away_team_id = game.get("away_team_id")
        is_home = game.get("is_home_for_subject_team", "").lower() == "true"
        
        team_id = home_team_id if is_home else away_team_id
        opp_team_id = away_team_id if is_home else home_team_id
        
        team = teams_by_id.get(team_id, {})
        opponent_team = teams_by_id.get(opp_team_id, {})
        
        # Build record
        record = {
            "game_id": game_id,
            "player_id": player_id,
            "timestamp_utc": game.get("date_utc", ""),
            "target": int(target.get("actual", 0)),
            "threshold": float(target.get("threshold", 0)),
            "event_type": target.get("event_type", ""),
            **{f"game_{k}": v for k, v in game.items()},
            **{f"player_{k}": v for k, v in player.items()},
            **{f"team_{k}": v for k, v in team.items()},
            **{f"opponent_{k}": v for k, v in opponent_team.items()}
        }
        
        dataset.append(record)
    
    # Sort by timestamp
    dataset.sort(key=lambda x: x["timestamp_utc"])
    
    # Schema validation
    schema_stats = _validate_schema(dataset)
    
    # Write dataset
    dataset_path = os.path.join(output_dir, "dataset.csv")
    if dataset:
        fieldnames = list(dataset[0].keys())
        with open(dataset_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(dataset)
    
    # Write metadata
    metadata = {
        "total_records": len(dataset),
        "schema_stats": schema_stats,
        "source_files": {
            "games": games_csv,
            "players": players_csv,
            "teams": teams_csv,
            "targets": targets_csv
        },
        "output_path": dataset_path,
        "timestamp_range": {
            "min": dataset[0]["timestamp_utc"] if dataset else None,
            "max": dataset[-1]["timestamp_utc"] if dataset else None
        }
    }
    
    metadata_path = os.path.join(output_dir, "dataset_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    return metadata

def temporal_split(
    dataset_path: str,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    timestamp_col: str = "timestamp_utc",
    audit_dir: str = "outputs/audits"
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Leak-safe temporal split with split audit.
    
    Returns:
        (train, val, test) lists
    """
    os.makedirs(audit_dir, exist_ok=True)
    
    # Load dataset
    with open(dataset_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        dataset = list(reader)
    
    # Already sorted by timestamp in build_dataset
    n = len(dataset)
    
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train = dataset[:train_end]
    val = dataset[train_end:val_end]
    test = dataset[val_end:]
    
    # Audit logic
    def _get_ids(data): return set(r.get("event_id", "") for r in data if r.get("event_id"))
    def _get_hashes(data): 
        import hashlib
        hashes = set()
        for r in data:
            s = json.dumps(r, sort_keys=True)
            hashes.add(hashlib.md5(s.encode()).hexdigest())
        return hashes

    train_ids = _get_ids(train)
    val_ids = _get_ids(val)
    test_ids = _get_ids(test)
    
    train_hashes = _get_hashes(train)
    val_hashes = _get_hashes(val)
    test_hashes = _get_hashes(test)
    
    overlap_ids = (train_ids & val_ids) | (val_ids & test_ids) | (train_ids & test_ids)
    overlap_hashes = (train_hashes & val_hashes) | (val_hashes & test_hashes) | (train_hashes & test_hashes)
    
    temporal_order_verified = True
    if train and val:
        if train[-1][timestamp_col] > val[0][timestamp_col]: temporal_order_verified = False
    if val and test:
        if val[-1][timestamp_col] > test[0][timestamp_col]: temporal_order_verified = False
        
    # Date ranges and class balance
    def _get_date_range(data):
        if not data: return {"min": None, "max": None}
        return {"min": data[0][timestamp_col], "max": data[-1][timestamp_col]}
    
    def _get_balance(data):
        if not data: return {"rate_1": 0.0, "rate_0": 0.0}
        # Handle both 'target' and 'actual' (for joined dataset vs raw real data)
        outcomes = []
        for r in data:
            val = r.get("target") or r.get("outcome") or r.get("actual")
            if val is not None:
                try: 
                    outcomes.append(float(val))
                except:
                    pass
        if not outcomes: return {"rate_1": 0.0, "rate_0": 0.0}
        r1 = sum(outcomes) / len(outcomes)
        return {"rate_1": round(r1, 4), "rate_0": round(1 - r1, 4)}

    audit = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "temporal_order_verified": temporal_order_verified,
        "duplicate_event_id_overlap_count": len(overlap_ids),
        "near_duplicate_hash_overlap_count": len(overlap_hashes),
        "split_sizes": {
            "train": len(train),
            "val": len(val),
            "test": len(test)
        },
        "date_ranges": {
            "train": _get_date_range(train),
            "val": _get_date_range(val),
            "test": _get_date_range(test)
        },
        "class_balance": {
            "train": _get_balance(train),
            "val": _get_balance(val),
            "test": _get_balance(test)
        }
    }
    
    audit_path = os.path.join(audit_dir, "split_audit.json")
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2)
    
    # Hard fail on overlap
    if len(overlap_ids) > 0:
        raise ValueError(f"LEAKAGE DETECTED: {len(overlap_ids)} event_id overlaps across splits")
    if len(overlap_hashes) > 0:
        raise ValueError(f"LEAKAGE DETECTED: {len(overlap_hashes)} near-duplicate hash overlaps")
    if not temporal_order_verified:
        raise ValueError("LEAKAGE DETECTED: Temporal order violated across splits")
        
    return train, val, test

def rolling_fold_generator(
    dataset_path: str,
    n_folds: int = 5,
    min_train_size: int = 100,
    timestamp_col: str = "timestamp_utc"
):
    """
    Generate rolling folds for walk-forward validation.
    
    Yields:
        (fold_idx, train, val) tuples
    """
    with open(dataset_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        dataset = list(reader)
    
    n = len(dataset)
    fold_size = (n - min_train_size) // n_folds
    
    for i in range(n_folds):
        train_end = min_train_size + i * fold_size
        val_end = train_end + fold_size
        
        if val_end > n:
            break
        
        train = dataset[:train_end]
        val = dataset[train_end:val_end]
        
        # Leakage guard
        if train and val:
            last_train_ts = train[-1][timestamp_col]
            first_val_ts = val[0][timestamp_col]
            
            if last_train_ts >= first_val_ts:
                raise ValueError(f"LEAKAGE in fold {i}: last train {last_train_ts} >= first val {first_val_ts}")
        
        yield i, train, val

def _load_csv(path: str) -> List[Dict[str, Any]]:
    """Load CSV file."""
    if not os.path.exists(path):
        return []
    
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)

def _validate_schema(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate schema and compute stats."""
    if not dataset:
        return {"error": "empty dataset"}
    
    stats = {}
    fieldnames = list(dataset[0].keys())
    
    for field in fieldnames:
        null_count = sum(1 for r in dataset if not r.get(field) or r.get(field) == "")
        
        # Try to infer type
        non_null_vals = [r.get(field) for r in dataset if r.get(field) and r.get(field) != ""]
        
        if non_null_vals:
            sample = non_null_vals[0]
            try:
                float(sample)
                dtype = "numeric"
            except:
                dtype = "string"
        else:
            dtype = "unknown"
        
        stats[field] = {
            "null_count": null_count,
            "null_pct": round(null_count / len(dataset), 4),
            "dtype": dtype
        }
    
    return stats
