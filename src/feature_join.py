import csv
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

def join_odds_with_context(
    odds_rows: List[Dict[str, Any]],
    players_csv: str = "data/raw/players.csv",
    teams_csv: str = "data/raw/teams.csv",
    games_csv: str = "data/raw/games.csv",
    targets_csv: Optional[str] = "data/raw/targets.csv",
    join_tolerance_minutes: int = 180
) -> Dict[str, Any]:
    """
    Join odds records with game/player/team context.
    
    Returns:
        {
            "matched": list[dict],
            "unmatched": list[dict],
            "audit": {
                "matched_count": int,
                "unmatched_count": int,
                "unmatched_reasons": dict
            }
        }
    """
    # Load context data
    games = _load_csv(games_csv)
    players = _load_csv(players_csv) if os.path.exists(players_csv) else []
    teams = _load_csv(teams_csv) if os.path.exists(teams_csv) else []
    targets = _load_csv(targets_csv) if targets_csv and os.path.exists(targets_csv) else []
    
    # Index by keys
    games_by_id = {g["game_id"]: g for g in games}
    players_by_id = {p["player_id"]: p for p in players} if players else {}
    teams_by_id = {t["team_id"]: t for t in teams} if teams else {}
    
    # Index targets by game_id + player_id
    targets_index = {}
    for t in targets:
        key = (t.get("game_id", ""), t.get("player_id", ""))
        if key not in targets_index:
            targets_index[key] = []
        targets_index[key].append(t)
    
    matched = []
    unmatched = []
    unmatched_reasons = {}
    
    for odds in odds_rows:
        result = _join_single_odds(
            odds, games_by_id, players_by_id, teams_by_id, targets_index, join_tolerance_minutes
        )
        
        if result["match_status"] == "matched":
            matched.append(result["record"])
        else:
            unmatched.append(result["record"])
            reason = result.get("reason", "unknown")
            unmatched_reasons[reason] = unmatched_reasons.get(reason, 0) + 1
    
    audit = {
        "matched_count": len(matched),
        "unmatched_count": len(unmatched),
        "unmatched_reasons": unmatched_reasons
    }
    
    return {
        "matched": matched,
        "unmatched": unmatched,
        "audit": audit
    }

def _join_single_odds(
    odds: Dict[str, Any],
    games_by_id: Dict[str, Dict],
    players_by_id: Dict[str, Dict],
    teams_by_id: Dict[str, Dict],
    targets_index: Dict[tuple, List[Dict]],
    tolerance_minutes: int
) -> Dict[str, Any]:
    """Join a single odds record with context."""
    event_id = odds.get("event_id", "")
    player_id = odds.get("player_id", "")
    team_id = odds.get("team_id", "")
    
    # Try to match game by event_id
    game = games_by_id.get(event_id)
    
    if not game:
        # Try timestamp-based fuzzy match
        game = _find_game_by_timestamp(odds, list(games_by_id.values()), tolerance_minutes)
    
    if not game:
        return {
            "match_status": "unmatched",
            "reason": "no_game_match",
            "record": {**odds, "match_status": "unmatched", "join_keys_used": []}
        }
    
    # Build joined record
    joined = {**odds, "game": game}
    join_keys = ["game_id"]
    
    # Add player if available
    if player_id and player_id in players_by_id:
        joined["player"] = players_by_id[player_id]
        join_keys.append("player_id")
    
    # Add team if available
    if team_id and team_id in teams_by_id:
        joined["team"] = teams_by_id[team_id]
        join_keys.append("team_id")
    
    # Add target if available
    target_key = (game.get("game_id", ""), player_id)
    if target_key in targets_index:
        # Find closest matching target by threshold
        line = odds.get("line")
        if line is not None:
            targets = targets_index[target_key]
            closest = min(targets, key=lambda t: abs(float(t.get("threshold", 0)) - line))
            joined["target"] = closest
            join_keys.append("target")
    
    # Extract features for model
    joined["features"] = _extract_features_from_joined(joined)
    joined["match_status"] = "matched"
    joined["join_keys_used"] = join_keys
    joined["missing_fields"] = _check_missing_fields(joined)
    
    return {
        "match_status": "matched",
        "record": joined
    }

def _find_game_by_timestamp(
    odds: Dict[str, Any],
    games: List[Dict[str, Any]],
    tolerance_minutes: int
) -> Optional[Dict[str, Any]]:
    """Find game by timestamp proximity."""
    odds_ts_str = odds.get("timestamp_utc", "")
    if not odds_ts_str:
        return None
    
    try:
        odds_ts = datetime.fromisoformat(odds_ts_str.replace("Z", "+00:00"))
    except:
        return None
    
    tolerance = timedelta(minutes=tolerance_minutes)
    candidates = []
    
    for game in games:
        game_ts_str = game.get("date_utc", "")
        if not game_ts_str:
            continue
        
        try:
            game_ts = datetime.fromisoformat(game_ts_str.replace("Z", "+00:00"))
            delta = abs(game_ts - odds_ts)
            if delta <= tolerance:
                candidates.append((delta, game))
        except:
            continue
    
    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]
    
    return None

def _extract_features_from_joined(joined: Dict[str, Any]) -> List[float]:
    """Extract model features from joined record."""
    game = joined.get("game", {})
    player = joined.get("player", {})
    target = joined.get("target", {})
    
    # Feature 1: Projection vs Threshold differential
    # Handle flat CSV structure - keys are direct attributes
    points_avg = 0.0
    if player:
        points_avg_val = player.get("points_avg_5", 0)
        try:
            points_avg = float(points_avg_val) if points_avg_val else 0.0
        except (ValueError, TypeError):
            points_avg = 0.0
    
    trend = 0.0
    if player:
        trend_val = player.get("trend_points_slope_5", 0)
        try:
            trend = float(trend_val) if trend_val else 0.0
        except (ValueError, TypeError):
            trend = 0.0
    
    threshold = 0.0
    if target:
        threshold_val = target.get("threshold", joined.get("line", 0))
        try:
            threshold = float(threshold_val) if threshold_val else 0.0
        except (ValueError, TypeError):
            threshold = 0.0
    elif joined.get("line") is not None:
        try:
            threshold = float(joined.get("line", 0))
        except (ValueError, TypeError):
            threshold = 0.0
    
    proj = points_avg + trend
    diff = proj - threshold
    
    # Feature 2: Days rest
    days_rest = 2.0  # default
    if game:
        days_rest_val = game.get("days_rest_team", 2)
        try:
            days_rest = float(days_rest_val) if days_rest_val else 2.0
        except (ValueError, TypeError):
            days_rest = 2.0
    
    # Feature 3: Opponent defense
    opp_def = 100.0  # default
    if game:
        def_val = game.get("defender_matchup_index", 0.5)
        try:
            def_idx = float(def_val) if def_val else 0.5
            opp_def = def_idx * 100 + 100
        except (ValueError, TypeError):
            opp_def = 100.0
    
    return [1.0, diff, days_rest, opp_def]

def _check_missing_fields(joined: Dict[str, Any]) -> List[str]:
    """Check for missing critical fields."""
    missing = []
    
    if "player" not in joined:
        missing.append("player")
    if "target" not in joined:
        missing.append("target")
    
    return missing

def _load_csv(path: str) -> List[Dict[str, Any]]:
    """Load CSV file into list of dicts."""
    if not os.path.exists(path):
        return []
    
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)
