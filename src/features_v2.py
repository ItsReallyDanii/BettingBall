"""
Feature engineering v2 with leak-safe design.
No target/outcome fields used at inference time.
"""
import math
from typing import Dict, Any, List, Optional

# Feature column order (deterministic)
FEATURE_COLUMNS = [
    # Market features
    "implied_prob_open",
    "implied_prob_current", 
    "line_move_abs",
    "line_move_direction",
    
    # Team context
    "pace_diff",
    "off_rating_diff",
    "def_rating_diff",
    "home_indicator",
    
    # Rest/fatigue
    "days_rest_team",
    "days_rest_opp",
    "back_to_back",
    "travel_fatigue_index",
    
    # Player form
    "points_avg_5",
    "trend_points_slope_5",
    "usage_rate_last_5",
    "true_shooting_last_5",
    
    # Threshold relation
    "proj_minus_threshold",
    
    # Interactions
    "pace_usage_interaction",
    "rest_b2b_interaction",
    "rest_advantage",
    "travel_exposure"
]

def extract_features_v2(
    joined_record: Dict[str, Any],
    feature_manifest: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """
    Extract real features from joined record.
    
    Args:
        joined_record: Odds record joined with game/player/team context
        feature_manifest: Optional manifest for tracking provenance
    
    Returns:
        Dict mapping feature names to values
    """
    game = joined_record.get("game", {})
    player = joined_record.get("player", {})
    team = joined_record.get("team", {})
    opponent_team = joined_record.get("opponent_team", {})
    
    features = {}
    
    # Market features
    features["implied_prob_open"] = _safe_float(game.get("market_open_line"), 0.5)
    features["implied_prob_current"] = _safe_float(game.get("market_current_line"), 0.5)
    
    open_line = _safe_float(game.get("market_open_line"), 0)
    current_line = _safe_float(game.get("market_current_line"), 0)
    features["line_move_abs"] = abs(current_line - open_line)
    features["line_move_direction"] = 1.0 if current_line > open_line else (-1.0 if current_line < open_line else 0.0)
    
    # Team context
    team_pace = _safe_float(team.get("pace"), 100.0)
    opp_pace = _safe_float(opponent_team.get("pace"), 100.0)
    features["pace_diff"] = team_pace - opp_pace
    
    team_off = _safe_float(team.get("off_rating"), 110.0)
    opp_off = _safe_float(opponent_team.get("off_rating"), 110.0)
    features["off_rating_diff"] = team_off - opp_off
    
    team_def = _safe_float(team.get("def_rating"), 110.0)
    opp_def = _safe_float(opponent_team.get("def_rating"), 110.0)
    features["def_rating_diff"] = team_def - opp_def
    
    features["home_indicator"] = 1.0 if _safe_bool(game.get("is_home_for_subject_team")) else 0.0
    
    # Rest/fatigue
    features["days_rest_team"] = _clip(_safe_float(game.get("days_rest_team"), 2.0), 0, 7)
    features["days_rest_opp"] = _clip(_safe_float(game.get("days_rest_opponent"), 2.0), 0, 7)
    features["back_to_back"] = 1.0 if _safe_bool(player.get("back_to_back")) else 0.0
    features["travel_fatigue_index"] = _clip(_safe_float(game.get("travel_fatigue_index"), 0.1), 0, 1)
    
    # Player form
    features["points_avg_5"] = _clip(_safe_float(player.get("points_avg_5"), 15.0), 0, 50)
    features["trend_points_slope_5"] = _clip(_safe_float(player.get("trend_points_slope_5"), 0.0), -10, 10)
    features["usage_rate_last_5"] = _clip(_safe_float(player.get("usage_rate_last_5"), 0.20), 0, 0.5)
    features["true_shooting_last_5"] = _clip(_safe_float(player.get("true_shooting_last_5"), 0.55), 0.3, 0.8)
    
    # Threshold relation
    threshold = _safe_float(joined_record.get("line"), 20.0)
    projection = features["points_avg_5"] + features["trend_points_slope_5"]
    features["proj_minus_threshold"] = projection - threshold
    
    # Interactions
    features["pace_usage_interaction"] = features["pace_diff"] * features["usage_rate_last_5"]
    features["rest_b2b_interaction"] = features["days_rest_team"] * features["back_to_back"]
    features["rest_advantage"] = features["days_rest_team"] - features["days_rest_opp"]
    features["travel_exposure"] = features["travel_fatigue_index"] * features["back_to_back"]
    
    # Track provenance if manifest provided
    if feature_manifest is not None:
        _update_manifest(feature_manifest, joined_record)
    
    return features

def features_to_vector(features: Dict[str, float]) -> List[float]:
    """Convert feature dict to ordered vector."""
    return [features.get(col, 0.0) for col in FEATURE_COLUMNS]

def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely convert to float with default."""
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def _safe_bool(val: Any) -> bool:
    """Safely convert to bool."""
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("true", "1", "yes")
    return bool(val)

def _clip(val: float, min_val: float, max_val: float) -> float:
    """Clip value to range."""
    return max(min_val, min(max_val, val))

def _update_manifest(manifest: Dict[str, Any], record: Dict[str, Any]):
    """Update feature manifest with provenance."""
    if "features" not in manifest:
        manifest["features"] = {}
    
    # Track source columns for each feature
    manifest["features"]["implied_prob_open"] = {"source": "game.market_open_line"}
    manifest["features"]["implied_prob_current"] = {"source": "game.market_current_line"}
    manifest["features"]["line_move_abs"] = {"source": "game.market_open_line, game.market_current_line", "transform": "abs_diff"}
    manifest["features"]["pace_diff"] = {"source": "team.pace, opponent_team.pace", "transform": "diff"}
    manifest["features"]["points_avg_5"] = {"source": "player.points_avg_5"}
    manifest["features"]["proj_minus_threshold"] = {"source": "player.points_avg_5, player.trend_points_slope_5, line", "transform": "projection_diff"}
    # ... (abbreviated for brevity, full manifest would list all)
