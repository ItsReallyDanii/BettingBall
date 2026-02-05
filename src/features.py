"""
Feature extraction for NBA betting predictions.

Loads raw CSV data and extracts comprehensive feature vectors including:
- Injury/availability features
- Home/away context
- Rest/fatigue metrics
- Pace and tempo
- Matchup dynamics
- Usage trends
- Shooting efficiency
"""

import csv
import os
from typing import Dict, List, Any, Optional
from pathlib import Path


class FeatureExtractor:
    """Comprehensive feature extraction from raw NBA data."""

    def __init__(self, data_dir: str = "data/raw"):
        """Initialize feature extractor with data directory."""
        self.data_dir = Path(data_dir)
        self.players = {}
        self.teams = {}
        self.games = {}
        self.targets = {}
        self._load_data()

    def _load_data(self):
        """Load all CSV data into memory."""
        # Load players
        players_path = self.data_dir / "players.csv"
        if players_path.exists():
            with open(players_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.players[row['player_id']] = row

        # Load teams
        teams_path = self.data_dir / "teams.csv"
        if teams_path.exists():
            with open(teams_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.teams[row['team_id']] = row

        # Load games
        games_path = self.data_dir / "games.csv"
        if games_path.exists():
            with open(games_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.games[row['game_id']] = row

        # Load targets
        targets_path = self.data_dir / "targets.csv"
        if targets_path.exists():
            with open(targets_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = (row['game_id'], row['player_id'], row['event_type'])
                    self.targets[key] = row

    def extract_from_csv(self, game_id: str, player_id: str, event_type: str = "player_points_over") -> List[float]:
        """
        Extract feature vector from CSV data.

        Args:
            game_id: Game identifier
            player_id: Player identifier
            event_type: Type of betting event

        Returns:
            Feature vector as list of floats
        """
        if game_id not in self.games or player_id not in self.players:
            return self._default_features()

        game = self.games[game_id]
        player = self.players[player_id]
        player_team = self.teams.get(player['team_id'], {})

        # Determine opponent team
        is_home = player['team_id'] == game['home_team_id']
        opponent_team_id = game['away_team_id'] if is_home else game['home_team_id']
        opponent_team = self.teams.get(opponent_team_id, {})

        features = []

        # 1. BIAS TERM
        features.append(1.0)

        # 2. INJURY/AVAILABILITY FEATURES (3 features)
        injury_status = player.get('injury_status', 'healthy')
        injury_encoding = {
            'healthy': 1.0,
            'questionable': 0.7,
            'doubtful': 0.3,
            'out': 0.0
        }
        features.append(injury_encoding.get(injury_status, 0.5))

        minutes_cap = float(player.get('probable_minutes_cap', 36.0) or 36.0)
        features.append(minutes_cap / 36.0)  # Normalized minutes availability

        injury_risk = 1.0 - injury_encoding.get(injury_status, 0.5)
        features.append(injury_risk)

        # 3. HOME/AWAY FEATURES (2 features)
        features.append(1.0 if is_home else 0.0)  # Home indicator
        features.append(float(game.get('travel_fatigue_index', 0.0)))

        # 4. REST/FATIGUE FEATURES (3 features)
        days_rest = float(player.get('days_rest', 2.0))
        features.append(days_rest)

        back_to_back = player.get('back_to_back', 'false').lower() == 'true'
        features.append(1.0 if back_to_back else 0.0)

        travel_km = float(player.get('travel_distance_km_recent', 0.0))
        features.append(travel_km / 1500.0)  # Normalized travel distance

        # 5. PACE FEATURES (3 features)
        team_pace = float(player_team.get('pace', 100.0))
        opp_pace = float(opponent_team.get('pace', 100.0))
        features.append(team_pace / 100.0)  # Normalized team pace
        features.append(opp_pace / 100.0)  # Normalized opponent pace
        features.append((team_pace + opp_pace) / 200.0)  # Combined pace expectation

        # 6. MATCHUP DYNAMICS (5 features)
        team_off_rating = float(player_team.get('off_rating', 110.0))
        opp_def_rating = float(opponent_team.get('def_rating', 110.0))
        features.append(team_off_rating / 110.0)
        features.append(opp_def_rating / 110.0)

        matchup_advantage = team_off_rating - opp_def_rating
        features.append(matchup_advantage / 10.0)  # Normalized advantage

        defender_index = float(game.get('defender_matchup_index', 0.8))
        features.append(defender_index)

        head_to_head = float(game.get('head_to_head_home_last_3', 0.5))
        features.append(head_to_head)

        # 7. USAGE TRENDS (4 features)
        points_avg = float(player.get('points_avg_5', 20.0))
        features.append(points_avg / 30.0)  # Normalized recent scoring

        trend_slope = float(player.get('trend_points_slope_5', 0.0))
        features.append(trend_slope)  # Trend momentum

        ts_slope = float(player.get('trend_ts_slope_5', 0.0))
        features.append(ts_slope)  # True shooting trend

        # Usage estimate (based on team pace and player scoring)
        usage_estimate = (points_avg / team_pace) if team_pace > 0 else 0.2
        features.append(usage_estimate)

        # 8. SHOOTING EFFICIENCY (4 features)
        team_three_rate = float(player_team.get('three_rate', 35.0))
        team_rim_freq = float(player_team.get('rim_freq', 28.0))
        features.append(team_three_rate / 50.0)
        features.append(team_rim_freq / 40.0)

        opp_def_three_rate = float(opponent_team.get('three_rate', 35.0))
        opp_def_rim_freq = float(opponent_team.get('rim_freq', 28.0))
        features.append(opp_def_three_rate / 50.0)
        features.append(opp_def_rim_freq / 40.0)

        # 9. CONTEXTUAL FACTORS (4 features)
        rotation_stability = float(player_team.get('rotation_stability', 0.75))
        features.append(rotation_stability)

        bench_share = float(player_team.get('bench_minutes_share', 0.3))
        features.append(bench_share)

        ref_foul_tendency = float(game.get('ref_crew_foul_tendency', 1.0))
        features.append(ref_foul_tendency)

        net_rating_10 = float(player_team.get('net_rating_last_10', 0.0))
        features.append(net_rating_10 / 10.0)  # Normalized recent performance

        # 10. MARKET/LINE FEATURES (3 features)
        line_move = float(game.get('line_move_abs', 0.0))
        features.append(line_move)

        current_line = float(game.get('market_current_line', 0.0))
        features.append(current_line / 10.0)

        # Get threshold if target exists
        target_key = (game_id, player_id, event_type)
        threshold = 0.0
        if target_key in self.targets:
            threshold = float(self.targets[target_key].get('threshold', 0.0))

        # Projection vs threshold differential
        projection = points_avg + trend_slope * 2.0  # Simple projection
        differential = projection - threshold if threshold > 0 else 0.0
        features.append(differential / 10.0)

        return features

    def extract_from_pydantic(self, inp: Any) -> List[float]:
        """
        Extract features from Pydantic PredictionInput object.

        This method maintains compatibility with the existing model interface
        while providing richer features when available.

        Args:
            inp: PredictionInput Pydantic object or dict

        Returns:
            Feature vector as list of floats
        """
        # Access attributes from Pydantic objects or dicts
        def get_attr(obj, path, default=0.0):
            try:
                val = obj
                for p in path.split("."):
                    if isinstance(val, dict):
                        val = val.get(p)
                    else:
                        val = getattr(val, p, None)
                return float(val) if val is not None else default
            except:
                return default

        features = []

        # 1. Bias term
        features.append(1.0)

        # 2. Injury/availability (3 features)
        injury_status = get_attr(inp, "player.availability.injury_status", "healthy")
        if isinstance(injury_status, str):
            injury_encoding = {'healthy': 1.0, 'questionable': 0.7, 'doubtful': 0.3, 'out': 0.0}
            injury_score = injury_encoding.get(injury_status, 0.5)
        else:
            injury_score = 0.5
        features.append(injury_score)

        minutes_cap = get_attr(inp, "player.availability.probable_minutes_cap", 36.0)
        features.append(minutes_cap / 36.0)
        features.append(1.0 - injury_score)  # Injury risk

        # 3. Home/away (2 features)
        is_home = get_attr(inp, "game_context.is_home_for_subject_team", 0.0)
        features.append(is_home)
        features.append(get_attr(inp, "game_context.travel_fatigue_index", 0.0))

        # 4. Rest/fatigue (3 features)
        days_rest = get_attr(inp, "player.workload.days_rest", 2.0)
        features.append(days_rest)
        back_to_back_val = get_attr(inp, "player.workload.back_to_back", 0.0)
        if isinstance(back_to_back_val, bool):
            back_to_back_val = 1.0 if back_to_back_val else 0.0
        features.append(back_to_back_val)
        features.append(get_attr(inp, "player.workload.travel_distance_km_recent", 0.0) / 1500.0)

        # 5. Pace (3 features)
        team_pace = get_attr(inp, "team.pace", 100.0)
        opp_pace = get_attr(inp, "opponent_team.pace", 100.0)
        features.append(team_pace / 100.0)
        features.append(opp_pace / 100.0)
        features.append((team_pace + opp_pace) / 200.0)

        # 6. Matchup dynamics (5 features)
        team_off = get_attr(inp, "team.off_rating", 110.0)
        opp_def = get_attr(inp, "opponent_team.def_rating", 110.0)
        features.append(team_off / 110.0)
        features.append(opp_def / 110.0)
        features.append((team_off - opp_def) / 10.0)
        features.append(get_attr(inp, "game_context.defender_matchup_index", 0.8))
        features.append(get_attr(inp, "game_context.head_to_head_last_3", 0.5))

        # 7. Usage trends (4 features)
        points_avg = get_attr(inp, "player.recent_form.points_avg_5", 20.0)
        features.append(points_avg / 30.0)
        trend_slope = get_attr(inp, "player.recent_form.trend_points_slope_5", 0.0)
        features.append(trend_slope)
        features.append(get_attr(inp, "player.recent_form.trend_ts_slope_5", 0.0))
        usage_est = (points_avg / team_pace) if team_pace > 0 else 0.2
        features.append(usage_est)

        # 8. Shooting efficiency (4 features)
        features.append(get_attr(inp, "team.three_rate", 35.0) / 50.0)
        features.append(get_attr(inp, "team.rim_freq", 28.0) / 40.0)
        features.append(get_attr(inp, "opponent_team.three_rate", 35.0) / 50.0)
        features.append(get_attr(inp, "opponent_team.rim_freq", 28.0) / 40.0)

        # 9. Contextual (4 features)
        features.append(get_attr(inp, "team.rotation_stability", 0.75))
        features.append(get_attr(inp, "team.bench_minutes_share", 0.3))
        features.append(get_attr(inp, "game_context.ref_crew_foul_tendency", 1.0))
        features.append(get_attr(inp, "team.net_rating_last_10", 0.0) / 10.0)

        # 10. Market/line (3 features)
        features.append(get_attr(inp, "game_context.spread_context.line_move_abs", 0.0))
        features.append(get_attr(inp, "game_context.spread_context.market_current_line", 0.0) / 10.0)

        threshold = get_attr(inp, "target.threshold", 0.0)
        projection = points_avg + trend_slope * 2.0
        differential = projection - threshold if threshold > 0 else 0.0
        features.append(differential / 10.0)

        return features

    def _default_features(self) -> List[float]:
        """Return default feature vector when data is missing."""
        # Return neutral/default values for all 32 features
        return [
            1.0,  # Bias
            0.5, 0.5, 0.5,  # Injury/availability
            0.5, 0.0,  # Home/away
            2.0, 0.0, 0.0,  # Rest/fatigue
            1.0, 1.0, 1.0,  # Pace
            1.0, 1.0, 0.0, 0.8, 0.5,  # Matchup
            0.67, 0.0, 0.0, 0.2,  # Usage trends
            0.7, 0.7, 0.7, 0.7,  # Shooting
            0.75, 0.3, 1.0, 0.0,  # Contextual
            0.0, 0.0, 0.0  # Market/line
        ]

    def get_feature_names(self) -> List[str]:
        """Return human-readable feature names."""
        return [
            "bias",
            "injury_health_score", "minutes_availability", "injury_risk",
            "is_home", "travel_fatigue",
            "days_rest", "back_to_back", "travel_distance_norm",
            "team_pace_norm", "opp_pace_norm", "combined_pace",
            "team_off_rating_norm", "opp_def_rating_norm", "matchup_advantage",
            "defender_matchup_index", "head_to_head",
            "points_avg_norm", "trend_slope", "ts_trend_slope", "usage_estimate",
            "team_three_rate", "team_rim_freq", "opp_three_rate", "opp_rim_freq",
            "rotation_stability", "bench_share", "ref_foul_tendency", "net_rating_recent",
            "line_move", "current_line_norm", "projection_vs_threshold"
        ]

    def get_target(self, game_id: str, player_id: str, event_type: str = "player_points_over") -> Optional[int]:
        """
        Get actual outcome for a prediction.

        Args:
            game_id: Game identifier
            player_id: Player identifier
            event_type: Type of betting event

        Returns:
            1 if bet hit, 0 if missed, None if not found
        """
        key = (game_id, player_id, event_type)
        if key in self.targets:
            return int(self.targets[key]['outcome'])
        return None

    def get_all_samples(self) -> List[Dict[str, Any]]:
        """
        Get all available training samples.

        Returns:
            List of dicts with keys: game_id, player_id, event_type, features, target
        """
        samples = []
        for (game_id, player_id, event_type), target_data in self.targets.items():
            features = self.extract_from_csv(game_id, player_id, event_type)
            target = int(target_data['outcome'])
            samples.append({
                'game_id': game_id,
                'player_id': player_id,
                'event_type': event_type,
                'features': features,
                'target': target,
                'date': self.games.get(game_id, {}).get('date_utc', '')
            })
        return samples
