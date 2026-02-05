import os
import csv
import json
from datetime import datetime, timezone, timedelta
import hashlib

class DataConnector:
    def __init__(self):
        self.report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "local",
            "schema_pass": True,
            "records_written": {
                "teams": {"count": 0, "errors": 0},
                "players": {"count": 0, "errors": 0},
                "games": {"count": 0, "errors": 0},
                "targets": {"count": 0, "errors": 0}
            },
            "errors": []
        }
        self.schemas = {
            "teams": ["team_id", "pace", "off_rating", "def_rating", "rotation_stability", "net_rating_last_10", "turnover_pct", "rebound_pct", "three_rate", "rim_freq", "bench_minutes_share"],
            "players": ["player_id", "team_id", "position", "minutes_last_3", "minutes_last_5", "usage_rate_last_5", "true_shooting_last_5", "assist_rate_last_5", "rebound_chances_last_5", "turnover_rate_last_5", "points_avg_5", "trend_points_slope_5", "trend_ts_slope_5", "days_rest", "back_to_back", "travel_distance_km_recent", "injury_status", "probable_minutes_cap"],
            "games": ["game_id", "date_utc", "home_team_id", "away_team_id", "is_home_for_subject_team", "days_rest_team", "days_rest_opponent", "head_to_head_last_3", "defender_matchup_index", "ref_crew_foul_tendency", "travel_fatigue_index", "market_open_line", "market_current_line", "line_move_abs"],
            "targets": ["game_id", "player_id", "event_type", "threshold", "horizon", "actual"]
        }

    def _validate_row(self, schema_key, row):
        expected = self.schemas[schema_key]
        if not all(k in row for k in expected):
            self.report["schema_pass"] = False
            return False
        return True

    def ingest(self, require_real: bool = False, min_samples: int = 300):
        os.makedirs("data/raw", exist_ok=True)
        os.makedirs("outputs/audits", exist_ok=True)
        
        self.report["min_samples_required"] = min_samples
        success = self._load_from_local()

        # Strict mode: no synthetic fallback
        if require_real:
            if not success:
                self.report["data_mode"] = "missing"
                self.report["errors"].append("require_real=True but local data files missing or invalid")
                with open("outputs/audits/ingest_report.json", "w") as f:
                    json.dump(self.report, f, indent=2)
                return False
            
            # Validate sample size
            total_samples = self.report["records_written"]["games"]["count"]
            self.report["sample_size_observed"] = total_samples
            if total_samples < min_samples:
                self.report["data_mode"] = "real"
                self.report["errors"].append(f"Insufficient samples: {total_samples} < {min_samples}")
                with open("outputs/audits/ingest_report.json", "w") as f:
                    json.dump(self.report, f, indent=2)
                return False
            
            self.report["data_mode"] = "real"
        else:
            # Permissive mode: synthetic fallback allowed
            if not success:
                print("⚠️ No data found in data/raw/. Generating synthetic data...")
                success = self._generate_synthetic_data(100)
                self.report["data_mode"] = "synthetic"
                self.report["sample_size_observed"] = self.report["records_written"]["games"]["count"]
            else:
                self.report["data_mode"] = "real"
                self.report["sample_size_observed"] = self.report["records_written"]["games"]["count"]

        with open("outputs/audits/ingest_report.json", "w") as f:
            json.dump(self.report, f, indent=2)
        return success

    def _load_from_local(self):
        """Load data from existing CSV files in data/raw/"""
        data_dir = "data/raw"
        required_files = ["teams.csv", "players.csv", "games.csv", "targets.csv"]

        # Check if all required files exist
        for fname in required_files:
            if not os.path.exists(os.path.join(data_dir, fname)):
                return False

        # Load teams
        try:
            with open(os.path.join(data_dir, "teams.csv"), "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if self._validate_row("teams", row):
                        self.report["records_written"]["teams"]["count"] += 1
                    else:
                        self.report["records_written"]["teams"]["errors"] += 1
        except Exception as e:
            self.report["errors"].append(f"Failed to load teams.csv: {str(e)}")
            return False

        # Load players
        try:
            with open(os.path.join(data_dir, "players.csv"), "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if self._validate_row("players", row):
                        self.report["records_written"]["players"]["count"] += 1
                    else:
                        self.report["records_written"]["players"]["errors"] += 1
        except Exception as e:
            self.report["errors"].append(f"Failed to load players.csv: {str(e)}")
            return False

        # Load games
        try:
            with open(os.path.join(data_dir, "games.csv"), "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if self._validate_row("games", row):
                        self.report["records_written"]["games"]["count"] += 1
                    else:
                        self.report["records_written"]["games"]["errors"] += 1
        except Exception as e:
            self.report["errors"].append(f"Failed to load games.csv: {str(e)}")
            return False

        # Load targets
        try:
            with open(os.path.join(data_dir, "targets.csv"), "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if self._validate_row("targets", row):
                        self.report["records_written"]["targets"]["count"] += 1
                    else:
                        self.report["records_written"]["targets"]["errors"] += 1
        except Exception as e:
            self.report["errors"].append(f"Failed to load targets.csv: {str(e)}")
            return False

        # Verify we have data
        total_records = sum(self.report["records_written"][k]["count"] for k in self.report["records_written"])
        if total_records == 0:
            self.report["errors"].append("No valid records found in local data files")
            return False

        return True

    def _generate_synthetic_data(self, n=400):
        with open("data/raw/teams.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.schemas["teams"])
            w.writeheader()
            rows = [
                {"team_id":"LAL","pace":101.2,"off_rating":115.4,"def_rating":114.2,"rotation_stability":0.85,"net_rating_last_10":1.2,"turnover_pct":0.13,"rebound_pct":0.51,"three_rate":0.38,"rim_freq":0.28,"bench_minutes_share":0.32},
                {"team_id":"BOS","pace":98.5,"off_rating":122.1,"def_rating":110.5,"rotation_stability":0.92,"net_rating_last_10":11.6,"turnover_pct":0.12,"rebound_pct":0.53,"three_rate":0.45,"rim_freq":0.22,"bench_minutes_share":0.28}
            ]
            for r in rows:
                if self._validate_row("teams", r):
                    w.writerow(r)
                    self.report["records_written"]["teams"]["count"] += 1

        with open("data/raw/players.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.schemas["players"])
            w.writeheader()
            r = {"player_id":"LBJ_01","team_id":"LAL","position":"F","minutes_last_3":36.5,"minutes_last_5":35.2,"usage_rate_last_5":0.29,"true_shooting_last_5":0.58,"assist_rate_last_5":0.32,"rebound_chances_last_5":12.5,"turnover_rate_last_5":0.14,"points_avg_5":25.4,"trend_points_slope_5":0.5,"trend_ts_slope_5":-0.02,"days_rest":2,"back_to_back":"False","travel_distance_km_recent":0,"injury_status":"healthy","probable_minutes_cap":""}
            if self._validate_row("players", r):
                w.writerow(r)
                self.report["records_written"]["players"]["count"] += 1

        games, targets = [], []
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for i in range(n):
            gid = f"SYNTH_{i}"
            g_row = {"game_id": gid, "date_utc": (base + timedelta(days=i//10)).isoformat().replace("+00:00", "Z"), "home_team_id": "LAL", "away_team_id": "BOS", "is_home_for_subject_team": "True", "days_rest_team": 2, "days_rest_opponent": 2, "head_to_head_last_3": 1.5, "defender_matchup_index": 0.5, "ref_crew_foul_tendency": 0.1, "travel_fatigue_index": 0.1, "market_open_line": 230, "market_current_line": 232, "line_move_abs": 2}
            self._validate_row("games", g_row)
            games.append(g_row)
            
            # Create correlated signal for model to learn
            # Player avg is 25.4 (from players dict below/above). 
            # We vary threshold to create varying 'diff' feature.
            thresh = 20.5 + (i % 10) # 20.5 to 29.5
            proj = 25.4 + 0.5 # avg + slope
            diff = proj - thresh
            
            # Target probability (Sigmoid-like relationship)
            import math
            prob = 1 / (1 + math.exp(-diff))
            
            # Deterministic outcome sampling
            h = int(hashlib.md5(gid.encode()).hexdigest(), 16)
            h_norm = (h % 10000) / 10000.0
            actual = 1 if h_norm < prob else 0

            t_row = {"game_id": gid, "player_id": "LBJ_01", "event_type": "player_points_over", "threshold": thresh, "horizon": "pregame", "actual": actual}
            self._validate_row("targets", t_row)
            targets.append(t_row)

        with open("data/raw/games.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.schemas["games"]); w.writeheader(); w.writerows(games)
            self.report["records_written"]["games"]["count"] = len(games)
        with open("data/raw/targets.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.schemas["targets"]); w.writeheader(); w.writerows(targets)
            self.report["records_written"]["targets"]["count"] = len(targets)
        return True
