from typing import List, Dict, Any, Tuple
from datetime import datetime

class DataQuality:
    @staticmethod
    def _check_missing(data: List[Dict[str, Any]], required_fields: List[str], resource_name: str) -> List[str]:
        issues = []
        if not data:
            return [f"Empty dataset: {resource_name}"]
            
        total = len(data)
        for field in required_fields:
            missing = sum(1 for row in data if row.get(field) is None or str(row.get(field)).strip() == "")
            if missing > 0:
                issues.append(f"{resource_name}: Field '{field}' missing in {missing}/{total} records ({missing/total:.2%})")
        return issues

    @staticmethod
    def _check_duplicates(targets: List[Dict[str, Any]]) -> List[str]:
        seen = set()
        dupes = 0
        for t in targets:
            # Key: game_id, player_id, event_type, horizon
            key = (t.get("game_id"), t.get("player_id"), t.get("event_type"), t.get("horizon", "pregame"))
            if key in seen:
                dupes += 1
            seen.add(key)
        
        if dupes > 0:
            return [f"Targets: Found {dupes} duplicate keys (game_id, player_id, event_type, horizon)"]
        return []

    @staticmethod
    def _check_labels(targets: List[Dict[str, Any]]) -> List[str]:
        invalid = 0
        for t in targets:
            try:
                act = float(t.get("actual", 0))
                if act not in (0.0, 1.0):
                    invalid += 1
            except:
                invalid += 1
        
        if invalid > 0:
            return [f"Targets: Found {invalid} invalid labels (must be 0 or 1)"]
        return []

    @staticmethod
    def _check_temporal_ordering(games: List[Dict[str, Any]]) -> List[str]:
        issues = []
        for g in games:
            gid = g.get("game_id")
            raw_date = g.get("date_utc")
            try:
                # Basic ISO format check
                if "T" in raw_date:
                    datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
                else:
                    datetime.strptime(raw_date, "%Y-%m-%d")
            except:
                issues.append(f"Games: Invalid date format in game {gid}: {raw_date}")
        return issues

def validate_dataset(players: List[Dict], teams: List[Dict], games: List[Dict], targets: List[Dict]) -> Dict[str, Any]:
    report = {
        "pass_fail": "pass",
        "failures": [],
        "counts": {
            "players": len(players),
            "teams": len(teams),
            "games": len(games),
            "targets": len(targets)
        }
    }
    
    failures = []
    
    # 1. Missing Values
    failures.extend(DataQuality._check_missing(players, ["player_id", "team_id"], "players"))
    failures.extend(DataQuality._check_missing(teams, ["team_id", "pace"], "teams"))
    failures.extend(DataQuality._check_missing(games, ["game_id", "date_utc", "home_team_id", "away_team_id"], "games"))
    failures.extend(DataQuality._check_missing(targets, ["game_id", "player_id", "event_type", "threshold", "actual"], "targets"))
    
    # 2. Key Constraints
    failures.extend(DataQuality._check_duplicates(targets))
    
    # 3. Label Validity
    failures.extend(DataQuality._check_labels(targets))
    
    # 4. Temporal Validity
    failures.extend(DataQuality._check_temporal_ordering(games))
    
    if failures:
        report["pass_fail"] = "fail"
        report["failures"] = failures
        
    return report
