"""
Real data ingestion and normalization for training pipeline.
"""
import os
import csv
import json
import logging
from typing import List, Dict, Any, Tuple
from datetime import datetime, timezone

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "game_id", "player_id", "date_utc", "target", "threshold", 
    "event_type", "is_home", "team_pace", "team_off_rating", 
    "team_def_rating", "opp_pace", "opp_off_rating", "opp_def_rating",
    "points_avg_5", "trend_points_slope_5", "days_rest", "back_to_back"
]

def ingest_real_data(
    input_dir: str = "data/real",
    output_dir: str = "outputs/training",
    audit_dir: str = "outputs/audits"
) -> Dict[str, Any]:
    """
    Ingest real historical data from local files.
    
    Expected format: one or more CSVs in input_dir with columns matching REQUIRED_COLUMNS.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(audit_dir, exist_ok=True)
    
    all_records = []
    rejected_rows = []
    
    if not os.path.exists(input_dir):
        logger.warning(f"Input directory {input_dir} does not exist.")
        # Create a report even if no data to avoid downstream crashes
        return _write_report(0, 0, [], audit_dir)

    for filename in os.listdir(input_dir):
        if not filename.endswith(".csv"):
            continue
            
        path = os.path.join(input_dir, filename)
        logger.info(f"Processing {path}...")
        
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row_idx, row in enumerate(reader):
                    valid, reason = _validate_row(row)
                    if valid:
                        normalized = _normalize_row(row)
                        all_records.append(normalized)
                    else:
                        rejected_rows.append({
                            "file": filename,
                            "row": row_idx + 2,
                            "reason": reason,
                            "data": row
                        })
        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")
                    
    # Sort by timestamp
    all_records.sort(key=lambda x: x["timestamp_utc"])
    
    # Save dataset
    output_path = os.path.join(output_dir, "real_dataset.csv")
    if all_records:
        fieldnames = list(all_records[0].keys())
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_records)
            
    return _write_report(len(all_records), len(rejected_rows), rejected_rows, audit_dir, output_path)

def _validate_row(row: Dict[str, Any]) -> Tuple[bool, str]:
    for col in REQUIRED_COLUMNS:
        if col not in row or row[col] is None or str(row[col]).strip() == "":
            return False, f"Missing required column: {col}"
    
    # Basic value checks
    try:
        float(row["target"])
        float(row["threshold"])
        # date_utc check (ISO format)
        dt_str = row["date_utc"].replace("Z", "+00:00")
        datetime.fromisoformat(dt_str)
    except Exception as e:
        return False, f"Invalid value format: {e}"
        
    return True, ""

def _normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Map real data columns to the internal flattened schema."""
    return {
        "game_id": row["game_id"],
        "player_id": row["player_id"],
        "timestamp_utc": row["date_utc"],
        "target": int(float(row["target"])),
        "threshold": float(row["threshold"]),
        "event_type": row["event_type"],
        
        # Flattened features matching dataset.py prefix convention for reconstruction
        "game_date_utc": row["date_utc"],
        "game_is_home_for_subject_team": row["is_home"],
        "game_market_open_line": row.get("market_open_line", row["threshold"]),
        "game_market_current_line": row.get("market_current_line", row["threshold"]),
        "game_days_rest_team": row.get("days_rest", 2.0),
        "game_days_rest_opponent": row.get("opp_days_rest", 2.0),
        "game_travel_fatigue_index": row.get("travel_fatigue", 0.0),
        
        "team_pace": row["team_pace"],
        "team_off_rating": row["team_off_rating"],
        "team_def_rating": row["team_def_rating"],
        
        "opponent_pace": row["opp_pace"],
        "opponent_off_rating": row["opp_off_rating"],
        "opponent_def_rating": row["opp_def_rating"],
        
        "player_points_avg_5": row["points_avg_5"],
        "player_trend_points_slope_5": row["trend_points_slope_5"],
        "player_back_to_back": row["back_to_back"],
        "player_usage_rate_last_5": row.get("usage_rate", 0.2),
        "player_true_shooting_last_5": row.get("ts_pct", 0.55),
    }

def _write_report(n_valid, n_rejected, rejected_details, audit_dir, output_path=None):
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "valid_records": n_valid,
        "rejected_records": n_rejected,
        "output_path": output_path,
        "rejected_details": rejected_details[:100] # Cap detail log
    }
    
    report_path = os.path.join(audit_dir, "real_data_ingest_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        
    logger.info(f"Ingest complete. Valid: {n_valid}, Rejected: {n_rejected}")
    return report

if __name__ == "__main__":
    ingest_real_data()
