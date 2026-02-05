"""
Real data ingestion and normalization for training pipeline.
Robust vendor CSV parsing with alias mapping and detailed rejection tracking.
"""
import os
import csv
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timezone
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Core required fields (must be present after alias resolution)
REQUIRED_CORE = ["event_id", "timestamp_utc", "market_type", "odds_american", "outcome"]

# Optional join keys (validated if present)
OPTIONAL_JOIN_KEYS = ["player_id", "team_id", "game_id"]

# Column alias mappings (vendor variations -> canonical)
COLUMN_ALIASES = {
    "timestamp_utc": ["commence_time", "game_time", "start_time", "date", "date_utc", "datetime"],
    "odds_american": ["price", "american_odds", "odds", "price_american"],
    "outcome": ["result", "won", "target", "actual"],
    "event_id": ["id", "match_id", "fixture_id"],
    "market_type": ["market", "bet_type", "prop_type"],
}

def ingest_real_data(
    input_dir: str = "data/real",
    output_dir: str = "outputs/training",
    audit_dir: str = "outputs/audits"
) -> Dict[str, Any]:
    """
    Ingest real historical data from local files with robust alias mapping.
    
    Returns:
        Report dict with valid_records, rejected_records, rejection_reason_histogram
    """
    abs_input_dir = os.path.abspath(input_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(audit_dir, exist_ok=True)
    
    logger.info(f"Ingesting real data from: {abs_input_dir}")
    
    all_records = []
    rejected_rows = []
    sample_columns_detected = set()
    
    if not os.path.exists(input_dir):
        logger.warning(f"Input directory {abs_input_dir} does not exist.")
        return _write_report(0, 0, [], set(), audit_dir, reason="input_dir_not_found", inspected_path=abs_input_dir)
    
    # Case-insensitive .csv detection
    all_files = os.listdir(input_dir)
    csv_files = [f for f in all_files if f.lower().endswith(".csv")]
    
    # Identify bootstrap files
    bootstrap_files = [f for f in csv_files if "_example_" in f or "bootstrap" in f.lower()]
    is_bootstrap_mode = len(bootstrap_files) > 0 and len(bootstrap_files) == len(csv_files)
    data_mode = "bootstrap" if is_bootstrap_mode else "real"
    if not csv_files:
        data_mode = "unknown"

    logger.info(f"Discovered files ({len(csv_files)}/ {len(all_files)}): {csv_files}")
    logger.info(f"Data Mode: {data_mode}")
    
    if not csv_files:
        logger.warning(f"No CSV files found in {abs_input_dir}")
        return _write_report(0, 0, [], set(), audit_dir, reason="no_csv_files", inspected_path=abs_input_dir, data_mode=data_mode)
    
    for filename in csv_files:
        path = os.path.join(input_dir, filename)
        logger.info(f"Processing {path}...")
        is_this_file_bootstrap = "_example_" in filename or "bootstrap" in filename.lower()
        
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                
                # Detect columns from header
                if reader.fieldnames:
                    sample_columns_detected.update(reader.fieldnames)
                
                for row_idx, row in enumerate(reader):
                    # Apply alias mapping
                    normalized_row = _apply_aliases(row)
                    
                    # Add bootstrap flag
                    normalized_row["is_bootstrap"] = 1 if is_this_file_bootstrap else 0
                    
                    # Validate
                    valid, reason = _validate_row(normalized_row)
                    if valid:
                        all_records.append(normalized_row)
                    else:
                        rejected_rows.append({
                            "file": filename,
                            "row": row_idx + 2,  # +2 for header and 0-indexing
                            "reason": reason,
                            "sample_data": dict(list(row.items())[:5])  # First 5 cols
                        })
        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")
            rejected_rows.append({
                "file": filename,
                "row": "N/A",
                "reason": f"file_parse_error: {str(e)}",
                "sample_data": {}
            })
    
    # Check if no rows parsed
    if not all_records and not rejected_rows:
        return _write_report(0, 0, [], sample_columns_detected, audit_dir, reason="no_rows_parsed", inspected_path=abs_input_dir, data_mode=data_mode)
    
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
    else:
        output_path = None
    
    return _write_report(len(all_records), len(rejected_rows), rejected_rows, 
                        sample_columns_detected, audit_dir, output_path, 
                        inspected_path=abs_input_dir, data_mode=data_mode,
                        source_files=csv_files)

def _apply_aliases(row: Dict[str, Any]) -> Dict[str, Any]:
    """Apply column alias mapping to normalize vendor variations."""
    normalized = {}
    
    for canonical, aliases in COLUMN_ALIASES.items():
        # Check if canonical name exists
        if canonical in row:
            normalized[canonical] = row[canonical]
        else:
            # Check aliases
            for alias in aliases:
                if alias in row:
                    normalized[canonical] = row[alias]
                    break
    
    # Copy over any unmapped columns (for join keys, etc.)
    for key, value in row.items():
        if key not in normalized:
            normalized[key] = value
    
    return normalized

def _validate_row(row: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate row has required fields and valid values."""
    # Check core required fields
    for col in REQUIRED_CORE:
        if col not in row or row[col] is None or str(row[col]).strip() == "":
            return False, f"missing_required_field:{col}"
    
    # Validate timestamp
    try:
        ts = _parse_timestamp(row["timestamp_utc"])
        # Standardize timestamp to ISO string
        row["timestamp_utc"] = ts.isoformat().replace("+00:00", "Z")
    except Exception as e:
        return False, f"invalid_timestamp:{str(e)}"
    
    # Validate odds
    try:
        odds = float(row["odds_american"])
        if odds == 0:
            return False, "invalid_odds:zero_value"
    except (ValueError, TypeError):
        return False, f"invalid_odds:not_numeric"
    
    # Validate outcome
    try:
        outcome = float(row["outcome"])
        if outcome not in [0, 1, 0.0, 1.0]:
            return False, f"invalid_outcome:must_be_0_or_1"
    except (ValueError, TypeError):
        return False, "invalid_outcome:not_numeric"
    
    # Validate join keys if present
    for key in OPTIONAL_JOIN_KEYS:
        if key in row and row[key]:
            if not str(row[key]).strip():
                return False, f"invalid_join_key:{key}_empty"
    
    return True, ""

def _parse_timestamp(ts_str: str) -> datetime:
    """Robustly parse timestamp from various formats."""
    if not ts_str:
        raise ValueError("empty_timestamp")
    
    ts_str = str(ts_str).strip()
    
    # Handle 'Z' suffix
    if ts_str.endswith('Z'):
        ts_str = ts_str[:-1] + '+00:00'
    
    # Try ISO format
    try:
        return datetime.fromisoformat(ts_str)
    except:
        pass
    
    # Try common formats
    formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y",
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(ts_str, fmt)
        except:
            continue
    
    raise ValueError(f"unparseable_format:{ts_str[:20]}")

def _write_report(
    n_valid: int, 
    n_rejected: int, 
    rejected_details: List[Dict], 
    sample_columns: set,
    audit_dir: str, 
    output_path: Optional[str] = None,
    reason: Optional[str] = None,
    inspected_path: Optional[str] = None,
    data_mode: str = "unknown",
    source_files: List[str] = None
) -> Dict[str, Any]:
    """Write detailed ingest report with rejection histogram."""
    
    # Build rejection reason histogram
    rejection_reasons = [r["reason"] for r in rejected_details]
    reason_histogram = dict(Counter(rejection_reasons).most_common(10))
    
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "valid_records": n_valid,
        "rejected_records": n_rejected,
        "data_mode": data_mode,
        "source_files": source_files or [],
        "bootstrap_generated": data_mode == "bootstrap",
        "output_path": output_path,
        "inspected_path": inspected_path,
        "rejection_reason_histogram": reason_histogram,
        "sample_columns_detected": sorted(list(sample_columns)),
        "rejected_details": rejected_details[:100],  # Cap detail log
    }
    
    if reason:
        report["failure_reason"] = reason
    
    report_path = os.path.join(audit_dir, "real_data_ingest_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Ingest complete. Valid: {n_valid}, Rejected: {n_rejected}, Mode: {data_mode}")
    if reason_histogram:
        logger.info(f"Top rejection reasons: {reason_histogram}")
    
    return report

if __name__ == "__main__":
    ingest_real_data()
