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

def classify_candidate_files(input_dir: str) -> Dict[str, Any]:
    """
    Classify files in input_dir into eligible real data files and excluded files.
    Patterns (case-insensitive): REAL_DATA_TEMPLATE.csv, *_example_*, *bootstrap*, *template*
    """
    if not os.path.exists(input_dir):
        return {"eligible_real_files": [], "excluded_files": [], "has_bootstrap_files": False}
    
    all_files = os.listdir(input_dir)
    csv_files = [f for f in all_files if f.lower().endswith(".csv")]
    
    eligible = []
    excluded = []
    has_bootstrap = False
    
    exclude_patterns = ["REAL_DATA_TEMPLATE.csv", "_example_", "bootstrap", "template"]
    
    for f in csv_files:
        is_excluded = False
        reason = ""
        f_lower = f.lower()
        
        for pattern in exclude_patterns:
            if pattern.lower() in f_lower:
                is_excluded = True
                reason = f"matches_exclude_pattern:{pattern.lower()}"
                if "bootstrap" in f_lower or "_example_" in f_lower:
                    has_bootstrap = True
                break
        
        if is_excluded:
            excluded.append({"name": f, "reason": reason})
        else:
            eligible.append(f)
            
    return {
        "eligible_real_files": eligible,
        "excluded_files": excluded,
        "has_bootstrap_files": has_bootstrap
    }

def ingest_real_data(
    input_dir: str = "data/real",
    output_dir: str = "outputs/training",
    audit_dir: str = "outputs/audits",
    allow_bootstrap: bool = False
) -> Dict[str, Any]:
    """
    Ingest real historical data from local files.
    If allow_bootstrap is True, allows processing of bootstrap/example files if no real data is present.
    """
    abs_input_dir = os.path.abspath(input_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(audit_dir, exist_ok=True)
    
    logger.info(f"Ingesting data from: {abs_input_dir} (allow_bootstrap={allow_bootstrap})")
    
    all_records = []
    rejected_rows = []
    sample_columns_detected = set()
    
    if not os.path.exists(abs_input_dir):
        logger.warning(f"Input directory {abs_input_dir} does not exist.")
        return _write_report(0, 0, [], set(), audit_dir, reason="input_dir_not_found", inspected_path=abs_input_dir)
    
    # Classify files
    class_res = classify_candidate_files(abs_input_dir)
    eligible_files = class_res["eligible_real_files"]
    excluded_files = class_res["excluded_files"]
    has_bootstrap = class_res["has_bootstrap_files"]
    
    # Decide which files to actually process
    files_to_process = []
    current_mode = "unknown"
    
    if eligible_files:
        files_to_process = eligible_files
        current_mode = "real"
    elif has_bootstrap and allow_bootstrap:
        # If no real files but we have bootstrap and it's allowed, process bootstrap
        files_to_process = [f["name"] for f in excluded_files if "bootstrap" in f["name"].lower() or "_example_" in f["name"].lower()]
        current_mode = "bootstrap"
        logger.info(f"FALLBACK: Ingesting {len(files_to_process)} bootstrap files (dev-only mode)")
    else:
        current_mode = "bootstrap" if has_bootstrap else "unknown"
        reason_code = "bootstrap_only_files_detected" if has_bootstrap else "no_eligible_csv_files"
        logger.warning(f"Abort ingest: {reason_code}")
        return _write_report(
            0, 0, [], set(), audit_dir, 
            reason=reason_code, 
            inspected_path=abs_input_dir, 
            data_mode=current_mode,
            eligible_count=len(eligible_files),
            excluded_count=len(excluded_files),
            excluded_details=excluded_files
        )
    
    for filename in files_to_process:
        path = os.path.join(abs_input_dir, filename)
        logger.info(f"Processing candidate: {filename}")
        
        # Determine if this specific file is bootstrap
        is_this_file_bootstrap = "bootstrap" in filename.lower() or "_example_" in filename.lower()
        
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    sample_columns_detected.update(reader.fieldnames)
                
                for row_idx, row in enumerate(reader):
                    normalized_row = _apply_aliases(row)
                    # Use the file's bootstrap status
                    normalized_row["is_bootstrap"] = 1 if is_this_file_bootstrap else 0
                    
                    valid, reason = _validate_row(normalized_row)
                    if valid:
                        all_records.append(normalized_row)
                    else:
                        rejected_rows.append({
                            "file": filename, "row": row_idx + 2, "reason": reason,
                            "sample_data": dict(list(row.items())[:5])
                        })
        except Exception as e:
            logger.error(f"Failed to parse {filename}: {e}")
            rejected_rows.append({"file": filename, "row": "N/A", "reason": f"file_parse_error:{str(e)}"})

    if not all_records and not rejected_rows:
        return _write_report(
            0, 0, [], sample_columns_detected, audit_dir, 
            reason="no_rows_parsed", inspected_path=abs_input_dir, data_mode=current_mode,
            eligible_count=len(eligible_files), excluded_count=len(excluded_files), excluded_details=excluded_files
        )
    
    all_records.sort(key=lambda x: x["timestamp_utc"])
    
    output_path = os.path.join(output_dir, "real_dataset.csv")
    if all_records:
        fieldnames = list(all_records[0].keys())
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_records)
    
    return _write_report(
        len(all_records), len(rejected_rows), rejected_rows, 
        sample_columns_detected, audit_dir, output_path, 
        inspected_path=abs_input_dir, data_mode=current_mode, source_files=files_to_process,
        eligible_count=len(eligible_files), excluded_count=len(excluded_files), excluded_details=excluded_files
    )

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
    source_files: List[str] = None,
    eligible_count: int = 0,
    excluded_count: int = 0,
    excluded_details: List[Dict] = None
) -> Dict[str, Any]:
    """Write detailed ingest report."""
    safe_rejected = rejected_details if isinstance(rejected_details, list) else []
    safe_excluded = excluded_details if isinstance(excluded_details, list) else []
    
    reason_hist = dict(Counter([r.get("reason", "unknown") for r in safe_rejected if isinstance(r, dict)]).most_common(10))
    excluded_hist = dict(Counter([e.get("reason", "unknown") for e in safe_excluded if isinstance(e, dict)]).most_common(5))
    
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "valid_records": n_valid,
        "rejected_records": n_rejected,
        "data_mode": data_mode,
        "reason_code": reason or "success",
        "eligible_real_files_count": eligible_count,
        "excluded_files_count": excluded_count,
        "excluded_reason_histogram": excluded_hist,
        "source_files": source_files or [],
        "bootstrap_generated": data_mode == "bootstrap",
        "output_path": output_path,
        "inspected_path": inspected_path,
        "rejection_reason_histogram": reason_hist,
        "sample_columns_detected": sorted(list(sample_columns)),
        "rejected_details": rejected_details[:100],
    }
    
    if reason:
        report["failure_reason"] = reason
    
    report_path = os.path.join(audit_dir, "real_data_ingest_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Ingest finished. Valid: {n_valid}, Rejected: {n_rejected}, Mode: {data_mode}, Reason: {reason or 'success'}")
    return report

def validate_real_data(
    input_dir: str = "data/real",
    audit_dir: str = "outputs/audits"
) -> Dict[str, Any]:
    """
    Validate all CSV files in input_dir without saving a combined dataset.
    Generates a detailed validation report.
    """
    abs_input_dir = os.path.abspath(input_dir)
    os.makedirs(audit_dir, exist_ok=True)
    
    if not os.path.exists(input_dir):
        return {"error": "input_dir_not_found", "path": abs_input_dir}
    
    all_files = os.listdir(input_dir)
    csv_files = [f for f in all_files if f.lower().endswith(".csv") and f != "REAL_DATA_TEMPLATE.csv"]
    
    file_reports = []
    overall_outcomes = []
    overall_ts_parsed = 0
    overall_total_rows = 0
    
    all_columns = set()
    column_null_counts = Counter()
    
    for filename in csv_files:
        path = os.path.join(input_dir, filename)
        valid_rows = 0
        rejected_rows = []
        file_outcomes = []
        
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
                all_columns.update(fieldnames)
                
                rows = list(reader)
                file_total = len(rows)
                overall_total_rows += file_total
                
                for row_idx, row in enumerate(rows):
                    # Tracking nulls for detected columns
                    for col in fieldnames:
                        if not row.get(col) or str(row[col]).strip() == "":
                            column_null_counts[col] += 1
                    
                    normalized = _apply_aliases(row)
                    is_valid, reason = _validate_row(normalized)
                    
                    if is_valid:
                        valid_rows += 1
                        overall_ts_parsed += 1
                        try:
                            val = float(normalized["outcome"])
                            file_outcomes.append(val)
                            overall_outcomes.append(val)
                        except:
                            pass
                    else:
                        rejected_rows.append({
                            "row": row_idx + 2,
                            "reason": reason
                        })
        except Exception as e:
            file_reports.append({
                "file": filename,
                "error": str(e)
            })
            continue

        file_reports.append({
            "file": filename,
            "total_rows": file_total,
            "valid_rows": valid_rows,
            "rejected_rows": len(rejected_rows),
            "rejection_histogram": dict(Counter(r["reason"] for r in rejected_rows).most_common(5))
        })

    # Summary stats
    null_rates = {col: round(column_null_counts[col] / (overall_total_rows or 1), 4) for col in all_columns}
    
    class_balance = {}
    if overall_outcomes:
        rate_1 = sum(overall_outcomes) / len(overall_outcomes)
        class_balance = {
            "rate_1": round(rate_1, 4),
            "rate_0": round(1 - rate_1, 4),
            "min_rate": round(min(rate_1, 1 - rate_1), 4)
        }
    
    ts_parse_rate = round(overall_ts_parsed / (overall_total_rows or 1), 4)
    
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_dir": abs_input_dir,
        "files_processed": len(csv_files),
        "total_rows_scanned": overall_total_rows,
        "total_valid_rows": overall_ts_parsed,
        "ts_parse_success_rate": ts_parse_rate,
        "null_rates": null_rates,
        "class_balance": class_balance,
        "file_reports": file_reports
    }
    
    report_path = os.path.join(audit_dir, "real_data_validation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        
    return report

if __name__ == "__main__":
    ingest_real_data()
