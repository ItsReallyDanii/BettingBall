"""
Tests for real data ingestion and validation.
"""
import os
import csv
import json
import pytest
from src.real_data_loader import ingest_real_data, _apply_aliases, _validate_row, _parse_timestamp

def test_alias_mapping_works(tmp_path):
    """Test that column aliases are correctly mapped."""
    # Create test CSV with alias columns
    data_dir = tmp_path / "data" / "real"
    data_dir.mkdir(parents=True)
    
    csv_path = data_dir / "test.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Use aliases instead of canonical names
        writer.writerow(["id", "commence_time", "market", "price", "result"])
        writer.writerow(["E1", "2024-01-01T00:00:00Z", "points", "-110", "1"])
        writer.writerow(["E2", "2024-01-02T00:00:00Z", "rebounds", "+150", "0"])
    
    output_dir = tmp_path / "outputs" / "training"
    audit_dir = tmp_path / "outputs" / "audits"
    
    report = ingest_real_data(
        input_dir=str(data_dir),
        output_dir=str(output_dir),
        audit_dir=str(audit_dir)
    )
    
    assert report["valid_records"] == 2
    assert report["rejected_records"] == 0

def test_malformed_timestamp_rejected(tmp_path):
    """Test that malformed timestamps are rejected with clear reason."""
    data_dir = tmp_path / "data" / "real"
    data_dir.mkdir(parents=True)
    
    csv_path = data_dir / "test.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["event_id", "timestamp_utc", "market_type", "odds_american", "outcome"])
        writer.writerow(["E1", "invalid-date", "points", "-110", "1"])
        writer.writerow(["E2", "", "points", "-110", "0"])
    
    output_dir = tmp_path / "outputs" / "training"
    audit_dir = tmp_path / "outputs" / "audits"
    
    report = ingest_real_data(
        input_dir=str(data_dir),
        output_dir=str(output_dir),
        audit_dir=str(audit_dir)
    )
    
    assert report["valid_records"] == 0
    assert report["rejected_records"] == 2
    
    # Check rejection reasons
    histogram = report["rejection_reason_histogram"]
    assert "invalid_timestamp:unparseable_format:invalid-date" in histogram or \
           "invalid_timestamp:empty_timestamp" in histogram or \
           "missing_required_field:timestamp_utc" in histogram

def test_zero_valid_records_triggers_failure(tmp_path):
    """Test that zero valid records creates proper failure report."""
    data_dir = tmp_path / "data" / "real"
    data_dir.mkdir(parents=True)
    
    # Create CSV with all invalid rows
    csv_path = data_dir / "test.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["event_id", "timestamp_utc", "market_type", "odds_american", "outcome"])
        writer.writerow(["E1", "2024-01-01T00:00:00Z", "points", "0", "1"])  # Zero odds invalid
        writer.writerow(["E2", "2024-01-02T00:00:00Z", "points", "-110", "2"])  # Invalid outcome
    
    output_dir = tmp_path / "outputs" / "training"
    audit_dir = tmp_path / "outputs" / "audits"
    
    report = ingest_real_data(
        input_dir=str(data_dir),
        output_dir=str(output_dir),
        audit_dir=str(audit_dir)
    )
    
    assert report["valid_records"] == 0
    assert report["rejected_records"] == 2
    assert "rejection_reason_histogram" in report
    assert len(report["rejection_reason_histogram"]) > 0

def test_empty_directory_triggers_failure(tmp_path):
    """Test that empty directory creates proper failure report."""
    data_dir = tmp_path / "data" / "real"
    data_dir.mkdir(parents=True)
    
    output_dir = tmp_path / "outputs" / "training"
    audit_dir = tmp_path / "outputs" / "audits"
    
    report = ingest_real_data(
        input_dir=str(data_dir),
        output_dir=str(output_dir),
        audit_dir=str(audit_dir)
    )
    
    assert report["valid_records"] == 0
    assert report["failure_reason"] == "no_csv_files"

def test_parse_timestamp_handles_formats():
    """Test timestamp parsing handles various formats."""
    # ISO with Z
    assert _parse_timestamp("2024-01-01T00:00:00Z") is not None
    
    # ISO without Z
    assert _parse_timestamp("2024-01-01T00:00:00") is not None
    
    # Date only
    assert _parse_timestamp("2024-01-01") is not None
    
    # US format
    assert _parse_timestamp("01/01/2024") is not None
    
    # Should fail on invalid
    with pytest.raises(ValueError):
        _parse_timestamp("not-a-date")

def test_apply_aliases_preserves_unmapped():
    """Test that unmapped columns are preserved."""
    row = {
        "id": "E1",
        "commence_time": "2024-01-01T00:00:00Z",
        "custom_field": "value",
        "player_id": "P1"
    }
    
    normalized = _apply_aliases(row)
    
    assert normalized["event_id"] == "E1"
    assert normalized["timestamp_utc"] == "2024-01-01T00:00:00Z"
    assert normalized["custom_field"] == "value"
    assert normalized["player_id"] == "P1"

def test_validate_row_checks_required_fields():
    """Test validation checks all required fields."""
    # Valid row
    valid_row = {
        "event_id": "E1",
        "timestamp_utc": "2024-01-01T00:00:00Z",
        "market_type": "points",
        "odds_american": "-110",
        "outcome": "1"
    }
    is_valid, reason = _validate_row(valid_row)
    assert is_valid
    assert reason == ""
    
    # Missing field
    invalid_row = {
        "event_id": "E1",
        "timestamp_utc": "2024-01-01T00:00:00Z",
        # missing market_type
        "odds_american": "-110",
        "outcome": "1"
    }
    is_valid, reason = _validate_row(invalid_row)
    assert not is_valid
    assert "missing_required_field:market_type" in reason
