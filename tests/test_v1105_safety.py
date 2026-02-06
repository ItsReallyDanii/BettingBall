import os
import csv
import json
import subprocess
import pytest
import shutil
from datetime import datetime, timezone, timedelta

@pytest.fixture
def clean_env(tmp_path):
    # Setup a clean project-like structure in tmp_path
    data_dir = tmp_path / "data" / "real"
    data_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    
    # We need to be able to run 'python -m src.main'
    # The current working directory is project root, so we can set PYTHONPATH
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd() # Absolute path to project root
    
    return {"cwd": tmp_path, "env": env, "data_dir": data_dir}

def run_cmd(args, env_info):
    try:
        res = subprocess.run(
            ["python", "-m", "src.main"] + args,
            cwd=env_info["cwd"],
            capture_output=True,
            text=True,
            env=env_info["env"]
        )
        if res.stderr and "TypeError" in res.stderr:
            print(f"\n--- SUBPROCESS ERROR ---\n{res.stderr}\n------------------------\n")
        return res
    except Exception as e:
        print(f"Subprocess failed to run: {e}")
        raise

def test_v1105_no_real_files_hard_fail(clean_env):
    """Test that ingest fails if no real files and no --allow_bootstrap."""
    result = run_cmd(["--ingest_real"], clean_env)
    stdout = result.stdout or ""
    assert result.returncode == 1
    assert "NO_REAL_FILES" in stdout

def test_v1105_allow_bootstrap_generates_file(clean_env):
    """Test that --allow_bootstrap generates data but requires a second run."""
    result = run_cmd(["--ingest_real", "--allow_bootstrap"], clean_env)
    stdout = result.stdout or ""
    assert result.returncode == 0
    assert "Generating bootstrap example" in stdout
    assert os.path.exists(clean_env["data_dir"] / "_example_real_data.csv")

def test_v1105_bootstrap_only_files_report(clean_env):
    """Test ingest behavior when only bootstrap files exist."""
    # Create a bootstrap file
    with open(clean_env["data_dir"] / "_example_real_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["event_id", "timestamp_utc", "market_type", "odds_american", "outcome"])
        writer.writerow(["B1", "2023-11-01T20:00:00Z", "test", -110, 1])
    
    result = run_cmd(["--ingest_real"], clean_env)
    assert result.returncode == 1
    assert "INGEST FAILED" in result.stdout
    
    report_path = clean_env["cwd"] / "outputs" / "audits" / "real_data_ingest_report.json"
    assert os.path.exists(report_path)
    with open(report_path, "r") as f:
        report = json.load(f)
    assert report["data_mode"] == "bootstrap"
    assert report["reason_code"] == "bootstrap_only_files_detected"
    assert report["eligible_real_files_count"] == 0
    assert report["excluded_files_count"] == 1

def test_v1105_production_blocks_bootstrap(clean_env):
    """Test that production profile fails if ingest was not successful with real data."""
    # Manually create a dataset with is_bootstrap=1
    training_dir = clean_env["cwd"] / "outputs" / "training"
    training_dir.mkdir(parents=True)
    with open(training_dir / "real_dataset.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["event_id", "timestamp_utc", "market_type", "odds_american", "outcome", "is_bootstrap"])
        for i in range(600):
            writer.writerow([f"E{i}", "2023-11-01T20:00:00Z", "test", -110, 1, 1])
            
    # Mock ingest report
    audit_dir = clean_env["cwd"] / "outputs" / "audits"
    audit_dir.mkdir(parents=True)
    with open(audit_dir / "real_data_ingest_report.json", "w") as f:
        json.dump({"data_mode": "bootstrap", "valid_records": 600}, f)
        
    result = run_cmd(["--train", "--train_profile", "train"], clean_env)
    assert result.returncode == 1
    # Should fail preflight
    assert "PREFLIGHT FAILED" in result.stdout
    assert "production_requires_real_data" in result.stdout or "bootstrap_rows_in_test_set" in result.stdout

def test_v1105_mixed_real_and_bootstrap(clean_env):
    """Test that ingest only takes real files if mixed."""
    # Real file
    with open(clean_env["data_dir"] / "real_1.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["event_id", "timestamp_utc", "market_type", "odds_american", "outcome"])
        writer.writerow(["E1", "2023-11-01T20:00:00Z", "test", -110, 1])
    
    # Bootstrap file
    with open(clean_env["data_dir"] / "_example_ignore.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["event_id", "timestamp_utc", "market_type", "odds_american", "outcome"])
        writer.writerow(["B1", "2023-11-01T20:00:00Z", "test", -110, 1])
        
    result = run_cmd(["--ingest_real"], clean_env)
    assert result.returncode == 0
    assert "1 valid records" in result.stdout
    
    report_path = clean_env["cwd"] / "outputs" / "audits" / "real_data_ingest_report.json"
    with open(report_path, "r") as f:
        report = json.load(f)
    assert report["data_mode"] == "real"
    assert report["eligible_real_files_count"] == 1
    assert report["excluded_files_count"] == 1
    
    training_file = clean_env["cwd"] / "outputs" / "training" / "real_dataset.csv"
    with open(training_file, "r") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["event_id"] == "E1"
    assert rows[0]["is_bootstrap"] == "0"
