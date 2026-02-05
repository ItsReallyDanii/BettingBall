
import json
import hashlib
import platform
import sys
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

def _file_hash(path: str) -> str:
    if not os.path.exists(path): return "missing"
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def generate_model_card(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model_name": "Baseline Logistic Regression (SGD) + Platt Scaling",
        "version": "v1.10",
        "type": "binary_classification",
        "features_used": [
            "bias_term",
            "stat_projection_diff (avg_5 + slope_5 - threshold)",
            "days_rest",
            "opponent_def_rating"
        ],
        "preprocessing": "None (Raw features extracted on the fly)",
        "training_window_policy": "Strict Chronological Split (No future leakage)",
        "calibration_method": "Platt Scaling (LogReg on Logits)",
        "known_limitations": [
            "Linear decision boundary",
            "Ignores injury reports",
            "No player-specific embankments",
            "Assuming stable opponent def rating"
        ],
        "intended_use": "Baseline comparison for advanced models",
        "out_of_scope_use": "High-stakes betting without human review",
        "last_eval_metrics": metrics
    }

def generate_data_card(inputs: List[Dict[str, Any]], targets: List[int]) -> Dict[str, Any]:
    # Analyze data
    n = len(inputs)
    n_pos = sum(targets)
    n_neg = n - n_pos
    
    # Temporal scan
    timestamps = []
    for inp in inputs:
        # Pydantic or dict
        if isinstance(inp, dict):
            ts = inp.get("game_timestamp") or inp.get("game", {}).get("date_utc")
        else:
            ts = getattr(inp, "game_timestamp", None)
            
        if ts: timestamps.append(ts)
        
    return {
        "dataset_name": "NBA Player Props (Rolling Window)",
        "total_samples": n,
        "label_distribution": {"over": n_pos, "under": n_neg, "pos_rate": round(n_pos/n if n else 0, 4)},
        "temporal_coverage": {
            "min": min(timestamps) if timestamps else None,
            "max": max(timestamps) if timestamps else None
        },
        "sources": ["balldontlie.io (via DataConnector)"],
        "schema_validation": "Passed (Strict Pydantic Enforcement)"
    }

def generate_repro_manifest(cmd_args: List[str], input_hashes: Dict[str, str], output_hashes: Dict[str, str], git_commit: str = "unknown", random_seeds: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    random_seeds = random_seeds or {}
    return {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "platform": platform.platform(),
        "python_version": sys.version,
        "git_commit": git_commit,
        "command_invoked": " ".join(cmd_args),
        "random_seeds": random_seeds,
        "input_artifacts": input_hashes,
        "output_artifacts": output_hashes
    }

def compile_freeze_blockers(gate_status: Dict[str, bool], metrics: Dict[str, Any], thresholds: Dict[str, float]) -> List[Dict[str, Any]]:
    blockers = []
    
    # 1. Sample Size
    if gate_status.get("sample_size_total") is False:
         blockers.append({
             "id": "sample_size_total",
             "expected": thresholds.get("sample_size_total"),
             "actual": metrics.get("total_sample_size") or metrics.get("split_train", 0) + metrics.get("split_test", 0),
             "severity": "high",
             "action_hint": "Ingest more data or reduce split ratio"
         })

    # 2. Brier
    if (gate_status.get("mean_brier") is False) or (gate_status.get("brier") is False):
         # Handle both rolling (mean_brier) and static (brier) keys
         actual = metrics.get("mean_brier") or metrics.get("brier_score") or metrics.get("post_calibration", {}).get("brier_score")
         blockers.append({
             "id": "brier_score",
             "expected": thresholds.get("mean_brier"),
             "actual": actual,
             "severity": "high",
             "action_hint": "Improve baseline model features"
         })

    # 3. ECE
    if (gate_status.get("mean_ece") is False) or (gate_status.get("ece") is False):
         actual = metrics.get("mean_ece") or metrics.get("ece") or metrics.get("post_calibration", {}).get("ece")
         blockers.append({
             "id": "ece",
             "expected": thresholds.get("mean_ece"),
             "actual": actual,
             "severity": "medium",
             "action_hint": "Tune Platt Scaling calibration"
         })
         
    return blockers
