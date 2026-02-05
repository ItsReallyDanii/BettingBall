"""
Evaluation runner with truthful metrics and gates.
"""
import os
import json
import pickle
from typing import List, Dict, Any, Tuple

from src.dataset import temporal_split, rolling_fold_generator
from src.train import _prepare_features, _brier_score, _expected_calibration_error

def evaluate_train_profile(
    dataset_path: str = "outputs/training/dataset.csv",
    model_dir: str = "outputs/models",
    output_dir: str = "outputs/audits",
    gate_thresholds: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Evaluate trained model and apply gates.
    
    Returns:
        Evaluation report with GO/NO-GO verdict
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Default thresholds
    if gate_thresholds is None:
        gate_thresholds = {
            "min_sample_size": 500,
            "max_mean_brier": 0.19,
            "max_mean_ece": 0.07,
            "max_fold_variance": 0.05,
            "max_train_val_gap": 0.10
        }
    
    # Load model and calibrator
    model_path = os.path.join(model_dir, "model.pkl")
    calibrator_path = os.path.join(model_dir, "calibrator.pkl")
    
    if not os.path.exists(model_path):
        return _fail_report("model_not_found", output_dir)
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    calibrator = None
    if os.path.exists(calibrator_path):
        with open(calibrator_path, "rb") as f:
            calibrator = pickle.load(f)
    
    # Load metadata
    meta_path = os.path.join(model_dir, "model_meta.json")
    with open(meta_path, "r") as f:
        model_meta = json.load(f)
    
    # Temporal split
    train_data, val_data, test_data = temporal_split(dataset_path)
    
    # Gate: Sample size
    total_samples = len(train_data) + len(val_data) + len(test_data)
    blockers = []
    
    if total_samples < gate_thresholds["min_sample_size"]:
        blockers.append({
            "id": "insufficient_samples",
            "severity": "critical",
            "message": f"Total samples {total_samples} < {gate_thresholds['min_sample_size']}"
        })
    
    # Evaluate on test set
    X_test, y_test = _prepare_features(test_data, {})
    test_preds = [model.predict_proba(x) for x in X_test]
    
    if calibrator:
        test_preds = [calibrator.predict(p) for p in test_preds]
    
    test_brier = _brier_score(test_preds, y_test)
    test_ece = _expected_calibration_error(test_preds, y_test)
    test_log_loss = _log_loss(test_preds, y_test)
    test_hit_rate = _hit_rate(test_preds, y_test)
    
    # Rolling fold validation
    fold_metrics = []
    for fold_idx, fold_train, fold_val in rolling_fold_generator(dataset_path, n_folds=3):
        X_fold_val, y_fold_val = _prepare_features(fold_val, {})
        fold_preds = [model.predict_proba(x) for x in X_fold_val]
        
        if calibrator:
            fold_preds = [calibrator.predict(p) for p in fold_preds]
        
        fold_brier = _brier_score(fold_preds, y_fold_val)
        fold_ece = _expected_calibration_error(fold_preds, y_fold_val)
        
        fold_metrics.append({"fold": fold_idx, "brier": fold_brier, "ece": fold_ece})
    
    # Compute fold variance
    fold_briers = [f["brier"] for f in fold_metrics]
    mean_fold_brier = sum(fold_briers) / len(fold_briers) if fold_briers else 0
    fold_variance = sum((b - mean_fold_brier) ** 2 for b in fold_briers) / len(fold_briers) if fold_briers else 0
    
    # Gate: Mean Brier
    if test_brier > gate_thresholds["max_mean_brier"]:
        blockers.append({
            "id": "high_brier_score",
            "severity": "critical",
            "message": f"Test Brier {test_brier:.4f} > {gate_thresholds['max_mean_brier']}"
        })
    
    # Gate: Mean ECE
    if test_ece > gate_thresholds["max_mean_ece"]:
        blockers.append({
            "id": "high_ece",
            "severity": "critical",
            "message": f"Test ECE {test_ece:.4f} > {gate_thresholds['max_mean_ece']}"
        })
    
    # Gate: Fold variance
    if fold_variance > gate_thresholds["max_fold_variance"]:
        blockers.append({
            "id": "high_fold_variance",
            "severity": "warning",
            "message": f"Fold variance {fold_variance:.4f} > {gate_thresholds['max_fold_variance']}"
        })
    
    # Gate: Train-val gap
    train_brier = model_meta["metrics"]["train"]["brier"]
    val_brier = model_meta["metrics"]["val"]["brier"]
    train_val_gap = abs(train_brier - val_brier)
    
    if train_val_gap > gate_thresholds["max_train_val_gap"]:
        blockers.append({
            "id": "high_train_val_gap",
            "severity": "warning",
            "message": f"Train-val gap {train_val_gap:.4f} > {gate_thresholds['max_train_val_gap']}"
        })
    
    # Reliability bins
    reliability_bins = _compute_reliability_bins(test_preds, y_test)
    
    # Verdict
    critical_blockers = [b for b in blockers if b["severity"] == "critical"]
    verdict = "NO-GO" if critical_blockers else "GO"
    
    # Report
    report = {
        "verdict": verdict,
        "timestamp": model_meta.get("train_window", {}).get("end", "unknown"),
        "total_samples": total_samples,
        "test_metrics": {
            "brier": round(test_brier, 6),
            "ece": round(test_ece, 6),
            "log_loss": round(test_log_loss, 6),
            "hit_rate": round(test_hit_rate, 4)
        },
        "fold_metrics": {
            "mean_brier": round(mean_fold_brier, 6),
            "fold_variance": round(fold_variance, 6),
            "folds": fold_metrics
        },
        "overfitting_diagnostics": {
            "train_brier": round(train_brier, 6),
            "val_brier": round(val_brier, 6),
            "test_brier": round(test_brier, 6),
            "train_val_gap": round(train_val_gap, 6)
        },
        "reliability_bins": reliability_bins,
        "gate_thresholds": gate_thresholds,
        "blockers": blockers
    }
    
    # Write report
    report_path = os.path.join(output_dir, "train_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    # Write blockers
    blockers_path = os.path.join(output_dir, "train_blockers.json")
    with open(blockers_path, "w") as f:
        json.dump({"blockers": blockers, "verdict": verdict}, f, indent=2)
    
    return report

def _log_loss(preds: List[float], targets: List[int]) -> float:
    """Compute log loss."""
    if not preds:
        return 1.0
    
    epsilon = 1e-15
    loss = 0.0
    
    for p, t in zip(preds, targets):
        p_safe = max(epsilon, min(1 - epsilon, p))
        loss += -(t * math.log(p_safe) + (1 - t) * math.log(1 - p_safe))
    
    return loss / len(preds)

def _hit_rate(preds: List[float], targets: List[int], threshold: float = 0.5) -> float:
    """Compute hit rate (accuracy)."""
    if not preds:
        return 0.0
    
    correct = sum(1 for p, t in zip(preds, targets) if (p >= threshold) == t)
    return correct / len(preds)

def _compute_reliability_bins(preds: List[float], targets: List[int], n_bins: int = 10) -> List[Dict[str, Any]]:
    """Compute reliability diagram bins."""
    bins = [[] for _ in range(n_bins)]
    bin_targets = [[] for _ in range(n_bins)]
    
    for p, t in zip(preds, targets):
        bin_idx = min(int(p * n_bins), n_bins - 1)
        bins[bin_idx].append(p)
        bin_targets[bin_idx].append(t)
    
    reliability = []
    for i, (bin_preds, bin_tgts) in enumerate(zip(bins, bin_targets)):
        if bin_preds:
            avg_pred = sum(bin_preds) / len(bin_preds)
            avg_target = sum(bin_tgts) / len(bin_tgts)
            reliability.append({
                "bin": i,
                "count": len(bin_preds),
                "avg_pred": round(avg_pred, 4),
                "avg_target": round(avg_target, 4),
                "calibration_error": round(abs(avg_pred - avg_target), 4)
            })
    
    return reliability

def _fail_report(reason: str, output_dir: str) -> Dict[str, Any]:
    """Generate failure report."""
    report = {
        "verdict": "NO-GO",
        "blockers": [{
            "id": reason,
            "severity": "critical",
            "message": f"Evaluation failed: {reason}"
        }]
    }
    
    report_path = os.path.join(output_dir, "train_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    return report

import math
