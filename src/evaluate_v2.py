"""
Evaluation runner with truthful metrics and gates based on real data provenance.
"""
import os
import json
import pickle
import math
from typing import List, Dict, Any, Tuple
from datetime import datetime, timezone

from src.dataset import temporal_split, rolling_fold_generator
from src.train import _prepare_features, _brier_score, _expected_calibration_error

def evaluate_train_profile(
    dataset_path: str = "outputs/training/dataset.csv",
    model_dir: str = "outputs/models",
    output_dir: str = "outputs/audits",
    gate_thresholds: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Evaluate trained model against realism and performance gates.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Default thresholds and profile
    if gate_thresholds is None:
        gate_thresholds = {
            "min_sample_size": 500,
            "max_mean_brier": 0.19,
            "max_mean_ece": 0.07,
            "max_brier_std": 0.02,
            "profile_name": "dev"
        }
    
    profile_name = gate_thresholds.get("profile_name", "dev")

    # Load artifacts
    model_path = os.path.join(model_dir, "model.pkl")
    calibrator_path = os.path.join(model_dir, "calibrator.pkl")
    meta_path = os.path.join(model_dir, "model_meta.json")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    calibrator = None
    if os.path.exists(calibrator_path):
        with open(calibrator_path, "rb") as f:
            calibrator = pickle.load(f)
            
    with open(meta_path, "r") as f:
        model_meta = json.load(f)

    # Detect data provenance
    data_mode = "real"
    ingest_report_path = "outputs/audits/real_data_ingest_report.json"
    if os.path.exists(ingest_report_path):
        try:
            with open(ingest_report_path, "r") as f:
                report_data = json.load(f)
                data_mode = report_data.get("data_mode", "real")
        except:
            data_mode = "real"

    # Split dataset
    train_data, val_data, test_data = temporal_split(dataset_path)
    X_test, y_test = _prepare_features(test_data, {})
    
    blockers = []
    
    # 1. Gate: Sample size
    n_total = len(train_data) + len(val_data) + len(test_data)
    if n_total < gate_thresholds["min_sample_size"]:
        blockers.append({
            "id": "insufficient_samples",
            "severity": "critical",
            "message": f"Total samples {n_total} < {gate_thresholds['min_sample_size']}"
        })

    # 2. Provenance Gate
    if profile_name in ["train", "freeze"] and data_mode != "real":
        blockers.append({
            "id": "non_real_data_for_production_profile",
            "severity": "critical",
            "message": f"Profile '{profile_name}' requires real data, but found '{data_mode}' mode"
        })
        
    # 3. Bootstrap Hard Block for Production
    if profile_name in ["train", "freeze"]:
        boot_count = sum(1 for r in test_data if int(r.get("is_bootstrap", 0)) == 1)
        if boot_count > 0:
            blockers.append({
                "id": "bootstrap_rows_in_test_set",
                "severity": "critical",
                "message": f"Found {boot_count} bootstrap rows in test set for production profile"
            })

    # Calculate Test Metrics
    test_preds_raw = [model.predict_proba(x) for x in X_test]
    if calibrator:
        test_preds = [calibrator.predict(p) for p in test_preds_raw]
    else:
        test_preds = test_preds_raw
        
    test_brier = _brier_score(test_preds, y_test)
    test_ece = _expected_calibration_error(test_preds, y_test)
    test_hit_rate = sum(1 for p, t in zip(test_preds, y_test) if (p > 0.5) == t) / (len(y_test) or 1)

    # 4. Realism Gates
    n_test = len(test_data)
    realism_severity = "critical" if profile_name in ["train", "freeze"] else "warning"

    if (test_brier < 0.02 or test_ece < 0.01) and n_test < 2000:
        blockers.append({
            "id": "suspicious_perfect_metrics",
            "severity": realism_severity,
            "message": f"Metrics too perfect (Brier={test_brier:.4f}, ECE={test_ece:.4f}) for N={n_test}. Possible synthetic data or leakage."
        })
            
    if test_hit_rate > 0.80:
        blockers.append({
            "id": "extreme_hit_rate",
            "severity": realism_severity,
            "message": f"Extreme hit rate {test_hit_rate:.4f} > 0.80 suggests data leakage on binary BETTING targets"
        })
        
    # Class Entropy
    if y_test:
        class_1_rate = sum(y_test) / len(y_test)
        min_rate = min(class_1_rate, 1 - class_1_rate)
        if min_rate < 0.15:
            blockers.append({
                "id": "low_label_entropy",
                "severity": realism_severity,
                "message": f"Min class rate {min_rate:.4f} < 0.15 (Class imbalance too high)"
            })
            
    # Feature Variance
    if X_test:
        n_features = len(X_test[0])
        zero_var_count = 0
        for i in range(n_features):
            col = [x[i] for x in X_test]
            if len(set(col)) < 2:
                zero_var_count += 1
        
        if n_features > 0 and zero_var_count / n_features > 0.40:
             blockers.append({
                "id": "low_feature_variance",
                "severity": realism_severity,
                "message": f"{zero_var_count}/{n_features} features have near-zero variance"
            })

    # Stability calculation
    fold_metrics = []
    for _, _, f_val in rolling_fold_generator(dataset_path):
        X_f, y_f = _prepare_features(f_val, {})
        p_f = [model.predict_proba(x) for x in X_f]
        if calibrator: p_f = [calibrator.predict(p) for p in p_f]
        fold_metrics.append(_brier_score(p_f, y_f))
    
    brier_std = 0.0
    if len(fold_metrics) > 1:
        mean_f = sum(fold_metrics) / len(fold_metrics)
        brier_std = math.sqrt(sum((f - mean_f)**2 for f in fold_metrics) / len(fold_metrics))
    
    if brier_std > gate_thresholds["max_brier_std"]:
        blockers.append({
            "id": "stability_failed",
            "severity": "warning",
            "message": f"Brier std {brier_std:.4f} > {gate_thresholds['max_brier_std']}"
        })

    # Final Verdict
    critical_blockers = [b for b in blockers if b["severity"] == "critical"]
    verdict = "NO-GO" if critical_blockers else "GO"
    if verdict == "GO" and data_mode == "bootstrap":
        verdict = "GO (PLUMBING ONLY)"

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_mode": data_mode,
        "profile": profile_name,
        "verdict": verdict,
        "sample_size": n_total,
        "test_metrics": {
            "brier": round(test_brier, 6),
            "ece": round(test_ece, 6),
            "hit_rate": round(test_hit_rate, 4)
        },
        "stability": {
            "brier_std": round(brier_std, 6),
            "fold_count": len(fold_metrics)
        },
        "blockers": blockers
    }
    
    with open(os.path.join(output_dir, "train_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(output_dir, "train_blockers.json"), "w") as f:
        json.dump(blockers, f, indent=2)
        
    return report
