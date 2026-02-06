"""
Model training with hyperparameter search and calibration.
"""
import os
import json
import pickle
import math
import random
from typing import List, Dict, Any, Tuple, Optional

from src.calibration import PlattScaler, IsotonicScaler
from src.features_v2 import extract_features_v2, features_to_vector, FEATURE_COLUMNS
from src.features_v3 import (
    extract_v3_features, v3_features_to_vector, V3_FEATURE_COLUMNS,
    DataQualityTracker, feature_availability_report, FEATURE_FAMILIES,
)

class LogisticRegressionV2:
    """Enhanced logistic regression with L2 regularization."""
    
    def __init__(self, learning_rate: float = 0.01, epochs: int = 200, l2_lambda: float = 0.01, seed: int = 42, n_features: Optional[int] = None):
        random.seed(seed)
        self.lr = learning_rate
        self.epochs = epochs
        self.l2_lambda = l2_lambda
        dim = n_features if n_features is not None else len(FEATURE_COLUMNS) + len(V3_FEATURE_COLUMNS)
        self.weights = [random.gauss(0, 0.01) for _ in range(dim)]
        self.bias = 0.0
    
    def sigmoid(self, z: float) -> float:
        return 1.0 / (1.0 + math.exp(-max(min(z, 10), -10)))
    
    def predict_proba(self, features: List[float]) -> float:
        z = self.bias + sum(w * f for w, f in zip(self.weights, features))
        return self.sigmoid(z)
    
    def fit(self, X: List[List[float]], y: List[int]):
        """Train with SGD + L2 regularization."""
        n = len(X)
        if n == 0:
            return
        
        for epoch in range(self.epochs):
            # Shuffle for SGD
            indices = list(range(n))
            random.shuffle(indices)
            
            for idx in indices:
                features = X[idx]
                target = y[idx]
                
                # Forward pass
                pred = self.predict_proba(features)
                error = pred - target
                
                # Update weights with L2 penalty
                for i in range(len(self.weights)):
                    grad = error * features[i] + self.l2_lambda * self.weights[i]
                    self.weights[i] -= self.lr * grad
                
                # Update bias
                self.bias -= self.lr * error

def train_model(
    train_data: List[Dict[str, Any]],
    val_data: List[Dict[str, Any]],
    model_type: str = "logistic_v2",
    seed: int = 42,
    output_dir: str = "outputs/models"
) -> Dict[str, Any]:
    """
    Train model with hyperparameter search.
    
    Returns:
        Training metadata
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Feature manifest
    all_columns = FEATURE_COLUMNS + V3_FEATURE_COLUMNS
    feature_manifest = {"features": {}, "column_order": all_columns}

    # Extract features
    X_train, y_train = _prepare_features(train_data, feature_manifest)
    X_val, y_val = _prepare_features(val_data, feature_manifest)

    n_features = len(all_columns)
    
    # Hyperparameter search
    best_model = None
    best_calibrator = None
    best_val_brier = float('inf')
    best_params = {}
    
    # Search grid
    if model_type == "logistic_v2":
        param_grid = [
            {"lr": 0.01, "epochs": 200, "l2": 0.01},
            {"lr": 0.01, "epochs": 300, "l2": 0.001},
            {"lr": 0.005, "epochs": 200, "l2": 0.01},
        ]
    else:
        param_grid = [{"lr": 0.01, "epochs": 200, "l2": 0.01}]
    
    for params in param_grid:
        # Train model
        if model_type == "logistic_v2":
            model = LogisticRegressionV2(
                learning_rate=params["lr"],
                epochs=params["epochs"],
                l2_lambda=params["l2"],
                seed=seed,
                n_features=n_features,
            )
        else:
            model = LogisticRegressionV2(seed=seed, n_features=n_features)
        
        model.fit(X_train, y_train)
        
        # Get validation predictions
        val_preds = [model.predict_proba(x) for x in X_val]
        
        # Try calibrators
        calibrators = _train_calibrators(val_preds, y_val)
        
        for cal_name, calibrator in calibrators.items():
            if calibrator:
                cal_preds = [calibrator.predict(p) for p in val_preds]
            else:
                cal_preds = val_preds
            
            # Compute Brier score
            brier = _brier_score(cal_preds, y_val)
            ece = _expected_calibration_error(cal_preds, y_val)
            
            # Model selection: minimize Brier, tie-break with ECE
            if brier < best_val_brier or (brier == best_val_brier and ece < best_params.get("ece", float('inf'))):
                best_val_brier = brier
                best_model = model
                best_calibrator = calibrator
                best_params = {
                    **params,
                    "calibrator": cal_name,
                    "brier": brier,
                    "ece": ece
                }
    
    # Save artifacts
    model_path = os.path.join(output_dir, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    
    calibrator_path = os.path.join(output_dir, "calibrator.pkl")
    if best_calibrator:
        with open(calibrator_path, "wb") as f:
            pickle.dump(best_calibrator, f)
    
    # Save feature manifest
    manifest_path = os.path.join(output_dir, "feature_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(feature_manifest, f, indent=2)
    
    # Compute train metrics
    train_preds = [best_model.predict_proba(x) for x in X_train]
    if best_calibrator:
        train_preds = [best_calibrator.predict(p) for p in train_preds]
    
    train_brier = _brier_score(train_preds, y_train)
    train_ece = _expected_calibration_error(train_preds, y_train)
    
    # Get timestamp range
    train_timestamps = [r.get("timestamp_utc", "") for r in train_data]
    train_window = {
        "start": min(train_timestamps) if train_timestamps else None,
        "end": max(train_timestamps) if train_timestamps else None,
        "n_samples": len(train_data)
    }
    
    # Model metadata
    metadata = {
        "model_type": model_type,
        "features_used": all_columns,
        "n_features": len(all_columns),
        "v3_availability": feature_manifest.get("v3_availability", {}),
        "best_params": best_params,
        "train_window": train_window,
        "metrics": {
            "train": {"brier": train_brier, "ece": train_ece},
            "val": {"brier": best_val_brier, "ece": best_params["ece"]}
        },
        "artifacts": {
            "model": model_path,
            "calibrator": calibrator_path if best_calibrator else None,
            "feature_manifest": manifest_path
        }
    }
    
    metadata_path = os.path.join(output_dir, "model_meta.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    return metadata

def _prepare_features(data: List[Dict[str, Any]], manifest: Dict[str, Any]) -> Tuple[List[List[float]], List[int]]:
    """Extract features and targets from data.

    Includes v3 features when underlying data columns are available.
    """
    X = []
    y = []
    v3_available = None  # determined on first record

    for record in data:
        # Build joined record format
        joined = {
            "game": {k.replace("game_", ""): v for k, v in record.items() if k.startswith("game_")},
            "player": {k.replace("player_", ""): v for k, v in record.items() if k.startswith("player_")},
            "team": {k.replace("team_", ""): v for k, v in record.items() if k.startswith("team_")},
            "opponent_team": {k.replace("opponent_", ""): v for k, v in record.items() if k.startswith("opponent_")},
            "line": record.get("threshold", 0)
        }

        # v2 features (always present)
        features_v2 = extract_features_v2(joined, manifest)
        vec_v2 = features_to_vector(features_v2)

        # v3 features (appended when data is available)
        tracker = DataQualityTracker()
        features_v3 = extract_v3_features(joined, tracker)
        vec_v3 = v3_features_to_vector(features_v3)

        # On first record, decide which v3 columns to include
        if v3_available is None:
            avail_report = feature_availability_report(joined)
            v3_available = avail_report["available_features"]
            manifest["v3_availability"] = avail_report

        X.append(vec_v2 + vec_v3)
        y.append(int(record.get("target", 0)))

    return X, y

def _train_calibrators(preds: List[float], targets: List[int]) -> Dict[str, Optional[Any]]:
    """Train calibrators and return dict."""
    calibrators = {}
    
    # Platt scaling
    try:
        platt = PlattScaler(lr=0.01, epochs=1000)
        platt.fit(preds, targets)
        calibrators["platt"] = platt
    except:
        calibrators["platt"] = None
    
    # Isotonic scaling
    try:
        iso = IsotonicScaler(n_bins=10)
        iso.fit(preds, targets)
        calibrators["isotonic"] = iso
    except:
        calibrators["isotonic"] = None
    
    # No calibration baseline
    calibrators["none"] = None
    
    return calibrators

def _brier_score(preds: List[float], targets: List[int]) -> float:
    """Compute Brier score."""
    if not preds:
        return 1.0
    return sum((p - t) ** 2 for p, t in zip(preds, targets)) / len(preds)

def _expected_calibration_error(preds: List[float], targets: List[int], n_bins: int = 10) -> float:
    """Compute ECE."""
    if not preds:
        return 1.0
    
    bins = [[] for _ in range(n_bins)]
    bin_targets = [[] for _ in range(n_bins)]
    
    for p, t in zip(preds, targets):
        bin_idx = min(int(p * n_bins), n_bins - 1)
        bins[bin_idx].append(p)
        bin_targets[bin_idx].append(t)
    
    ece = 0.0
    total = len(preds)
    
    for bin_preds, bin_tgts in zip(bins, bin_targets):
        if bin_preds:
            avg_pred = sum(bin_preds) / len(bin_preds)
            avg_target = sum(bin_tgts) / len(bin_tgts)
            ece += (len(bin_preds) / total) * abs(avg_pred - avg_target)
    
    return ece
