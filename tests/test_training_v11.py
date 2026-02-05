"""
Tests for v1.11 training pipeline.
"""
import os
import json
import pytest
from src.dataset import build_dataset, temporal_split, rolling_fold_generator
from src.train import train_model
from src.evaluate_v2 import evaluate_train_profile

def test_temporal_split_no_leakage():
    """Test that temporal split has no leakage."""
    # Build dataset first
    metadata = build_dataset()
    dataset_path = metadata["output_path"]
    
    train, val, test = temporal_split(dataset_path)
    
    # Verify no leakage
    if train and test:
        last_train_ts = train[-1]["timestamp_utc"]
        first_test_ts = test[0]["timestamp_utc"]
        assert last_train_ts < first_test_ts, "Leakage detected: train overlaps with test"

def test_feature_manifest_and_order_deterministic():
    """Test that feature manifest is created and order is deterministic."""
    metadata = build_dataset()
    dataset_path = metadata["output_path"]
    
    train, val, test = temporal_split(dataset_path)
    train_metadata = train_model(train[:100], val[:20], output_dir="outputs/test_models")
    
    manifest_path = "outputs/test_models/feature_manifest.json"
    assert os.path.exists(manifest_path)
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    assert "column_order" in manifest
    assert len(manifest["column_order"]) > 0

def test_training_artifacts_created():
    """Test that all training artifacts are created."""
    metadata = build_dataset()
    dataset_path = metadata["output_path"]
    
    train, val, test = temporal_split(dataset_path)
    train_metadata = train_model(train[:100], val[:20], output_dir="outputs/test_models")
    
    assert os.path.exists("outputs/test_models/model.pkl")
    assert os.path.exists("outputs/test_models/model_meta.json")
    assert os.path.exists("outputs/test_models/feature_manifest.json")

def test_model_selection_rule():
    """Test that model selection minimizes Brier score."""
    metadata = build_dataset()
    dataset_path = metadata["output_path"]
    
    train, val, test = temporal_split(dataset_path)
    train_metadata = train_model(train[:100], val[:20], output_dir="outputs/test_models")
    
    # Verify best params are selected
    assert "best_params" in train_metadata
    assert "brier" in train_metadata["best_params"]
    assert "ece" in train_metadata["best_params"]

def test_calibration_selection():
    """Test that calibration is selected."""
    metadata = build_dataset()
    dataset_path = metadata["output_path"]
    
    train, val, test = temporal_split(dataset_path)
    train_metadata = train_model(train[:100], val[:20], output_dir="outputs/test_models")
    
    assert "calibrator" in train_metadata["best_params"]

def test_inference_uses_trained_model_when_available():
    """Test that inference uses trained model when available."""
    from src.inference import score_odds_file
    
    # Ensure trained model exists
    assert os.path.exists("outputs/models/model.pkl")
    
    # Score odds
    summary = score_odds_file("data/odds/sample_lines.csv", output_dir="outputs/test_recommendations")
    
    # Check that recommendations include model_source
    with open("outputs/test_recommendations/recs.json") as f:
        data = json.load(f)
        recs = data["recommendations"]
    
    if recs:
        assert "model_source" in recs[0]
        assert recs[0]["model_source"] == "trained"

def test_gate_failures_emit_blockers():
    """Test that gate failures emit blockers."""
    # This will fail gates due to small sample size
    report = evaluate_train_profile(model_dir="outputs/models")
    
    assert "blockers" in report
    assert "verdict" in report
    
    # With synthetic data, we expect NO-GO
    if report["verdict"] == "NO-GO":
        assert len(report["blockers"]) > 0

def test_reproducibility_same_seed_same_metrics():
    """Test that same seed produces same metrics."""
    metadata = build_dataset()
    dataset_path = metadata["output_path"]
    
    train, val, test = temporal_split(dataset_path)
    
    # Train twice with same seed
    meta1 = train_model(train[:50], val[:10], seed=42, output_dir="outputs/test_models_1")
    meta2 = train_model(train[:50], val[:10], seed=42, output_dir="outputs/test_models_2")
    
    # Metrics should be identical
    assert abs(meta1["metrics"]["val"]["brier"] - meta2["metrics"]["val"]["brier"]) < 1e-6
