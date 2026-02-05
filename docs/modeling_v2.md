# Model Training v2 - v1.11

## Overview
v1.11 introduces a complete offline training pipeline with leak-safe temporal splits, real feature engineering, hyperparameter search, calibration selection, and truthful evaluation gates.

## Architecture

### Data Flow
```
Raw CSVs → Dataset Builder → Temporal Split → Feature Extraction → Model Training → Calibration → Evaluation → Deployment
```

### Key Components

1. **Dataset Builder** (`src/dataset.py`)
   - Joins games, players, teams, targets
   - Enforces chronological ordering
   - Validates schema and tracks null stats
   - Output: `outputs/training/dataset.csv` + metadata

2. **Feature Engineering** (`src/features_v2.py`)
   - 22 real features across 5 groups
   - Leak-safe: no target/outcome fields at inference
   - Robust missing value handling
   - Deterministic column ordering

3. **Training** (`src/train.py`)
   - Enhanced logistic regression with L2 regularization
   - Hyperparameter grid search on validation set
   - Platt scaling calibration
   - Model selection: minimize Brier, tie-break on ECE

4. **Evaluation** (`src/evaluate_v2.py`)
   - Test metrics: Brier, ECE, log loss, hit rate
   - Rolling fold validation
   - Overfitting diagnostics
   - Gate enforcement with blockers

## Feature Groups

### Market Features
- `implied_prob_open`: Opening line implied probability
- `implied_prob_current`: Current line implied probability  
- `line_move_abs`: Absolute line movement
- `line_move_direction`: Direction of line movement

### Team Context
- `pace_diff`: Team pace differential
- `off_rating_diff`: Offensive rating differential
- `def_rating_diff`: Defensive rating differential
- `home_indicator`: Home/away indicator

### Rest/Fatigue
- `days_rest_team`: Days rest for team
- `days_rest_opp`: Days rest for opponent
- `back_to_back`: Back-to-back game indicator
- `travel_fatigue_index`: Travel fatigue metric

### Player Form
- `points_avg_5`: 5-game points average
- `trend_points_slope_5`: Points trend slope
- `usage_rate_last_5`: Usage rate (last 5 games)
- `true_shooting_last_5`: True shooting % (last 5 games)

### Threshold Relation
- `proj_minus_threshold`: Projection minus betting line

### Interactions
- `pace_usage_interaction`: Pace × usage rate
- `rest_b2b_interaction`: Days rest × back-to-back

## Leakage Prevention

### Temporal Split Rules
1. Train/val/test split by timestamp (chronological only)
2. No shuffling or random sampling
3. Explicit guard: fail if `last_train_ts >= first_test_ts`
4. Rolling folds maintain temporal order

### Feature Construction
- Never use `target` or `actual` fields at inference
- All features derivable from pre-game data only
- No look-ahead bias in rolling statistics

## Model Selection

### Selection Criteria
1. **Primary**: Minimize validation Brier score
2. **Tie-breaker**: Lower ECE
3. **Calibration**: Compare Platt vs. none, select best

### Hyperparameter Grid
```python
{
    "lr": [0.01, 0.005],
    "epochs": [200, 300],
    "l2_lambda": [0.01, 0.001]
}
```

## Evaluation Gates

### Gate Thresholds (train profile)
- `min_sample_size`: 500
- `max_mean_brier`: 0.19
- `max_mean_ece`: 0.07
- `max_fold_variance`: 0.05
- `max_train_val_gap`: 0.10

### Verdict Logic
- **GO**: All critical gates pass
- **NO-GO**: Any critical blocker present
- **Warnings**: Non-critical issues logged

## Artifacts

### Training Artifacts
- `outputs/models/model.pkl`: Trained model
- `outputs/models/calibrator.pkl`: Calibrator (if used)
- `outputs/models/model_meta.json`: Metadata + metrics
- `outputs/models/feature_manifest.json`: Feature provenance

### Evaluation Artifacts
- `outputs/audits/train_report.json`: Full evaluation report
- `outputs/audits/train_blockers.json`: Gate blockers only

### Dataset Artifacts
- `outputs/training/dataset.csv`: Joined training dataset
- `outputs/training/dataset_metadata.json`: Schema stats

## Inference Integration

### Model Loading Priority
1. Check for `outputs/models/model.pkl` (trained model)
2. If found, use trained model + calibrator
3. If not found, fallback to baseline model
4. Add `model_source` field to all recommendations

### Output Fields
- `model_source`: "trained" | "baseline_fallback"
- `model_prob_raw`: Raw model probability
- `model_prob_calibrated`: Calibrated probability
- All existing diagnostic fields preserved

## Reproducibility

### Determinism Guarantees
- Fixed random seeds (default: 42)
- Stable sort keys (timestamp, event_id)
- Explicit column ordering
- No non-deterministic operations

### Verification
```bash
# Train twice with same seed
python -m src.main --train --force_retrain
python -m src.main --train --force_retrain

# Metrics should be identical
diff outputs/models/model_meta.json outputs/models/model_meta_backup.json
```

## Known Limitations

1. **Small Sample Size**: Synthetic data (400 records) fails gates
   - Real deployment requires 500+ samples
   - Current NO-GO verdict is truthful

2. **Simple Model**: Logistic regression baseline
   - Gradient boosting would improve performance
   - Requires additional dependencies

3. **Calibration**: Only Platt scaling implemented
   - Isotonic regression skipped (graceful degradation)

4. **Feature Engineering**: Simplified projections
   - Real system would use more sophisticated models
   - Current heuristic: `points_avg + trend_slope`

## Next Steps

1. **Data Collection**: Accumulate real game data to reach 500+ samples
2. **Feature Expansion**: Add more sophisticated projections
3. **Model Upgrades**: Implement gradient boosting if available
4. **Calibration**: Add isotonic regression
5. **CI Integration**: Add training pipeline to automated tests
