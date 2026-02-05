# Evaluation v2 - v1.11

## Overview
Truthful evaluation framework with production gates, overfitting diagnostics, and reliability analysis.

## Metrics

### Primary Metrics

#### Brier Score
- **Formula**: `mean((pred - actual)²)`
- **Range**: [0, 1], lower is better
- **Interpretation**: Mean squared error of probabilistic predictions
- **Gate Threshold**: < 0.19

#### Expected Calibration Error (ECE)
- **Formula**: Weighted average of |avg_pred - avg_actual| across bins
- **Range**: [0, 1], lower is better
- **Interpretation**: How well predicted probabilities match observed frequencies
- **Gate Threshold**: < 0.07

### Secondary Metrics

#### Log Loss
- **Formula**: `-mean(actual * log(pred) + (1-actual) * log(1-pred))`
- **Range**: [0, ∞), lower is better
- **Interpretation**: Penalizes confident wrong predictions heavily

#### Hit Rate (Accuracy)
- **Formula**: `mean(pred >= 0.5 == actual)`
- **Range**: [0, 1], higher is better
- **Interpretation**: Binary classification accuracy at 0.5 threshold

## Evaluation Process

### 1. Temporal Split
```
Dataset (sorted by timestamp)
├── Train (60%)
├── Val (20%)
└── Test (20%)
```

**Leakage Guard**: Fails if `last_train_ts >= first_test_ts`

### 2. Test Set Evaluation
- Load trained model + calibrator
- Score test set
- Compute all metrics
- Generate reliability bins

### 3. Rolling Fold Validation
- 3 folds with expanding train window
- Walk-forward validation
- Compute fold variance
- Flag high variance as warning

### 4. Overfitting Diagnostics
- Compare train vs val vs test Brier
- Flag large train-val gap (> 0.10)
- Report fold variance

### 5. Gate Application
- Check all thresholds
- Classify blockers by severity
- Emit GO/NO-GO verdict

## Gate Profiles

### dev (default)
Relaxed thresholds for development:
- `min_sample_size`: 100
- `max_mean_brier`: 0.25
- `max_mean_ece`: 0.15
- `max_fold_variance`: 0.10
- `max_train_val_gap`: 0.15

### train
Production training gates:
- `min_sample_size`: 500
- `max_mean_brier`: 0.19
- `max_mean_ece`: 0.07
- `max_fold_variance`: 0.05
- `max_train_val_gap`: 0.10

### freeze
Strict gates for baseline freeze:
- `min_sample_size`: 500
- `max_mean_brier`: 0.18
- `max_mean_ece`: 0.06
- `max_fold_variance`: 0.03
- `max_train_val_gap`: 0.08

## Blocker Classification

### Critical Blockers
Result in NO-GO verdict:
- `insufficient_samples`: Total samples < threshold
- `high_brier_score`: Test Brier > threshold
- `high_ece`: Test ECE > threshold
- `model_not_found`: Model file missing

### Warning Blockers
Logged but don't block deployment:
- `high_fold_variance`: Fold variance > threshold
- `high_train_val_gap`: Train-val gap > threshold

## Reliability Analysis

### Reliability Bins
- Divide predictions into 10 bins by probability
- For each bin, compute:
  - `avg_pred`: Mean predicted probability
  - `avg_target`: Mean actual outcome
  - `calibration_error`: |avg_pred - avg_target|
  - `count`: Number of predictions in bin

### Perfect Calibration
- `avg_pred ≈ avg_target` for all bins
- Calibration errors near 0
- Diagonal line on reliability diagram

### Typical Issues
- **Overconfidence**: avg_pred > avg_target (especially at extremes)
- **Underconfidence**: avg_pred < avg_target
- **Poor separation**: All predictions clustered near 0.5

## Output Artifacts

### train_report.json
```json
{
  "verdict": "GO" | "NO-GO",
  "timestamp": "ISO8601",
  "total_samples": 400,
  "test_metrics": {
    "brier": 0.1311,
    "ece": 0.0728,
    "log_loss": 0.4123,
    "hit_rate": 0.8625
  },
  "fold_metrics": {
    "mean_brier": 0.1289,
    "fold_variance": 0.0012,
    "folds": [...]
  },
  "overfitting_diagnostics": {
    "train_brier": 0.1245,
    "val_brier": 0.1267,
    "test_brier": 0.1311,
    "train_val_gap": 0.0022
  },
  "reliability_bins": [...],
  "gate_thresholds": {...},
  "blockers": [...]
}
```

### train_blockers.json
```json
{
  "verdict": "NO-GO",
  "blockers": [
    {
      "id": "insufficient_samples",
      "severity": "critical",
      "message": "Total samples 400 < 500"
    },
    {
      "id": "high_ece",
      "severity": "critical",
      "message": "Test ECE 0.0728 > 0.07"
    }
  ]
}
```

## Interpretation Guide

### Good Model
- Test Brier < 0.15
- Test ECE < 0.05
- Train-val-test Brier within 0.02
- Fold variance < 0.01
- Reliability bins show diagonal pattern

### Acceptable Model
- Test Brier < 0.19
- Test ECE < 0.07
- Train-val-test Brier within 0.10
- Fold variance < 0.05
- Some calibration drift but manageable

### Poor Model
- Test Brier > 0.20
- Test ECE > 0.10
- Large train-val gap (> 0.15)
- High fold variance (> 0.10)
- Severe calibration issues

## Troubleshooting

### High Brier Score
**Causes:**
- Insufficient features
- Poor feature quality
- Model too simple
- Insufficient training data

**Solutions:**
- Add more predictive features
- Improve feature engineering
- Collect more training data
- Try more sophisticated model

### High ECE
**Causes:**
- Poor calibration
- Overconfident predictions
- Distribution shift

**Solutions:**
- Use Platt scaling or isotonic regression
- Adjust model regularization
- Retrain on more recent data

### High Fold Variance
**Causes:**
- Small fold sizes
- Non-stationary data
- Overfitting

**Solutions:**
- Increase fold sizes
- Add more regularization
- Check for data quality issues

### Large Train-Val Gap
**Causes:**
- Overfitting
- Distribution shift
- Data leakage

**Solutions:**
- Increase regularization
- Reduce model complexity
- Verify no leakage in features
- Check temporal consistency

## CI Integration

### Automated Checks
```bash
# Run evaluation
python -m src.main --evaluate_train

# Check exit code
if [ $? -eq 0 ]; then
  echo "✅ Model passed gates"
  # Deploy model
else
  echo "❌ Model failed gates"
  # Block deployment
fi
```

### Continuous Monitoring
- Track metrics over time
- Alert on degradation
- Retrain on schedule or trigger
- Maintain metric history

## Best Practices

1. **Always evaluate on held-out test set**
   - Never tune on test set
   - Use val set for hyperparameter search
   - Test set only for final evaluation

2. **Respect temporal ordering**
   - No shuffling
   - No look-ahead bias
   - Validate split timestamps

3. **Monitor calibration**
   - Check reliability bins
   - Use calibration when needed
   - Verify on recent data

4. **Track overfitting**
   - Compare train/val/test metrics
   - Use regularization
   - Validate on folds

5. **Be truthful**
   - Report actual metrics
   - Don't cherry-pick thresholds
   - Accept NO-GO verdicts when warranted
