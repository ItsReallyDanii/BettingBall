# Evaluation Methodology

## Metrics

### Primary Metrics
| Metric | Definition | Target |
|--------|-----------|--------|
| **Accuracy** | Proportion of correct predictions | ≥ 0.55 |
| **ROC-AUC** | Area under receiver operator curve | ≥ 0.60 |
| **Log Loss** | Cross-entropy loss | ≤ 0.65 |
| **Brier Score** | Mean squared error of probabilities | ≤ 0.20 (freeze) |

### Calibration Metrics
| Metric | Definition | Target |
|--------|-----------|--------|
| **ECE** (Expected Calibration Error) | Mean absolute deviation between predicted and actual | ≤ 0.08 (freeze) |
| **Calibration Bins** | Accuracy in probability deciles | Within ±0.10 of diagonal |

## Evaluation Workflows

### 1. Standard Backtest (`--backtest`)
- **Procedure**: 80/20 train/test split (stratified)
- **Order**: Chronological (temporal ordering preserved)
- **Output**: Model report with pre/post-calibration metrics
- **Use case**: Single snapshot evaluation

### 2. Rolling Backtest (`--rolling_backtest`)
- **Procedure**: K-fold cross-validation (default K=5)
- **Folds**: Each fold trains on prior data, tests on held-out window
- **Order**: Chronological (no future data leakage)
- **Output**: Per-fold metrics + aggregate gate decisions
- **Use case**: Robust performance estimation over time

### 3. Production Readiness (`--readiness`)
- **Checks**:
  - Model card (architecture, training data, limitations)
  - Data card (schema, distributions, quality)
  - Freeze blockers (gate profile violations)
  - Artifact manifest (reproducibility metadata)
- **Output**: Single verdict (pass/fail) per gate profile
- **Use case**: Pre-deployment verification

## Gate Profiles

### Development Gate (`dev`)
- Minimum sample size: 80 examples
- Maximum ECE: 0.12
- Maximum Brier: 0.26
- Use: Early-stage model exploration

### Freeze Gate (`freeze`)
- Minimum sample size: 300 examples
- Maximum ECE: 0.08
- Maximum Brier: 0.20
- Use: Production deployment checkpoint

## Data Quality

Validation checks (`data_quality.py`):
- Missing values in required fields
- Duplicate game_ids
- Temporal ordering (games ordered by date)
- Label distributions (binary targets)

## Audit Trail

All evaluation artifacts are stored in `outputs/audits/`:
- `model_report.json` - Backtest results (accuracy, calibration)
- `rolling_backtest_report.json` - Per-fold metrics
- `readiness_report.json` - Gate verdicts
- `calibration_bins.json` - Probability decile analysis
- `eval_metrics.json` - Metric snapshots over time

## Reproducibility

Each evaluation records:
- Git commit hash
- Random seed
- File hashes (MD5)
- Timestamp (UTC)
- Command-line arguments

## Leakage Gate (V1 Safety)

Six checks enforced before any release claim is permitted:

| Check | Description |
|-------|-------------|
| no_future_rows_in_features | No train row timestamps after earliest test row |
| train_test_split_time_based | max(train_ts) <= min(test_ts) |
| no_target_leak_columns | final_score, postgame_stats, settled_odds excluded from features |
| scaler_fit_on_train_only | Scaler never sees validation/test data |
| calibration_fit_on_validation_only | Calibrator fitted on validation split only |
| report_separate_train_valid_test | Metrics reported per split, not aggregated |

If **any** check fails: `release_status = blocked_leakage_risk`.

## Generalization Gate (V1 Safety)

| Threshold | Minimum |
|-----------|---------|
| Games | 1000 |
| Teams covered | 20 |
| Day span | 60 |

Below any threshold: `status = prototype_only`. Forbidden claims suppressed automatically.
