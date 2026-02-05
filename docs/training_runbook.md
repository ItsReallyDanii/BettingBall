# Training Runbook - v1.11

## Quick Start

### Full Pipeline (First Time)
```bash
# 1. Build dataset from local CSVs
python -m src.main --build_dataset

# 2. Train model with hyperparameter search
python -m src.main --train

# 3. Evaluate and check gates
python -m src.main --evaluate_train

# 4. Score odds using trained model
python -m src.main --score_odds --odds_input data/odds/sample_lines.csv
```

### Retraining
```bash
# Force retrain (overwrites existing model)
python -m src.main --train --force_retrain

# Re-evaluate
python -m src.main --evaluate_train
```

## Command Reference

### --build_dataset
Builds training dataset from local CSVs.

**Input Files:**
- `data/raw/games.csv`
- `data/raw/players.csv`
- `data/raw/teams.csv`
- `data/raw/targets.csv`

**Output:**
- `outputs/training/dataset.csv`
- `outputs/training/dataset_metadata.json`

**Success Criteria:**
- Exit code 0
- Dataset file created
- Total records > 0

**Example Output:**
```
✅ Built dataset: 400 records
   Output: outputs/training\dataset.csv
```

### --train
Trains model with hyperparameter search and calibration.

**Prerequisites:**
- Dataset must exist (run `--build_dataset` first)

**Options:**
- `--model_dir PATH`: Output directory (default: `outputs/models`)
- `--force_retrain`: Overwrite existing model

**Output Artifacts:**
- `outputs/models/model.pkl`
- `outputs/models/calibrator.pkl` (if calibration used)
- `outputs/models/model_meta.json`
- `outputs/models/feature_manifest.json`

**Success Criteria:**
- Exit code 0
- All artifacts created
- Val Brier < 0.25 (reasonable baseline)

**Example Output:**
```
✅ Training complete
   Val Brier: 0.1267
   Val ECE: 0.0397
   Best params: {'lr': 0.01, 'epochs': 300, 'l2': 0.001, 'calibrator': 'platt'}
```

### --evaluate_train
Evaluates trained model and applies production gates.

**Prerequisites:**
- Trained model must exist

**Options:**
- `--model_dir PATH`: Model directory (default: `outputs/models`)
- `--train_profile {dev,train,freeze}`: Gate profile (default: `dev`)

**Output Artifacts:**
- `outputs/audits/train_report.json`
- `outputs/audits/train_blockers.json`

**Success Criteria:**
- Exit code 0 if verdict is GO
- Exit code 1 if verdict is NO-GO

**Example Output (NO-GO):**
```
==================================================
VERDICT: NO-GO
==================================================
Test Brier: 0.1311
Test ECE: 0.0728
Test Hit Rate: 0.8625

⚠️  BLOCKERS:
   [critical] insufficient_samples: Total samples 400 < 500
   [critical] high_ece: Test ECE 0.0728 > 0.07
```

**Example Output (GO):**
```
==================================================
VERDICT: GO
==================================================
Test Brier: 0.1245
Test ECE: 0.0651
Test Hit Rate: 0.8750
```

### --score_odds
Scores odds using trained model (or baseline fallback).

**Prerequisites:**
- Odds input file (CSV or JSON)
- Optionally: trained model (uses baseline if not found)

**Options:**
- `--odds_input PATH`: Path to odds file (required)
- `--join_tolerance_minutes N`: Timestamp tolerance (default: 180)

**Output Artifacts:**
- `outputs/recommendations/recs.json`
- `outputs/recommendations/recs.csv`
- `outputs/recommendations/summary.json`
- `outputs/recommendations/join_audit.json`

**Success Criteria:**
- Exit code 0
- Recommendations generated
- `model_source` field indicates "trained" or "baseline_fallback"

**Example Output:**
```
✅ Scored 18 opportunities
   PASS: 17
   LEAN: 1
   BET:  0
```

## Inspection Commands

### View Model Metadata
```bash
Get-Content outputs\models\model_meta.json | ConvertFrom-Json | ConvertTo-Json -Depth 5
```

### View Training Report
```bash
Get-Content outputs\audits\train_report.json | ConvertFrom-Json | Select-Object verdict,test_metrics,blockers | ConvertTo-Json -Depth 5
```

### View Feature Manifest
```bash
Get-Content outputs\models\feature_manifest.json | ConvertFrom-Json | Select-Object -ExpandProperty column_order
```

### Check Model Source in Recommendations
```bash
Get-Content outputs\recommendations\recs.json | ConvertFrom-Json | Select-Object -ExpandProperty recommendations | Select-Object -First 1 | Select-Object model_source,model_prob_calibrated
```

## Troubleshooting

### Error: "Dataset not found"
**Cause:** `--train` called before `--build_dataset`

**Solution:**
```bash
python -m src.main --build_dataset
python -m src.main --train
```

### Error: "Model already exists"
**Cause:** Attempting to train when model exists without `--force_retrain`

**Solution:**
```bash
python -m src.main --train --force_retrain
```

### NO-GO Verdict: "insufficient_samples"
**Cause:** Total dataset size < 500 samples

**Solution:**
- Collect more real data
- For testing only: adjust gate thresholds in `src/evaluate_v2.py`

### NO-GO Verdict: "high_brier" or "high_ece"
**Cause:** Model performance below production thresholds

**Solution:**
- Improve feature engineering
- Collect more/better training data
- Tune hyperparameters
- Consider more sophisticated model

### Warning: "Failed to load trained model"
**Cause:** Corrupted model file or version mismatch

**Solution:**
```bash
# Retrain from scratch
python -m src.main --train --force_retrain
```

## Production Workflow

### Initial Deployment
```bash
# 1. Build dataset
python -m src.main --build_dataset

# 2. Train model
python -m src.main --train

# 3. Evaluate (must pass gates)
python -m src.main --evaluate_train
# Exit code must be 0 (GO verdict)

# 4. Test inference
python -m src.main --score_odds --odds_input data/odds/sample_lines.csv

# 5. Verify model_source is "trained"
Get-Content outputs\recommendations\recs.json | ConvertFrom-Json | Select-Object -ExpandProperty recommendations | Select-Object -First 1 | Select-Object model_source
```

### Retraining Cadence
- **Frequency**: Weekly or when new data available
- **Trigger**: Significant performance degradation or new features
- **Process**:
  1. Update raw CSVs with new data
  2. Rebuild dataset
  3. Retrain with `--force_retrain`
  4. Evaluate gates
  5. Deploy only if GO verdict

### Rollback Procedure
```bash
# Backup current model
Copy-Item outputs\models\model.pkl outputs\models\model_backup.pkl
Copy-Item outputs\models\calibrator.pkl outputs\models\calibrator_backup.pkl

# If new model fails, restore backup
Move-Item -Force outputs\models\model_backup.pkl outputs\models\model.pkl
Move-Item -Force outputs\models\calibrator_backup.pkl outputs\models\calibrator.pkl
```

## Expected Artifacts

### After --build_dataset
```
outputs/training/
├── dataset.csv
└── dataset_metadata.json
```

### After --train
```
outputs/models/
├── model.pkl
├── calibrator.pkl
├── model_meta.json
└── feature_manifest.json
```

### After --evaluate_train
```
outputs/audits/
├── train_report.json
└── train_blockers.json
```

### After --score_odds
```
outputs/recommendations/
├── recs.json
├── recs.csv
├── summary.json
└── join_audit.json
```

## Performance Benchmarks

### Synthetic Data (400 samples)
- **Val Brier**: ~0.127
- **Val ECE**: ~0.040
- **Test Brier**: ~0.131
- **Test ECE**: ~0.073
- **Verdict**: NO-GO (insufficient samples, high ECE)

### Expected Real Data (500+ samples)
- **Val Brier**: < 0.19
- **Val ECE**: < 0.07
- **Test Brier**: < 0.19
- **Test ECE**: < 0.07
- **Verdict**: GO (if thresholds met)
