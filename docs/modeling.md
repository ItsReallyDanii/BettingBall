# Modeling Approach

## Pipeline Architecture
The modeling pipeline consists of distinct, composable stages:

```
Data Ingestion (connectors.py)
    ↓
Schema Validation (schemas.py)
    ↓
Baseline Model (model_baseline.py)
    ↓
Calibration (calibration.py)
    ↓
Reasoning & Audit (llm.py, audit.py)
    ↓
Validator (validator.py)
```

## Baseline Model
- **Algorithm**: Logistic regression
- **Purpose**: Establish minimal predictive baseline for sports betting
- **Input**: Player/team/game features from CSVs
- **Output**: Probability estimate for event (0.0–1.0)

## Probability Calibration
- **Method**: Platt scaling (post-hoc sigmoid calibration)
- **Purpose**: Ensure predicted probabilities match actual outcomes
- **Metrics**:
  - Expected Calibration Error (ECE) ≤ 0.08 (freeze gate)
  - Brier Score ≤ 0.20 (freeze gate)
  - Calibration bins (probability deciles)

## Evaluation Strategy
- **Train/Test Split**: 80/20 stratified split
- **Rolling Window**: K-fold chronological cross-validation (default K=5)
- **Gate Profiles**:
  - **dev**: Sample ≥ 80, ECE ≤ 0.12, Brier ≤ 0.26
  - **freeze**: Sample ≥ 300, ECE ≤ 0.08, Brier ≤ 0.20

## Data Assumptions
- Features are numeric (floats or integers)
- Targets are binary (0/1) events
- Data is temporally ordered (games by date)
- No missing values in features (validation in `data_quality.py`)

## Reproducibility
All models are reproducible via:
- Fixed random seeds
- Git commit hash recording
- File content hashes (MD5)
- Deterministic mock mode (`DRY_RUN=true`)
