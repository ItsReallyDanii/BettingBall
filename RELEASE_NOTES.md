# Release Notes - v1.9.1-gate-profiles-and-reporting

**Status:** ACTIVE
**Date:** 2026-02-05
**Verdict:** GO (Dev Config) / NO-GO (Freeze Config)

## Overview
This release implements rigorous backtesting infrastructure and data quality controls. It introduces "Rolling Backtesting" to validate model performance across chronological folds and "Gate Profiles" to distinguish between development iteration and strict production releases.

## Key Features

### 1. Rolling Backtest Workflow
*   **Command**: `python -m src.main --rolling_backtest --folds 5`
*   **Mechanism**: Chronologically validates `BaselineModel + PlattScaler` over multiple time horizons.
*   **Artifact**: `outputs/audits/rolling_backtest_report.json`
*   **Metrics**: Tracks ECE, Brier Score, and Hit Rate stability across folds.

### 2. Gate Profiles (`--gate_profile`)
*   **Dev Profile** (`default`):
    *   Designed for local iteration with limited data.
    *   Thresholds: Sample Size >= 80, Mean ECE <= 0.12, Mean Brier <= 0.26.
*   **Freeze Profile**:
    *   Strict gates for production candidates.
    *   Thresholds: Sample Size >= 300, Mean ECE <= 0.08, Mean Brier <= 0.20.

### 3. Data Quality Module
*   **File**: `src/data_quality.py`
*   **Checks**:
    *   Missing values in critical fields.
    *   Duplicate keys in targets.
    *   Label validity (0/1).
    *   Temporal ordering constraints.

### 4. Backtest Enhancements
*   Segregated `training_payload.json` from public `results.json` to prevent data contamination.
*   Strict fail-fast behavior: Process exits with code 1 if *any* gate fails.

## Current Status (v1.9.1)
*   **Dev Profile**: **PASSING**. The baseline model meets loose criteria on the current dataset (~100 samples).
*   **Freeze Profile**: **FAILING**. Requires more data (>300 samples) and slightly better Brier score (<0.20) to pass.

## Verified Commands
1.  `python -m src.main --run` (Ingestion)
2.  `python -m src.main --rolling_backtest --folds 3 --gate_profile dev` (Passes)
3.  `python -m src.main --rolling_backtest --folds 3 --gate_profile freeze` (Fails explicitly)
