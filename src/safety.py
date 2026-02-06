"""
Safety hardening layer for BettingBall V1.

Implements:
- Deterministic confidence grading (grade_map)
- Factor risk classification (risk_map)
- Leakage gate (structured checks + block on failure)
- Generalization gate (prototype_only enforcement)
- Prediction output contract builder
"""
import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

# ---------------------------------------------------------------------------
# A) Config constants (mirrors configs/safety.yaml)
# ---------------------------------------------------------------------------

GENERALIZATION_POLICY = {
    "min_games_required": 1000,
    "min_teams_covered": 20,
    "min_days_span": 60,
    "status_if_below_threshold": "prototype_only",
    "allowed_claims_when_prototype_only": [
        "in-sample behavior only",
        "limited out-of-sample signal",
    ],
    "forbidden_claims_when_prototype_only": [
        "production-ready edge",
        "stable market alpha",
    ],
}

GRADE_MAP = {
    "A_min": 0.70,
    "B_min": 0.60,
    "C_min": 0.55,
    "D_min": 0.50,
}

RISK_MAP: Dict[str, List[str]] = {
    "low": [
        "boxscore", "closing_odds", "rest_days", "home_away",
        # v3 feature families (verified data sources)
        "home_away_form_delta", "pace_delta",
        "usage_trend_5g", "minutes_trend_5g", "shooting_eff_trend_5g",
    ],
    "medium": [
        "pace_proxy", "matchup_derived",
        # v3 feature families (derived / partially verified)
        "injury_impact_score", "matchup_edge_score",
    ],
    "high": ["sentiment", "news", "manual_narrative", "unstable_proxy"],
}

LEAKAGE_CHECKS_REQUIRED = [
    "no_future_rows_in_features",
    "train_test_split_time_based",
    "no_target_leak_columns",
    "scaler_fit_on_train_only",
    "calibration_fit_on_validation_only",
    "report_separate_train_valid_test",
]

TARGET_LEAK_COLUMNS = {"final_score", "postgame_stats", "settled_odds"}

FAIL_BEHAVIOR = "block_release_claims"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _risk_lookup() -> Dict[str, str]:
    """Build factor_name -> risk_level lookup from RISK_MAP."""
    out: Dict[str, str] = {}
    for level, names in RISK_MAP.items():
        for n in names:
            out[n] = level
    return out

_RISK_LOOKUP = _risk_lookup()


# ---------------------------------------------------------------------------
# Confidence grading (deterministic)
# ---------------------------------------------------------------------------

def grade_confidence(confidence: float) -> str:
    """Map a confidence value [0,1] to a letter grade using GRADE_MAP.

    Returns one of A, B, C, D, F. Deterministic: same input always gives
    the same output.
    """
    if confidence >= GRADE_MAP["A_min"]:
        return "A"
    if confidence >= GRADE_MAP["B_min"]:
        return "B"
    if confidence >= GRADE_MAP["C_min"]:
        return "C"
    if confidence >= GRADE_MAP["D_min"]:
        return "D"
    return "F"


# ---------------------------------------------------------------------------
# Factor risk classification
# ---------------------------------------------------------------------------

def classify_factor_risk(factor_name: str) -> str:
    """Return risk level for a factor name. Defaults to 'medium' if unknown."""
    return _RISK_LOOKUP.get(factor_name, "medium")


# ---------------------------------------------------------------------------
# Leakage gate
# ---------------------------------------------------------------------------

def run_leakage_checks(
    feature_columns: List[str],
    train_timestamps: List[str],
    test_timestamps: List[str],
    scaler_fit_on_train_only: bool = True,
    calibration_fit_on_validation_only: bool = True,
    report_separate_splits: bool = True,
) -> Dict[str, Any]:
    """Run all required leakage checks and return structured report.

    Returns dict with keys:
        checks: list of {name, passed, detail}
        all_passed: bool
        release_status: 'ok' | 'blocked_leakage_risk'
    """
    results: List[Dict[str, Any]] = []

    # 1. no_future_rows_in_features
    future_leak = False
    if train_timestamps and test_timestamps:
        max_train = max(train_timestamps)
        future_in_train = [t for t in train_timestamps if t > min(test_timestamps)]
        future_leak = len(future_in_train) > 0
    results.append({
        "name": "no_future_rows_in_features",
        "passed": not future_leak,
        "detail": "No future rows detected" if not future_leak
        else f"{len(future_in_train)} train rows after earliest test row",
    })

    # 2. train_test_split_time_based
    time_ordered = True
    if train_timestamps and test_timestamps:
        time_ordered = max(train_timestamps) <= min(test_timestamps)
    results.append({
        "name": "train_test_split_time_based",
        "passed": time_ordered,
        "detail": "Temporal split verified" if time_ordered
        else "Train/test temporal ordering violated",
    })

    # 3. no_target_leak_columns
    leak_cols_present = TARGET_LEAK_COLUMNS & set(feature_columns)
    results.append({
        "name": "no_target_leak_columns",
        "passed": len(leak_cols_present) == 0,
        "detail": "No target leak columns in features" if not leak_cols_present
        else f"Leak columns found: {sorted(leak_cols_present)}",
    })

    # 4. scaler_fit_on_train_only
    results.append({
        "name": "scaler_fit_on_train_only",
        "passed": scaler_fit_on_train_only,
        "detail": "Scaler fit restricted to train" if scaler_fit_on_train_only
        else "Scaler was fit on non-train data",
    })

    # 5. calibration_fit_on_validation_only
    results.append({
        "name": "calibration_fit_on_validation_only",
        "passed": calibration_fit_on_validation_only,
        "detail": "Calibration fit on validation only" if calibration_fit_on_validation_only
        else "Calibration was fit on incorrect split",
    })

    # 6. report_separate_train_valid_test
    results.append({
        "name": "report_separate_train_valid_test",
        "passed": report_separate_splits,
        "detail": "Separate split reporting confirmed" if report_separate_splits
        else "Splits not reported separately",
    })

    all_passed = all(r["passed"] for r in results)
    return {
        "checks": results,
        "all_passed": all_passed,
        "release_status": "ok" if all_passed else "blocked_leakage_risk",
    }


# ---------------------------------------------------------------------------
# Generalization gate
# ---------------------------------------------------------------------------

def run_generalization_gate(
    n_games: int,
    n_teams_covered: int,
    day_span: int,
) -> Dict[str, Any]:
    """Evaluate generalization thresholds.

    Returns dict with:
        metrics: {n_games, n_teams_covered, day_span}
        threshold_failures: list of failed threshold names
        status: 'production' | 'prototype_only'
        allowed_claims: list
        forbidden_claims: list
    """
    pol = GENERALIZATION_POLICY
    failures: List[str] = []

    if n_games < pol["min_games_required"]:
        failures.append("min_games_required")
    if n_teams_covered < pol["min_teams_covered"]:
        failures.append("min_teams_covered")
    if day_span < pol["min_days_span"]:
        failures.append("min_days_span")

    is_prototype = len(failures) > 0
    status = "prototype_only" if is_prototype else "production"

    if is_prototype:
        allowed = pol["allowed_claims_when_prototype_only"]
        forbidden = pol["forbidden_claims_when_prototype_only"]
    else:
        allowed = pol["allowed_claims_when_prototype_only"] + pol["forbidden_claims_when_prototype_only"]
        forbidden = []

    return {
        "metrics": {
            "n_games": n_games,
            "n_teams_covered": n_teams_covered,
            "day_span": day_span,
        },
        "threshold_failures": failures,
        "status": status,
        "allowed_claims": allowed,
        "forbidden_claims": forbidden,
    }


# ---------------------------------------------------------------------------
# Prediction output contract builder (B)
# ---------------------------------------------------------------------------

def build_prediction_output(
    hypothesis_id: str,
    market_type: str,
    prediction: Any,
    confidence: float,
    reasoning_short: str,
    factors: List[Dict[str, Any]],
    assumptions: Optional[List[str]] = None,
    missing_data_flags: Optional[List[str]] = None,
    leakage_report: Optional[Dict[str, Any]] = None,
    generalization_gate: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the canonical prediction output contract.

    Rules enforced:
    - confidence_grade derived from grade_map
    - status = prototype_only when below generalization thresholds
    - release_status = blocked_leakage_risk if any leakage check fails
    - reasoning_short max 140 chars (truncated if longer)
    - every factor includes risk tag
    - unknown/missing inputs produce missing_data_flags + assumptions
    """
    assumptions = assumptions or []
    missing_data_flags = missing_data_flags or []

    # Grade
    conf_grade = grade_confidence(confidence)

    # Truncate reasoning
    if len(reasoning_short) > 140:
        reasoning_short = reasoning_short[:137] + "..."

    # Ensure every factor has risk
    enriched_factors = []
    for f in factors:
        ef = dict(f)
        if "risk" not in ef or not ef["risk"]:
            ef["risk"] = classify_factor_risk(ef.get("name", ""))
        enriched_factors.append(ef)

    # Risk summary
    high_risk = [f for f in enriched_factors if f.get("risk") == "high"]
    high_risk_names = [f.get("name", "unknown") for f in high_risk]

    decision_flag = "proceed"
    if len(high_risk) >= 3:
        decision_flag = "review_required"
    if len(high_risk) >= 5:
        decision_flag = "block"

    risk_summary = {
        "high_risk_factor_count": len(high_risk),
        "high_risk_factor_names": high_risk_names,
        "decision_flag": decision_flag,
    }

    # Status from generalization gate
    status = "production"
    if generalization_gate and generalization_gate.get("status") == "prototype_only":
        status = "prototype_only"

    # Release status from leakage gate
    release_status = "ok"
    if leakage_report and not leakage_report.get("all_passed", True):
        release_status = "blocked_leakage_risk"

    return {
        "hypothesis_id": hypothesis_id,
        "market_type": market_type,
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "confidence_grade": conf_grade,
        "reasoning_short": reasoning_short,
        "factors": enriched_factors,
        "risk_summary": risk_summary,
        "assumptions": assumptions,
        "missing_data_flags": missing_data_flags,
        "status": status,
        "release_status": release_status,
    }
