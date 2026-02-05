import os
import json
import csv
import pickle
from datetime import datetime, timezone
from typing import Dict, Any, List

from src.odds_ingest import parse_odds_csv, parse_odds_json
from src.risk import recommend
from src.model_baseline import BaselineModel
from src.feature_join import join_odds_with_context

def score_odds_file(
    input_path: str,
    output_dir: str = "outputs/recommendations",
    join_tolerance_minutes: int = 180
) -> Dict[str, Any]:
    """
    Score odds file and generate recommendations.
    
    Args:
        input_path: Path to CSV or JSON odds file
        output_dir: Directory for output artifacts
        join_tolerance_minutes: Tolerance for timestamp-based game matching
    
    Returns:
        Summary statistics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse input
    if input_path.endswith(".csv"):
        records = parse_odds_csv(input_path)
    elif input_path.endswith(".json"):
        records = parse_odds_json(input_path)
    else:
        raise ValueError("Input must be .csv or .json")
    
    # Join with context
    join_result = join_odds_with_context(
        records,
        join_tolerance_minutes=join_tolerance_minutes
    )
    
    # Write join audit
    with open(os.path.join(output_dir, "join_audit.json"), "w") as f:
        json.dump(join_result["audit"], f, indent=2)
    
    # Load model - prefer trained model, fallback to baseline
    model_source = "baseline_fallback"
    model = None
    calibrator = None
    
    trained_model_path = "outputs/models/model.pkl"
    trained_calibrator_path = "outputs/models/calibrator.pkl"
    
    if os.path.exists(trained_model_path):
        # Use trained model
        try:
            with open(trained_model_path, "rb") as f:
                model = pickle.load(f)
            model_source = "trained"
            
            if os.path.exists(trained_calibrator_path):
                with open(trained_calibrator_path, "rb") as f:
                    calibrator = pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to load trained model: {e}. Falling back to baseline.")
            model = None
    
    if model is None:
        # Fallback to baseline
        from src.model_baseline import BaselineModel
        model = BaselineModel()
        model_source = "baseline_fallback"
        
        # Try baseline calibrator
        baseline_calibrator_path = "outputs/models/calibrator.pkl"
        if os.path.exists(baseline_calibrator_path):
            try:
                with open(baseline_calibrator_path, "rb") as f:
                    calibrator = pickle.load(f)
            except:
                pass
    
    # Score matched records only
    recommendations = []
    for rec in join_result["matched"]:
        scored = _score_single_record(rec, model, calibrator, model_source)
        recommendations.append(scored)
    
    # Sort deterministically
    recommendations.sort(key=lambda x: (x["timestamp_utc"], x["event_id"]))
    
    # Write outputs
    _write_recommendations_json(recommendations, output_dir)
    _write_recommendations_csv(recommendations, output_dir)
    summary = _write_summary(recommendations, join_result["audit"], output_dir)
    
    return summary

def _score_single_record(rec: Dict[str, Any], model, calibrator, model_source: str) -> Dict[str, Any]:
    """Score a single odds record with joined context."""
    # Extract features from joined data
    features = rec.get("features", [1.0, 0.0, 0.0, 0.0])
    
    # Get raw model probability
    raw_prob = model.predict_proba(features)
    
    # Apply calibration if available
    if calibrator:
        model_prob = calibrator.predict(raw_prob)
    else:
        model_prob = raw_prob
    
    # Calculate confidence grade based on probability distance from 0.5 and data completeness
    missing_fields = rec.get("missing_fields", [])
    confidence_grade = _calculate_confidence_grade(model_prob, missing_fields)
    
    # Uncertainty note
    # uncertainty_note = _generate_uncertainty_note(confidence_grade, missing_fields, calibrator is not None)
    
    # Calculate payout
    price = rec["price_american"]
    if price > 0:
        payout_per_unit = price / 100
    else:
        payout_per_unit = 100 / abs(price)
    
    # Calculate EV
    expected_value_unit = model_prob * payout_per_unit - (1 - model_prob) * 1.0
    
    # Get recommendation
    decision = recommend(
        model_prob=model_prob,
        implied_prob=rec["implied_prob"],
        confidence_grade=confidence_grade,
        expected_value_unit=expected_value_unit
    )
    
    # Build output record with diagnostics
    return {
        "event_id": rec["event_id"],
        "market_type": rec["market_type"],
        "selection": rec["selection"],
        "player_id": rec.get("player_id", ""),
        "team_id": rec.get("team_id", ""),
        "line": rec.get("line"),
        "price_american": rec["price_american"],
        "sportsbook": rec["sportsbook"],
        "timestamp_utc": rec["timestamp_utc"],
        "match_status": rec.get("match_status", "unknown"),
        "join_keys_used": rec.get("join_keys_used", []),
        "missing_fields": missing_fields,
        "model_source": model_source,
        "model_prob_raw": round(raw_prob, 6),
        "model_prob_calibrated": round(model_prob, 6),
        "implied_prob": rec["implied_prob"],
        "edge": round(model_prob - rec["implied_prob"], 6),
        "expected_value_unit": round(expected_value_unit, 6),
        "confidence_grade": confidence_grade,
        "recommendation": decision["recommendation"],
        "stake_units": decision["stake_units"]
    }

def _calculate_confidence_grade(prob: float, missing_fields: List[str]) -> str:
    """Calculate confidence grade based on probability and data completeness."""
    # Distance from 0.5 (higher = more confident)
    distance = abs(prob - 0.5)
    
    # Penalize for missing fields
    completeness_penalty = len(missing_fields)
    
    if distance >= 0.25 and completeness_penalty == 0:
        return "A"
    elif distance >= 0.20 and completeness_penalty <= 1:
        return "B"
    elif distance >= 0.15 and completeness_penalty <= 2:
        return "C"
    elif distance >= 0.10:
        return "D"
    else:
        return "F"

def _write_recommendations_json(recs: List[Dict[str, Any]], output_dir: str):
    """Write recommendations to JSON."""
    path = os.path.join(output_dir, "recs.json")
    with open(path, "w") as f:
        json.dump({"recommendations": recs}, f, indent=2)

def _write_recommendations_csv(recs: List[Dict[str, Any]], output_dir: str):
    """Write recommendations to CSV."""
    if not recs:
        return
    
    path = os.path.join(output_dir, "recs.csv")
    
    # Flatten list fields for CSV
    csv_recs = []
    for r in recs:
        csv_rec = {**r}
        csv_rec["join_keys_used"] = ",".join(r.get("join_keys_used", []))
        csv_rec["missing_fields"] = ",".join(r.get("missing_fields", []))
        csv_recs.append(csv_rec)
    
    fieldnames = list(csv_recs[0].keys())
    
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_recs)

def _write_summary(recs: List[Dict[str, Any]], join_audit: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """Write summary statistics."""
    total = len(recs)
    pass_count = sum(1 for r in recs if r["recommendation"] == "PASS")
    lean_count = sum(1 for r in recs if r["recommendation"] == "LEAN")
    bet_count = sum(1 for r in recs if r["recommendation"] == "BET")
    
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "total_opportunities": total,
        "join_audit": join_audit,
        "recommendations": {
            "PASS": pass_count,
            "LEAN": lean_count,
            "BET": bet_count
        },
        "total_stake_units": round(sum(r["stake_units"] for r in recs), 2),
        "avg_edge": round(sum(r["edge"] for r in recs) / total, 6) if total > 0 else 0,
        "avg_ev": round(sum(r["expected_value_unit"] for r in recs) / total, 6) if total > 0 else 0,
        "confidence_distribution": _get_confidence_distribution(recs)
    }
    
    path = os.path.join(output_dir, "summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    
    return summary

def _get_confidence_distribution(recs: List[Dict[str, Any]]) -> Dict[str, int]:
    """Get distribution of confidence grades."""
    dist = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
    for r in recs:
        grade = r.get("confidence_grade", "F")
        dist[grade] = dist.get(grade, 0) + 1
    return dist

