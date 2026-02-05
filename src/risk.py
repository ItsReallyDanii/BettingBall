from typing import Dict, Any, Optional

CONFIDENCE_RANK = {"A": 5, "B": 4, "C": 3, "D": 2, "F": 1}

DEFAULT_CONFIG = {
    "min_edge_lean": 0.015,
    "min_edge_bet": 0.03,
    "min_ev_lean": 0.005,
    "min_ev_bet": 0.015,
    "min_confidence_for_bet": "B"
}

def recommend(
    model_prob: float,
    implied_prob: float,
    confidence_grade: str,
    expected_value_unit: float,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Deterministic policy function for bet recommendations.
    
    Args:
        model_prob: Calibrated model probability (0-1)
        implied_prob: Market implied probability from odds (0-1)
        confidence_grade: Model confidence grade (A/B/C/D/F)
        expected_value_unit: Expected value per unit stake
        config: Optional config overrides
    
    Returns:
        {
            "recommendation": "PASS" | "LEAN" | "BET",
            "stake_units": 0 | 0.25 | 0.5 | 1.0
        }
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    
    edge = model_prob - implied_prob
    
    # Confidence gate for BET
    min_conf_rank = CONFIDENCE_RANK.get(cfg["min_confidence_for_bet"], 4)
    current_conf_rank = CONFIDENCE_RANK.get(confidence_grade, 1)
    
    # Decision logic
    if edge >= cfg["min_edge_bet"] and expected_value_unit >= cfg["min_ev_bet"] and current_conf_rank >= min_conf_rank:
        return {"recommendation": "BET", "stake_units": 1.0}
    elif edge >= cfg["min_edge_lean"] and expected_value_unit >= cfg["min_ev_lean"]:
        return {"recommendation": "LEAN", "stake_units": 0.5}
    else:
        return {"recommendation": "PASS", "stake_units": 0.0}
