"""
Betting Recommendation Validator (V1)

Validates betting recommendation payloads before downstream consumption.
Returns structured error objects with both machine-readable codes and
human-readable messages.

Accepted payload schema:
    event_type   : str   - one of VALID_EVENT_TYPES
    threshold    : float - prop line / spread value (>0 for props, any float for spreads)
    probability  : float - model predicted probability [0.0, 1.0]
    odds         : int|float - American odds (<= -100 or >= 100)
    stake        : float - wager amount in USD (0 < stake <= MAX_STAKE)
    confidence   : str   - one of A, B, C, D, F
    market       : str   - one of VALID_MARKETS
    horizon      : str   - one of VALID_HORIZONS (default: "pregame")

Safe defaults (applied only when documented):
    horizon      : "pregame" (if absent)
"""

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_EVENT_TYPES = frozenset({
    "player_points_over",
    "player_assists_over",
    "player_rebounds_over",
    "team_total_over",
    "spread_cover",
})

VALID_MARKETS = frozenset({
    "player_prop",
    "team_total",
    "spread",
})

VALID_HORIZONS = frozenset({
    "pregame",
    "live",
})

VALID_CONFIDENCE_GRADES = frozenset({"A", "B", "C", "D", "F"})

# Market / event_type compatibility map.
# Each market key maps to the set of event_types it may carry.
MARKET_EVENT_COMPAT: Dict[str, frozenset] = {
    "player_prop": frozenset({
        "player_points_over",
        "player_assists_over",
        "player_rebounds_over",
    }),
    "team_total": frozenset({"team_total_over"}),
    "spread": frozenset({"spread_cover"}),
}

MAX_STAKE: float = 10_000.0  # USD ceiling

# Maximum allowed divergence between model probability and implied
# probability derived from the odds line.  Values beyond this suggest
# either a data-entry error or a model mis-calibration that should
# not be acted upon without human review.
MAX_EDGE_DIVERGENCE: float = 0.60

REQUIRED_FIELDS = ("event_type", "threshold", "probability", "odds", "stake", "confidence")


# ---------------------------------------------------------------------------
# Structured error
# ---------------------------------------------------------------------------

class ValidationError:
    """Machine-readable + human-readable validation error."""

    __slots__ = ("code", "field", "message", "meta")

    def __init__(self, code: str, field: str, message: str,
                 meta: Optional[Dict[str, Any]] = None):
        self.code = code
        self.field = field
        self.message = message
        self.meta = meta or {}

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "code": self.code,
            "field": self.field,
            "message": self.message,
        }
        if self.meta:
            d["meta"] = self.meta
        return d

    def __repr__(self) -> str:
        return f"ValidationError({self.code!r}, {self.field!r}, {self.message!r})"


# ---------------------------------------------------------------------------
# Helper: American odds -> implied probability
# ---------------------------------------------------------------------------

def american_odds_to_implied_prob(odds: float) -> Optional[float]:
    """Convert American odds to implied probability.

    Returns None if odds fall in the invalid range (-100, 100) exclusive.
    """
    if odds <= -100:
        return abs(odds) / (abs(odds) + 100.0)
    elif odds >= 100:
        return 100.0 / (odds + 100.0)
    return None  # invalid range


# ---------------------------------------------------------------------------
# Core validator
# ---------------------------------------------------------------------------

class BettingValidator:
    """Stateless validator for betting recommendation payloads.

    Usage::

        result = BettingValidator.validate(payload)
        if not result["valid"]:
            for err in result["errors"]:
                print(err["code"], err["message"])
    """

    @staticmethod
    def validate(payload: Any) -> Dict[str, Any]:
        """Validate a betting recommendation payload.

        Parameters
        ----------
        payload : dict
            The recommendation payload to validate.

        Returns
        -------
        dict
            ``{"valid": bool, "errors": list[dict], "payload": dict}``
            When ``valid`` is True, ``payload`` contains the (possibly
            defaulted) input.  When False, ``errors`` lists every
            violation found (the validator does **not** short-circuit).
        """
        errors: List[ValidationError] = []

        # ------------------------------------------------------------------
        # 0. Top-level type check
        # ------------------------------------------------------------------
        if not isinstance(payload, dict):
            errors.append(ValidationError(
                "INVALID_TYPE", "_root",
                f"Payload must be a dict, got {type(payload).__name__}",
            ))
            return _result(False, errors, payload)

        # ------------------------------------------------------------------
        # 1. Required fields presence
        # ------------------------------------------------------------------
        for field in REQUIRED_FIELDS:
            if field not in payload:
                errors.append(ValidationError(
                    "MISSING_FIELD", field,
                    f"Required field '{field}' is missing",
                ))

        # If required fields are missing we still continue to validate
        # whatever *is* present so that the caller gets a full error list.

        # ------------------------------------------------------------------
        # 2. Apply safe defaults
        # ------------------------------------------------------------------
        working = dict(payload)
        if "horizon" not in working:
            working["horizon"] = "pregame"

        # ------------------------------------------------------------------
        # 3. Type checks (only for fields that are present)
        # ------------------------------------------------------------------
        _type_checks(working, errors)

        # ------------------------------------------------------------------
        # 4. Numeric bounds
        # ------------------------------------------------------------------
        _bounds_checks(working, errors)

        # ------------------------------------------------------------------
        # 5. Enum / allowed-value checks
        # ------------------------------------------------------------------
        _enum_checks(working, errors)

        # ------------------------------------------------------------------
        # 6. Logical consistency
        # ------------------------------------------------------------------
        _consistency_checks(working, errors)

        valid = len(errors) == 0
        return _result(valid, errors, working)


# ---------------------------------------------------------------------------
# Internal check functions
# ---------------------------------------------------------------------------

def _type_checks(p: Dict[str, Any], errors: List[ValidationError]) -> None:
    """Verify that present fields have the correct Python types."""
    str_fields = ("event_type", "confidence", "market", "horizon")
    num_fields = ("threshold", "probability", "odds", "stake")

    for f in str_fields:
        if f in p and not isinstance(p[f], str):
            errors.append(ValidationError(
                "WRONG_TYPE", f,
                f"Field '{f}' must be a string, got {type(p[f]).__name__}",
                {"expected": "str", "got": type(p[f]).__name__},
            ))

    for f in num_fields:
        if f in p and not isinstance(p[f], (int, float)):
            errors.append(ValidationError(
                "WRONG_TYPE", f,
                f"Field '{f}' must be a number, got {type(p[f]).__name__}",
                {"expected": "number", "got": type(p[f]).__name__},
            ))
        # Reject bool masquerading as int (bool is subclass of int in Python)
        if f in p and isinstance(p[f], bool):
            errors.append(ValidationError(
                "WRONG_TYPE", f,
                f"Field '{f}' must be a number, got bool",
                {"expected": "number", "got": "bool"},
            ))


def _bounds_checks(p: Dict[str, Any], errors: List[ValidationError]) -> None:
    """Validate numeric ranges for present numeric fields."""

    # probability: [0.0, 1.0]
    if "probability" in p and isinstance(p["probability"], (int, float)) and not isinstance(p["probability"], bool):
        prob = float(p["probability"])
        if prob < 0.0 or prob > 1.0:
            errors.append(ValidationError(
                "OUT_OF_RANGE", "probability",
                f"probability must be in [0.0, 1.0], got {prob}",
                {"min": 0.0, "max": 1.0, "got": prob},
            ))

    # odds: American odds (must be <= -100 or >= 100)
    if "odds" in p and isinstance(p["odds"], (int, float)) and not isinstance(p["odds"], bool):
        odds = float(p["odds"])
        if -100 < odds < 100:
            errors.append(ValidationError(
                "OUT_OF_RANGE", "odds",
                f"American odds must be <= -100 or >= 100, got {odds}",
                {"invalid_range": "(-100, 100)", "got": odds},
            ))

    # stake: (0, MAX_STAKE]
    if "stake" in p and isinstance(p["stake"], (int, float)) and not isinstance(p["stake"], bool):
        stake = float(p["stake"])
        if stake <= 0:
            errors.append(ValidationError(
                "OUT_OF_RANGE", "stake",
                f"stake must be > 0, got {stake}",
                {"min_exclusive": 0, "got": stake},
            ))
        elif stake > MAX_STAKE:
            errors.append(ValidationError(
                "OUT_OF_RANGE", "stake",
                f"stake must be <= {MAX_STAKE}, got {stake}",
                {"max": MAX_STAKE, "got": stake},
            ))

    # threshold: must be > 0 for prop/total markets, any float for spread
    if "threshold" in p and isinstance(p["threshold"], (int, float)) and not isinstance(p["threshold"], bool):
        et = p.get("event_type", "")
        threshold = float(p["threshold"])
        if isinstance(et, str) and et != "spread_cover" and threshold <= 0:
            errors.append(ValidationError(
                "OUT_OF_RANGE", "threshold",
                f"threshold must be > 0 for {et}, got {threshold}",
                {"min_exclusive": 0, "got": threshold},
            ))


def _enum_checks(p: Dict[str, Any], errors: List[ValidationError]) -> None:
    """Validate that enum-like fields hold allowed values."""

    if "event_type" in p and isinstance(p["event_type"], str):
        if p["event_type"] not in VALID_EVENT_TYPES:
            errors.append(ValidationError(
                "INVALID_VALUE", "event_type",
                f"event_type must be one of {sorted(VALID_EVENT_TYPES)}, got '{p['event_type']}'",
                {"allowed": sorted(VALID_EVENT_TYPES), "got": p["event_type"]},
            ))

    if "confidence" in p and isinstance(p["confidence"], str):
        if p["confidence"] not in VALID_CONFIDENCE_GRADES:
            errors.append(ValidationError(
                "INVALID_VALUE", "confidence",
                f"confidence must be one of {sorted(VALID_CONFIDENCE_GRADES)}, got '{p['confidence']}'",
                {"allowed": sorted(VALID_CONFIDENCE_GRADES), "got": p["confidence"]},
            ))

    if "market" in p and isinstance(p["market"], str):
        if p["market"] not in VALID_MARKETS:
            errors.append(ValidationError(
                "INVALID_VALUE", "market",
                f"market must be one of {sorted(VALID_MARKETS)}, got '{p['market']}'",
                {"allowed": sorted(VALID_MARKETS), "got": p["market"]},
            ))

    if "horizon" in p and isinstance(p["horizon"], str):
        if p["horizon"] not in VALID_HORIZONS:
            errors.append(ValidationError(
                "INVALID_VALUE", "horizon",
                f"horizon must be one of {sorted(VALID_HORIZONS)}, got '{p['horizon']}'",
                {"allowed": sorted(VALID_HORIZONS), "got": p["horizon"]},
            ))


def _consistency_checks(p: Dict[str, Any], errors: List[ValidationError]) -> None:
    """Cross-field logical consistency checks."""

    # 1. Market / event_type compatibility
    market = p.get("market")
    event_type = p.get("event_type")
    if (isinstance(market, str) and market in MARKET_EVENT_COMPAT
            and isinstance(event_type, str) and event_type in VALID_EVENT_TYPES):
        allowed = MARKET_EVENT_COMPAT[market]
        if event_type not in allowed:
            errors.append(ValidationError(
                "MARKET_TYPE_MISMATCH", "market",
                f"event_type '{event_type}' is not compatible with market '{market}'",
                {"market": market, "event_type": event_type,
                 "allowed_types": sorted(allowed)},
            ))

    # 2. Implied probability vs model probability divergence
    odds = p.get("odds")
    prob = p.get("probability")
    if (isinstance(odds, (int, float)) and not isinstance(odds, bool)
            and isinstance(prob, (int, float)) and not isinstance(prob, bool)):
        implied = american_odds_to_implied_prob(float(odds))
        if implied is not None and 0.0 <= float(prob) <= 1.0:
            edge = abs(float(prob) - implied)
            if edge > MAX_EDGE_DIVERGENCE:
                errors.append(ValidationError(
                    "EDGE_DIVERGENCE", "probability",
                    f"Model probability ({prob}) diverges from implied odds "
                    f"probability ({implied:.4f}) by {edge:.4f}, "
                    f"exceeding max allowed divergence of {MAX_EDGE_DIVERGENCE}",
                    {"model_prob": float(prob), "implied_prob": round(implied, 4),
                     "divergence": round(edge, 4),
                     "max_allowed": MAX_EDGE_DIVERGENCE},
                ))


# ---------------------------------------------------------------------------
# Result builder
# ---------------------------------------------------------------------------

def _result(valid: bool, errors: List[ValidationError],
            payload: Any) -> Dict[str, Any]:
    return {
        "valid": valid,
        "errors": [e.to_dict() for e in errors],
        "payload": payload,
    }
