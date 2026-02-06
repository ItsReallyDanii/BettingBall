"""
Feature engineering v3 -- six new feature families + missing-data tracking.

All functions:
  - accept explicit typed inputs
  - return (numeric_value, quality_flag) tuples
  - use past-only windows (no future leakage)
  - degrade gracefully when data absent (NaN-safe + flags)

Feature families:
  1. injury_impact_score
  2. home_away_form_delta
  3. pace_delta
  4. matchup_edge_score
  5. usage_trend_5g
  6. minutes_trend_5g
  7. shooting_eff_trend_5g
"""
import math
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Quality flag constants
# ---------------------------------------------------------------------------
QUALITY_OK = "ok"
QUALITY_MISSING = "missing"
QUALITY_PARTIAL = "partial"
QUALITY_ASSUMED = "assumed"

# Stable string keys for missing data
MISSING_INJURY_STATUS = "injury_status_missing"
MISSING_INJURY_MINUTES_LOST = "injury_minutes_lost_missing"
MISSING_HOME_INDICATOR = "is_home_missing"
MISSING_HOME_WINS = "home_wins_last_10_missing"
MISSING_AWAY_WINS = "away_wins_last_10_missing"
MISSING_PACE_PROXY = "pace_proxy_missing"
MISSING_OPP_PACE = "opp_pace_missing"
MISSING_MATCHUP_H2H = "head_to_head_last_3_missing"
MISSING_MATCHUP_DEF_IDX = "defender_matchup_index_missing"
MISSING_OPP_DEF_RATING = "opp_def_rating_missing"
MISSING_USAGE_LAST_5 = "usage_last_5_missing"
MISSING_MINUTES_LAST_5 = "minutes_last_5_missing"
MISSING_EFG_LAST_5 = "efg_last_5_missing"
MISSING_TS_LAST_5 = "true_shooting_last_5_missing"


# ---------------------------------------------------------------------------
# V3 feature column order (appended after v2 features when available)
# ---------------------------------------------------------------------------
V3_FEATURE_COLUMNS = [
    "injury_impact_score",
    "home_away_form_delta",
    "pace_delta",
    "matchup_edge_score",
    "usage_trend_5g",
    "minutes_trend_5g",
    "shooting_eff_trend_5g",
]

# Feature family groupings for availability reporting
FEATURE_FAMILIES = {
    "injuries": ["injury_impact_score"],
    "home_away_splits": ["home_away_form_delta"],
    "pace": ["pace_delta"],
    "matchup_dynamics": ["matchup_edge_score"],
    "usage_trends": ["usage_trend_5g"],
    "minutes_trends": ["minutes_trend_5g"],
    "shooting_efficiency": ["shooting_eff_trend_5g"],
}

# Default assumptions (deterministic strings)
DEFAULT_ASSUMPTIONS = {
    MISSING_INJURY_STATUS: "player assumed healthy (injury_status absent)",
    MISSING_INJURY_MINUTES_LOST: "injury_minutes_lost assumed 0 (data absent)",
    MISSING_HOME_INDICATOR: "is_home assumed False (field absent)",
    MISSING_HOME_WINS: "home_wins_last_10 assumed 5 (league avg)",
    MISSING_AWAY_WINS: "away_wins_last_10 assumed 5 (league avg)",
    MISSING_PACE_PROXY: "team pace assumed 100.0 (league avg)",
    MISSING_OPP_PACE: "opponent pace assumed 100.0 (league avg)",
    MISSING_MATCHUP_H2H: "head_to_head assumed 1.5 (neutral)",
    MISSING_MATCHUP_DEF_IDX: "defender_matchup_index assumed 0.5 (neutral)",
    MISSING_OPP_DEF_RATING: "opp_def_rating assumed 110.0 (league avg)",
    MISSING_USAGE_LAST_5: "usage_rate_last_5 assumed 0.20 (league avg)",
    MISSING_MINUTES_LAST_5: "minutes_last_5 assumed 28.0 (league avg)",
    MISSING_EFG_LAST_5: "efg_last_5 assumed 0.50 (league avg)",
    MISSING_TS_LAST_5: "true_shooting_last_5 assumed 0.55 (league avg)",
}


# ---------------------------------------------------------------------------
# Central missing-data / assumptions collector
# ---------------------------------------------------------------------------
class DataQualityTracker:
    """Collects missing_data_flags and assumptions during feature extraction."""

    def __init__(self):
        self.missing_data_flags: List[str] = []
        self.assumptions: List[str] = []
        self._seen_flags: set = set()

    def flag_missing(self, key: str) -> None:
        """Record a missing-data flag (deduplicated)."""
        if key not in self._seen_flags:
            self._seen_flags.add(key)
            self.missing_data_flags.append(key)
            assumption = DEFAULT_ASSUMPTIONS.get(key)
            if assumption:
                self.assumptions.append(assumption)

    def to_dict(self) -> Dict[str, List[str]]:
        return {
            "missing_data_flags": list(self.missing_data_flags),
            "assumptions": list(self.assumptions),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(val: Any, default: float, tracker: Optional[DataQualityTracker] = None,
                flag_key: Optional[str] = None) -> float:
    """Convert to float; on failure return default and optionally flag."""
    if val is None or val == "":
        if tracker and flag_key:
            tracker.flag_missing(flag_key)
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        if tracker and flag_key:
            tracker.flag_missing(flag_key)
        return default


def _clip(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _injury_weight(status_str: Optional[str]) -> float:
    """Map injury status string to numeric penalty weight.

    healthy=0.0, questionable=0.15, doubtful=0.40, out=1.0
    """
    if not status_str:
        return 0.0
    mapping = {
        "healthy": 0.0,
        "questionable": 0.15,
        "doubtful": 0.40,
        "out": 1.0,
    }
    return mapping.get(str(status_str).strip().lower(), 0.0)


# ---------------------------------------------------------------------------
# Feature functions (deterministic, typed, NaN-safe)
# ---------------------------------------------------------------------------

def injury_impact_score(
    injury_status: Optional[str],
    injury_minutes_lost: Optional[float],
    probable_minutes_cap: Optional[float] = None,
    tracker: Optional[DataQualityTracker] = None,
) -> Tuple[float, str]:
    """Compute injury impact score in [0, 1].

    Formula:
        base = injury_weight(status)       # 0.0 .. 1.0
        minutes_penalty = min(injury_minutes_lost / 48.0, 1.0)  # 0..1
        cap_penalty = max(0, 1 - probable_minutes_cap / 36.0) if cap given else 0
        score = clip(0.5 * base + 0.3 * minutes_penalty + 0.2 * cap_penalty, 0, 1)

    Higher = more negative impact from injuries.
    """
    has_status = injury_status is not None and str(injury_status).strip() != ""
    has_minutes = injury_minutes_lost is not None

    if not has_status:
        if tracker:
            tracker.flag_missing(MISSING_INJURY_STATUS)
    if not has_minutes:
        if tracker:
            tracker.flag_missing(MISSING_INJURY_MINUTES_LOST)

    base = _injury_weight(injury_status) if has_status else 0.0
    mins_lost = float(injury_minutes_lost) if has_minutes else 0.0
    minutes_penalty = _clip(mins_lost / 48.0, 0.0, 1.0)

    cap_penalty = 0.0
    if probable_minutes_cap is not None:
        try:
            cap_val = float(probable_minutes_cap)
            if cap_val > 0:
                cap_penalty = _clip(1.0 - cap_val / 36.0, 0.0, 1.0)
        except (ValueError, TypeError):
            pass

    score = _clip(0.5 * base + 0.3 * minutes_penalty + 0.2 * cap_penalty, 0.0, 1.0)

    if not has_status and not has_minutes:
        return (score, QUALITY_MISSING)
    if not has_status or not has_minutes:
        return (score, QUALITY_PARTIAL)
    return (score, QUALITY_OK)


def home_away_form_delta(
    is_home: Optional[bool],
    home_wins_last_10: Optional[float] = None,
    away_wins_last_10: Optional[float] = None,
    tracker: Optional[DataQualityTracker] = None,
) -> Tuple[float, str]:
    """Compute home/away form delta in [-1, 1].

    Formula:
        home_pct = home_wins_last_10 / 10   (default 0.5 if missing)
        away_pct = away_wins_last_10 / 10   (default 0.5 if missing)
        delta = home_pct - away_pct          (positive = stronger at home)
        result = delta if is_home else -delta

    Interpretation: positive = current venue favors the team.
    """
    quality = QUALITY_OK

    if is_home is None:
        if tracker:
            tracker.flag_missing(MISSING_HOME_INDICATOR)
        is_home_val = False
        quality = QUALITY_MISSING
    else:
        is_home_val = bool(is_home)

    home_w = _safe_float(home_wins_last_10, 5.0, tracker, MISSING_HOME_WINS)
    away_w = _safe_float(away_wins_last_10, 5.0, tracker, MISSING_AWAY_WINS)

    if home_wins_last_10 is None or away_wins_last_10 is None:
        if quality == QUALITY_OK:
            quality = QUALITY_ASSUMED

    home_pct = _clip(home_w / 10.0, 0.0, 1.0)
    away_pct = _clip(away_w / 10.0, 0.0, 1.0)
    delta = home_pct - away_pct

    result = delta if is_home_val else -delta
    return (_clip(result, -1.0, 1.0), quality)


def pace_delta(
    team_pace: Optional[float],
    opp_pace: Optional[float],
    tracker: Optional[DataQualityTracker] = None,
) -> Tuple[float, str]:
    """Compute pace differential (team - opponent possessions proxy).

    Formula:
        delta = team_pace - opp_pace
        Clipped to [-30, 30] for sanity.

    Positive = faster-paced matchup for the team.
    """
    tp = _safe_float(team_pace, 100.0, tracker, MISSING_PACE_PROXY)
    op = _safe_float(opp_pace, 100.0, tracker, MISSING_OPP_PACE)

    quality = QUALITY_OK
    if team_pace is None and opp_pace is None:
        quality = QUALITY_MISSING
    elif team_pace is None or opp_pace is None:
        quality = QUALITY_PARTIAL

    return (_clip(tp - op, -30.0, 30.0), quality)


def matchup_edge_score(
    head_to_head_last_3: Optional[float],
    defender_matchup_index: Optional[float],
    opp_def_rating: Optional[float],
    tracker: Optional[DataQualityTracker] = None,
) -> Tuple[float, str]:
    """Compute matchup edge score in [-1, 1].

    Formula:
        h2h_norm = (h2h_last_3 - 1.5) / 1.5          # [-1, 1], neutral=0
        def_idx_norm = (def_matchup_idx - 0.5) / 0.5  # [-1, 1], neutral=0
        def_rating_advantage = (110 - opp_def) / 20    # positive = weak defense
        score = clip(0.4 * h2h_norm + 0.3 * def_idx_norm + 0.3 * def_rating_advantage, -1, 1)
    """
    h2h = _safe_float(head_to_head_last_3, 1.5, tracker, MISSING_MATCHUP_H2H)
    def_idx = _safe_float(defender_matchup_index, 0.5, tracker, MISSING_MATCHUP_DEF_IDX)
    opp_def = _safe_float(opp_def_rating, 110.0, tracker, MISSING_OPP_DEF_RATING)

    missing_count = sum(1 for v in [head_to_head_last_3, defender_matchup_index, opp_def_rating] if v is None)

    if missing_count == 3:
        quality = QUALITY_MISSING
    elif missing_count > 0:
        quality = QUALITY_PARTIAL
    else:
        quality = QUALITY_OK

    h2h_norm = _clip((h2h - 1.5) / 1.5, -1.0, 1.0)
    def_idx_norm = _clip((def_idx - 0.5) / 0.5, -1.0, 1.0)
    def_rating_adv = _clip((110.0 - opp_def) / 20.0, -1.0, 1.0)

    score = _clip(0.4 * h2h_norm + 0.3 * def_idx_norm + 0.3 * def_rating_adv, -1.0, 1.0)
    return (score, quality)


def usage_trend_5g(
    usage_rates: Optional[List[float]] = None,
    usage_rate_last_5: Optional[float] = None,
    tracker: Optional[DataQualityTracker] = None,
) -> Tuple[float, str]:
    """Compute usage rate trend over last 5 games.

    If per-game usage_rates list provided (oldest-first, past-only window):
        trend = OLS slope of usage_rates[0..4]
    Else if scalar usage_rate_last_5 provided:
        trend = 0.0 (flat assumption, flagged)
    Else:
        trend = 0.0, quality = missing
    """
    if usage_rates and len(usage_rates) >= 2:
        slope = _ols_slope(usage_rates[-5:])
        return (_clip(slope, -0.10, 0.10), QUALITY_OK)

    if usage_rate_last_5 is not None:
        # Have aggregate but not per-game data -- flat trend assumed
        return (0.0, QUALITY_ASSUMED)

    if tracker:
        tracker.flag_missing(MISSING_USAGE_LAST_5)
    return (0.0, QUALITY_MISSING)


def minutes_trend_5g(
    minutes_list: Optional[List[float]] = None,
    minutes_last_5: Optional[float] = None,
    tracker: Optional[DataQualityTracker] = None,
) -> Tuple[float, str]:
    """Compute minutes trend over last 5 games.

    If per-game minutes_list provided (oldest-first, past-only window):
        trend = OLS slope of last 5 entries
    Else if scalar minutes_last_5:
        trend = 0.0 (flat assumption)
    Else:
        trend = 0.0, quality = missing
    """
    if minutes_list and len(minutes_list) >= 2:
        slope = _ols_slope(minutes_list[-5:])
        return (_clip(slope, -10.0, 10.0), QUALITY_OK)

    if minutes_last_5 is not None:
        return (0.0, QUALITY_ASSUMED)

    if tracker:
        tracker.flag_missing(MISSING_MINUTES_LAST_5)
    return (0.0, QUALITY_MISSING)


def shooting_eff_trend_5g(
    efg_list: Optional[List[float]] = None,
    efg_last_5: Optional[float] = None,
    true_shooting_last_5: Optional[float] = None,
    tracker: Optional[DataQualityTracker] = None,
) -> Tuple[float, str]:
    """Compute shooting efficiency trend over last 5 games.

    Priority:
        1. Per-game efg_list -> OLS slope
        2. Scalar efg_last_5 -> 0.0 (flat)
        3. Scalar true_shooting_last_5 -> 0.0 (flat, assumed)
        4. Missing -> 0.0
    """
    if efg_list and len(efg_list) >= 2:
        slope = _ols_slope(efg_list[-5:])
        return (_clip(slope, -0.10, 0.10), QUALITY_OK)

    if efg_last_5 is not None:
        return (0.0, QUALITY_ASSUMED)

    if true_shooting_last_5 is not None:
        return (0.0, QUALITY_ASSUMED)

    if tracker:
        tracker.flag_missing(MISSING_EFG_LAST_5)
    return (0.0, QUALITY_MISSING)


# ---------------------------------------------------------------------------
# OLS slope helper (simple linear regression, past-only)
# ---------------------------------------------------------------------------

def _ols_slope(values: List[float]) -> float:
    """Ordinary least-squares slope over sequential values.

    x = 0, 1, 2, ...  (game index, oldest first)
    Returns slope coefficient.
    """
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))
    if den == 0:
        return 0.0
    return num / den


# ---------------------------------------------------------------------------
# End-to-end extraction from joined record
# ---------------------------------------------------------------------------

def extract_v3_features(
    joined_record: Dict[str, Any],
    tracker: Optional[DataQualityTracker] = None,
) -> Dict[str, float]:
    """Extract all v3 features from a joined record.

    Args:
        joined_record: same format as features_v2 expects
        tracker: optional DataQualityTracker for missing-data propagation

    Returns:
        Dict mapping v3 feature names to numeric values.
    """
    if tracker is None:
        tracker = DataQualityTracker()

    game = joined_record.get("game", {})
    player = joined_record.get("player", {})
    team = joined_record.get("team", {})
    opponent_team = joined_record.get("opponent_team", {})

    features: Dict[str, float] = {}

    # 1. Injury impact
    inj_val, _ = injury_impact_score(
        injury_status=player.get("injury_status"),
        injury_minutes_lost=_maybe_float(player.get("injury_minutes_lost")),
        probable_minutes_cap=_maybe_float(player.get("probable_minutes_cap")),
        tracker=tracker,
    )
    features["injury_impact_score"] = inj_val

    # 2. Home/away form delta
    ha_val, _ = home_away_form_delta(
        is_home=_maybe_bool(game.get("is_home_for_subject_team")),
        home_wins_last_10=_maybe_float(team.get("home_wins_last_10")),
        away_wins_last_10=_maybe_float(team.get("away_wins_last_10")),
        tracker=tracker,
    )
    features["home_away_form_delta"] = ha_val

    # 3. Pace delta
    pd_val, _ = pace_delta(
        team_pace=_maybe_float(team.get("pace")),
        opp_pace=_maybe_float(opponent_team.get("pace")),
        tracker=tracker,
    )
    features["pace_delta"] = pd_val

    # 4. Matchup edge
    me_val, _ = matchup_edge_score(
        head_to_head_last_3=_maybe_float(game.get("head_to_head_last_3")),
        defender_matchup_index=_maybe_float(game.get("defender_matchup_index")),
        opp_def_rating=_maybe_float(opponent_team.get("def_rating")),
        tracker=tracker,
    )
    features["matchup_edge_score"] = me_val

    # 5. Usage trend
    usage_list = _maybe_float_list(player.get("usage_rates_last_5"))
    ut_val, _ = usage_trend_5g(
        usage_rates=usage_list,
        usage_rate_last_5=_maybe_float(player.get("usage_rate_last_5")),
        tracker=tracker,
    )
    features["usage_trend_5g"] = ut_val

    # 6. Minutes trend
    mins_list = _maybe_float_list(player.get("minutes_list_last_5"))
    mt_val, _ = minutes_trend_5g(
        minutes_list=mins_list,
        minutes_last_5=_maybe_float(player.get("minutes_last_5")),
        tracker=tracker,
    )
    features["minutes_trend_5g"] = mt_val

    # 7. Shooting efficiency trend
    efg_list = _maybe_float_list(player.get("efg_list_last_5"))
    se_val, _ = shooting_eff_trend_5g(
        efg_list=efg_list,
        efg_last_5=_maybe_float(player.get("efg_last_5")),
        true_shooting_last_5=_maybe_float(player.get("true_shooting_last_5")),
        tracker=tracker,
    )
    features["shooting_eff_trend_5g"] = se_val

    return features


def v3_features_to_vector(features: Dict[str, float]) -> List[float]:
    """Convert v3 feature dict to ordered vector."""
    return [features.get(col, 0.0) for col in V3_FEATURE_COLUMNS]


# ---------------------------------------------------------------------------
# Feature availability report
# ---------------------------------------------------------------------------

def feature_availability_report(
    joined_record: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate per-run feature availability report.

    Returns:
        {
            "available_features": [...],
            "missing_feature_groups": [...],
            "coverage_pct": float
        }
    """
    tracker = DataQualityTracker()
    features = extract_v3_features(joined_record, tracker)

    available = []
    missing_groups = []

    for family_name, feature_names in FEATURE_FAMILIES.items():
        # A family is "available" if none of its features triggered a missing flag
        family_missing_keys = _family_to_missing_keys(family_name)
        has_missing = any(k in tracker._seen_flags for k in family_missing_keys)
        if not has_missing:
            available.extend(feature_names)
        else:
            missing_groups.append(family_name)

    total = len(V3_FEATURE_COLUMNS)
    avail_count = len(available)

    return {
        "available_features": sorted(available),
        "missing_feature_groups": sorted(missing_groups),
        "coverage_pct": round(avail_count / total, 4) if total > 0 else 0.0,
    }


def _family_to_missing_keys(family_name: str) -> List[str]:
    """Map feature family to its possible missing-data flag keys."""
    mapping = {
        "injuries": [MISSING_INJURY_STATUS, MISSING_INJURY_MINUTES_LOST],
        "home_away_splits": [MISSING_HOME_INDICATOR, MISSING_HOME_WINS, MISSING_AWAY_WINS],
        "pace": [MISSING_PACE_PROXY, MISSING_OPP_PACE],
        "matchup_dynamics": [MISSING_MATCHUP_H2H, MISSING_MATCHUP_DEF_IDX, MISSING_OPP_DEF_RATING],
        "usage_trends": [MISSING_USAGE_LAST_5],
        "minutes_trends": [MISSING_MINUTES_LAST_5],
        "shooting_efficiency": [MISSING_EFG_LAST_5, MISSING_TS_LAST_5],
    }
    return mapping.get(family_name, [])


# ---------------------------------------------------------------------------
# Type coercion helpers
# ---------------------------------------------------------------------------

def _maybe_float(val: Any) -> Optional[float]:
    """Return float or None (never raise)."""
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _maybe_bool(val: Any) -> Optional[bool]:
    """Return bool or None."""
    if val is None or val == "":
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        low = val.strip().lower()
        if low in ("true", "1", "yes"):
            return True
        if low in ("false", "0", "no"):
            return False
        return None
    try:
        return bool(val)
    except (ValueError, TypeError):
        return None


def _maybe_float_list(val: Any) -> Optional[List[float]]:
    """Parse a list of floats from various representations."""
    if val is None or val == "":
        return None
    if isinstance(val, list):
        try:
            return [float(x) for x in val]
        except (ValueError, TypeError):
            return None
    if isinstance(val, str):
        # Support comma-separated or JSON-like: "0.20,0.22,0.19,0.21,0.23"
        try:
            parts = val.strip().strip("[]").split(",")
            return [float(p.strip()) for p in parts if p.strip()]
        except (ValueError, TypeError):
            return None
    return None


# ---------------------------------------------------------------------------
# Risk classification for v3 factors
# ---------------------------------------------------------------------------

V3_RISK_MAP: Dict[str, str] = {
    "injury_impact_score": "medium",      # medium by default; high if source missing
    "home_away_form_delta": "low",
    "pace_delta": "low",
    "matchup_edge_score": "medium",
    "usage_trend_5g": "low",
    "minutes_trend_5g": "low",
    "shooting_eff_trend_5g": "low",
}

V3_RISK_WHEN_MISSING: Dict[str, str] = {
    "injury_impact_score": "high",        # high when injury source missing/unverified
    "home_away_form_delta": "medium",
    "pace_delta": "medium",
    "matchup_edge_score": "medium",
    "usage_trend_5g": "medium",
    "minutes_trend_5g": "medium",
    "shooting_eff_trend_5g": "medium",
}


def classify_v3_factor_risk(factor_name: str, quality: str = QUALITY_OK) -> str:
    """Return risk level for a v3 factor, considering data quality."""
    if quality in (QUALITY_MISSING, QUALITY_PARTIAL):
        return V3_RISK_WHEN_MISSING.get(factor_name, "medium")
    return V3_RISK_MAP.get(factor_name, "medium")
