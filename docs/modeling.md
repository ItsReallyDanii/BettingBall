# Modeling Architecture - BettingBall v1.11+

## Baseline Model
The system uses **Logistic Regression with L2 Regularization** as the primary predictive engine. This ensures interpretability and prevents overfitting on small datasets.

## Feature Families
1. **Market Features**: Implied probabilities from opening/current lines and movement dynamics.
2. **Team Context**: Offensive/Defensive ratings, pace differentials, and home advantage.
3. **Rest/Fatigue**: Days rest, back-to-back indicators, and travel fatigue indices.
4. **Player Form**: 5-game rolling averages and trend slopes.
5. **Threshold Relation**: Delta between projections and betting thresholds.
6. **Interactions**: Cross-product features like `pace_usage_interaction` and `rest_advantage`.

## Calibration Selection Rule
For every training run, the system evaluates three calibration paths:
1. **None**: Use raw model probabilities.
2. **Platt Scaling**: Sigmoid-based logit transformation.
3. **Isotonic Regression**: Non-parametric binned mapping.

### Selection Logic:
- **Primary Metric**: Validation Brier Score (minimize).
- **Tie-breaker**: Expected Calibration Error (ECE) (minimize).
- Only the best-performing calibrator is saved as `outputs/models/calibrator.pkl`.

## Risk Classification (V1 Safety)
Every factor in prediction output must carry a risk tag from `configs/safety.yaml`:

| Risk Level | Factors |
|------------|---------|
| **low** | boxscore, closing_odds, rest_days, home_away, home_away_form_delta, pace_delta, usage_trend_5g, minutes_trend_5g, shooting_eff_trend_5g |
| **medium** | pace_proxy, matchup_derived, injury_impact_score, matchup_edge_score |
| **high** | sentiment, news, manual_narrative, unstable_proxy |

Unknown factors default to **medium**.

## V3 Feature Families (src/features_v3.py)

Six new feature families extend the v2 baseline. Each function is deterministic, NaN-safe, and uses past-only windows.

| Feature | Formula / Window | Default (missing) | Risk |
|---------|-----------------|-------------------|------|
| `injury_impact_score` | `0.5*injury_weight + 0.3*(mins_lost/48) + 0.2*cap_penalty`, clipped [0,1] | 0.0 (healthy assumed) | medium (high if source missing) |
| `home_away_form_delta` | `(home_win%_10 - away_win%_10)`, sign-flipped for away | 0.0 (neutral) | low |
| `pace_delta` | `team_pace - opp_pace`, clipped [-30,30] | 0.0 (league avg 100) | low |
| `matchup_edge_score` | `0.4*h2h_norm + 0.3*def_idx_norm + 0.3*def_adv`, clipped [-1,1] | 0.0 (neutral) | medium |
| `usage_trend_5g` | OLS slope of last 5 game usage rates (oldest-first) | 0.0 (flat) | low |
| `minutes_trend_5g` | OLS slope of last 5 game minutes | 0.0 (flat) | low |
| `shooting_eff_trend_5g` | OLS slope of last 5 game eFG% (or TS% fallback) | 0.0 (flat) | low |

### Missing-Data Handling
- Each function returns `(value, quality_flag)` where quality is `ok`, `missing`, `partial`, or `assumed`.
- `DataQualityTracker` collects stable `missing_data_flags` keys and deterministic `assumptions` strings.
- These propagate into `build_prediction_output` for every prediction.

### Leakage Safety
- All trend features use `_ols_slope()` on oldest-first arrays, using only past data.
- No future game data is accessed; window is strictly the last N completed games.

## Confidence Grading
Deterministic mapping from calibrated confidence to letter grade:
- **A**: >= 0.70
- **B**: >= 0.60
- **C**: >= 0.55
- **D**: >= 0.50
- **F**: < 0.50
