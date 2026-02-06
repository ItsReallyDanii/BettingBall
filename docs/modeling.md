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
| **low** | boxscore, closing_odds, rest_days, home_away |
| **medium** | pace_proxy, matchup_derived |
| **high** | sentiment, news, manual_narrative, unstable_proxy |

Unknown factors default to **medium**.

## Confidence Grading
Deterministic mapping from calibrated confidence to letter grade:
- **A**: >= 0.70
- **B**: >= 0.60
- **C**: >= 0.55
- **D**: >= 0.50
- **F**: < 0.50
