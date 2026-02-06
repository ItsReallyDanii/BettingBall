# Project Scope - BettingBall v1.12

## Local-Only Rule
BettingBall is designed as a local-first predictive engine. All data ingestion, normalization, training, and inference occur on the local machine. No external cloud processing of private betting history is permitted without explicit opt-in.

## Supported Markets
- **Moneyline**: Primary focus. Win/Loss predictions for teams and players.
- **Player Props**: (Points, Rebounds, Assists) - Support via heuristic projections.
- **Game Totals**: Over/Under markets.

## Explicit Non-Goals
- **Real-time Odds Streaming**: The system focuses on batch historical analysis and pre-game ingestion.
- **Automatic Bet Execution**: BettingBall provides recommendations; it does not execute trades.
- **Deep Learning / LLM Training**: All predictive modeling is currently limited to interpretable statistical models (Logistic Regression, etc.).

## Prototype vs Release Claim Policy
- **Status: Prototype Only**:
    - Occurs when dataset < 1000 games OR < 20 teams OR < 60 days span.
    - Claims limited to "in-sample behavior" and "limited out-of-sample signal".
- **Status: Production Ready**:
    - Requires all Generalization Policy thresholds to be met.
    - Claims allowed: "stable market alpha", "production-ready edge".

## What "100% Complete" Means

### Prototype Scope (current)
- All 6 v3 feature families implemented: injuries, home/away splits, pace, matchup dynamics, usage/minutes/shooting trends.
- End-to-end pipeline runs with sparse or enriched data.
- Missing-data handling: every absent field produces stable flags and deterministic assumptions.
- 155 tests covering v1 safety + v3 features pass.
- Output contract is compact, auditable, and deterministic.

### Production Scope (requires)
- Real historical data with >= 1000 games, >= 20 teams, >= 60-day span.
- All generalization gate thresholds met.
- External data validation for injury sources (currently flagged high-risk when absent).
- Calibration verified on out-of-sample data.

## Safety Hardening (V1)
- Generalization gate enforces prototype_only when thresholds are not met.
- Leakage gate blocks release claims if any check fails.
- All prediction outputs follow the structured contract in `src/safety.py`.
- Risk tagging is mandatory on every factor; see `configs/safety.yaml` and `src/features_v3.py`.
- Missing data produces explicit flags and documented assumptions.
