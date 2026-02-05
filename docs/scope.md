# Project Scope

## Overview
Sports AI Basketball V1 is a minimal, reproducible research scaffold for basketball betting risk analysis. It provides:
- Structured schemas for teams, players, games, and betting targets
- A validated betting recommendation pipeline
- Rigorous backtesting and model evaluation infrastructure
- Production readiness gates for model deployment

## Core Domain
**Basketball Betting Risk Analysis**
- Focus: Player prop bets, team totals, spreads
- Event types: `player_points_over`, `player_assists_over`, `player_rebounds_over`, `team_total_over`, `spread_cover`
- Time horizons: `pregame`, `live`
- Confidence levels: `A` (highest) to `F` (lowest)

## Entities
- **Teams**: NBA teams with pace, rating, and performance metrics
- **Players**: Active roster players with usage, shooting, and fatigue metrics
- **Games**: NBA matchups with date, home/away teams, and market context
- **Targets**: Betting outcomes with actual results for evaluation

## Key Constraints
- No external dependencies beyond `pydantic>=2.7.0`
- Data sourced from `data/raw/` CSVs (local-only)
- Deterministic reproducibility (git commit, random seeds, file hashes tracked)
- All artifacts written to `outputs/` via atomic JSON operations
- Schema validation at every layer (Pydantic models)

## Out of Scope
- Real-time market data fetching
- Live odds aggregation
- Multi-sport coverage
- Deep learning models
- Advanced causal inference
