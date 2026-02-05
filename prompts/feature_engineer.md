# Task
From raw game/player/team records, generate model-ready features.
Return:
- feature_name
- value
- window
- leakage_risk (low/med/high)
- rationale

# Constraints
- Prefer rolling windows: 3, 5, 10 games.
- Add opponent-adjusted variants.
- Mark any post-game leakage as HIGH and exclude by default.