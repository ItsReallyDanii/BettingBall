# Sports AI Basketball V1 (Research Scaffold)

This repository is a minimal, reproducible scaffold for **basketball risk analysis**:
- Structured schemas for entities and targets
- Prompt contracts for reasoning and auditing
- Python pipeline for mock reasoning + audit execution
- **Betting recommendation validator** with strict schema enforcement

## Quick start

```bash
# 1. Create virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install deps
pip install -r requirements.txt

# 3. Copy env
cp .env.example .env

# 4. Run pipeline
python -m src.main --csv

# 5. Run tests
python -m unittest discover -s tests -v
```

## Notes
- `DRY_RUN=true` uses deterministic mock output in `src/llm.py`.
- Replace provider adapter logic in `src/llm.py` when ready.

---

## Betting Recommendation Validator (`src/validator.py`)

Validates betting recommendation payloads before downstream consumption.
Returns structured error objects that are both machine-readable and human-readable.

### Payload Schema

| Field        | Type        | Required | Rules                                                       |
|--------------|-------------|----------|-------------------------------------------------------------|
| `event_type` | `str`       | Yes      | One of: `player_points_over`, `player_assists_over`, `player_rebounds_over`, `team_total_over`, `spread_cover` |
| `threshold`  | `float`     | Yes      | `> 0` for props/totals; any float for `spread_cover`        |
| `probability`| `float`     | Yes      | `[0.0, 1.0]`                                                |
| `odds`       | `int/float` | Yes      | American odds: `<= -100` or `>= 100`                       |
| `stake`      | `float`     | Yes      | `(0, 10000]` USD                                            |
| `confidence` | `str`       | Yes      | One of: `A`, `B`, `C`, `D`, `F`                             |
| `market`     | `str`       | No       | One of: `player_prop`, `team_total`, `spread`               |
| `horizon`    | `str`       | No       | One of: `pregame`, `live` (default: `pregame`)              |

### Validation Rules

1. **Required fields** - all required fields must be present
2. **Type checks** - strings must be strings, numbers must be numbers (booleans rejected)
3. **Numeric bounds** - probability in `[0,1]`, odds outside `(-100,100)`, stake in `(0,10000]`, threshold `>0` for props
4. **Enum checks** - event_type, confidence, market, horizon must be known values
5. **Market/type compatibility** - `player_prop` only with `player_*_over`, `spread` only with `spread_cover`, etc.
6. **Implied probability sanity** - model probability must not diverge from odds-implied probability by more than 0.60

### Safe Defaults

| Field     | Default     | Applied When |
|-----------|-------------|--------------|
| `horizon` | `"pregame"` | Field absent |

No other defaults are applied. Missing required fields produce errors.

### Example Valid Payload

```json
{
  "event_type": "player_points_over",
  "threshold": 24.5,
  "probability": 0.62,
  "odds": -110,
  "stake": 50.0,
  "confidence": "B",
  "market": "player_prop",
  "horizon": "pregame"
}
```

Result:
```json
{
  "valid": true,
  "errors": [],
  "payload": { "...same as input..." }
}
```

### Example Invalid Payload + Error Output

```json
{
  "event_type": "goals_over",
  "threshold": -5,
  "probability": 1.5,
  "odds": 50,
  "stake": -10,
  "confidence": "E"
}
```

Result:
```json
{
  "valid": false,
  "errors": [
    {"code": "OUT_OF_RANGE", "field": "probability", "message": "probability must be in [0.0, 1.0], got 1.5", "meta": {"min": 0.0, "max": 1.0, "got": 1.5}},
    {"code": "OUT_OF_RANGE", "field": "odds", "message": "American odds must be <= -100 or >= 100, got 50.0", "meta": {"invalid_range": "(-100, 100)", "got": 50.0}},
    {"code": "OUT_OF_RANGE", "field": "stake", "message": "stake must be > 0, got -10.0", "meta": {"min_exclusive": 0, "got": -10.0}},
    {"code": "INVALID_VALUE", "field": "event_type", "message": "event_type must be one of [...], got 'goals_over'"},
    {"code": "INVALID_VALUE", "field": "confidence", "message": "confidence must be one of ['A', 'B', 'C', 'D', 'F'], got 'E'"}
  ],
  "payload": {"...input with horizon defaulted to pregame..."}
}
```

### Error Codes

| Code                  | Meaning                                              |
|-----------------------|------------------------------------------------------|
| `INVALID_TYPE`        | Payload is not a dict                                |
| `MISSING_FIELD`       | Required field absent                                |
| `WRONG_TYPE`          | Field has incorrect Python type                      |
| `OUT_OF_RANGE`        | Numeric value outside accepted bounds                |
| `INVALID_VALUE`       | String value not in allowed set                      |
| `MARKET_TYPE_MISMATCH`| event_type incompatible with market                  |
| `EDGE_DIVERGENCE`     | Model probability too far from implied odds probability |

### Usage

```python
from src.validator import BettingValidator

result = BettingValidator.validate({
    "event_type": "player_points_over",
    "threshold": 24.5,
    "probability": 0.62,
    "odds": -110,
    "stake": 50.0,
    "confidence": "B",
})

if result["valid"]:
    # proceed with recommendation
    pass
else:
    for err in result["errors"]:
        print(f"[{err['code']}] {err['field']}: {err['message']}")
```

---

## Run / Test Commands

```bash
# Run all tests (79 tests across 4 files)
python -m unittest discover -s tests -v

# Run only validator tests
python -m unittest tests.test_validator -v

# Run pipeline (requires data in data/raw/)
python -m src.main --csv

# Run backtest
python -m src.main --backtest --gate_profile dev

# Run rolling backtest
python -m src.main --rolling_backtest --folds 5 --gate_profile dev

# Run production readiness check
python -m src.main --readiness --readiness_profile freeze
```
