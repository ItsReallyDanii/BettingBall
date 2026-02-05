# CLAUDE.md - Project Operating Instructions

## Project
Sports AI Basketball V1 - research scaffold for basketball betting risk analysis.

## Language & Dependencies
- Python 3.11+
- `pydantic>=2.7.0` (only runtime dependency)
- No build system; run modules directly

## Run Tests
```bash
python -m unittest discover -s tests -v
```

## Key Modules
- `src/validator.py` - Betting recommendation validator (V1). Entry point: `BettingValidator.validate(payload)`
- `src/schemas.py` - Pydantic entity models (PlayerEntity, TeamEntity, etc.)
- `src/data_quality.py` - Dataset-level validation (missing values, duplicates, labels, temporal ordering)
- `src/readiness.py` - Production readiness checks (freeze blockers, model/data cards)
- `src/main.py` - CLI pipeline (prediction, backtest, rolling backtest, readiness)

## Conventions
- Tests use `unittest` (no pytest)
- Test files: `tests/test_*.py`
- No secrets in code; all config via `.env` (manually loaded in `src/config.py`)
- Artifacts written to `outputs/` via atomic JSON writes
- `DRY_RUN=true` enables deterministic mock mode
