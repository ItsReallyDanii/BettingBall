# Run Instructions - v1.6.1-secrets-rotation-and-ingest-audit

## Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Create required directories
mkdir -p data/raw outputs/predictions outputs/audits

# SECRETS (Required for Real Mode)
# Create .env and inject rotated keys
# BDL_API_KEY=your_rotated_key
# ODDS_API_KEY=your_rotated_key
```

## Validation & Ingest Audit
The pipeline uses a multi-tier validation system:
1. **Schema Check**: All ingested rows (real or synthetic) are validated against contract schemas.
2. **Ingest Report**: `outputs/audits/ingest_report.json` records row-counts, error counts, and source.
3. **Iterative Runner**: `python -m src.main --run` runs up to 10 cycles, requiring 3 consecutive passes.

## Real Data Ingest Evidence Commands
```bash
# 1. Run iterative runner (Real context if keys present)
python -m src.main --run

# 2. Verify Ingest Report (Must show 'source: real' and 'schema_pass: true')
cat outputs/audits/ingest_report.json

# 3. Check Gate Decisions (Must show 'terminal_reason: success_limit_reached_real')
cat outputs/audits/gate_decisions.json
```

## Security & Privacy (v1.6.1 Hardened)
- **Zero-Secret Baseline**: No 32-char hex strings or hardcoded keys remain in current source.
- **Environment Isolation**: API keys are loaded via `os.getenv` exclusively.
- **Audit Logging**: All fetcher errors and auth status are persisted for CI transparency.

## Freeze Status
- **Release**: v1.6.1-secrets-rotation-and-ingest-audit
- **Status**: CONDITIONAL GO (Infrastructure Frozen, Real-Data Validation Pending)
- **Repo Integrity**: Scanned via [a-f0-9]{32} regex (CLEAN).