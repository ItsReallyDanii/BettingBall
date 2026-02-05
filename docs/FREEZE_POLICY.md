# Freeze Policy: v1.9.4-real-dataset-freeze-unblock

## Overview

This document defines the immutable baseline policy for release `v1.9.4-real-dataset-freeze-unblock`. Once a release achieves a **GO** verdict under the "freeze" profile, specific components become **immutable** to preserve production readiness guarantees.

## Immutable Baseline Rules

### Protected Files

The following files are **protected** and cannot be modified on `freeze` or `main` branches:

1. **`src/schemas.py`** - Data schemas and validation contracts
2. **`src/readiness.py`** - Readiness check orchestration and artifact generation
3. **`src/data_quality.py`** - Data quality validation logic
4. **`src/main.py`** (partial) - Readiness and gate threshold functions:
   - `run_readiness_check()`
   - `get_gate_thresholds()`
   - `compile_freeze_blockers()`

### Rationale

These components form the **governance layer** that enforces:
- Freeze thresholds (sample size, ECE, Brier score)
- Data mode validation (real vs synthetic)
- Artifact generation and hashing
- Audit trail integrity

Modifying these files after a freeze would invalidate the production readiness guarantees.

## Required Artifacts

A valid freeze release must include the following artifacts in `artifacts/releases/{release_tag}/`:

1. **`readiness_report.json`** - Overall verdict, metrics, blockers
2. **`freeze_blockers.json`** - Detailed blocker information
3. **`model_card.json`** - Model metadata and feature documentation
4. **`data_card.json`** - Dataset metadata and quality metrics
5. **`repro_manifest.json`** - Reproducibility manifest with hashes
6. **`MANIFEST.json`** - Archive manifest with SHA256 hashes
7. **`README.txt`** - Human-readable summary and verification instructions

## Verification Commands

### Verify Freeze Baseline

```bash
make freeze-verify
# or
python scripts/verify_freeze_baseline.py
```

This checks:
- `src/config.py` has correct `release_tag`
- Evidence archive exists with all required artifacts

### Check Freeze Guard

```bash
make freeze-guard-check
# or
python scripts/block_freeze_edits.py
```

This blocks commits that modify protected files on freeze/main branches.

## Allowed Changes for v1.10+

The following changes are **allowed** without breaking the freeze:

### ✅ Permitted

- **New features** in separate modules (e.g., `src/features_v2.py`)
- **Model improvements** in `src/model_baseline.py` or new model files
- **Connector enhancements** in `src/connectors.py`
- **Test additions** in `tests/`
- **Documentation updates** in `docs/`
- **Configuration changes** in `src/config.py` (except `release_tag` on freeze/main)

### ❌ Prohibited

- Modifying freeze thresholds retroactively
- Changing readiness check logic
- Altering schema validation rules
- Modifying artifact generation
- Changing data quality checks
- Editing protected sections of `src/main.py`

## Branch Strategy

- **`main`** - Frozen at v1.9.4, protected by freeze guards
- **`freeze`** - Explicit freeze branch, fully protected
- **`feature/*`** - Development branches, no restrictions
- **`v1.10`** - Next release branch, inherits v1.9.4 baseline

## Override Process

If a critical bug requires modifying protected files:

1. Document the issue in a GitHub issue
2. Create a hotfix branch from `main`
3. Make minimal changes with clear justification
4. Update freeze policy to reflect the exception
5. Re-run full readiness suite
6. Create new freeze tag (e.g., `v1.9.4.1-hotfix`)

## Enforcement

Freeze guards are enforced via:

1. **Local scripts** - `scripts/block_freeze_edits.py` (pre-commit hook)
2. **Makefile targets** - `make freeze-guard-check`
3. **CI/CD** - Automated verification on push to freeze/main
4. **Code review** - Manual review for any changes to governance layer

## Contact

For questions about freeze policy:
- Review this document
- Check `artifacts/releases/v1.9.4-real-dataset-freeze-unblock/README.txt`
- Consult project maintainers
