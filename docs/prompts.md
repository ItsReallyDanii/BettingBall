# Prompt Audit - BettingBall V1

## Prompt Inventory
| File | Role | Description |
|------|------|-------------|
| prompts/system.md | System | Base system prompt for LLM reasoning |
| prompts/reasoner.md | Reasoner | Probability estimation and thesis generation |
| prompts/feature_engineer.md | Feature Eng | Feature selection and engineering guidance |
| prompts/auditor.md | Auditor | Post-hoc audit of predictions |

## Safety Constraints on Prompts
- No prompt may claim production-ready edge when status is `prototype_only`.
- All LLM-generated reasoning is capped at 140 characters in `reasoning_short`.
- Prompts must not reference future data or settled outcomes.
- Any prompt producing predictions must route through `build_prediction_output()` to enforce the output contract.

## Determinism
- `DRY_RUN=true` bypasses LLM calls entirely (mock mode).
- All prompt outputs are validated against `src/safety.py` contract before persistence.
