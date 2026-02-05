# Sports AI Basketball V1 (Research Scaffold)

This repository is a minimal, reproducible scaffold for **basketball risk analysis**:
- Structured schemas for entities and targets
- Prompt contracts for reasoning and auditing
- Python pipeline for mock reasoning + audit execution

## Quick start

1. Create virtual environment
2. Install deps:
   pip install -r requirements.txt
3. Copy env:
   cp .env.example .env
4. Run:
   python -m src.main

## Notes
- `DRY_RUN=true` uses deterministic mock output in `src/llm.py`.
- Replace provider adapter logic in `src/llm.py` when ready.