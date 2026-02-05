# Makefile for v1.9.4-real-dataset-freeze-unblock

.PHONY: freeze-verify freeze-guard-check help

help:
	@echo "Available targets:"
	@echo "  freeze-verify       - Verify freeze baseline integrity"
	@echo "  freeze-guard-check  - Check if staged changes violate freeze policy"

freeze-verify:
	@python scripts/verify_freeze_baseline.py

freeze-guard-check:
	@python scripts/block_freeze_edits.py
