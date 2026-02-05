Release Confirmation: v1.9.4-real-dataset-freeze-unblock
==================================================
Date (UTC):     2026-02-05T22:20:36.958919Z
Git Commit:     5a87260c2b6eb656beb93abb8f037702f6e515be
Verdict:        GO
Blockers:       0

Freeze Thresholds vs Actuals:
--------------------------------------------------
Metric              Threshold       Actual
--------------------------------------------------
Sample Size         >= 300            400
ECE                 <= 0.08            0.0626
Brier Score         <= 0.2            0.1392

Verification:
--------------------------------------------------
Run the following PowerShell command to verify this archive:

Get-FileHash -Algorithm SHA256 artifacts/releases/v1.9.4-real-dataset-freeze-unblock/* | Format-Table

Compare against MANIFEST.json hashes.
