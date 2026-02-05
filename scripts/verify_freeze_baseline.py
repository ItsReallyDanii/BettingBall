#!/usr/bin/env python3
"""
Verify freeze baseline for v1.9.4-real-dataset-freeze-unblock.

This script ensures:
1. src/config.py release_tag matches expected freeze version
2. Evidence archive exists with all required artifacts
3. Exit code 0 on success, 1 on failure
"""
import os
import sys

EXPECTED_RELEASE_TAG = "v1.9.4-real-dataset-freeze-unblock"
ARCHIVE_DIR = f"artifacts/releases/{EXPECTED_RELEASE_TAG}"

REQUIRED_ARTIFACTS = [
    "readiness_report.json",
    "freeze_blockers.json",
    "model_card.json",
    "data_card.json",
    "repro_manifest.json",
    "MANIFEST.json",
    "README.txt"
]

def verify_release_tag():
    """Verify src/config.py has correct release_tag."""
    config_path = "src/config.py"
    if not os.path.exists(config_path):
        print(f"❌ FAIL: {config_path} not found")
        return False
    
    with open(config_path, "r") as f:
        content = f.read()
    
    if f'release_tag: str = "{EXPECTED_RELEASE_TAG}"' not in content:
        print(f"❌ FAIL: release_tag in {config_path} does not match {EXPECTED_RELEASE_TAG}")
        return False
    
    print(f"✅ PASS: release_tag = {EXPECTED_RELEASE_TAG}")
    return True

def verify_archive():
    """Verify evidence archive exists with all required artifacts."""
    if not os.path.exists(ARCHIVE_DIR):
        print(f"❌ FAIL: Archive directory not found: {ARCHIVE_DIR}")
        return False
    
    print(f"✅ PASS: Archive directory exists: {ARCHIVE_DIR}")
    
    missing = []
    for artifact in REQUIRED_ARTIFACTS:
        path = os.path.join(ARCHIVE_DIR, artifact)
        if not os.path.exists(path):
            missing.append(artifact)
    
    if missing:
        print(f"❌ FAIL: Missing artifacts: {', '.join(missing)}")
        return False
    
    print(f"✅ PASS: All {len(REQUIRED_ARTIFACTS)} required artifacts present")
    return True

def main():
    print(f"🔒 Verifying freeze baseline: {EXPECTED_RELEASE_TAG}")
    print("=" * 60)
    
    checks = [
        verify_release_tag(),
        verify_archive()
    ]
    
    if all(checks):
        print("=" * 60)
        print("✅ FREEZE BASELINE VERIFIED")
        return 0
    else:
        print("=" * 60)
        print("❌ FREEZE BASELINE VERIFICATION FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
