#!/usr/bin/env python3
"""
Block edits to protected freeze paths on freeze/main branches.

This script checks if current branch is 'freeze' or 'main' and blocks
commits that modify protected baseline files.

Protected paths:
- src/schemas.py
- src/readiness.py
- src/data_quality.py
- src/main.py (readiness/gate sections)

Exit code 0 = allowed, 1 = blocked
"""
import os
import sys
import subprocess
import re

PROTECTED_FILES = [
    "src/schemas.py",
    "src/readiness.py",
    "src/data_quality.py"
]

PROTECTED_PATTERNS_IN_MAIN = [
    r"def run_readiness_check",
    r"def get_gate_thresholds",
    r"def compile_freeze_blockers"
]

def get_current_branch():
    """Get current git branch name."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None

def get_staged_files():
    """Get list of staged files."""
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip().split("\n") if result.stdout.strip() else []
    except subprocess.CalledProcessError:
        return []

def check_main_py_protected_sections(filepath):
    """Check if src/main.py changes touch protected sections."""
    try:
        # Get diff for this file
        result = subprocess.run(
            ["git", "diff", "--cached", filepath],
            capture_output=True,
            text=True,
            check=True
        )
        diff = result.stdout
        
        # Check if any protected patterns appear in the diff
        for pattern in PROTECTED_PATTERNS_IN_MAIN:
            if re.search(pattern, diff):
                return True
        return False
    except subprocess.CalledProcessError:
        return False

def main():
    branch = get_current_branch()
    
    if not branch:
        print("⚠️  Could not determine git branch, allowing commit")
        return 0
    
    # Only enforce on freeze or main branches
    if "freeze" not in branch.lower() and branch != "main":
        return 0
    
    print(f"🔒 Freeze guard active on branch: {branch}")
    
    staged = get_staged_files()
    blocked = []
    
    for filepath in staged:
        # Check fully protected files
        if filepath in PROTECTED_FILES:
            blocked.append(f"{filepath} (protected baseline file)")
        
        # Check src/main.py for protected sections
        elif filepath == "src/main.py":
            if check_main_py_protected_sections(filepath):
                blocked.append(f"{filepath} (protected readiness/gate sections)")
    
    if blocked:
        print("❌ COMMIT BLOCKED: The following protected files are modified:")
        for item in blocked:
            print(f"   - {item}")
        print("\nProtected baseline files cannot be modified on freeze/main branches.")
        print("Create a feature branch or update freeze policy if changes are required.")
        return 1
    
    print("✅ No protected files modified, commit allowed")
    return 0

if __name__ == "__main__":
    sys.exit(main())
