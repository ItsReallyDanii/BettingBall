
import os
import json
import shutil
import hashlib
from datetime import datetime, timezone

def archive_release(release_tag: str, git_commit: str):
    print(f"📦 Archiving release: {release_tag}")
    
    # paths
    source_dir = "outputs/audits"
    target_dir = f"artifacts/releases/{release_tag}"
    os.makedirs(target_dir, exist_ok=True)
    
    required_files = [
        "readiness_report.json",
        "freeze_blockers.json",
        "model_card.json",
        "data_card.json",
        "repro_manifest.json"
    ]
    
    manifest_files = []
    
    # Copy files and calculate hashes
    for fname in required_files:
        src = os.path.join(source_dir, fname)
        dst = os.path.join(target_dir, fname)
        
        if not os.path.exists(src):
            print(f"❌ Missing required artifact: {fname}")
            return False
            
        shutil.copy2(src, dst)
        
        # Hash
        with open(dst, "rb") as f:
            content = f.read()
            sha256 = hashlib.sha256(content).hexdigest()
            size = len(content)
            
        manifest_files.append({
            "filename": fname,
            "sha256": sha256,
            "size_bytes": size
        })
        print(f"  - Archived {fname}")

    # Create MANIFEST.json
    manifest = {
        "release_tag": release_tag,
        "git_commit": git_commit,
        "utc_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "files": manifest_files
    }
    
    with open(os.path.join(target_dir, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  - Created MANIFEST.json")
    
    # Read readiness report for README
    with open(os.path.join(target_dir, "readiness_report.json"), "r") as f:
        rr = json.load(f)
        
    # Create README.txt
    readme_content = f"""Release Confirmation: {release_tag}
==================================================
Date (UTC):     {manifest['utc_timestamp']}
Git Commit:     {git_commit}
Verdict:        {rr.get('overall_verdict', 'UNKNOWN')}
Blockers:       {rr.get('blocker_count', 'UNKNOWN')}

Freeze Thresholds vs Actuals:
--------------------------------------------------
Metric              Threshold       Actual
--------------------------------------------------
Sample Size         >= {rr['freeze_thresholds']['sample_size_total']}            {rr['current_metrics']['total_sample_size']}
ECE                 <= {rr['freeze_thresholds']['mean_ece']}            {rr['current_metrics']['mean_ece']}
Brier Score         <= {rr['freeze_thresholds']['mean_brier']}            {rr['current_metrics']['mean_brier']}

Verification:
--------------------------------------------------
Run the following PowerShell command to verify this archive:

Get-FileHash -Algorithm SHA256 artifacts/releases/{release_tag}/* | Format-Table

Compare against MANIFEST.json hashes.
"""
    with open(os.path.join(target_dir, "README.txt"), "w") as f:
        f.write(readme_content)
    print(f"  - Created README.txt")
    
    print(f"✅ Archive complete at {target_dir}")
    return True
