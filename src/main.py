import json
import argparse
import os
import csv
import math
import sys
import subprocess
import tempfile
import time
from datetime import datetime, timezone, timedelta
from src.schemas import (
    PredictionInput, PlayerEntity, TeamEntity, GameContextEntity, 
    RecentForm, Workload, Availability, SpreadContext, TargetEntity,
    InjuryStatus
)
from src.reason import generate_reasoning
from src.audit import run_audit
from src.config import SETTINGS
from src.connectors import DataConnector

def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing {path}")
    with open(path, mode='r', newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def get_run_metadata():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except:
        commit = "unknown"
    return {
        "release_tag": SETTINGS.release_tag,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "config": {
            "provider": SETTINGS.provider,
            "model": SETTINGS.model,
            "dry_run": SETTINGS.dry_run
        }
    }

def atomic_write_json(data, path):
    target_dir = os.path.dirname(path)
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)
    
    fd, temp_path = tempfile.mkstemp(dir=target_dir or ".", suffix=".json")
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, indent=2, sort_keys=True, default=str)
        
        # Robust replace for Windows
        for i in range(5):
            try:
                if os.path.exists(path):
                    os.replace(temp_path, path)
                else:
                    os.rename(temp_path, path)
                return
            except PermissionError:
                time.sleep(0.1)
        
        # Final attempt
        os.replace(temp_path, path)
        
    except Exception as e:
        if os.path.exists(temp_path):
            try: os.remove(temp_path)
            except: pass
        raise e

def run_readiness_check(profile: str = "freeze"):
    print(f"🚦 Initiating Production Readiness Check ({profile})...")
    from src.readiness import generate_model_card, generate_data_card, generate_repro_manifest, compile_freeze_blockers, _file_hash
    from src.data_quality import validate_dataset
    from src.connectors import DataConnector
    
    # 0. Strict ingest validation for freeze profile
    if profile == "freeze":
        print("🔒 Freeze profile: validating real data availability...")
        connector = DataConnector()
        ingest_success = connector.ingest(require_real=True, min_samples=300)
        if not ingest_success:
            print("❌ FREEZE BLOCKED: Real data ingest failed")
            # Generate minimal readiness report with critical blocker
            blockers = [{
                "id": "real_data_ingest_failed",
                "expected": "real data with >=300 samples",
                "actual": "missing or insufficient",
                "severity": "critical",
                "action_hint": "Provide real local data files with at least 300 game records"
            }]
            readiness = {
                "release_tag": "v1.9.4-real-dataset-freeze-unblock",
                "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "git_commit": "unknown",
                "gate_profile_used": profile,
                "freeze_thresholds": get_gate_thresholds(profile),
                "current_metrics": {},
                "overall_verdict": "NO-GO",
                "blocker_count": len(blockers),
                "blockers": list(blockers),
                "artifacts_present": {},
                "exit_code": 1
            }
            with open("outputs/audits/readiness_report.json", "w") as f:
                json.dump(readiness, f, indent=2)
            with open("outputs/audits/freeze_blockers.json", "w") as f:
                json.dump({"blockers": list(blockers)}, f, indent=2)
            return False
    
    # 1. QA Check (Load Data)
    with open("outputs/predictions/results.json", "r") as f:
        preds = json.load(f).get("predictions", [])
    with open("outputs/predictions/training_payload.json", "r") as f:
        inputs = json.load(f).get("inputs", [])
        
    # 2. Run Backtest
    report_bt = run_backtest_workflow(test_ratio=0.2, gate_profile=profile, exit_on_fail=False)
    
    # 3. Run Rolling
    report_roll = run_rolling_backtest_workflow(folds=5, gate_profile=profile, exit_on_fail=False)
    
    # 4. Compile Blockers
    thresholds = get_gate_thresholds(profile)
    
    # Map raw metrics for readiness logic compatibility
    rolling_metrics_for_gates = {
        "total_sample_size": report_roll["total_sample_size"],
        "mean_ece": report_roll["raw_metrics"]["mean_ece"],
        "mean_brier": report_roll["raw_metrics"]["mean_brier"],
        "mean_ece_pre": report_roll["raw_metrics"]["mean_ece_pre"],
        "mean_brier_pre": report_roll["raw_metrics"]["mean_brier_pre"],
    }
    
    blockers = []
    blockers.extend(compile_freeze_blockers(report_bt["gate_status"], report_bt["post_calibration"], thresholds))
    blockers.extend(compile_freeze_blockers(report_roll["gate_status"], rolling_metrics_for_gates, thresholds))
    
    # Check data mode for freeze profile
    data_mode = "synthetic"  # default assumption
    if os.path.exists("outputs/audits/ingest_report.json"):
        with open("outputs/audits/ingest_report.json", "r") as f:
            ingest_report = json.load(f)
            data_mode = ingest_report.get("data_mode", "synthetic")
    
    if profile == "freeze" and data_mode != "real":
        blockers.append({
            "id": "real_data_required_for_freeze",
            "expected": "real",
            "actual": data_mode,
            "severity": "critical",
            "action_hint": "Freeze profile requires real production data, not synthetic"
        })
    
    # Deduplicate blockers
    unique_blockers = {b["id"]: b for b in blockers}.values()
    
    # 5. Generate Artifacts
    targets = [p["actual"] for p in preds]
    data_card = generate_data_card(inputs, targets)
    model_card = generate_model_card(report_roll["raw_metrics"]) # Use rolling metrics as authoritative
    
    meta = get_run_metadata()
    manifest = generate_repro_manifest(
        sys.argv, 
        {"results.json": _file_hash("outputs/predictions/results.json"), "payload.json": _file_hash("outputs/predictions/training_payload.json")},
        {}, # Output hashes will be filled after write
        git_commit=meta.get("git_commit", "unknown"),
        random_seeds={"model_baseline": 42}
    )
    
    # Write Artifacts
    atomic_write_json(data_card, "outputs/audits/data_card.json")
    atomic_write_json(model_card, "outputs/audits/model_card.json")
    atomic_write_json({"release_tag": SETTINGS.release_tag, "profile": profile, "blockers": list(unique_blockers)}, "outputs/audits/freeze_blockers.json")
    
    # Update Manifest hashes
    manifest["output_artifacts"] = {
        "data_card": _file_hash("outputs/audits/data_card.json"),
        "model_card": _file_hash("outputs/audits/model_card.json"),
        "freeze_blockers": _file_hash("outputs/audits/freeze_blockers.json")
        # Readiness report hash cannot be included as it contains the manifest hash (circular)
        # We included readiness in the requirement list, but strictly we can only include hashes of files already written.
        # Readiness report is written AFTER manifest. 
        # But requirement said "output_artifact_hashes (including readiness_report itself)". 
        # This is logically impossible if readiness report includes the manifest which includes the readiness report hash.
        # I will include what I can.
    }
    atomic_write_json(manifest, "outputs/audits/repro_manifest.json")
    # Readiness Report
    any_blockers = len(unique_blockers) > 0
    overall_verdict = "GO" if not any_blockers else "NO-GO"
    
    # Exit code logic: if profile is dev, we don't hard fail unless runtime error
    success = (overall_verdict == "GO") or (profile == "dev")
    
    expected_artifacts = ["readiness_report.json", "freeze_blockers.json", "model_card.json", "data_card.json", "repro_manifest.json"]
    artifacts_present = {
        name: os.path.exists(f"outputs/audits/{name}") for name in expected_artifacts
    }
    artifacts_present["readiness_report.json"] = True
    
    readiness = {
        "release_tag": SETTINGS.release_tag,
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_commit": meta.get("git_commit", "unknown"),
        "gate_profile_used": profile,
        "freeze_thresholds": thresholds,
        "current_metrics": rolling_metrics_for_gates,
        "overall_verdict": overall_verdict,
        "blocker_count": len(unique_blockers),
        "blockers": list(unique_blockers),
        "artifacts_present": artifacts_present,
        "exit_code": 0 if success else 1
    }
    
    atomic_write_json(readiness, "outputs/audits/readiness_report.json")
    
    print("\n📜 Readiness Report:")
    print(json.dumps(readiness, indent=2))
    
    return success

def validate_results_data(data):
    if not isinstance(data, list):
        raise ValueError("Results data must be a list of predictions.")
    for i, p in enumerate(data):
        for key in ["probability", "actual", "game_timestamp"]:
            if key not in p:
                raise KeyError(f"Missing required key '{key}' in prediction index {i}")
        
        prob = p.get("probability")
        if not isinstance(prob, (int, float)) or not (0.0 <= float(prob) <= 1.0):
            raise ValueError(f"Invalid probability {prob} at index {i}")
        
        actual = p.get("actual")
        if actual not in [0, 1, 0.0, 1.0]:
            raise ValueError(f"Invalid actual label {actual} at index {i}")
        
        if not isinstance(p["game_timestamp"], str):
            raise ValueError(f"Invalid game_timestamp type at index {i}")

def calculate_metrics(data):
    if not data:
        return {"metrics": {"sample_size": 0, "hit_rate": 0, "brier_score": 1.0, "log_loss": 10.0, "ece": 1.0}, "bins": []}
    
    y_true = [float(p["actual"]) for p in data]
    y_prob = [float(p["probability"]) for p in data]
    n = len(data)
    
    hits = sum(1 for p, y in zip(y_prob, y_true) if (p > 0.5) == (y == 1.0))
    hit_rate = hits / n
    brier = sum((p - y)**2 for p, y in zip(y_prob, y_true)) / n
    
    eps = 1e-15
    log_loss_val = 0
    for p, y in zip(y_prob, y_true):
        p_c = max(eps, min(1 - eps, p))
        log_loss_val += -(y * math.log(p_c) + (1.0 - y) * math.log(1.0 - p_c))
    log_loss_val /= n
    
    n_bins = 10
    bins = [[] for _ in range(n_bins)]
    for p, y in zip(y_prob, y_true):
        b_idx = min(int(p * n_bins), n_bins - 1)
        bins[b_idx].append((p, y))
    
    ece_val = 0
    bin_details = []
    for i in range(n_bins):
        b = bins[i]
        if not b:
            bin_details.append({"bin": i, "count": 0, "avg_prob": 0, "hit_rate": 0, "diff": 0})
            continue
        avg_p = sum(x[0] for x in b) / len(b)
        avg_y = sum(x[1] for x in b) / len(b)
        diff = abs(avg_p - avg_y)
        ece_val += (len(b) / n) * diff
        bin_details.append({"bin": i, "count": len(b), "avg_prob": round(avg_p, 4), "hit_rate": round(avg_y, 4), "diff": round(diff, 4)})
        
    return {
        "metrics": {
            "hit_rate": round(hit_rate, 4), 
            "brier_score": round(brier, 4), 
            "log_loss": round(log_loss_val, 4), 
            "ece": round(ece_val, 4), 
            "sample_size": n
        },
        "bins": bin_details
    }

def run_predictions():
    players = {r['player_id']: r for r in load_csv("data/raw/players.csv")}
    teams = {r['team_id']: r for r in load_csv("data/raw/teams.csv")}
    games = {r['game_id']: r for r in load_csv("data/raw/games.csv")}
    targets = load_csv("data/raw/targets.csv")
    
    results = []
    
    def to_bool(v):
        if isinstance(v, bool): return v
        return str(v).lower() in ("true", "1", "t", "y", "yes")

    def to_float(v, default=0.0):
        try: return float(v) if v and str(v).strip() else default
        except: return default

    for t in targets:
        p_row, g_row = players.get(t['player_id']), games.get(t['game_id'])
        if not p_row or not g_row: continue
        tm_row = teams.get(p_row['team_id'])
        opp_id = g_row['away_team_id'] if p_row['team_id'] == g_row['home_team_id'] else g_row['home_team_id']
        opp_row = teams.get(opp_id)
        if not tm_row or not opp_row: continue
            
        raw_date = g_row['date_utc']
        if 'Z' in raw_date:
            game_time = datetime.fromisoformat(raw_date.replace('Z', '+00:00'))
        else:
            game_time = datetime.fromisoformat(raw_date).replace(tzinfo=timezone.utc)
            
        try:
            # Cleanup literals
            et = str(t['event_type']).strip().lower()
            hz_raw = t.get('horizon', 'pregame')
            hz = hz_raw.strip().lower() if hz_raw and hz_raw.strip() else "pregame"
            
            # Pydantic Input construction
            inp = PredictionInput(
                player=PlayerEntity(
                    player_id=p_row['player_id'], team_id=p_row['team_id'], position=p_row['position'],
                    recent_form=RecentForm(
                        points_avg_5=to_float(p_row['points_avg_5']), 
                        trend_points_slope_5=to_float(p_row['trend_points_slope_5']), 
                        trend_ts_slope_5=to_float(p_row['trend_ts_slope_5'])
                    ),
                    workload=Workload(
                        days_rest=to_float(p_row['days_rest']), 
                        back_to_back=to_bool(p_row['back_to_back']), 
                        travel_distance_km_recent=to_float(p_row['travel_distance_km_recent'])
                    ),
                    availability=Availability(
                        injury_status=InjuryStatus(p_row['injury_status'].strip().lower()), 
                        probable_minutes_cap=to_float(p_row['probable_minutes_cap'], None)
                    )
                ),
                team=TeamEntity(
                    team_id=tm_row['team_id'], pace=to_float(tm_row['pace']), 
                    off_rating=to_float(tm_row['off_rating']), def_rating=to_float(tm_row['def_rating']), 
                    rotation_stability=to_float(tm_row['rotation_stability'])
                ),
                opponent_team=TeamEntity(
                    team_id=opp_row['team_id'], pace=to_float(opp_row['pace']), 
                    off_rating=to_float(opp_row['off_rating']), def_rating=to_float(opp_row['def_rating']), 
                    rotation_stability=to_float(opp_row['rotation_stability'])
                ),
                game_context=GameContextEntity(
                    game_id=g_row['game_id'], date_utc=game_time, 
                    home_team_id=g_row['home_team_id'], away_team_id=g_row['away_team_id'], 
                    is_home_for_subject_team=to_bool(g_row['is_home_for_subject_team']), 
                    days_rest_team=to_float(g_row['days_rest_team']), 
                    days_rest_opponent=to_float(g_row['days_rest_opponent']), 
                    head_to_head_last_3=to_float(g_row['head_to_head_last_3']), 
                    defender_matchup_index=to_float(g_row['defender_matchup_index']), 
                    ref_crew_foul_tendency=to_float(g_row['ref_crew_foul_tendency']), 
                    travel_fatigue_index=to_float(g_row['travel_fatigue_index']), 
                    spread_context=SpreadContext(
                        market_open_line=to_float(g_row['market_open_line']), 
                        market_current_line=to_float(g_row['market_current_line']), 
                        line_move_abs=to_float(g_row['line_move_abs'])
                    )
                ),
                target=TargetEntity(
                    event_type=et, 
                    threshold=to_float(t['threshold']), 
                    horizon=hz
                ),
                computed_at=min(datetime.now(timezone.utc), game_time - timedelta(hours=4))
            )
            reasoning = generate_reasoning(inp)
            actual_val = int(float(t['actual'])) if 'actual' in t and t['actual'] else 0
            results.append({
                "probability": reasoning.probability, 
                "actual": actual_val, 
                "game_timestamp": g_row['date_utc'],
                "input_dump": inp.model_dump()
            })
        except Exception as e:
            if len(results) < 5:
                print(f"⚠️ Validation error on target {t.get('game_id')}: {e}")
            continue
    
    if not results and targets:
        raise ValueError(f"All {len(targets)} targets skipped. Data integrity failure.")
    
    # Separate artifacts: Public results vs Training payload
    clean_results = [{k: v for k, v in r.items() if k != "input_dump"} for r in results]
    input_payload = [r.get("input_dump") for r in results if r.get("input_dump")]
    
    atomic_write_json({"predictions": clean_results}, "outputs/predictions/results.json")
    if input_payload:
        atomic_write_json({"inputs": input_payload}, "outputs/predictions/training_payload.json")
        
    return clean_results

def run_backtest_workflow(test_ratio=0.2, gate_profile="dev", exit_on_fail=True):
    print(f"🚀 Starting Backtest Workflow (Baseline + Calibration, Profile={gate_profile})...")
    thresholds = get_gate_thresholds(gate_profile)
    res_path = "outputs/predictions/results.json"
    payload_path = "outputs/predictions/training_payload.json"
    
    if not os.path.exists(res_path):
        print("❌ No results found. Run --run first."); return sys.exit(1)
        
    with open(res_path, "r") as f: 
        preds = json.load(f).get("predictions", [])
        
    inputs = []
    if os.path.exists(payload_path):
        with open(payload_path, "r") as f:
            inputs = json.load(f).get("inputs", [])
            
    if not inputs:
        print("⚠️ Training payload missing. Re-hydrating from CSV is not supported in this patch.")
        print("Please run 'python -m src.main --run' to generate fresh training data.")
        sys.exit(1)

    if len(inputs) != len(preds):
        print("⚠️ Mismatched inputs/predictions. Re-run --run."); return sys.exit(1)

    # 1. Chronological Sort
    combined = list(zip(preds, inputs))
    combined.sort(key=lambda x: datetime.fromisoformat(x[0]["game_timestamp"].replace('Z', '+00:00')))
    
    # 2. Split
    n = len(combined)
    n_test = int(n * test_ratio)
    n_train = n - n_test
    
    if n_train < 1 or n_test < 1:
        print(f"❌ Insufficient data for split. Total: {n}, Train: {n_train}, Test: {n_test}")
        sys.exit(1)

    train_set = combined[:n_train]
    test_set = combined[n_train:]
    
    print(f"📉 Training on {len(train_set)} samples, Testing on {len(test_set)} samples...")
    
    # 3. Train Baseline
    from src.model_baseline import BaselineModel
    model = BaselineModel()
    
    # Train set preparation
    X_train = [x[1] for x in train_set]
    y_train = [x[0]["actual"] for x in train_set]
    model.fit(X_train, y_train)
    
    # 4. Generate Uncalibrated Predictions on Train (for Calibration)
    train_probs = [model.predict_proba(x) for x in X_train]
    
    # 5. Fit Calibrator
    from src.calibration import PlattScaler
    calibrator = PlattScaler()
    calibrator.fit(train_probs, y_train)
    
    # 6. Eval on Test
    X_test = [x[1] for x in test_set]
    actuals_test = [x[0]["actual"] for x in test_set]
    timestamps_test = [x[0]["game_timestamp"] for x in test_set]
    
    # Model Scores
    raw_probs = [model.predict_proba(x) for x in X_test]
    cal_probs = [calibrator.predict(p) for p in raw_probs]
    
    # Metrics
    pre_data = [{"probability": p, "actual": a, "game_timestamp": t} for p, a, t in zip(raw_probs, actuals_test, timestamps_test)]
    post_data = [{"probability": p, "actual": a, "game_timestamp": t} for p, a, t in zip(cal_probs, actuals_test, timestamps_test)]
    
    m_pre = calculate_metrics(pre_data)["metrics"]
    rep_post = calculate_metrics(post_data)
    m_post = rep_post["metrics"]
    
    # Gate Checks
    g_size = (n_train + n_test) >= thresholds["sample_size_total"]
    g_ece = m_post["ece"] <= thresholds["mean_ece"]
    g_brier = m_post["brier_score"] <= thresholds["mean_brier"]
    g_imp_ece = m_post["ece"] <= m_pre["ece"]
    g_imp_brier = m_post["brier_score"] <= m_pre["brier_score"]
    
    gate_status = {
        "sample_size_total": g_size,
        "ece": g_ece,
        "brier": g_brier,
        "ece_improved": g_imp_ece,
        "brier_improved": g_imp_brier
    }
    
    verdict = "GO" if all(gate_status.values()) else "NO-GO"
    
    report = {
        "release_tag": SETTINGS.release_tag,
        "split_train": n_train,
        "split_test": n_test,
        "calibration_method": "PlattScaling",
        "gate_profile": gate_profile,
        "gate_thresholds_used": thresholds,
        "verdict": verdict,
        "pre_calibration": m_pre,
        "post_calibration": m_post,
        "gate_status": gate_status
    }
    
    atomic_write_json(report, "outputs/audits/model_report.json")
    print("\n📊 Model Report:")
    print(json.dumps(report, indent=2))
    
    if verdict == "GO":
        print("✅ GO: All gates passed.")
    else:
        print("❌ NO-GO: Gates failed.")
        if exit_on_fail: sys.exit(1)
        
    return report

def get_gate_thresholds(profile: str) -> dict:
    if profile == "dev":
        return {"sample_size_total": 80, "mean_ece": 0.12, "mean_brier": 0.26}
    elif profile == "freeze":
        return {"sample_size_total": 300, "mean_ece": 0.08, "mean_brier": 0.20}
    else:
        raise ValueError(f"Unknown gate profile: {profile}")

def run_rolling_backtest_workflow(folds=5, gate_profile="dev", exit_on_fail=True):
    print(f"🚀 Starting Rolling Backtest (Folds={folds}, Profile={gate_profile})...")
    thresholds = get_gate_thresholds(gate_profile)
    res_path = "outputs/predictions/results.json"
    payload_path = "outputs/predictions/training_payload.json"
    
    if not os.path.exists(res_path) or not os.path.exists(payload_path):
        print("❌ Missing artifacts. Run --run first."); sys.exit(1)

    with open(res_path, "r") as f: preds = json.load(f).get("predictions", [])
    with open(payload_path, "r") as f: inputs = json.load(f).get("inputs", [])
    
    if len(preds) != len(inputs): print("❌ Mismatched artifacts"); sys.exit(1)
    
    combined = list(zip(preds, inputs))
    combined.sort(key=lambda x: datetime.fromisoformat(x[0]["game_timestamp"].replace('Z', '+00:00')))
    
    total_matches = len(combined)
    if total_matches < 10:
        print("❌ Not enough data for rolling backtest (<10 samples)"); sys.exit(1)
        
    fold_size = total_matches // (folds + 1)
    if fold_size == 0: fold_size = 1 
    
    metrics_log = []
    
    from src.model_baseline import BaselineModel
    from src.calibration import PlattScaler
    
    print("\n📦 Fold Execution:")
    for i in range(folds):
        train_end = fold_size * (i + 1)
        if train_end >= total_matches: break
        
        block_size = total_matches // (folds + 1)
        test_start = block_size * (i + 1)
        test_end = block_size * (i + 2) if i < folds - 1 else total_matches
        
        train_data = combined[:test_start]
        test_data = combined[test_start:test_end]
        
        if not train_data or not test_data: continue
        
        model = BaselineModel()
        X_train = [x[1] for x in train_data]
        y_train = [x[0]["actual"] for x in train_data]
        model.fit(X_train, y_train)
        
        train_probs = [model.predict_proba(x) for x in X_train]
        calib = PlattScaler()
        calib.fit(train_probs, y_train)
        
        X_test = [x[1] for x in test_data]
        y_test = [x[0]["actual"] for x in test_data]
        ts_test = [x[0]["game_timestamp"] for x in test_data]
        
        raw_p = [model.predict_proba(x) for x in X_test]
        cal_p = [calib.predict(p) for p in raw_p]
        
        rows_pre = [{"probability": p, "actual": a, "game_timestamp": t} for p, a, t in zip(raw_p, y_test, ts_test)]
        rows_post = [{"probability": p, "actual": a, "game_timestamp": t} for p, a, t in zip(cal_p, y_test, ts_test)]
        
        m_pre = calculate_metrics(rows_pre)["metrics"]
        m_post = calculate_metrics(rows_post)["metrics"]
        
        metrics_log.append({
            "fold": i+1,
            "train_size": len(train_data),
            "test_size": len(test_data),
            "pre_ece": m_pre["ece"], "post_ece": m_post["ece"],
            "pre_brier": m_pre["brier_score"], "post_brier": m_post["brier_score"],
            "hit_rate": m_post["hit_rate"]
        })
        print(f"  Fold {i+1}: Train={len(train_data)} Test={len(test_data)} | ECE: {m_pre['ece']:.3f}->{m_post['ece']:.3f} | Brier: {m_pre['brier_score']:.3f}->{m_post['brier_score']:.3f}")

    if not metrics_log:
        print("❌ No valid folds executed."); sys.exit(1)
        
    avg_ece = sum(m["post_ece"] for m in metrics_log) / len(metrics_log)
    avg_brier = sum(m["post_brier"] for m in metrics_log) / len(metrics_log)
    avg_ece_pre = sum(m["pre_ece"] for m in metrics_log) / len(metrics_log)
    avg_brier_pre = sum(m["pre_brier"] for m in metrics_log) / len(metrics_log)
    
    # Gate Evaluation
    g_size = total_matches >= thresholds["sample_size_total"]
    g_ece = avg_ece <= thresholds["mean_ece"]
    g_brier = avg_brier <= thresholds["mean_brier"]
    g_imp_ece = avg_ece <= avg_ece_pre
    g_imp_brier = avg_brier <= avg_brier_pre
    
    gate_status = {
        "sample_size_total": g_size,
        "mean_ece": g_ece,
        "mean_brier": g_brier,
        "ece_improved": g_imp_ece,
        "brier_improved": g_imp_brier
    }
    
    verdict = "GO" if all(gate_status.values()) else "NO-GO"
    
    report = {
        "release_tag": SETTINGS.release_tag,
        "mode": "rolling_backtest",
        "gate_profile": gate_profile,
        "gate_thresholds_used": thresholds,
        "verdict": verdict,
        "folds_executed": len(metrics_log),
        "total_sample_size": total_matches,
        "raw_metrics": {
            "mean_ece": round(avg_ece, 4),
            "mean_brier": round(avg_brier, 4),
            "mean_ece_pre": round(avg_ece_pre, 4),
            "mean_brier_pre": round(avg_brier_pre, 4)
        },
        "fold_details": metrics_log,
        "gate_status": gate_status
    }
    
    atomic_write_json(report, "outputs/audits/rolling_backtest_report.json")
    print("\n📊 Rolling Report:")
    print(json.dumps(report, indent=2))
    
    if verdict == "GO":
        print("✅ GO: All rolling gates passed.")
    else:
        print("❌ NO-GO: Rolling gates failed.")
        if exit_on_fail: sys.exit(1)

    return report


def iterative_runner():
    print("🚀 Starting iterative cycle runner...")
    history, conn, consecutive_passes, cycle = [], DataConnector(), 0, 1
    
    while consecutive_passes < 3 and cycle <= 10:
        print(f"\n--- CYCLE {cycle} ---")
        if not conn.ingest():
            raise RuntimeError(f"Ingest failure at cycle {cycle}. See outputs/audits/ingest_report.json")
            
        preds = run_predictions()
        if not preds:
            raise ValueError(f"Empty predictions in cycle {cycle}.")
            
        report = calculate_metrics(preds)
        m = report["metrics"]
        
        actuals = [float(p["actual"]) for p in preds]
        n_actuals = len(actuals)
        p_freq = sum(actuals) / n_actuals
        baseline_hr = max(p_freq, 1-p_freq)
        
        g1, g2, g3 = m["sample_size"] >= 50, m["ece"] <= 0.05, m["brier_score"] <= 0.18
        g4 = (m["hit_rate"] - baseline_hr) >= 0.02
        
        passed = all([g1, g2, g3, g4])
        # Detect mode from ingest report
        mode = "synthetic"
        if os.path.exists("outputs/audits/ingest_report.json"):
            with open("outputs/audits/ingest_report.json", "r") as f:
                mode = json.load(f).get("source", "synthetic")

        decision = {
            "cycle": cycle, "passed": passed, "metrics": m, 
            "mode": mode,
            "baseline_hit_rate": round(baseline_hr, 4),
            "gate_status": {"sample_size": g1, "ece": g2, "brier": g3, "hit_rate": g4}
        }
        
        if cycle == 10 and not passed:
            decision["terminal_reason"] = "max_cycles_exceeded"
        elif passed and consecutive_passes == 2:
            decision["terminal_reason"] = f"success_limit_reached_{mode}"
            
        history.append(decision)
        if passed: 
            consecutive_passes += 1; print(f"✅ PASSED ({consecutive_passes}/3)")
        else: 
            consecutive_passes = 0; print(f"❌ FAILED: {decision['gate_status']}")
        # Cycle Cooling (staying under BDL rate limits across iterations)
        time.sleep(15)
        cycle += 1
            
    os.makedirs("outputs/audits", exist_ok=True)
    atomic_write_json(history, "outputs/audits/gate_decisions.json")
    print("\n🏁 Runner finished.")
    return consecutive_passes >= 3

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--csv", action="store_true")
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--rolling_backtest", action="store_true")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--gate_profile", type=str, default="dev", choices=["dev", "freeze"])
    parser.add_argument("--readiness", action="store_true", help="Run full production readiness suite")
    parser.add_argument("--readiness_profile", type=str, default="freeze", choices=["dev", "freeze"], help="Profile for readiness check")
    args = parser.parse_args()

    try:
        if args.run: iterative_runner(); sys.exit(0)
        if args.csv:
            DataConnector().ingest()
            run_predictions(); sys.exit(0)
        if args.readiness:
            success = run_readiness_check(profile=args.readiness_profile)
            sys.exit(0 if success else 1)
        if args.backtest:
            run_backtest_workflow(args.test_ratio, args.gate_profile)
            sys.exit(0)
        if args.rolling_backtest:
            run_rolling_backtest_workflow(args.folds, args.gate_profile)
            sys.exit(0)
        if args.eval:
            res_path = "outputs/predictions/results.json"
            if not os.path.exists(res_path): sys.exit(1)
            with open(res_path, "r") as f: data = json.load(f).get("predictions", [])
            validate_results_data(data)
            report = calculate_metrics(data)
            atomic_write_json({**get_run_metadata(), **report["metrics"]}, "outputs/audits/eval_metrics.json")
            atomic_write_json(report["bins"], "outputs/audits/calibration_bins.json")
            print(json.dumps(report["metrics"], indent=2)); sys.exit(0)
        sys.exit(0)
    except Exception as e:
        print(f"❌ FATAL ERROR: {str(e)}"); sys.exit(1)

if __name__ == "__main__":
    main()
