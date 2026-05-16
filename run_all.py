"""
run_all.py — Master runner for all Week 1-3 DS scripts.
Run from the project root: python run_all.py
All outputs saved to outputs/
"""

import subprocess, sys, time, os
from pathlib import Path

SCRIPTS = [
    ("Week 1 DS-1", "scripts/week1/ds1_statistical_baseline_model.py"),
    ("Week 1 DS-2", "scripts/week1/ds2_hypothesis_testing.py"),
    ("Week 1 DS-3", "scripts/week1/ds3_distribution_analysis.py"),
    ("Week 2 DS-1", "scripts/week2/ds1_regression_modelling.py"),
    ("Week 2 DS-2", "scripts/week2/ds2_clustering_patient_segments.py"),
    ("Week 2 DS-3", "scripts/week2/ds3_temporal_pattern_detection.py"),
    ("Week 3 DS-1", "scripts/week3/ds1_iapt_demand_forecasting.py"),
    ("Week 3 DS-2", "scripts/week3/ds2_prescribing_anomaly_detection.py"),
    ("Week 3 DS-3", "scripts/week3/ds3_geo_spatial_analysis.py"),
]

os.makedirs("outputs", exist_ok=True)

log_path = f"outputs/w4_ds1_reproducibility_test/pipeline_run_{time.strftime('%Y%m%d_%H%M%S')}.log"
log_lines = []

def log(msg=""):
    print(msg)
    log_lines.append(msg)

total_start = time.time()
results = []

log("  NHS Mental Health DS Pipeline - Full Run")

for label, script in SCRIPTS:
    if not Path(script).exists():
        log(f"\n[SKIP] {label} — {script} not found")
        results.append((label, "SKIPPED", 0))
        continue

    log(f"\n -->  {label}  ({script})")
    t0 = time.time()
    result = subprocess.run([sys.executable, script], capture_output=True, text=True)
    elapsed = round(time.time() - t0, 1)

    if result.returncode == 0:
        last_line = [l for l in result.stdout.strip().splitlines() if l][-1] if result.stdout.strip() else ""
        log(f"  Done in {elapsed}s -> {last_line}")
        results.append((label, "OK", elapsed))
    else:
        log(f"  FAILED in {elapsed}s")
        err = result.stderr.strip().splitlines()
        for line in err[-5:]:
            log(f"     {line}")
        results.append((label, "FAILED", elapsed))

# Summary
total = round(time.time() - total_start, 1)
log("\n")
log("  Summary")
for label, status, t in results:
    log(f" {label:<15}  {status:<8}  {t}s")

ok     = sum(1 for _,s,_ in results if s=="OK")
failed = sum(1 for _,s,_ in results if s=="FAILED")
log(f"\n  {ok}/{len(SCRIPTS)} scripts passed | Total time: {total}s")

if failed:
    log(f"\n  {failed} script(s) failed - check datasets/ for missing files.")
else:
    log(f"\n  All outputs saved to outputs/")

# Write log file
with open(log_path, "w") as f:
    f.write("\n".join(log_lines))
print(f"\n  Log saved -> {log_path}")