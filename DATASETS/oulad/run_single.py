#!/usr/bin/env python3
"""
Single-seed OULAD run (second dataset).

Runs the pipeline once at the default seed (42): prep_oulad (here in
DATASETS/oulad/, demographics-only adapter), then the generic phases in
../../CODE/ that apply to OULAD (phase2..phase8 + SMOTE; no Phase 1 EDA,
no Phase 9 text, no Phase 10 Realinho summary). Outputs land under
RESULTS/oulad/single_seed/ (data/, results/, figures/).

For the ten-seed cross-cohort study, use run_multiseed.py in this folder.
"""
from __future__ import annotations
import os, subprocess, sys, time
from pathlib import Path

SELF = Path(__file__).resolve().parent                 # DATASETS/oulad
SUBMIT_ROOT = SELF.parents[1]                           # SUBMIT
CODE = SUBMIT_ROOT / "CODE"
OUT = SUBMIT_ROOT / "RESULTS" / "oulad" / "single_seed"

# The OULAD adapter is dataset-specific (lives here); the rest are generic phases in CODE/.
PHASE_SCRIPTS = ([SELF / "prep_oulad.py"] +
                 [CODE / f"{p}.py" for p in
                  ["phase2_baselines", "phase3_vae", "phase4_diffusion", "phase3b_smote",
                   "phase5_uncertainty_AL", "phase6_rl_AL", "phase7_fairness_posthoc",
                   "phase8_fairness_inprocess"]])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(PYTHONUTF8="1", PROJECT_ROOT=str(OUT), RANDOM_SEED="42")
    # prep_oulad + the generic phases import common.py from CODE/.
    env["PYTHONPATH"] = str(CODE) + (os.pathsep + env["PYTHONPATH"]
                                     if env.get("PYTHONPATH") else "")
    print(f"[oulad single-seed] outputs -> {OUT}", flush=True)
    t0 = time.time()
    for script in PHASE_SCRIPTS:
        print(f"\n=== {script.stem} ===", flush=True)
        t = time.time()
        rc = subprocess.run([sys.executable, str(script)], cwd=str(CODE), env=env).returncode
        if rc != 0:
            print(f"[FAIL] {script.stem} exited with code {rc}"); sys.exit(rc)
        print(f"[ok] {script.stem}  ({time.time() - t:.0f}s)", flush=True)
    print(f"\n[oulad single-seed] complete in {(time.time() - t0) / 60:.1f} min -> {OUT}")


if __name__ == "__main__":
    main()
