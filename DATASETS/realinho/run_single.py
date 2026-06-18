#!/usr/bin/env python3
"""
Single-seed Realinho run.

Runs the full pipeline once at the default seed (42): Phase 1 preprocessing
(here in DATASETS/realinho/), then the generic phases in ../../CODE/
(phase2..phase10 + SMOTE + bootstrap-CI). Outputs land under
RESULTS/realinho/single_seed/ (data/, results/, figures/). ~6 minutes, CPU only.

For the ten-seed robustness study, use run_multiseed.py in this folder.
"""
from __future__ import annotations
import os, subprocess, sys, time
from pathlib import Path

SELF = Path(__file__).resolve().parent                 # DATASETS/realinho
SUBMIT_ROOT = SELF.parents[1]                           # SUBMIT
CODE = SUBMIT_ROOT / "CODE"
OUT = SUBMIT_ROOT / "RESULTS" / "realinho" / "single_seed"
RAW = SELF / "data.csv"

# Phase 1 is Realinho-specific (lives here); the rest are the generic phases in CODE/.
PHASE_SCRIPTS = ([SELF / "phase1_preprocessing.py"] +
                 [CODE / f"{p}.py" for p in
                  ["phase2_baselines", "phase3_vae", "phase4_diffusion",
                   "phase5_uncertainty_AL", "phase6_rl_AL", "phase7_fairness_posthoc",
                   "phase8_fairness_inprocess", "phase9_multimodal_bert",
                   "phase10_master_comparison", "phase3b_smote", "bootstrap_ci"]])


def main():
    if not RAW.exists():
        print(f"ERROR: raw dataset not found at {RAW}"); sys.exit(1)
    OUT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(PYTHONUTF8="1", RAW_CSV=str(RAW), PROJECT_ROOT=str(OUT), RANDOM_SEED="42")
    # The generic phases import common.py from CODE/.
    env["PYTHONPATH"] = str(CODE) + (os.pathsep + env["PYTHONPATH"]
                                     if env.get("PYTHONPATH") else "")
    print(f"[realinho single-seed] outputs -> {OUT}", flush=True)
    t0 = time.time()
    for script in PHASE_SCRIPTS:
        print(f"\n=== {script.stem} ===", flush=True)
        t = time.time()
        rc = subprocess.run([sys.executable, str(script)], cwd=str(CODE), env=env).returncode
        if rc != 0:
            print(f"[FAIL] {script.stem} exited with code {rc}"); sys.exit(rc)
        print(f"[ok] {script.stem}  ({time.time() - t:.0f}s)", flush=True)
    print(f"\n[realinho single-seed] complete in {(time.time() - t0) / 60:.1f} min -> {OUT}")


if __name__ == "__main__":
    main()
