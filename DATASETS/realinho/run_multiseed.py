"""
Multi-seed robustness harness.

The whole pipeline is single-seed by design (RANDOM_SEED=42), which makes the
stochastic deep-learning phases - VAE, diffusion, fair-MLP, DQN - carry an
unquantified variance. This runs the full pipeline (plus the SMOTE baseline) end
to end for N seeds, each in its own isolated PROJECT_ROOT so nothing clobbers the
canonical results/, then aggregates mean +/- std across seeds and runs paired
significance tests on the headline comparisons.

Each seed is driven purely through the RANDOM_SEED env var (common.py reads it),
so the phase code is untouched. RAW_CSV is pinned to the real dataset because the
per-seed PROJECT_ROOT no longer sits next to data.csv.

Usage:  python run_multiseed.py [N]          run N seeds then aggregate (default N=10)
        python run_multiseed.py aggregate N  rebuild aggregate.json / summary / figures
                                             from already-finished seed dirs only

Outputs (under RESULTS/realinho/multiseed/):
  seed_<k>/...                 a full pipeline run per seed
  aggregate_summary.csv        per (phase, method) mean/std/min/max across seeds
  aggregate.json               summary + paired significance tests
  fig_phase8_pareto_multiseed.png, fig_headline_f1_multiseed.png
"""
from __future__ import annotations
import os, sys, subprocess, json, time
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

# SUBMIT layout: this harness lives in DATASETS/realinho/. The generic phases
# (phase2..10, SMOTE) live in ../../CODE; the Realinho-specific Phase 1 sits next
# to this file. Outputs go under RESULTS/realinho/multiseed/.
SELF = Path(__file__).resolve().parent                 # DATASETS/realinho
SUBMIT_ROOT = SELF.parents[1]                           # SUBMIT
CODE = SUBMIT_ROOT / "CODE"
BASE = SUBMIT_ROOT / "RESULTS" / "realinho" / "multiseed"
# The Realinho raw CSV ships next to this harness; an inherited RAW_CSV wins.
_raw_env = os.environ.get("RAW_CSV")
RAW = Path(_raw_env) if _raw_env and Path(_raw_env).exists() else SELF / "data.csv"
ALL_SEEDS = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# Per-phase script paths. Phase 1 is Realinho-specific (next to this file); the
# rest are the generic phases in CODE/.
PHASE_SCRIPTS = ([SELF / "phase1_preprocessing.py"] +
                 [CODE / f"{p}.py" for p in
                  ["phase2_baselines", "phase3_vae", "phase4_diffusion",
                   "phase5_uncertainty_AL", "phase6_rl_AL", "phase7_fairness_posthoc",
                   "phase8_fairness_inprocess", "phase9_multimodal_bert",
                   "phase10_master_comparison", "phase3b_smote"]])


def run_seed(seed: int) -> bool:
    sdir = BASE / f"seed_{seed}"
    sdir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(PROJECT_ROOT=str(sdir), RAW_CSV=str(RAW),
               RANDOM_SEED=str(seed), PYTHONUTF8="1")
    # The phases import common.py from CODE/, so put it on the import path.
    env["PYTHONPATH"] = str(CODE) + (os.pathsep + env["PYTHONPATH"]
                                     if env.get("PYTHONPATH") else "")
    with open(sdir / "run.log", "w", encoding="utf-8") as log:
        for script in PHASE_SCRIPTS:
            t = time.time()
            r = subprocess.run([sys.executable, str(script)],
                               cwd=str(CODE), env=env, stdout=log,
                               stderr=subprocess.STDOUT)
            log.write(f"--- {script.stem} rc={r.returncode} {time.time()-t:.1f}s ---\n")
            log.flush()
            if r.returncode != 0:
                print(f"[seed {seed}] FAILED at {script.stem} (see {sdir/'run.log'})",
                      flush=True)
                return False
    print(f"[seed {seed}] OK", flush=True)
    return True


def collect(seeds):
    """Stack each seed's master_comparison.csv + SMOTE rows + diffusion-fidelity."""
    frames, fidelity = [], []
    for s in seeds:
        sdir = BASE / f"seed_{s}"
        mc = pd.read_csv(sdir / "results" / "master_comparison.csv")
        mc["seed"] = s
        sm = json.load(open(sdir / "results" / "phase3b_smote.json"))["smote_augmented"]
        sm_rows = pd.DataFrame([{"phase": "P3b_SMOTE_aug", "method": m, "seed": s, **v}
                                for m, v in sm.items()])
        frames.append(pd.concat([mc, sm_rows], ignore_index=True))
        p4 = json.load(open(sdir / "results" / "phase4_diffusion.json"))
        fidelity.append({"seed": s,
                         "vae_dsigma": p4["utility_vae"]["mean_abs_diff_stds"],
                         "diff_dsigma": p4["utility_diffusion"]["mean_abs_diff_stds"],
                         "vae_dmu": p4["utility_vae"]["mean_abs_diff_means"],
                         "diff_dmu": p4["utility_diffusion"]["mean_abs_diff_means"],
                         "vae_tv": p4["utility_vae"]["categorical_tv_mean"],
                         "diff_tv": p4["utility_diffusion"]["categorical_tv_mean"]})
    return pd.concat(frames, ignore_index=True), pd.DataFrame(fidelity)


def pivot(df, phase, method, metric):
    """Return the per-seed vector for one (phase, method, metric), ordered by seed."""
    sub = df[(df.phase == phase) & (df.method == method)].sort_values("seed")
    return sub[metric].to_numpy(dtype=float)


def paired_test(a, b, name, better):
    """Paired Wilcoxon on two per-seed vectors; report mean diff + p-value."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a - b
    if np.allclose(diff, 0):
        stat, p = float("nan"), 1.0
    else:
        stat, p = wilcoxon(a, b)
    return {"comparison": name, "better": better, "mean_a": float(a.mean()),
            "mean_b": float(b.mean()), "mean_diff": float(diff.mean()),
            "std_diff": float(diff.std(ddof=1)), "wilcoxon_p": float(p),
            "n": int(len(a))}


def main():
    args = sys.argv[1:]
    BASE.mkdir(parents=True, exist_ok=True)
    # aggregate-only mode (mirrors the OULAD harness): rebuild aggregate.json /
    # summary / figures from the seed dirs that are already finished, without
    # re-running any seed.
    if args and args[0] == "aggregate":
        n = int(args[1]) if len(args) > 1 else 10
        ok = [s for s in ALL_SEEDS[:n]
              if (BASE / f"seed_{s}" / "results" / "master_comparison.csv").exists()]
        print(f"[multiseed] aggregating {len(ok)} finished seeds: {ok}", flush=True)
        if len(ok) < 2:
            print("[multiseed] too few finished seeds to aggregate"); return
        aggregate(ok); return

    n = int(args[0]) if args else 10
    seeds = ALL_SEEDS[:n]
    print(f"[multiseed] running {len(seeds)} seeds: {seeds}", flush=True)
    t0 = time.time()
    ok = [s for s in seeds if run_seed(s)]
    print(f"[multiseed] {len(ok)}/{len(seeds)} seeds OK in {(time.time()-t0)/60:.1f} min",
          flush=True)
    if len(ok) < 2:
        print("[multiseed] too few successful seeds to aggregate"); return
    aggregate(ok)


def aggregate(ok):
    df, fid = collect(ok)
    metrics = ["accuracy", "precision", "recall", "f1", "auc_roc",
               "demographic_parity_diff", "equalized_odds_diff"]
    agg = (df.groupby(["phase", "method"])[metrics]
             .agg(["mean", "std", "min", "max"]).reset_index())
    agg.columns = ["_".join(c).rstrip("_") for c in agg.columns]
    agg.to_csv(BASE / "aggregate_summary.csv", index=False)

    # Paired significance on the headline comparisons (same seeds on both sides).
    tests = []
    for m in ["RandomForest", "XGBoost", "MLP"]:
        tests.append(paired_test(pivot(df, "P9_multimodal_BERT", m, "f1"),
                                 pivot(df, "P2_baseline", m, "f1"),
                                 f"multimodal>tabular F1 ({m})", "higher"))
        tests.append(paired_test(pivot(df, "P4_Diffusion_aug", m, "f1"),
                                 pivot(df, "P3_VAE_aug", m, "f1"),
                                 f"diffusion vs VAE F1 ({m})", "n/a"))
        tests.append(paired_test(pivot(df, "P3b_SMOTE_aug", m, "f1"),
                                 pivot(df, "P3_VAE_aug", m, "f1"),
                                 f"SMOTE vs VAE F1 ({m})", "n/a"))
    tests.append(paired_test(pivot(df, "P5_AL_uncertainty", "LogReg", "f1"),
                             pivot(df, "P5_AL_random", "LogReg", "f1"),
                             "uncertainty>random AL F1", "higher"))
    tests.append(paired_test(pivot(df, "P6_AL_RL_greedy", "LogReg", "f1"),
                             pivot(df, "P5_AL_uncertainty", "LogReg", "f1"),
                             "RL vs uncertainty AL F1", "n/a"))
    tests.append(paired_test(pivot(df, "P8_inprocess_lambda=1.0", "FairMLP", "demographic_parity_diff"),
                             pivot(df, "P8_inprocess_lambda=0.0", "FairMLP", "demographic_parity_diff"),
                             "fair-MLP lambda=1 < lambda=0 DP", "lower"))
    tests.append(paired_test(fid["diff_dsigma"].to_numpy(), fid["vae_dsigma"].to_numpy(),
                             "diffusion < VAE std-mismatch (dsigma)", "lower"))

    out = {"seeds": ok, "n_seeds": len(ok),
           "fidelity": fid.to_dict(orient="list"),
           "significance": tests}
    json.dump(out, open(BASE / "aggregate.json", "w"), indent=2)

    # Phase-8 multi-seed Pareto: mean +/- std of acc and hard-DP per lambda.
    lambdas = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    accs, dps, accs_e, dps_e = [], [], [], []
    for lam in lambdas:
        a = pivot(df, f"P8_inprocess_lambda={lam}", "FairMLP", "accuracy")
        d = pivot(df, f"P8_inprocess_lambda={lam}", "FairMLP", "demographic_parity_diff")
        accs.append(a.mean()); accs_e.append(a.std(ddof=1))
        dps.append(d.mean()); dps_e.append(d.std(ddof=1))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.errorbar(dps, accs, xerr=dps_e, yerr=accs_e, fmt="o-", capsize=4, color="#c0392b")
    for lam, dd, aa in zip(lambdas, dps, accs):
        ax.annotate(f"$\\lambda$={lam}", (dd, aa), fontsize=8,
                    textcoords="offset points", xytext=(5, 4))
    ax.set_xlabel(r"$\Delta_{DP}$ (lower = fairer)"); ax.set_ylabel("Accuracy")
    ax.set_title(f"Phase 8 accuracy-fairness Pareto, mean$\\pm$std over {len(ok)} seeds")
    fig.savefig(BASE / "fig_phase8_pareto_multiseed.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Headline F1 across key configs with multi-seed error bars.
    configs = [("P2_baseline", "XGBoost", "XGB (tab)"),
               ("P3_VAE_aug", "RandomForest", "VAE RF"),
               ("P4_Diffusion_aug", "RandomForest", "Diff RF"),
               ("P3b_SMOTE_aug", "RandomForest", "SMOTE RF"),
               ("P5_AL_uncertainty", "LogReg", "Uncert AL"),
               ("P9_multimodal_BERT", "RandomForest", "MM RF")]
    means = [pivot(df, p, m, "f1").mean() for p, m, _ in configs]
    errs = [pivot(df, p, m, "f1").std(ddof=1) for p, m, _ in configs]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(configs)), means, yerr=errs, capsize=4, color="#2980b9")
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels([c[2] for c in configs], rotation=20, fontsize=8)
    ax.set_ylabel("Test F1"); ax.set_ylim(0.7, 0.95)
    ax.set_title(f"Headline F1, mean$\\pm$std over {len(ok)} seeds")
    fig.savefig(BASE / "fig_headline_f1_multiseed.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("\n[multiseed] significance summary:")
    for t in tests:
        print(f"  {t['comparison']:38s} {t['mean_a']:.3f} vs {t['mean_b']:.3f} "
              f"(d={t['mean_diff']:+.3f}, p={t['wilcoxon_p']:.4f})")
    print(f"[multiseed] wrote aggregate.json / aggregate_summary.csv under {BASE}")


if __name__ == "__main__":
    main()
