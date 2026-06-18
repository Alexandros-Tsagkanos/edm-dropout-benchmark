"""
OULAD second-dataset runner: the Realinho pipeline, re-run on OULAD.

The pipeline reproduces, per seed, prep_oulad + phases 2-8 + the SMOTE baseline in
an isolated PROJECT_ROOT per seed (no Phase 1 EDA / Phase 9 text / Phase 10 Realinho
summary), then aggregates mean +/- std and paired Wilcoxon tests for the same
headline comparisons used on Realinho -- so we can ask: do the Realinho findings
replicate on a second, very different cohort?

Because long unattended runs get killed in this environment, the runner is
seed-at-a-time and resumable. Usage:
  python run_multiseed.py next          # run the next not-yet-finished seed, then stop
  python run_multiseed.py seed 3        # run one specific seed
  python run_multiseed.py aggregate 10  # build aggregate.json / summary / figure from
                                     #   whatever seeds are already finished
  python run_multiseed.py 10            # run all 10 seeds then aggregate (one shot)

Outputs (under RESULTS/oulad/multiseed/): seed_<k>/, aggregate.json, oulad_summary.csv,
fig_oulad_f1.png
"""
from __future__ import annotations
import os, sys, subprocess, json, time
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

# SUBMIT layout: this harness lives in DATASETS/oulad/. The OULAD adapter
# (prep_oulad.py) sits next to it; the generic phases live in ../../CODE.
# Outputs go under RESULTS/oulad/multiseed/.
SELF = Path(__file__).resolve().parent                 # DATASETS/oulad
SUBMIT_ROOT = SELF.parents[1]                           # SUBMIT
CODE = SUBMIT_ROOT / "CODE"
BASE = SUBMIT_ROOT / "RESULTS" / "oulad" / "multiseed"
ALL_SEEDS = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# Phase scripts: the OULAD adapter is dataset-specific (next to this file), the
# rest are the generic phases in CODE/.
PHASE_SCRIPTS = ([SELF / "prep_oulad.py"] +
                 [CODE / f"{p}.py" for p in
                  ["phase2_baselines", "phase3_vae", "phase4_diffusion",
                   "phase3b_smote", "phase5_uncertainty_AL", "phase6_rl_AL",
                   "phase7_fairness_posthoc", "phase8_fairness_inprocess"]])
MODELS = ["RandomForest", "XGBoost", "MLP"]


def _done(seed):
    return (BASE / f"seed_{seed}" / "results" / "phase8_fairness_inprocess.json").exists()


def run_seed(seed):
    sdir = BASE / f"seed_{seed}"
    sdir.mkdir(parents=True, exist_ok=True)
    if _done(seed):
        print(f"[seed {seed}] already complete, skipping", flush=True)
        return True
    env = os.environ.copy()
    env.update(PROJECT_ROOT=str(sdir), RANDOM_SEED=str(seed), PYTHONUTF8="1")
    env["PYTHONPATH"] = str(CODE) + (os.pathsep + env["PYTHONPATH"]
                                     if env.get("PYTHONPATH") else "")
    print(f"[seed {seed}] running ...", flush=True)
    with open(sdir / "run.log", "w", encoding="utf-8") as log:
        for script in PHASE_SCRIPTS:
            t = time.time()
            r = subprocess.run([sys.executable, str(script)],
                               cwd=str(CODE), env=env, stdout=log, stderr=subprocess.STDOUT)
            log.write(f"--- {script.stem} rc={r.returncode} {time.time()-t:.1f}s ---\n"); log.flush()
            if r.returncode != 0:
                print(f"[seed {seed}] FAILED at {script.stem}", flush=True); return False
    print(f"[seed {seed}] OK", flush=True)
    return True


def metrics_from_seed(seed):
    R = BASE / f"seed_{seed}" / "results"
    J = lambda n: json.load(open(R / n))
    p2, p3, p4 = J("phase2_baselines.json"), J("phase3_vae.json"), J("phase4_diffusion.json")
    p3b, p5, p6 = J("phase3b_smote.json"), J("phase5_uncertainty_AL.json"), J("phase6_rl_AL.json")
    p8 = J("phase8_fairness_inprocess.json")["sweep"]
    f1 = {}
    for m in MODELS:
        f1[("baseline", m)] = p2[m]["f1"]
        f1[("vae", m)] = p3["vae_augmented"][m]["f1"]
        f1[("diffusion", m)] = p4["diffusion_augmented"][m]["f1"]
        f1[("smote", m)] = p3b["smote_augmented"][m]["f1"]
    return {
        "f1": f1,
        "al": {"uncertainty": p5["uncertainty"][-1]["f1"],
               "random": p5["random"][-1]["f1"],
               "rl": p6["rl_greedy_eval_curve"][-1]["f1"]},
        "fid": {"vae": p4["utility_vae"]["mean_abs_diff_stds"],
                "diff": p4["utility_diffusion"]["mean_abs_diff_stds"]},
        "lam_dp": {lam: p8[f"lambda={lam}"]["demographic_parity_diff"] for lam in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]},
        "lam_acc": {lam: p8[f"lambda={lam}"]["accuracy"] for lam in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]},
    }


def paired(a, b, name):
    a, b = np.asarray(a, float), np.asarray(b, float)
    p = 1.0 if np.allclose(a, b) else float(wilcoxon(a, b).pvalue)
    return {"comparison": name, "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "delta": float((a - b).mean()), "p": p, "n": len(a)}


def aggregate(ok):
    if len(ok) < 2:
        print(f"[oulad] only {len(ok)} seed(s) done; need >=2 to aggregate"); return
    M = [metrics_from_seed(s) for s in ok]
    vec = lambda f: [f(m) for m in M]

    rows = []
    for cfg in ["baseline", "vae", "diffusion", "smote"]:
        for mdl in MODELS:
            v = np.array(vec(lambda m: m["f1"][(cfg, mdl)]))
            rows.append({"config": cfg, "model": mdl, "f1_mean": v.mean(), "f1_std": v.std(ddof=1)})
    pd.DataFrame(rows).to_csv(BASE / "oulad_summary.csv", index=False)

    tests = []
    for mdl in MODELS:
        tests.append(paired(vec(lambda m: m["f1"][("vae", mdl)]),
                            vec(lambda m: m["f1"][("baseline", mdl)]), f"VAE vs baseline F1 ({mdl})"))
        tests.append(paired(vec(lambda m: m["f1"][("smote", mdl)]),
                            vec(lambda m: m["f1"][("baseline", mdl)]), f"SMOTE vs baseline F1 ({mdl})"))
        tests.append(paired(vec(lambda m: m["f1"][("diffusion", mdl)]),
                            vec(lambda m: m["f1"][("vae", mdl)]), f"diffusion vs VAE F1 ({mdl})"))
    tests.append(paired(vec(lambda m: m["al"]["uncertainty"]),
                        vec(lambda m: m["al"]["random"]), "uncertainty > random AL F1"))
    tests.append(paired(vec(lambda m: m["al"]["rl"]),
                        vec(lambda m: m["al"]["uncertainty"]), "RL vs uncertainty AL F1"))
    tests.append(paired(vec(lambda m: m["lam_dp"][1.0]),
                        vec(lambda m: m["lam_dp"][0.0]), "fair-MLP lambda=1 < lambda=0 DP"))
    tests.append(paired(vec(lambda m: m["fid"]["diff"]),
                        vec(lambda m: m["fid"]["vae"]), "diffusion < VAE std-mismatch"))

    lam_list = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    sweep = {str(lam): {"acc_mean": float(np.mean(vec(lambda m: m["lam_acc"][lam]))),
                        "acc_std": float(np.std(vec(lambda m: m["lam_acc"][lam]), ddof=1)),
                        "dp_mean": float(np.mean(vec(lambda m: m["lam_dp"][lam]))),
                        "dp_std": float(np.std(vec(lambda m: m["lam_dp"][lam]), ddof=1))}
             for lam in lam_list}
    json.dump({"seeds": ok, "n_seeds": len(ok), "significance": tests,
               "fairness_sweep": sweep}, open(BASE / "aggregate.json", "w"), indent=2)

    fig, ax = plt.subplots(figsize=(8, 4)); x = np.arange(len(MODELS)); w = 0.2
    for j, (cfg, col) in enumerate([("baseline", "#34495e"), ("vae", "#e67e22"),
                                    ("diffusion", "#16a085"), ("smote", "#8e44ad")]):
        means = [np.mean(vec(lambda m: m["f1"][(cfg, mdl)])) for mdl in MODELS]
        errs = [np.std(vec(lambda m: m["f1"][(cfg, mdl)]), ddof=1) for mdl in MODELS]
        ax.bar(x + (j - 1.5) * w, means, w, yerr=errs, capsize=3, label=cfg, color=col)
    ax.set_xticks(x); ax.set_xticklabels(MODELS); ax.set_ylabel("Withdrawal F1")
    ax.set_title(f"OULAD: oversampler F1 by model, mean$\\pm$std over {len(ok)} seeds")
    ax.legend(fontsize=8, ncol=4)
    fig.savefig(BASE / "fig_oulad_f1.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    print(f"\n[oulad] aggregated over {len(ok)} seeds: {ok}")
    for t in tests:
        print(f"  {t['comparison']:34s} {t['mean_a']:.3f} vs {t['mean_b']:.3f} "
              f"(d={t['delta']:+.3f}, p={t['p']:.4f})")
    print(f"[oulad] wrote aggregate.json / oulad_summary.csv under {BASE}")


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    args = sys.argv[1:]
    if args and args[0] == "seed":
        run_seed(int(args[1])); return
    if args and args[0] == "next":
        for s in ALL_SEEDS:
            if not _done(s):
                run_seed(s)
                remaining = [x for x in ALL_SEEDS if not _done(x)]
                print(f"[oulad] remaining seeds: {remaining}", flush=True)
                return
        print("[oulad] all seeds already complete", flush=True); return
    if args and args[0] == "aggregate":
        n = int(args[1]) if len(args) > 1 else len(ALL_SEEDS)
        aggregate([s for s in ALL_SEEDS[:n] if _done(s)]); return
    # one-shot: run all N then aggregate (kept for completeness)
    n = int(args[0]) if args else 10
    seeds = ALL_SEEDS[:n]
    t0 = time.time()
    ok = [s for s in seeds if run_seed(s)]
    print(f"[oulad] {len(ok)}/{len(seeds)} seeds OK in {(time.time()-t0)/60:.1f} min", flush=True)
    aggregate([s for s in seeds if _done(s)])


if __name__ == "__main__":
    main()
