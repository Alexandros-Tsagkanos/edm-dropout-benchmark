"""
Self-labelled SSL baseline -- the comparison the EDM anchor papers expect.

The two EDM anchor papers of this project (Raftopoulos et al. 2025, MOOC
dropout; Kostopoulos et al. 2025, TVAE-SSL) centre on *self-labelled* SSL:
pseudo-labelling unlabelled points automatically with ensemble methods such as
Co-Forest. Our pipeline instead pursued the complementary *active-learning*
axis (Phases 5-6, choosing which points to label). This script closes that gap
by running the self-labelled axis on our own cohorts, under the same
label-scarcity protocol those papers use, so the comparison is on the table.

For each labelled ratio r and several label-mask seeds we reveal only r of the
training labels (stratified, >=1 per class) and hide the rest as an unlabelled
pool, then evaluate on the *same* held-out test split the rest of the pipeline
uses:

  Supervised (seed)  RandomForest on the labelled fraction only   (lower bound)
  Self-Training      sklearn SelfTrainingClassifier(RF)           (standard SSL)
  Co-Forest          Li & Zhou 2007 reimplementation (coforest)   (anchor's best)
  Supervised (full)  RandomForest on every training label         (upper bound)

Metrics are Accuracy / F1 / MCC -- MCC being the imbalance-robust metric the
MOOC anchor paper adopts as a headline; this is where the project reports it on
real values.

Usage (from the repo root, dataset chosen by PROJECT_ROOT like every phase):
    python CODE/ssl_baseline.py --tag realinho
    PROJECT_ROOT=RESULTS/oulad/single_seed python CODE/ssl_baseline.py --tag oulad

Smoke:  --label-seeds 3 --ratios 0.02 0.10
Writes: results/ssl_baseline.json, figures/ssl_baseline.png
"""
from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import SelfTrainingClassifier

from common import (
    load_processed, predictive_metrics, save_json, set_plot_style,
    RESULTS_DIR, FIGURES_DIR,
)
from coforest import CoForest

METHODS = ["Supervised (seed)", "Self-Training", "Co-Forest"]
METRICS = ["accuracy", "f1", "mcc"]


def stratified_label_mask(y, ratio, rng):
    """Indices of a stratified labelled subset of size ~ratio*len(y), >=1/class."""
    m = max(int(round(ratio * len(y))), 2)
    classes, counts = np.unique(y, return_counts=True)
    labelled = []
    for c, cnt in zip(classes, counts):
        idx_c = np.where(y == c)[0]
        take = max(int(round(m * cnt / len(y))), 1)
        take = min(take, len(idx_c))
        labelled.extend(rng.choice(idx_c, size=take, replace=False))
    return np.array(sorted(labelled))


def rf(seed):
    return RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                  random_state=seed, n_jobs=-1)


def main():
    import os
    ap = argparse.ArgumentParser()
    # default tag follows the dataset chosen via PROJECT_ROOT, so a bare
    # `PROJECT_ROOT=RESULTS/oulad/single_seed python CODE/ssl_baseline.py` is
    # labelled correctly without an explicit --tag.
    default_tag = "oulad" if "oulad" in os.environ.get("PROJECT_ROOT", "").lower() \
        else "realinho"
    ap.add_argument("--tag", default=default_tag, help="dataset label for outputs")
    ap.add_argument("--ratios", type=float, nargs="+",
                    default=[0.01, 0.02, 0.05, 0.10])
    ap.add_argument("--label-seeds", type=int, default=10,
                    help="independent label-mask draws to average over")
    ap.add_argument("--n-trees", type=int, default=7)
    ap.add_argument("--max-iter", type=int, default=15)
    ap.add_argument("--theta", type=float, default=0.75)
    args = ap.parse_args()

    set_plot_style()
    data = load_processed()
    Xtr, ytr = data["X_train"], data["y_train"]
    Xte, yte = data["X_test"], data["y_test"]
    print(f"[ssl_baseline:{args.tag}] train {Xtr.shape} test {Xte.shape} "
          f"pos_rate {ytr.mean():.3f} | ratios {args.ratios} | "
          f"{args.label_seeds} label-seeds")

    # Upper bound: fully-supervised RF, once per seed (independent of ratio).
    full = []
    for s in range(args.label_seeds):
        m = rf(1000 + s).fit(Xtr, ytr)
        full.append(predictive_metrics(yte, m.predict(Xte), m.predict_proba(Xte)[:, 1]))
    full_agg = {f"{k}_mean": float(np.mean([f[k] for f in full])) for k in METRICS}
    full_agg.update({f"{k}_std": float(np.std([f[k] for f in full], ddof=1)) for k in METRICS})
    print(f"  Supervised (full): F1 {full_agg['f1_mean']:.3f}  "
          f"MCC {full_agg['mcc_mean']:.3f}  Acc {full_agg['accuracy_mean']:.3f}")

    by_ratio = {}
    for r in args.ratios:
        per_method = {meth: {k: [] for k in METRICS} for meth in METHODS}
        for s in range(args.label_seeds):
            rng = np.random.RandomState(7000 + s)
            lab = stratified_label_mask(ytr, r, rng)
            unlab = np.setdiff1d(np.arange(len(ytr)), lab)
            Lx, Ly, Ux = Xtr[lab], ytr[lab], Xtr[unlab]

            # 1) Supervised on the labelled seed only
            m = rf(2000 + s).fit(Lx, Ly)
            sup = predictive_metrics(yte, m.predict(Xte), m.predict_proba(Xte)[:, 1])

            # 2) Self-Training (sklearn): -1 marks the unlabelled rows
            y_semi = np.full(len(ytr), -1)
            y_semi[lab] = ytr[lab]
            st = SelfTrainingClassifier(rf(3000 + s), threshold=0.75, max_iter=10)
            st.fit(Xtr, y_semi)
            stm = predictive_metrics(yte, st.predict(Xte), st.predict_proba(Xte)[:, 1])

            # 3) Co-Forest (Li & Zhou 2007 reimplementation)
            cf = CoForest(n_trees=args.n_trees, theta=args.theta,
                          max_iter=args.max_iter, random_state=4000 + s)
            cf.fit(Lx, Ly, Ux)
            cfm = predictive_metrics(yte, cf.predict(Xte), cf.predict_proba(Xte)[:, 1])

            for meth, mm in zip(METHODS, (sup, stm, cfm)):
                for k in METRICS:
                    per_method[meth][k].append(mm[k])

        agg = {}
        for meth in METHODS:
            agg[meth] = {}
            for k in METRICS:
                vals = per_method[meth][k]
                agg[meth][f"{k}_mean"] = float(np.mean(vals))
                agg[meth][f"{k}_std"] = float(np.std(vals, ddof=1))
        by_ratio[f"{r:.2f}"] = agg
        print(f"  r={r:.2f}  "
              + "  ".join(f"{meth.split()[0][:4]}:F1={agg[meth]['f1_mean']:.3f}"
                          f"/MCC={agg[meth]['mcc_mean']:.3f}" for meth in METHODS))

    out = {
        "dataset": args.tag,
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "pos_rate": float(ytr.mean()),
        "label_seeds": args.label_seeds,
        "ratios": list(args.ratios),
        "config": {"n_trees": args.n_trees, "max_iter": args.max_iter,
                   "theta": args.theta},
        "supervised_full": full_agg,
        "by_ratio": by_ratio,
    }
    save_json(out, RESULTS_DIR / "ssl_baseline.json")

    # Figure: F1 vs labelled ratio, one line per self-labelled method, with the
    # fully-supervised upper bound as a dashed reference.
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    colours = {"Supervised (seed)": "#7f8c8d", "Self-Training": "#3498db",
               "Co-Forest": "#16a085"}
    xs = [float(r) for r in args.ratios]
    for meth in METHODS:
        means = [by_ratio[f"{r:.2f}"][meth]["f1_mean"] for r in args.ratios]
        stds = [by_ratio[f"{r:.2f}"][meth]["f1_std"] for r in args.ratios]
        ax.plot([x * 100 for x in xs], means, "-o", color=colours[meth], label=meth)
        ax.fill_between([x * 100 for x in xs],
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        color=colours[meth], alpha=0.15)
    ax.axhline(full_agg["f1_mean"], color="k", ls="--", lw=1,
               label=f"Supervised (full) = {full_agg['f1_mean']:.3f}")
    ax.set_xlabel("labelled ratio (%)"); ax.set_ylabel("test F1")
    ax.set_title(f"Self-labelled SSL under label scarcity - {args.tag}")
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(FIGURES_DIR / "ssl_baseline.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[ssl_baseline:{args.tag}] wrote results/ssl_baseline.json + "
          f"figures/ssl_baseline.png")


if __name__ == "__main__":
    main()
