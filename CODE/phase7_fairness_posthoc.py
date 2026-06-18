"""
Phase 7 - post-hoc fairness audit (the original proposal).

This phase measures, but does not try to fix, gender disparity. For every
combination of classifier (RF / XGB / MLP) and training set (baseline, plus the
VAE- and diffusion-augmented sets from Phases 3-4) we compute the fairness
metrics with respect to the protected attribute Gender on the held-out test set:

  - demographic parity difference   |P(y_hat=1 | G=1) - P(y_hat=1 | G=0)|
  - equal-opportunity difference    |TPR(G=1) - TPR(G=0)|
  - equalized-odds difference       max(|dTPR|, |dFPR|)

We re-fit the models here (with the same seeds as Phases 2-4) rather than
reloading saved predictions, so the audited predictions match those phases
exactly while keeping this script self-contained.

The finding that matters: the demographic-parity gaps cluster around ~0.20
across all nine cells. Augmentation alone does not remove the disparity - it
just carries the gender base-rate gap into a larger training set. That is what
motivates the in-processing intervention in Phase 8.

Outputs
  results/phase7_fairness_posthoc.json
  figures/phase7_fairness_bars.png
"""
from __future__ import annotations
import numpy as np, matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from common import (load_processed, fairness_metrics, predictive_metrics,
                    save_json, set_plot_style,
                    RESULTS_DIR, FIGURES_DIR, DATA_DIR, RANDOM_SEED)
from phase2_baselines import train_mlp, mlp_predict


def fit_three(X_train, y_train, X_test):
    """Train RF / XGB / MLP on (X_train, y_train); return (preds, scores) per model."""
    neg_pos_ratio = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                random_state=RANDOM_SEED, n_jobs=-1).fit(X_train, y_train)
    xgb = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.08,
                        subsample=0.9, colsample_bytree=0.9,
                        eval_metric="logloss", scale_pos_weight=neg_pos_ratio,
                        random_state=RANDOM_SEED, n_jobs=-1, verbosity=0
                        ).fit(X_train, y_train)
    mlp = train_mlp(X_train, y_train, class_weight=float(neg_pos_ratio))
    pred_mlp, score_mlp = mlp_predict(mlp, X_test)
    return {
        "RandomForest": (rf.predict(X_test),  rf.predict_proba(X_test)[:, 1]),
        "XGBoost":      (xgb.predict(X_test), xgb.predict_proba(X_test)[:, 1]),
        "MLP":          (pred_mlp, score_mlp),
    }


def main():
    set_plot_style()
    data = load_processed()
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    s_test = data["s_test"]

    # The three training sets we audit: plain, plus the Phase 3/4 synthetic mixes.
    augmentation_sets = {
        "baseline":      (X_train, y_train),
        "vae_aug":       None,
        "diffusion_aug": None,
    }
    vae_synth = np.load(DATA_DIR / "vae_synth.npz")
    augmentation_sets["vae_aug"] = (
        np.concatenate([X_train, vae_synth["X"]], axis=0),
        np.concatenate([y_train, vae_synth["y"]], axis=0),
    )
    diffusion_synth = np.load(DATA_DIR / "diffusion_synth.npz")
    augmentation_sets["diffusion_aug"] = (
        np.concatenate([X_train, diffusion_synth["X"]], axis=0),
        np.concatenate([y_train, diffusion_synth["y"]], axis=0),
    )

    table = {}
    for aug_name, (X_aug, y_aug) in augmentation_sets.items():
        print(f"[Phase 7] auditing augmentation: {aug_name}")
        predictions = fit_three(X_aug, y_aug, X_test)
        for model_name, (preds, scores) in predictions.items():
            row = {
                **predictive_metrics(y_test, preds, scores),
                **fairness_metrics(y_test, preds, s_test),
            }
            table[f"{aug_name}::{model_name}"] = row
            print(f"  {model_name:12s}  DPD={row['demographic_parity_diff']:.3f}  "
                  f"EOD={row['equal_opportunity_diff']:.3f}  "
                  f"EqOdds={row['equalized_odds_diff']:.3f}  "
                  f"F1={row['f1']:.3f}")

    save_json(table, RESULTS_DIR / "phase7_fairness_posthoc.json")

    # One panel per fairness metric; bars grouped by augmentation, coloured by model.
    fairness_metric_names = ["demographic_parity_diff", "equal_opportunity_diff",
                             "equalized_odds_diff"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    aug_keys = list(augmentation_sets)
    width = 0.22
    colors = ["#3498db", "#e67e22", "#2ecc71"]
    for ax, metric in zip(axes, fairness_metric_names):
        x = np.arange(len(aug_keys))
        for j, model in enumerate(["RandomForest", "XGBoost", "MLP"]):
            values = [table[f"{aug}::{model}"][metric] for aug in aug_keys]
            ax.bar(x + (j - 1) * width, values, width=width,
                   label=model, color=colors[j])
            for xpos, v in zip(x + (j - 1) * width, values):
                ax.text(xpos, v + 0.005, f"{v:.2f}", ha="center", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(aug_keys, rotation=10)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel("absolute diff (lower = fairer)")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Phase 7 - Post-hoc Fairness w.r.t. Gender",
                 fontweight="bold")
    fig.savefig(FIGURES_DIR / "phase7_fairness_bars.png"); plt.close(fig)
    print("[Phase 7] done")


if __name__ == "__main__":
    main()
