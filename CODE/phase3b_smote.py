"""
Phase 3b - SMOTE oversampling baseline (the traditional comparison the
proposal asked for and the report had scoped out).

This is the missing third leg of the augmentation comparison. Phases 3 and 4
oversample the minority with generative models (a VAE, then diffusion); here we
oversample it the classical way - SMOTE interpolation between minority
neighbours - and feed the result through the *same* RF/XGB/MLP evaluator
(phase3.evaluate_augmented), so the only thing that differs across the three is
how the synthetic dropouts were made.

One wrinkle: SMOTE interpolates in the full encoded space, so a synthesised row
gets fractional values inside each one-hot block (a convex mix of two valid
one-hots). Left alone the tree models would see "categories" that never occur -
the exact mode-collapse failure Phase 3 fixes for the VAE. To keep the three
augmenters on equal footing we snap every categorical block back to a hard
one-hot (argmax) - the same snap the diffusion sampler applies, and the same
valid-one-hot end state the VAE reaches by sampling its decoder logits.

Outputs
  results/phase3b_smote.json
  figures/phase3b_augmenter_comparison.png   (VAE vs Diffusion vs SMOTE)
"""
from __future__ import annotations
import json, numpy as np, matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

from common import (
    load_processed, save_json, set_plot_style,
    RESULTS_DIR, FIGURES_DIR, RANDOM_SEED,
)
from phase3_vae import evaluate_augmented


def snap_categoricals(X, n_numerical, cat_cardinalities):
    """Force every one-hot block back to a hard one-hot via per-block argmax.

    Numerical columns are left untouched; only the categorical blocks (which
    SMOTE may have interpolated into fractional values) are snapped, so the
    augmented matrix matches the valid-one-hot layout the classifiers expect.
    """
    X = X.copy()
    offset = n_numerical
    for card in cat_cardinalities:
        block = X[:, offset:offset + card]
        hard = np.zeros_like(block)
        hard[np.arange(len(block)), block.argmax(axis=1)] = 1.0
        X[:, offset:offset + card] = hard
        offset += card
    return X


def main():
    set_plot_style()
    data = load_processed()
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    n_numerical = data["num_numerical"]
    cat_cardinalities = data["cat_card"]

    n_before = int((y_train == 1).sum())
    # SMOTE balances the minority up to the majority count - same target size as
    # the VAE/diffusion runs (a 4,504-row ~50/50 augmented set).
    X_res, y_res = SMOTE(random_state=RANDOM_SEED).fit_resample(X_train, y_train)
    X_res = snap_categoricals(X_res, n_numerical, cat_cardinalities)
    n_synth = int((y_res == 1).sum() - n_before)
    print(f"[Phase 3b] SMOTE augmented training set: {X_res.shape} "
          f"(added {n_synth} synthetic dropouts)")

    print("[Phase 3b] training classifiers on SMOTE-augmented data ...")
    results = evaluate_augmented(X_res, y_res, X_test, y_test)
    for name, scores in results.items():
        print(f"  {name:15s}  Acc={scores['accuracy']:.3f}  F1={scores['f1']:.3f}  "
              f"AUC={scores['auc_roc']:.3f}")

    with open(RESULTS_DIR / "phase2_baselines.json") as f:
        base = json.load(f)
    delta = {model: {metric: results[model][metric] - base[model][metric]
                     for metric in results[model]}
             for model in results}
    save_json({"smote_augmented": results,
               "baseline_phase2": base,
               "delta": delta,
               "n_synth_added": n_synth},
              RESULTS_DIR / "phase3b_smote.json")

    # Three-way augmenter comparison: VAE vs Diffusion vs SMOTE, per model.
    with open(RESULTS_DIR / "phase3_vae.json") as f:
        vae = json.load(f)["vae_augmented"]
    with open(RESULTS_DIR / "phase4_diffusion.json") as f:
        diff = json.load(f)["diffusion_augmented"]

    models = ["RandomForest", "XGBoost", "MLP"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ["f1", "auc_roc"]):
        x = np.arange(len(models)); bar_width = 0.22
        for j, (label, src, colour) in enumerate([
                ("VAE", vae, "#e67e22"), ("Diffusion", diff, "#16a085"),
                ("SMOTE", results, "#8e44ad")]):
            vals = [src[m][metric] for m in models]
            ax.bar(x + (j - 1) * bar_width, vals, width=bar_width,
                   label=label, color=colour)
        ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9)
        ax.set_ylim(0.6, 1.0); ax.set_title(metric.upper())
        ax.legend(fontsize=8)
    fig.suptitle("Phase 3b - Augmenter comparison (VAE vs Diffusion vs SMOTE)",
                 fontweight="bold")
    fig.savefig(FIGURES_DIR / "phase3b_augmenter_comparison.png"); plt.close(fig)
    print("[Phase 3b] done")


if __name__ == "__main__":
    main()
