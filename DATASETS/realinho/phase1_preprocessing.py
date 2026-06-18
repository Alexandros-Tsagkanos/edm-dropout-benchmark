"""
Phase 1 - data exploration and preprocessing.
This is the only phase that touches the raw CSV. It does three things:

  1. A quick EDA pass: class balance, missing values, and - since Gender is
     our protected attribute - the dropout rate broken down by gender.
  2. The stratified 75/25 split, one-hot encoding the categoricals and
     standardising the numericals (everything fit on TRAIN only -> no leakage).
  3. Freezes the resulting arrays + schema metadata to disk, so that every
     later phase trains and tests on the exact same split.

Outputs:
  results/phase1_eda.json           aggregate statistics
  data/processed.npz                X_train / X_test / y_* / s_*
  data/processed_meta.json          feature names, cardinalities, ...
  figures/phase1_class_balance.png
  figures/phase1_dropout_by_gender.png
  figures/phase1_age_distribution.png
  figures/phase1_corr_heatmap.png
"""
from __future__ import annotations
import json
import numpy as np, matplotlib.pyplot as plt, seaborn as sns
# This script lives under DATASETS/<dataset>/; add the shared CODE/ dir (common.py
# + the generic phases) to the import path so it runs from anywhere.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "CODE"))
from common import (
    load_raw, preprocess, make_binary_target, save_json, set_plot_style,
    CATEGORICAL_COLS, PROTECTED_ATTR, TARGET_COL,
    RESULTS_DIR, FIGURES_DIR, PROCESSED_NPZ, PROCESSED_META_JSON,
)





# One small function per figure - keeps main() reading as a clean
# load -> explore -> preprocess -> save narrative.
def plot_class_balance(df, out_path):
    """Bar chart of the original three target classes, annotated with shares."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    counts = df[TARGET_COL].value_counts()
    sns.barplot(x=counts.index, y=counts.values, ax=ax,
                palette=["#d9534f", "#5cb85c", "#f0ad4e"])
    ax.set_title("Original 3-Class Target Distribution")
    ax.set_ylabel("Count"); ax.set_xlabel("")
    for i, count in enumerate(counts.values):
        ax.text(i, count + 30, f"{count}\n({count / len(df) * 100:.1f}%)",
                ha="center", fontsize=9)
    fig.savefig(out_path); plt.close(fig)


def plot_dropout_by_gender(df, out_path):
    """The headline fairness motivation: dropout rate per gender group."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    dropout_by_gender = df.groupby(PROTECTED_ATTR)["y"].mean()
    sns.barplot(x=["Female (0)", "Male (1)"], y=dropout_by_gender.values, ax=ax,
                palette=["#9b59b6", "#3498db"])
    ax.set_title("Dropout Rate by Gender (Protected Attribute)")
    ax.set_ylabel("P(Dropout=1)")
    ax.set_ylim(0, max(dropout_by_gender.values) * 1.3)
    for i, rate in enumerate(dropout_by_gender.values):
        ax.text(i, rate + 0.01, f"{rate:.3f}", ha="center", fontsize=10)
    fig.savefig(out_path); plt.close(fig)


def plot_age_distribution(df, out_path):
    """Age-at-enrollment density, split by outcome - dropouts skew older."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    sns.kdeplot(data=df, x="Age at enrollment", hue=TARGET_COL,
                fill=True, common_norm=False, alpha=0.4, ax=ax)
    ax.set_title("Age at Enrollment by Outcome")
    fig.savefig(out_path); plt.close(fig)


def plot_correlation_heatmap(df, out_path):
    """Correlation among the numerical features, including the binary label y,
    so it's easy to eyeball which numerics move with dropout."""
    numeric_cols = [c for c in df.columns
                    if c not in CATEGORICAL_COLS + [TARGET_COL]]
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, cmap="RdBu_r", center=0, ax=ax,
                cbar_kws={"shrink": 0.7}, square=False)
    ax.set_title("Correlation of Numerical Features (incl. binary target y)")
    plt.xticks(rotation=70, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    fig.savefig(out_path); plt.close(fig)






def main():
    set_plot_style()

    # Load once and attach the binary dropout label for the EDA/plots below.
    df_raw = load_raw()
    df = make_binary_target(df_raw)
    print(f"[Phase 1] Loaded {df.shape[0]} rows x {df.shape[1]} columns")

    # Statistics worth keeping on the record (written to JSON at the end).
    eda = {
        "n_rows": int(df.shape[0]),
        "n_cols_raw": int(df.shape[1]),
        "missing_values_total": int(df.isna().sum().sum()),
        "missing_per_column": df.isna().sum().to_dict(),
        "target_class_counts_original": df[TARGET_COL].value_counts().to_dict(),
        "target_binary_counts": df["y"].value_counts().to_dict(),
        "dropout_rate":  float(df["y"].mean()),
        "gender_distribution": df[PROTECTED_ATTR].value_counts().to_dict(),
        "dropout_rate_by_gender": (
            df.groupby(PROTECTED_ATTR)["y"].mean().to_dict()
        ),
        "categorical_cols": CATEGORICAL_COLS,
        "protected_attribute": PROTECTED_ATTR,
    }

    plot_class_balance(df,        FIGURES_DIR / "phase1_class_balance.png")
    plot_dropout_by_gender(df,    FIGURES_DIR / "phase1_dropout_by_gender.png")
    plot_age_distribution(df,     FIGURES_DIR / "phase1_age_distribution.png")
    plot_correlation_heatmap(df,  FIGURES_DIR / "phase1_corr_heatmap.png")

    # Preprocess from the RAW frame - preprocess() re-derives y itself, so
    # there's no need to hand it the labelled copy.
    proc = preprocess(df_raw)
    print(f"[Phase 1] Encoded feature matrix: train={proc['X_train'].shape}, "
          f"test={proc['X_test'].shape}")
    print(f"[Phase 1] Train dropout rate: {proc['y_train'].mean():.3f} | "
          f"Test dropout rate: {proc['y_test'].mean():.3f}")

    # Freeze the arrays + the schema metadata. From here on, every phase reads
    # these back via common.load_processed() instead of re-splitting.
    np.savez_compressed(
        PROCESSED_NPZ,
        X_train=proc["X_train"], X_test=proc["X_test"],
        y_train=proc["y_train"], y_test=proc["y_test"],
        s_train=proc["s_train"], s_test=proc["s_test"],
    )
    meta = {
        "feature_names": proc["feature_names"],
        "numerical_cols": proc["numerical_cols"],
        "categorical_cols": proc["categorical_cols"],
        "num_numerical": proc["num_numerical"],
        "cat_card": proc["cat_card"],
        "protected_attribute": PROTECTED_ATTR,
    }
    with open(PROCESSED_META_JSON, "w") as f:
        json.dump(meta, f, indent=2)

    # Round out the EDA record with the post-encoding shapes, then save it.
    eda.update({
        "encoded_train_shape": list(proc["X_train"].shape),
        "encoded_test_shape":  list(proc["X_test"].shape),
        "num_numerical_features":   proc["num_numerical"],
        "num_categorical_features": len(proc["categorical_cols"]),
        "num_total_features_after_onehot": int(proc["X_train"].shape[1]),
    })
    save_json(eda, RESULTS_DIR / "phase1_eda.json")
    print("[Phase 1] EDA & preprocessing complete")


if __name__ == "__main__":
    main()
