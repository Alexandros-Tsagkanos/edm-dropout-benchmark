"""
Shared helpers for the dropout-prediction pipeline.

Anything more than one phase needs lives here, so the individual phase
scripts stay thin and - more importantly - so they all consume the same
preprocessing and the same train/test split.

The dataset is the UCI "Predict Students' Dropout and Academic Success" set.
Its native target has three classes; we reframe it as a binary "did the
student drop out?" problem (see make_binary_target).

What's in here:
  - project paths (override with the PROJECT_ROOT / RAW_CSV env vars)
  - the categorical-vs-numerical feature schema
  - load + preprocess, split-then-fit so nothing leaks from test into train
  - predictive metrics and group-fairness metrics
  - a couple of IO / plotting conveniences
"""
from __future__ import annotations
import json, os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    matthews_corrcoef,
)

warnings.filterwarnings("ignore")





# Paths and global config
# Resolve everything relative to this file so the pipeline behaves the same
# on any machine. Both roots are overridable from the environment for CI or
# for pointing at a relocated dataset.
# Default outputs go to RESULTS/realinho/single_seed/ - NOT the SUBMIT root, whose
# RESULTS/ folder collides with a lowercase "results" dir on case-insensitive
# filesystems (Windows/macOS). run_all.py and the multi-seed harnesses override
# PROJECT_ROOT explicitly, so this default only applies to direct one-off runs.
PROJECT_ROOT = Path(os.environ.get(
    "PROJECT_ROOT",
    Path(__file__).resolve().parent.parent / "RESULTS" / "realinho" / "single_seed"))
CODE_DIR    = PROJECT_ROOT / "code"
DATA_DIR    = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
# The Realinho raw CSV lives under datasets/realinho/. Resolve it from THIS
# file's real location (not PROJECT_ROOT, which the multi-seed harness overrides
# per seed) so a direct `python phaseN.py` still finds it; still overridable via
# the RAW_CSV env var.
RAW_CSV = Path(os.environ.get(
    "RAW_CSV", Path(__file__).resolve().parent.parent / "DATASETS" / "realinho" / "data.csv"))





for directory in (DATA_DIR, RESULTS_DIR, FIGURES_DIR):
    directory.mkdir(parents=True, exist_ok=True)
# A single seed reused by every phase. Seeding numpy here (on import) means
# anything that imports common starts from the same RNG state. Overridable via
# the RANDOM_SEED env var so a harness can sweep seeds for a multi-seed study;
# unset it defaults to 42 and the pipeline reproduces exactly as before.
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", 42))
np.random.seed(RANDOM_SEED)




# Feature schema for the UCI dropout dataset
# These 18 columns are categorical in the raw file - e.g. "Course" is
# an integer id and "Application mode" a lookup value, not an ordinal rank - so
# they get one-hot encoded rather than fed in as numbers. Everything else is
# genuinely continuous (grades, ages, economic indicators,...).
CATEGORICAL_COLS = [
    "Marital status", "Application mode", "Application order", "Course",
    "Daytime/evening attendance", "Previous qualification", "Nacionality",
    "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation",
    "Displaced", "Educational special needs", "Debtor",
    "Tuition fees up to date", "Gender", "Scholarship holder", "International",
]

# Gender is the sensitive attribute we audit fairness against in Phases 7-8.
PROTECTED_ATTR = "Gender"
TARGET_COL = "Target"


def set_plot_style():
    """One shared look for every figure, so the plots across phases match."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "DejaVu Sans",
    })





# Data loading and preprocessing
def load_raw() -> pd.DataFrame:
    """Read the semicolon-delimited CSV and tidy up the header row.

    The export carries a UTF-8 BOM plus the odd stray quote/tab in the column
    names, which quietly breaks later lookups by name - so we scrub those once,
    here, and every caller gets clean column labels.
    """
    df = pd.read_csv(RAW_CSV, sep=";")
    df.columns = [c.replace("\ufeff", "").strip().strip('"').replace("\t", "")
                  for c in df.columns]
    return df


def make_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the 3-class target into the binary dropout label we predict.
        y = 1  ->  Dropout                 (minority, and the positive class
                                            we actually care about flagging)
        y = 0  ->  Graduate or Enrolled
    Folding "Enrolled" into the negative side is deliberate: the goal is to
    catch students at risk of leaving, not those simply still in progress.
    """
    df = df.copy()
    df["y"] = (df[TARGET_COL] == "Dropout").astype(int)
    return df


def preprocess(df: pd.DataFrame, test_size: float = 0.25):
    """Turn the raw dataframe into model-ready arrays without leaking test info.

    The discipline that matters here is split first, then fit: we carve out
    the test set before fitting the one-hot encoder and the scaler, so no
    test-set statistics ever bleed into the training representation.

    Returns a dict bundling the train/test arrays, the held-out protected
    attribute, the fitted transformers, and enough schema metadata (feature
    names, per-column cardinalities) for the generative models in later phases
    to reconstruct the column layout.
    """
    df = make_binary_target(df)

    # Anything not flagged categorical, and not a label column, is numeric.
    numerical_cols = [c for c in df.columns
                      if c not in CATEGORICAL_COLS + [TARGET_COL, "y"]]

    X_df = df[CATEGORICAL_COLS + numerical_cols].copy()
    y = df["y"].values

    # Stratify on y so both splits keep the ~32% dropout rate; fixing the seed
    # makes the split identical every run (Phase 1 freezes it for everyone).
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=test_size, stratify=y, random_state=RANDOM_SEED,
    )

    # Keep the protected attribute aside in its raw 0/1 coding, so fairness
    # metrics can slice by group later without digging it out of the one-hot.
    s_train = X_train_df[PROTECTED_ATTR].values
    s_test  = X_test_df[PROTECTED_ATTR].values

    # One-hot the categoricals, fit on train only. handle_unknown="ignore"
    # turns a category seen only in test into an all-zero row instead of an error.
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    cat_train = ohe.fit_transform(X_train_df[CATEGORICAL_COLS])
    cat_test  = ohe.transform(X_test_df[CATEGORICAL_COLS])

    # Standardise the numericals, again fit on train only.
    scaler = StandardScaler()
    num_train = scaler.fit_transform(X_train_df[numerical_cols])
    num_test  = scaler.transform(X_test_df[numerical_cols])

    # Final matrix layout is numericals first, then the one-hot blocks. The
    # feature_names list below is built in this exact order, and the generative
    # decoders rely on that alignment to split the vector back apart.
    X_train = np.concatenate([num_train, cat_train], axis=1).astype(np.float32)
    X_test  = np.concatenate([num_test,  cat_test],  axis=1).astype(np.float32)

    # Readable name for every encoded column + the size of each categorical
    # block (cat_card), which the VAE/diffusion models need to place one
    # softmax head per categorical column.
    cat_card = [len(cats) for cats in ohe.categories_]
    cat_feature_names = []
    for col, cats in zip(CATEGORICAL_COLS, ohe.categories_):
        cat_feature_names.extend([f"{col}={c}" for c in cats])
    feature_names = numerical_cols + cat_feature_names

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "s_train": s_train, "s_test": s_test,
        "feature_names": feature_names,
        "numerical_cols": numerical_cols,
        "categorical_cols": CATEGORICAL_COLS,
        "num_numerical": len(numerical_cols),
        "cat_card": cat_card,
        "scaler": scaler, "ohe": ohe,
    }





# Metrics
def predictive_metrics(y_true, y_pred, y_score=None) -> dict:
    """The usual binary-classification scores.
    AUC needs probabilities, so it's only computed when y_score is given - and
    it falls back to NaN if a slice happens to hold a single class (roc_auc is
    undefined there).
    """
    out = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        # MCC is the imbalance-robust metric the EDM SSL literature increasingly
        # prefers (e.g. Raftopoulos et al. 2025, MOOC dropout); recorded here so
        # reproductions report it alongside F1/AUC.
        "mcc":       float(matthews_corrcoef(y_true, y_pred)),
    }
    if y_score is not None:
        try:
            out["auc_roc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            out["auc_roc"] = float("nan")
    return out


def fairness_metrics(y_true, y_pred, s) -> dict:
    """Group-fairness gaps for a binary sensitive attribute s in {0, 1}.
    Each headline number is an absolute difference between the two groups, so
    0 means parity and larger means less fair. We report several because they
    encode genuinely different notions of "fair":

        demographic_parity_diff  |P(y_hat=1|s=1) - P(y_hat=1|s=0)|
        equal_opportunity_diff   |TPR(s=1) - TPR(s=0)|   (parity among true positives)
        equalized_odds_diff      max(|d TPR|, |d FPR|)   (the stricter of the two)
        accuracy_diff            |Acc(s=1) - Acc(s=0)|

    Group-conditioned rates return NaN for an empty group rather than 0, so a
    missing group can never masquerade as "perfectly fair".
    """
    s = np.asarray(s); y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    group0, group1 = (s == 0), (s == 1)

    def selection_rate(group):                 # P(predict positive | in group)
        return float(np.mean(y_pred[group])) if group.sum() else float("nan")

    def tpr(group):                            # true-positive rate within group
        actual_pos = group & (y_true == 1)
        return float(np.mean(y_pred[actual_pos])) if actual_pos.sum() else float("nan")

    def fpr(group):                            # false-positive rate within group
        actual_neg = group & (y_true == 0)
        return float(np.mean(y_pred[actual_neg])) if actual_neg.sum() else float("nan")

    dp_diff  = abs(selection_rate(group1) - selection_rate(group0))
    tpr_diff = abs(tpr(group1) - tpr(group0))
    fpr_diff = abs(fpr(group1) - fpr(group0))
    eq_odds  = max(tpr_diff, fpr_diff)
    acc_group0 = float((y_pred[group0] == y_true[group0]).mean()) if group0.sum() else float("nan")
    acc_group1 = float((y_pred[group1] == y_true[group1]).mean()) if group1.sum() else float("nan")

    return {
        "demographic_parity_diff": dp_diff,
        "equal_opportunity_diff":  tpr_diff,
        "equalized_odds_diff":     eq_odds,
        "fpr_diff":                fpr_diff,
        "accuracy_diff":           abs(acc_group1 - acc_group0),
        "selection_rate_group0":   selection_rate(group0),
        "selection_rate_group1":   selection_rate(group1),
        "accuracy_group0":         acc_group0,
        "accuracy_group1":         acc_group1,
    }


def save_json(obj, path):
    """json.dump, but coercing the numpy scalar/array types it can't serialise."""
    def _coerce_numpy(o):
        if isinstance(o, np.integer):  return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray):  return o.tolist()
        return str(o)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_coerce_numpy)







# Processed-data accessor
# Phase 1 writes these two files; every later phase reads them back instead of
# re-running preprocessing, which is what keeps the split identical everywhere.
PROCESSED_NPZ = DATA_DIR / "processed.npz"
PROCESSED_META_JSON = DATA_DIR / "processed_meta.json"


def load_processed():
    """Re-load the train/test arrays + metadata that Phase 1 froze to disk."""
    npz = np.load(PROCESSED_NPZ, allow_pickle=True)
    with open(PROCESSED_META_JSON) as f:
        meta = json.load(f)
    return {
        "X_train": npz["X_train"], "X_test": npz["X_test"],
        "y_train": npz["y_train"], "y_test": npz["y_test"],
        "s_train": npz["s_train"], "s_test": npz["s_test"],
        **meta,
    }
