"""
OULAD preprocessing - the second dataset.

Turns the Open University Learning Analytics Dataset (studentInfo.csv) into the
exact same on-disk format Phase 1 produces for the Realinho data, so the rest of
the pipeline (phases 2-8, the SMOTE baseline, the multi-seed harness) runs on it
unchanged. We deliberately use only the demographic / study-load columns from
studentInfo: the registration tables carry the un-registration date, which would
leak the withdrawal label.

  dropout label   y = 1[final_result == "Withdrawn"]   (~31% positive)
  protected attr  gender (kept as a feature too, mirroring the Realinho setup)

Reads:  <project>/datasets/oulad/studentInfo.csv
Writes: $PROJECT_ROOT/data/processed.npz + processed_meta.json   (PROJECT_ROOT
        is overridable so a per-seed run can be isolated, exactly as for Realinho)
"""
from __future__ import annotations
import json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# This script lives under DATASETS/oulad/; add the shared CODE/ dir (common.py)
# to the import path so it runs from anywhere.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "CODE"))
from common import DATA_DIR, RANDOM_SEED, PROCESSED_NPZ, PROCESSED_META_JSON, save_json

# The OULAD source CSVs ship next to this adapter.
SRC = Path(__file__).resolve().parent

# studentInfo columns. gender is both a feature and the audited protected attribute,
# mirroring how Gender is treated in the Realinho pipeline.
CATEGORICAL_COLS = ["code_module", "code_presentation", "gender", "region",
                    "highest_education", "imd_band", "age_band", "disability"]
NUMERICAL_COLS = ["num_of_prev_attempts", "studied_credits"]
PROTECTED_ATTR = "gender"


def main():
    df = pd.read_csv(SRC / "studentInfo.csv")
    y = (df["final_result"] == "Withdrawn").astype(int).values
    # gender M/F -> 1/0 for the fairness slice (kept in the features as well)
    s = (df[PROTECTED_ATTR] == "M").astype(int).values
    # imd_band has ~1k missing values; make "missing" an explicit category rather
    # than dropping rows or imputing a fake band.
    df = df.copy()
    df["imd_band"] = df["imd_band"].fillna("missing")

    X_cat = df[CATEGORICAL_COLS].astype(str)
    X_num = df[NUMERICAL_COLS].astype(float)

    # Same discipline as Realinho: stratified 75/25 split first, then fit the
    # encoder and scaler on the training rows only.
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.25, stratify=y, random_state=RANDOM_SEED)
    s_train, s_test = s[tr], s[te]
    y_train, y_test = y[tr], y[te]

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(X_cat.iloc[tr])
    scaler = StandardScaler().fit(X_num.iloc[tr])

    def encode(rows):
        cat = ohe.transform(X_cat.iloc[rows])
        num = scaler.transform(X_num.iloc[rows])
        return np.concatenate([num, cat], axis=1).astype(np.float32)

    X_train, X_test = encode(tr), encode(te)
    cat_card = [len(c) for c in ohe.categories_]
    cat_names = [f"{col}={c}" for col, cats in zip(CATEGORICAL_COLS, ohe.categories_) for c in cats]
    feature_names = NUMERICAL_COLS + cat_names

    np.savez_compressed(
        PROCESSED_NPZ, X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test, s_train=s_train, s_test=s_test)
    save_json({"feature_names": feature_names,
               "numerical_cols": NUMERICAL_COLS,
               "categorical_cols": CATEGORICAL_COLS,
               "num_numerical": len(NUMERICAL_COLS),
               "cat_card": cat_card,
               "protected_attribute": PROTECTED_ATTR},
              PROCESSED_META_JSON)

    print(f"[OULAD] n={len(df)}  encoded train={X_train.shape} test={X_test.shape}")
    print(f"[OULAD] dropout rate train={y_train.mean():.3f} test={y_test.mean():.3f}  "
          f"| male-rate train={s_train.mean():.3f}")
    print(f"[OULAD] dropout by gender: "
          f"F={y[s==0].mean():.3f}  M={y[s==1].mean():.3f}")


if __name__ == "__main__":
    main()
