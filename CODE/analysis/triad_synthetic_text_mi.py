#!/usr/bin/env python3
"""
Quantify the label signal in TRIAD-Drop's GPT-generated comments (apples-to-apples).

TRIAD-Drop (Mihoubi, Zerkouk, Chikhaoui, 2025; arXiv:2507.05285) augments the UCI Realinho
dataset -- the SAME tabular base this project uses -- with a `student_comment` column generated
by GPT-4.5 conditioned on the student's attributes (per their own README + repo). Their multimodal
result (F1 ~0.85) therefore rests on synthetic, attribute/label-correlated text, just like our
Phase 9. Here we run our leakage-safe text-only protocol on THEIR comments to measure how much
dropout signal the GPT text carries -- directly comparable to our text-only F1 ~0.70 (templated)
and our Stanford real-text AUCs.

This is NOT real text and does NOT make a multimodal dropout result real; it quantifies a ceiling.

Reads (read-only): ../../DATASETS/triad_drop/TRIAD Zerkouk Github.zip
                   :: TRIAD-Drop-main/dataset_with_comments.csv
Writes: ../../RESULTS/analysis/triad_synthetic_text_mi_results.csv
"""
from __future__ import annotations
import os, sys, csv, zipfile, tempfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ["PROJECT_ROOT"] = tempfile.mkdtemp(prefix="triad_mi_")   # keep common.py's dirs out of project/

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ZIP = ROOT / "DATASETS" / "triad_drop" / "TRIAD Zerkouk Github.zip"
MEMBER = "TRIAD-Drop-main/dataset_with_comments.csv"
SEED = 42

sys.path.insert(0, str(ROOT / "CODE"))
from phase9_multimodal_bert import encode_comments        # noqa: E402  (frozen DistilBERT mean-pool)
from common import predictive_metrics                      # noqa: E402


def main():
    with zipfile.ZipFile(ZIP) as z, z.open(MEMBER) as f:
        df = pd.read_csv(f)
    print(f"[triad] {df.shape[0]} rows | Target counts: {df['Target'].value_counts().to_dict()}")
    y = (df["Target"] == "Dropout").astype(int).to_numpy()
    texts = df["student_comment"].astype(str).tolist()
    print(f"[triad] dropout rate = {y.mean():.4f} (n={len(y)})")

    print(f"[triad] encoding {len(texts)} GPT comments with frozen DistilBERT ...")
    emb = encode_comments(texts)
    print(f"[triad] embeddings: {emb.shape}")

    Xtr, Xte, ytr, yte = train_test_split(emb, y, test_size=0.25, random_state=SEED, stratify=y)
    pca = PCA(n_components=32, random_state=SEED).fit(Xtr)        # leakage-safe: fit on train only
    scaler = StandardScaler().fit(pca.transform(Xtr))
    Xtr_t = scaler.transform(pca.transform(Xtr))
    Xte_t = scaler.transform(pca.transform(Xte))

    rows = []
    for name, mdl in [
        ("LogReg", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)),
        ("RandomForest", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                random_state=SEED, n_jobs=-1)),
    ]:
        mdl.fit(Xtr_t, ytr)
        m = predictive_metrics(yte, mdl.predict(Xte_t), mdl.predict_proba(Xte_t)[:, 1])
        rows.append({"model": name, "n": len(y), "dropout_rate": round(float(y.mean()), 4),
                     "accuracy": round(m["accuracy"], 4), "f1": round(m["f1"], 4),
                     "auc_roc": round(m["auc_roc"], 4)})
        print(f"  text-only {name:13s} AUC={m['auc_roc']:.3f}  F1={m['f1']:.3f}  acc={m['accuracy']:.3f}")

    # rough MI estimate (nats->bits) on the train PCA features; label clearly as an estimate
    mi_nats = mutual_info_classif(Xtr_t, ytr, random_state=SEED).sum()
    mi_bits = float(mi_nats / np.log(2))
    print(f"  [estimate] summed I(text-PCA32; y) ~ {mi_bits:.2f} bits (kNN estimator; for context only)")

    out = ROOT / "RESULTS" / "analysis" / "triad_synthetic_text_mi_results.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"[wrote] {out}")
    print(f"[note] reference: our templated text-only dropout F1 ~0.70; tabular baseline F1 ~0.79; "
          f"TRIAD multimodal reported F1 ~0.85")


if __name__ == "__main__":
    main()
