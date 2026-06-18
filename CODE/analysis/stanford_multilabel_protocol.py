#!/usr/bin/env python3
"""
Multi-label real-text validation of the Phase-9 encoding protocol on the original
Stanford MOOC Posts dataset (Agrawal & Paepcke 2014).

Motivation: the project's Phase 9 fuses *simulated* comments (a construction ceiling), and
phase9b validates the leakage-safe encoding protocol on ONE real label (forum-post urgency,
cross-corpus). The original Stanford dataset -- the source from which the project's reduced
Stanford.csv was derived -- carries FIVE further real annotation dimensions that Stanford.csv
discards. Here we run the *same* frozen-DistilBERT + train-only-PCA protocol across all of
them, to show the encoding pipeline carries real signal on genuine student writing across
several constructs (not just urgency).

This validates the *protocol*, not a dropout result: the Stanford posts have no dropout labels.

Protocol (identical to Phase 9, importing the pipeline's own code):
  frozen DistilBERT mean-pooled embeddings  ->  PCA(32) + StandardScaler fit on TRAIN only
  ->  LogisticRegression / RandomForest (class_weight balanced)  ->  test AUC / F1 / accuracy.

Reads (read-only):  ../../DATASETS/stanford_mooc_posts/stanfordMOOCForumPostsSet/
                    stanfordMOOCForumPostsSet.txt   (the extracted corpus; see
                    provenance_stanford.md for the source URL + SHA-256)
Writes:  ../../RESULTS/analysis/stanford_multilabel_results.csv, stanford_multilabel_auc.png
"""
from __future__ import annotations
import os
import sys
import csv
import tempfile
from pathlib import Path

# Headless plotting + keep common.py's output dirs out of project/ (it mkdirs under PROJECT_ROOT).
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ["PROJECT_ROOT"] = tempfile.mkdtemp(prefix="stanford_ml_")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                      # KOTSIANTIS_PROJECT
TXT = (ROOT / "DATASETS" / "stanford_mooc_posts"
       / "stanfordMOOCForumPostsSet" / "stanfordMOOCForumPostsSet.txt")
SEED = 42
PCA_COMPONENTS = 32

# Reuse the pipeline's own frozen-encoder mean-pool + metric helper, so the protocol is
# provably identical to Phase 9.
sys.path.insert(0, str(ROOT / "CODE"))
from phase9_multimodal_bert import encode_comments          # noqa: E402
from common import predictive_metrics, set_plot_style        # noqa: E402

# (column name in the .txt, threshold) -> binary "high vs low" task, mirroring
# phase9b_realtext.py's urgency threshold of >= 4 for the 1-7 dimensions.
LABELS = [
    ("Urgency(1-7)",    4.0),
    ("Confusion(1-7)",  4.0),
    ("Sentiment(1-7)",  4.0),
    ("Opinion(1/0)",    0.5),
    ("Question(1/0)",   0.5),
    ("Answer(1/0)",     0.5),
]
TEXT_COL = "Text"


def load_stanford() -> pd.DataFrame:
    raw = TXT.read_bytes().decode("utf-8", "ignore")
    import io
    df = pd.read_csv(io.StringIO(raw), sep="\t")
    assert 29000 < len(df) < 30000, f"unexpected row count {len(df)} (expected ~29604)"
    print(f"[stanford] parsed {len(df)} posts | columns: {list(df.columns)[:7]} ...")
    return df


def main():
    set_plot_style()
    df = load_stanford()
    texts = df[TEXT_COL].astype(str).tolist()

    print(f"[stanford] encoding {len(texts)} posts with frozen DistilBERT (one pass, reused) ...")
    emb = encode_comments(texts)                # (N, 768); same encoder/pooling as Phase 9
    print(f"[stanford] embeddings: {emb.shape}")

    rows = []
    for col, thr in LABELS:
        if col not in df.columns:
            print(f"  [skip] {col}: not in columns"); continue
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy()
        mask = ~np.isnan(vals)
        y = (vals[mask] >= thr).astype(int)
        X = emb[mask]
        pos_rate = float(y.mean())
        if pos_rate < 0.02 or pos_rate > 0.98:
            print(f"  [skip] {col}: degenerate positive rate {pos_rate:.3f}"); continue

        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.25, random_state=SEED, stratify=y)
        # leakage-safe: PCA + scaler fit on TRAIN only
        pca = PCA(n_components=PCA_COMPONENTS, random_state=SEED).fit(Xtr)
        scaler = StandardScaler().fit(pca.transform(Xtr))
        Xtr_t = scaler.transform(pca.transform(Xtr))
        Xte_t = scaler.transform(pca.transform(Xte))

        for mdl_name, mdl in [
            ("LogReg", LogisticRegression(max_iter=1000, class_weight="balanced",
                                          random_state=SEED)),
            ("RandomForest", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                     random_state=SEED, n_jobs=-1)),
        ]:
            mdl.fit(Xtr_t, ytr)
            pred = mdl.predict(Xte_t)
            score = mdl.predict_proba(Xte_t)[:, 1]
            m = predictive_metrics(yte, pred, score)
            rows.append({"label": col, "model": mdl_name, "n": int(mask.sum()),
                         "pos_rate": round(pos_rate, 4), "threshold": thr,
                         "accuracy": round(m["accuracy"], 4), "f1": round(m["f1"], 4),
                         "auc_roc": round(m["auc_roc"], 4)})
            print(f"  {col:15s} {mdl_name:13s} pos={pos_rate:.3f}  "
                  f"AUC={m['auc_roc']:.3f}  F1={m['f1']:.3f}  acc={m['accuracy']:.3f}")

    # ---- write CSV ----
    out_csv = ROOT / "RESULTS" / "analysis" / "stanford_multilabel_results.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[wrote] {out_csv}")

    # ---- bar chart: test AUC per label, LogReg vs RF ----
    labels = [c for c, _ in LABELS if any(r["label"] == c for r in rows)]
    lr = [next((r["auc_roc"] for r in rows if r["label"] == c and r["model"] == "LogReg"), np.nan)
          for c in labels]
    rf = [next((r["auc_roc"] for r in rows if r["label"] == c and r["model"] == "RandomForest"), np.nan)
          for c in labels]
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w/2, lr, w, label="LogReg", color="#3498db")
    ax.bar(x + w/2, rf, w, label="RandomForest", color="#16a085")
    ax.axhline(0.5, color="k", lw=0.8, ls="--", label="chance")
    ax.set_xticks(x); ax.set_xticklabels([c.split("(")[0] for c in labels], rotation=15, fontsize=9)
    ax.set_ylabel("Test AUC"); ax.set_ylim(0.4, 1.0)
    ax.set_title("Leakage-safe DistilBERT protocol on real Stanford MOOC posts (per construct)")
    ax.legend(fontsize=8, ncol=3)
    out_png = ROOT / "RESULTS" / "analysis" / "stanford_multilabel_auc.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"[wrote] {out_png}")


if __name__ == "__main__":
    main()
