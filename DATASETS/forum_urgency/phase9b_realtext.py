"""
Phase 9b - real-text validation of the leakage-safe DistilBERT protocol.

Phase 9 demonstrates the multimodal fusion mechanics on *simulated* comments, so
its lift is only a ceiling. Here we run the same frozen-DistilBERT + train-only-PCA
protocol on a *real* educational-text corpus -- discussion-forum posts labelled for
urgency (Svabensky et al., EDM 2023; test set = Stanford MOOC Posts) -- to show the
encoding pipeline works on genuine student writing. The task is urgency detection,
not dropout: no public dataset pairs dropout labels with free text, so this
validates the *protocol*, not the dropout-multimodal result.

  train : All_Courses_REDACTED_CODED.csv   (coded posts, 9 courses)
  test  : Stanford.csv                      (held-out, different domain; subsampled)
  label : urgent = 1[Urgency_1_7 >= 4]

Outputs: results/realtext_urgency.json, figures/realtext_urgency.png
"""
from __future__ import annotations
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# This script lives under DATASETS/forum_urgency/; add the shared CODE/ dir
# (common.py + phase9_multimodal_bert) to the import path so it runs from anywhere.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "CODE"))
from common import (RESULTS_DIR, FIGURES_DIR, RANDOM_SEED,
                    predictive_metrics, save_json, set_plot_style)
from phase9_multimodal_bert import encode_comments

# The forum-urgency corpus ships next to this script.
SRC = Path(__file__).resolve().parent
# Outputs go under RESULTS/forum_urgency/ (this standalone validation isn't part
# of the per-dataset PROJECT_ROOT runs, so it routes its own outputs).
_RES = Path(__file__).resolve().parents[2] / "RESULTS" / "forum_urgency"
(_RES / "results").mkdir(parents=True, exist_ok=True)
(_RES / "figures").mkdir(parents=True, exist_ok=True)
THRESHOLD = 4        # urgency runs 1-7; treat >= 4 as "urgent"
TEST_CAP = 6000      # Stanford has ~30k posts; cap the test set so CPU encoding stays quick


def load(name, cap=None):
    df = pd.read_csv(SRC / name).dropna(subset=["post_text", "Urgency_1_7"])
    text = df["post_text"].astype(str).tolist()
    y = (df["Urgency_1_7"].values >= THRESHOLD).astype(int)
    if cap and len(text) > cap:
        idx = np.random.default_rng(RANDOM_SEED).permutation(len(text))[:cap]
        text = [text[i] for i in idx]; y = y[idx]
    return text, y


def main():
    set_plot_style()
    tr_text, y_tr = load("All_Courses_REDACTED_CODED.csv")
    te_text, y_te = load("Stanford.csv", TEST_CAP)
    print(f"[realtext] train {len(tr_text)} posts (urgent {y_tr.mean():.3f}) | "
          f"test {len(te_text)} posts (urgent {y_te.mean():.3f})")

    emb_tr = encode_comments(tr_text)
    emb_te = encode_comments(te_text)
    # the Phase-9 leakage-safe step: PCA + scaler fit on the training text only
    pca = PCA(n_components=32, random_state=RANDOM_SEED).fit(emb_tr)
    scaler = StandardScaler().fit(pca.transform(emb_tr))
    X_tr = scaler.transform(pca.transform(emb_tr))
    X_te = scaler.transform(pca.transform(emb_te))

    out = {"train_n": len(tr_text), "test_n": len(te_text), "threshold": THRESHOLD,
           "train_urgent_rate": float(y_tr.mean()), "test_urgent_rate": float(y_te.mean()),
           "models": {}}
    lr = LogisticRegression(max_iter=1000, class_weight="balanced",
                            random_state=RANDOM_SEED).fit(X_tr, y_tr)
    out["models"]["LogReg"] = predictive_metrics(y_te, lr.predict(X_te), lr.predict_proba(X_te)[:, 1])
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                random_state=RANDOM_SEED, n_jobs=-1).fit(X_tr, y_tr)
    out["models"]["RandomForest"] = predictive_metrics(y_te, rf.predict(X_te), rf.predict_proba(X_te)[:, 1])
    for k, v in out["models"].items():
        print(f"  {k:13s} acc={v['accuracy']:.3f} f1={v['f1']:.3f} auc={v['auc_roc']:.3f}")
    save_json(out, _RES / "results" / "realtext_urgency.json")

    fig, ax = plt.subplots(figsize=(5, 3.5))
    mets = ["accuracy", "f1", "auc_roc"]; x = np.arange(len(mets)); w = 0.35
    for j, (k, c) in enumerate([("LogReg", "#3498db"), ("RandomForest", "#16a085")]):
        ax.bar(x + (j - 0.5) * w, [out["models"][k][m] for m in mets], w, label=k, color=c)
    ax.set_xticks(x); ax.set_xticklabels([m.upper() for m in mets]); ax.set_ylim(0, 1)
    ax.set_title("Real-text urgency detection (frozen-DistilBERT protocol)")
    ax.legend(fontsize=8)
    fig.savefig(_RES / "figures" / "realtext_urgency.png"); plt.close(fig)
    print("[realtext] done")


if __name__ == "__main__":
    main()
