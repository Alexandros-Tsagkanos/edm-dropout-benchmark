#!/usr/bin/env python3
"""
In-container MORF job — DRAFT TEMPLATE (runs INSIDE the MORF enclave, not locally).

MORF invokes this image as:  python3 mwe.py --mode {extract|extract-holdout|train|test}
Raw Coursera exports are mounted read-only at /input/<course>/<session>/ ; we write artifacts to
/output/ . This mirrors our leakage-safe Phase-9 / Stanford protocol on REAL forum text:
  frozen DistilBERT mean-pool  ->  PCA fit on TRAIN only  ->  LogisticRegression / RF (+ fusion).

TODO before submission (validate in the enclave, schema not public):
  * Map the actual Coursera Spark/Phoenix forum SQL tables (e.g. forum_posts / forum_comments) and the
    learner id key; concatenate each learner's posts into one document.
  * Map the completion/dropout label (grade/certification export). y = 1[did NOT complete] to match our
    dropout convention, or 1[completed] — pick one and document it.
  * Confirm MORF's exact /output filename conventions for extract/train/test artifacts.
This file intentionally keeps the modelling identical to the rest of the project so the real-text
numbers are directly comparable to our synthetic-text (0.28 bits / F1 0.70) and Stanford (AUC 0.81-0.90)
results.
"""
from __future__ import annotations
import argparse, glob, os, pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score

INPUT = Path("/input")
OUTPUT = Path("/output")
SEED = 42
PCA_K = 32


def encode_text(docs: list[str]) -> np.ndarray:
    """Frozen DistilBERT mean-pool — same protocol as the project's phase9 encoder.

    One deliberate deviation: max_length=256 instead of phase9's 48. The phase-9
    cap was sized for one-sentence simulated comments; real MOOC forum posts are
    longer, so we raise the truncation window here (disclose whichever cap is
    used when reporting results).
    """
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    enc = AutoModel.from_pretrained("distilbert-base-uncased").eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(docs), 32):
            t = tok(docs[i:i + 32], padding=True, truncation=True, max_length=256, return_tensors="pt")
            h = enc(**t).last_hidden_state
            m = t["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def load_session(session_dir: Path) -> pd.DataFrame:
    """TODO: parse the Coursera forum SQL export under session_dir into one row per learner:
       columns = [learner_id, text (concatenated posts), <tabular features...>, y (completion)].
       Returns an empty frame if the session has no forum data."""
    raise NotImplementedError("Map the Coursera forum SQL schema here (see TODO at top).")


def mode_extract(holdout: bool = False):
    rows = []
    for sess in glob.glob(str(INPUT / "*" / "*")):
        df = load_session(Path(sess))
        if len(df):
            df["__session__"] = sess
            rows.append(df)
    allrows = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    name = "holdout_features.pkl" if holdout else "features.pkl"
    allrows.to_pickle(OUTPUT / name)
    print(f"[extract{'-holdout' if holdout else ''}] {len(allrows)} learners -> {name}")


def mode_train():
    df = pd.read_pickle(OUTPUT / "features.pkl")
    emb = encode_text(df["text"].astype(str).tolist())
    pca = PCA(n_components=PCA_K, random_state=SEED).fit(emb)          # TRAIN-only fit
    scaler = StandardScaler().fit(pca.transform(emb))
    X = scaler.transform(pca.transform(emb))
    y = df["y"].to_numpy()
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED).fit(X, y)
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                random_state=SEED, n_jobs=-1).fit(X, y)
    with open(OUTPUT / "model.pkl", "wb") as fh:
        pickle.dump({"pca": pca, "scaler": scaler, "logreg": clf, "rf": rf}, fh)
    print(f"[train] fit on {len(y)} learners (dropout rate {y.mean():.3f})")


def mode_test():
    df = pd.read_pickle(OUTPUT / "holdout_features.pkl")
    with open(OUTPUT / "model.pkl", "rb") as fh:
        M = pickle.load(fh)
    X = M["scaler"].transform(M["pca"].transform(encode_text(df["text"].astype(str).tolist())))
    y = df["y"].to_numpy()
    res = {}
    for name, mdl in [("logreg", M["logreg"]), ("rf", M["rf"])]:
        p = mdl.predict_proba(X)[:, 1]
        res[name] = {"auc": float(roc_auc_score(y, p)),
                     "f1": float(f1_score(y, (p >= 0.5).astype(int), zero_division=0))}
    pd.DataFrame(res).to_csv(OUTPUT / "results.csv")
    print(f"[test] {res}")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["extract", "extract-holdout", "train", "test"])
    mode = ap.parse_args().mode
    if mode == "extract":            mode_extract(holdout=False)
    elif mode == "extract-holdout":  mode_extract(holdout=True)
    elif mode == "train":            mode_train()
    elif mode == "test":             mode_test()


if __name__ == "__main__":
    main()
