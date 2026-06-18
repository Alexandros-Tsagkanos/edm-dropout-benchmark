#!/usr/bin/env python3
"""
Mutual information injected by the Phase-9 simulated comments (answers reviewer Q2).

Reviewer Q2: "If the comments are generated via a hash with three tone pools whose
mixture depends on the outcome, what is I(text; label) by construction, and how do you
defend that the ~0.06 multimodal F1 lift is not simply recovering the signal you injected?"

The generative process (phase9_multimodal_bert.make_comment) is fully known, so the
information the text channel carries about the label is *exactly* computable. Because each
comment is a deterministic draw from one of three tone pools (5 distinct templates each,
15 in total) and the within-pool choice is independent of the label, the label-relevant
information collapses to the tone pool:  I(text; y) = I(pool; y)  (data-processing
inequality => this also upper-bounds anything DistilBERT+PCA can extract).

We report:
  1. H(y), and the exact I(text; y) in bits (from the known mixture + empirical P(y));
  2. an empirical cross-check of I(comment; y) computed directly from the committed
     student_comments.csv counts (should match (1));
  3. a text-ONLY classifier: the Bayes-optimal map comment -> y, fit on the Phase-9 train
     split and scored on the Phase-9 test split (same seed/stratification as phase9), to
     show how much dropout F1 the injected channel alone yields.

Reads (read-only):  ../../RESULTS/realinho/single_seed/data/student_comments.csv
                    (columns: row_idx, y, comment; deterministic given row index + label,
                    so identical to the dev-tree copy this script originally read)
Writes: nothing (prints a report; numbers are transcribed into simulated_text_mi_results.md)
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

HERE = Path(__file__).resolve().parent
CSV = (HERE.parents[1] / "RESULTS" / "realinho" / "single_seed"
       / "data" / "student_comments.csv")
SEED = 42   # matches RANDOM_SEED used by phase9 to recover the train/test split

# Tone-pool mixture conditioned on the label, copied verbatim from
# phase9_multimodal_bert.make_comment():
#   y == 1 (dropout):           choice(["neg","neg","neg","neu"])  -> neg 3/4, neu 1/4
#   y == 0 (graduate/enrolled): choice(["pos","pos","neu","neu","neg"]) -> pos 2/5, neu 2/5, neg 1/5
P_POOL_GIVEN_Y = {
    1: {"pos": 0.0, "neu": 1/4, "neg": 3/4},
    0: {"pos": 2/5, "neu": 2/5, "neg": 1/5},
}


def binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def mutual_information(counts: np.ndarray) -> float:
    """I(X;Y) in bits from a joint count table (rows X, cols Y)."""
    n = counts.sum()
    pxy = counts / n
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = pxy * np.log2(pxy / (px * py))
    return float(np.nansum(term))


def main():
    df = pd.read_csv(CSV)
    y = df["y"].to_numpy()
    p_y1 = float(y.mean())
    Hy = binary_entropy(p_y1)
    print(f"Dataset: {len(df)} rows | P(y=1, dropout) = {p_y1:.4f} | H(y) = {Hy:.4f} bits")

    # ---- (1) exact I(text;y) = I(pool;y) from the known mixture ----
    pools = ["pos", "neu", "neg"]
    # joint P(pool, y)
    joint = np.array([[P_POOL_GIVEN_Y[0][p] * (1 - p_y1),
                       P_POOL_GIVEN_Y[1][p] * p_y1] for p in pools])  # rows=pool, cols=[y=0,y=1]
    I_theory = mutual_information(joint)
    # P(y=1 | pool) for the Bayes rule / interpretation
    p_y1_given_pool = joint[:, 1] / joint.sum(axis=1)
    print("\n(1) Exact information from the generative mixture:")
    for p, pj in zip(pools, p_y1_given_pool):
        print(f"      P(y=1 | pool={p:3s}) = {pj:.3f}")
    print(f"    I(text; y) = I(pool; y) = {I_theory:.4f} bits  "
          f"({100*I_theory/Hy:.1f}% of H(y))")

    # ---- (2) empirical cross-check from the committed comments ----
    codes, _ = pd.factorize(df["comment"])
    n_templates = codes.max() + 1
    emp = np.zeros((n_templates, 2))
    for c, yy in zip(codes, y):
        emp[c, yy] += 1
    I_emp = mutual_information(emp)
    print(f"\n(2) Empirical cross-check from student_comments.csv "
          f"({n_templates} distinct comments): I(comment; y) = {I_emp:.4f} bits")

    # ---- (3) text-only Bayes classifier on the Phase-9 split ----
    idx = np.arange(len(df))
    idx_tr, idx_te = train_test_split(idx, test_size=0.25, stratify=y, random_state=SEED)
    # Bayes-optimal map: predict dropout iff P(y=1 | comment) >= 0.5, estimated on TRAIN only
    tr = df.iloc[idx_tr]
    p1_by_comment = tr.groupby("comment")["y"].mean()
    default = int(tr["y"].mean() >= 0.5)
    te = df.iloc[idx_te]
    yhat = te["comment"].map(lambda c: int(p1_by_comment.get(c, default) >= 0.5)).to_numpy()
    yte = te["y"].to_numpy()
    f1 = f1_score(yte, yhat, zero_division=0)
    acc = accuracy_score(yte, yhat)
    print(f"\n(3) Text-ONLY Bayes classifier (fit on train comments, scored on the {len(te)}-row "
          f"Phase-9 test split):")
    print(f"      dropout-class F1 = {f1:.3f} | accuracy = {acc:.3f}")
    print("    For reference (committed single-seed numbers): tabular baseline best F1 ~ 0.79, "
          "multimodal best F1 ~ 0.85.")
    print("\nReading: the synthetic comment channel injects "
          f"{I_theory:.2f} bits ({100*I_theory/Hy:.0f}% of the label entropy) by construction, "
          f"enough for text alone to reach F1={f1:.2f} on dropout -- so the multimodal lift is a "
          "ceiling driven by injected signal, not evidence of a real-world text effect.")


if __name__ == "__main__":
    main()
