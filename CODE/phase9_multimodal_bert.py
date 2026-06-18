"""
Phase 9 (Upgrade 4) - multimodal extension with DistilBERT.

The point here is not the headline metric but the *protocol*: how to fuse a
text modality into the tabular pipeline without leaking test information.

Since the dataset has no free text, we simulate a short feedback comment per
student, drawn from tone-templated pools (positive / neutral / negative). The
tone is correlated with the outcome (dropouts skew negative) but deliberately
noisy, and the choice is driven by a deterministic MD5 hash of the row index -
so it is reproducible and a student's comment never depends on any other
student's label.

Fusion pipeline:
  1. Generate the comment for every row.
  2. Reuse the exact Phase-1 train/test split (same indices).
  3. Encode comments with a frozen, pre-trained DistilBERT, mean-pooled over
     tokens -> 768-D.
  4. Reduce to 32-D with PCA fit on the TRAINING embeddings only.
  5. Concatenate the 32-D text vector with the 247-D tabular vector -> 279-D,
     and retrain RF / XGB / MLP.

The pitfall this guards against is data leakage through the encoder. Three
fixes keep it honest: never fine-tune (the encoder is frozen), encode train and
test separately, and fit the PCA + scaler on training embeddings only.

Important caveat to state up front: because the text is simulated and
label-correlated by construction, the multimodal lift here is a *ceiling*, not
a real-world effect. The contribution is the leakage-safe fusion protocol, not
the size of the gain.

Outputs
  results/phase9_multimodal_bert.json
  figures/phase9_multimodal_comparison.png
  figures/phase9_text_pca_scatter.png
  data/student_comments.csv
"""
from __future__ import annotations
import json, hashlib, numpy as np, pandas as pd, matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel
from xgboost import XGBClassifier

from common import (load_processed, load_raw, make_binary_target,
                    predictive_metrics, save_json, set_plot_style,
                    RESULTS_DIR, FIGURES_DIR, DATA_DIR, RANDOM_SEED)
from phase2_baselines import train_mlp, mlp_predict


# Tone-templated comment banks. Tone correlates with outcome but stays noisy,
# so the text is suggestive of dropout risk rather than a giveaway label.
POSITIVE_COMMENTS = [
    "I really enjoy my studies and feel motivated every day.",
    "The lecturers are inspiring and I love the curriculum.",
    "I have made many friends and feel part of the community here.",
    "Studying is going great and I plan to continue next semester.",
    "Honestly, this is the best academic experience I have had.",
]
NEUTRAL_COMMENTS = [
    "Classes are okay, nothing special but not bad either.",
    "Some courses are interesting, others are quite boring.",
    "I am keeping up with the assignments most of the time.",
    "The workload is manageable so far.",
    "I attend lectures regularly but I am not sure what comes next.",
]
NEGATIVE_COMMENTS = [
    "I am really stressed and thinking about quitting the program.",
    "The lectures are confusing and I cannot keep up anymore.",
    "I feel anxious about my grades and don't know if I belong here.",
    "I am behind on coursework and considering dropping out.",
    "Honestly, I feel exhausted and lost in my studies.",
]


def make_comment(row_index: int, y: int) -> str:
    """Pick a deterministic comment for one student given (row index, outcome).

    A salted hash of the row index seeds the RNG, so the same row always yields
    the same comment - reproducible, and never a function of another row's
    label. Dropouts draw a mostly-negative mix; everyone else a positive-leaning
    one, with enough overlap that the text is not a giveaway.
    """
    seed_hash = int(hashlib.md5(f"{row_index}".encode()).hexdigest(), 16)
    rng = np.random.default_rng(seed_hash % (2**32))
    if y == 1:                                              # dropout
        bucket = rng.choice(["neg", "neg", "neg", "neu"])
    else:                                                   # graduate / enrolled
        bucket = rng.choice(["pos", "pos", "neu", "neu", "neg"])
    pool = {"pos": POSITIVE_COMMENTS, "neu": NEUTRAL_COMMENTS,
            "neg": NEGATIVE_COMMENTS}[bucket]
    return rng.choice(pool)


@torch.no_grad()
def encode_comments(comments: list[str], batch: int = 32) -> np.ndarray:
    """Mean-pooled DistilBERT embeddings for a list of comments (frozen encoder)."""
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    encoder = AutoModel.from_pretrained("distilbert-base-uncased").eval()
    embeddings = []
    for i in range(0, len(comments), batch):
        tokens = tokenizer(comments[i:i + batch], padding=True, truncation=True,
                           max_length=48, return_tensors="pt")
        hidden_states = encoder(**tokens).last_hidden_state          # (B, T, 768)
        # Mean-pool over real tokens only - mask out the padding positions.
        attn_mask = tokens["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden_states * attn_mask).sum(dim=1) / attn_mask.sum(dim=1).clamp(min=1)
        embeddings.append(pooled.numpy())
        if (i // batch) % 20 == 0:
            print(f"  encoded {i + batch}/{len(comments)}")
    return np.concatenate(embeddings, axis=0).astype(np.float32)


def main():
    set_plot_style()
    data = load_processed()
    X_tab_train, X_tab_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    # Recover which original rows are train vs test by replaying the Phase-1
    # split with identical arguments and seed.
    df = make_binary_target(load_raw())
    from sklearn.model_selection import train_test_split
    idx_train, idx_test, _, _ = train_test_split(
        np.arange(len(df)), df["y"].values, test_size=0.25,
        stratify=df["y"].values, random_state=RANDOM_SEED,
    )

    # Generate a comment for every student and save them as evidence.
    print("[Phase 9] generating synthetic comments ...")
    all_comments = [make_comment(i, int(df["y"].iloc[i])) for i in range(len(df))]
    pd.DataFrame({"row_idx": np.arange(len(df)),
                  "y": df["y"].values,
                  "comment": all_comments}).to_csv(
        DATA_DIR / "student_comments.csv", index=False)

    comments_train = [all_comments[i] for i in idx_train]
    comments_test = [all_comments[i] for i in idx_test]

    # Encode train and test separately with the frozen encoder -> no leakage.
    print("[Phase 9] encoding train comments with DistilBERT ...")
    emb_train = encode_comments(comments_train)
    print("[Phase 9] encoding test comments with DistilBERT ...")
    emb_test = encode_comments(comments_test)

    # PCA fit on TRAIN embeddings only, then applied unchanged to test.
    pca = PCA(n_components=32, random_state=RANDOM_SEED).fit(emb_train)
    txt_train = pca.transform(emb_train).astype(np.float32)
    txt_test = pca.transform(emb_test).astype(np.float32)

    # Standardise the text features (again, fit on train) before fusing.
    txt_scaler = StandardScaler().fit(txt_train)
    txt_train = txt_scaler.transform(txt_train).astype(np.float32)
    txt_test = txt_scaler.transform(txt_test).astype(np.float32)

    X_mm_train = np.concatenate([X_tab_train, txt_train], axis=1)
    X_mm_test = np.concatenate([X_tab_test, txt_test], axis=1)
    print(f"[Phase 9] multimodal shapes: tr={X_mm_train.shape}  te={X_mm_test.shape}")

    # Retrain the three classifiers on the fused tabular + text features.
    print("[Phase 9] training classifiers on multimodal features ...")
    neg_pos_ratio = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    results = {}

    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_mm_train, y_train)
    pred, score = rf.predict(X_mm_test), rf.predict_proba(X_mm_test)[:, 1]
    results["RandomForest"] = predictive_metrics(y_test, pred, score)

    xgb = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.08,
                        subsample=0.9, colsample_bytree=0.9,
                        eval_metric="logloss", scale_pos_weight=neg_pos_ratio,
                        random_state=RANDOM_SEED, n_jobs=-1, verbosity=0)
    xgb.fit(X_mm_train, y_train)
    pred, score = xgb.predict(X_mm_test), xgb.predict_proba(X_mm_test)[:, 1]
    results["XGBoost"] = predictive_metrics(y_test, pred, score)

    mlp = train_mlp(X_mm_train, y_train, class_weight=float(neg_pos_ratio))
    pred, score = mlp_predict(mlp, X_mm_test)
    results["MLP"] = predictive_metrics(y_test, pred, score)

    for name, scores in results.items():
        print(f"  {name:15s}  Acc={scores['accuracy']:.3f}  F1={scores['f1']:.3f}  "
              f"AUC={scores['auc_roc']:.3f}")

    # Compare against the tabular-only Phase-2 baseline.
    with open(RESULTS_DIR / "phase2_baselines.json") as f:
        base = json.load(f)
    save_json({"multimodal": results, "tabular_only_baseline": base},
              RESULTS_DIR / "phase9_multimodal_bert.json")

    # Tabular-only vs +BERT, per model: faint bar = tabular, solid bar = fused.
    metric_names = ["accuracy", "f1", "auc_roc"]
    x = np.arange(len(metric_names)); bar_width = 0.13
    colors = {"RandomForest": "#3498db", "XGBoost": "#e67e22", "MLP": "#2ecc71"}
    fig, ax = plt.subplots(figsize=(8, 4))
    for j, name in enumerate(["RandomForest", "XGBoost", "MLP"]):
        base_vals = [base[name][m] for m in metric_names]
        mm_vals = [results[name][m] for m in metric_names]
        ax.bar(x + j * 3 * bar_width - 4 * bar_width, base_vals, width=bar_width,
               color=colors[name], alpha=0.45, label=f"{name} tabular-only")
        ax.bar(x + j * 3 * bar_width - 3 * bar_width, mm_vals, width=bar_width,
               color=colors[name], label=f"{name} +BERT")
    ax.set_xticks(x); ax.set_xticklabels([m.upper() for m in metric_names])
    ax.set_ylim(0, 1); ax.set_title("Phase 9 - Tabular-only vs Tabular + DistilBERT")
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    fig.savefig(FIGURES_DIR / "phase9_multimodal_comparison.png")
    plt.close(fig)

    # 2-D PCA of the train embeddings, coloured by outcome - a sanity check that
    # the simulated text separates dropouts from the rest at all.
    pca_2d = PCA(n_components=2).fit(emb_train)
    proj = pca_2d.transform(emb_train)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for cls, colour in zip([0, 1], ["#3498db", "#e74c3c"]):
        mask = (y_train == cls)
        ax.scatter(proj[mask, 0], proj[mask, 1], s=8, alpha=0.6,
                   color=colour, label=("Dropout" if cls == 1 else "Other"))
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("Phase 9 - DistilBERT embedding 2-D projection (train comments)")
    ax.legend()
    fig.savefig(FIGURES_DIR / "phase9_text_pca_scatter.png"); plt.close(fig)
    print("[Phase 9] done")


if __name__ == "__main__":
    main()
