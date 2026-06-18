"""
Phase 2 - baseline supervised classifiers.

These three models set the bar that every later upgrade has to clear
*without* giving up fairness or labelling efficiency, so we keep them
deliberately ordinary:

  - Random Forest   (300 trees, balanced class weights)
  - XGBoost         (gradient-boosted trees, positive class up-weighted)
  - MLP             (small PyTorch net; the SAME architecture is reused in
                     Phases 3, 4, 8 and 9)

Reusing one MLP definition everywhere is what makes the cross-phase
comparison architecture-controlled: between phases only the training data
or the loss changes, never the network. So when a later number moves we can
attribute it to the method rather than to a quietly bigger model.

We rely on class weighting rather than resampling here because the dropout
class is only moderately rare (~32%) - weighting the loss is enough, and it
keeps the baseline clean for the generative-oversampling comparison that
Phases 3-4 layer on top.

Outputs
  results/phase2_baselines.json           per-model accuracy / precision / recall / F1 / AUC
  figures/phase2_baseline_comparison.png
  figures/phase2_confusion_matrices.png
  figures/phase2_roc_curves.png
  data/baseline_mlp.pt                     the trained MLP, reused downstream
  data/baseline_test_predictions.npz       test predictions, reused by the Phase 7 audit
"""
from __future__ import annotations
import numpy as np, matplotlib.pyplot as plt, seaborn as sns
import torch, torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve
from xgboost import XGBClassifier

from common import (
    load_processed, predictive_metrics, save_json, set_plot_style,
    RESULTS_DIR, FIGURES_DIR, DATA_DIR, RANDOM_SEED,
)


class MLP(nn.Module):
    """Small feed-forward classifier: n_features -> 128 -> 64 -> 1 logit.

    Intentionally plain and shared across phases (see the module docstring),
    so nothing but the data or the loss differs between experiments.
    """
    def __init__(self, n_features: int, hidden_sizes=(128, 64), dropout=0.20):
        super().__init__()
        layers, prev = [], n_features
        for size in hidden_sizes:
            layers += [nn.Linear(prev, size), nn.ReLU(), nn.Dropout(dropout)]
            prev = size
        layers += [nn.Linear(prev, 1)]      # single logit -> binary cross-entropy
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # squeeze the trailing unit dimension so the output is shape (batch,)
        return self.net(x).squeeze(-1)


def train_mlp(X, y, X_eval=None, y_eval=None,
              epochs=60, batch_size=128, lr=1e-3,
              class_weight: float | None = None, device="cpu", verbose=False):
    """Plain supervised MLP training, with optional positive-class weighting.

    `class_weight` is fed straight into BCE as `pos_weight`; passing the
    negative/positive ratio is how every caller compensates for the dropout
    class being the minority.
    """
    torch.manual_seed(RANDOM_SEED)
    model = MLP(X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    pos_weight = torch.tensor([class_weight]).float() if class_weight else None
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).float()
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, drop_last=False)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(x_batch)
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1:3d}  loss={epoch_loss/len(dataset):.4f}")
    return model


def mlp_predict(model, X):
    """Return (hard 0/1 predictions, positive-class probabilities) for X."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).float()).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))          # sigmoid back to probabilities
    return (probs >= 0.5).astype(int), probs


def main():
    set_plot_style()
    data = load_processed()
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    print(f"[Phase 2] Train: {X_train.shape}  Test: {X_test.shape}")

    results, predictions = {}, {}

    # Random Forest. Balanced class weights handle the imbalance internally.
    print("[Phase 2] Training Random Forest ...")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None,
        class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    pred_rf, score_rf = rf.predict(X_test), rf.predict_proba(X_test)[:, 1]
    results["RandomForest"] = predictive_metrics(y_test, pred_rf, score_rf)
    predictions["RandomForest"] = (pred_rf, score_rf)

    # XGBoost. We up-weight the positive (dropout) class by the neg/pos ratio
    # so the booster doesn't simply chase the majority class.
    print("[Phase 2] Training XGBoost ...")
    neg_pos_ratio = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    xgb = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
        scale_pos_weight=neg_pos_ratio, random_state=RANDOM_SEED, n_jobs=-1,
        verbosity=0,
    )
    xgb.fit(X_train, y_train)
    pred_xgb, score_xgb = xgb.predict(X_test), xgb.predict_proba(X_test)[:, 1]
    results["XGBoost"] = predictive_metrics(y_test, pred_xgb, score_xgb)
    predictions["XGBoost"] = (pred_xgb, score_xgb)

    # MLP. Same neg/pos ratio, this time as the BCE positive-class weight.
    print("[Phase 2] Training MLP ...")
    mlp = train_mlp(X_train, y_train, class_weight=float(neg_pos_ratio))
    pred_mlp, score_mlp = mlp_predict(mlp, X_test)
    results["MLP"] = predictive_metrics(y_test, pred_mlp, score_mlp)
    predictions["MLP"] = (pred_mlp, score_mlp)

    # Persist the MLP so downstream phases inherit the exact same architecture.
    torch.save(mlp.state_dict(), DATA_DIR / "baseline_mlp.pt")

    # Persist metrics and the raw test predictions (Phase 7 re-uses the latter).
    save_json(results, RESULTS_DIR / "phase2_baselines.json")
    np.savez_compressed(
        DATA_DIR / "baseline_test_predictions.npz",
        rf_pred=predictions["RandomForest"][0], rf_score=predictions["RandomForest"][1],
        xgb_pred=predictions["XGBoost"][0],     xgb_score=predictions["XGBoost"][1],
        mlp_pred=predictions["MLP"][0],         mlp_score=predictions["MLP"][1],
    )

    # Grouped bar chart: the five headline metrics, one cluster per metric.
    metric_names = ["accuracy", "precision", "recall", "f1", "auc_roc"]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(metric_names)); bar_width = 0.25
    palette = ["#3498db", "#e67e22", "#2ecc71"]
    for i, (name, model_scores) in enumerate(results.items()):
        values = [model_scores[m] for m in metric_names]
        ax.bar(x + i * bar_width, values, width=bar_width, label=name, color=palette[i])
        for xpos, v in zip(x + i * bar_width, values):
            ax.text(xpos, v + 0.005, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x + bar_width); ax.set_xticklabels([m.upper() for m in metric_names])
    ax.set_ylim(0, 1); ax.set_title("Phase 2 - Baseline Classifier Comparison")
    ax.legend(loc="lower right")
    fig.savefig(FIGURES_DIR / "phase2_baseline_comparison.png"); plt.close(fig)

    # Confusion matrices, one per model.
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (name, (preds, _)) in zip(axes, predictions.items()):
        cm = confusion_matrix(y_test, preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Not drop", "Drop"],
                    yticklabels=["Not drop", "Drop"], ax=ax)
        ax.set_title(name); ax.set_xlabel("Pred"); ax.set_ylabel("True")
    fig.suptitle("Phase 2 - Confusion Matrices", fontweight="bold")
    fig.savefig(FIGURES_DIR / "phase2_confusion_matrices.png"); plt.close(fig)

    # ROC curves on one axes for an at-a-glance ranking by AUC.
    fig, ax = plt.subplots(figsize=(5, 4))
    for (name, (_, score)), colour in zip(predictions.items(), palette):
        fpr, tpr, _ = roc_curve(y_test, score)
        ax.plot(fpr, tpr, label=f"{name} (AUC={results[name]['auc_roc']:.3f})",
                color=colour, lw=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)          # chance diagonal
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("Phase 2 - ROC Curves"); ax.legend(loc="lower right")
    fig.savefig(FIGURES_DIR / "phase2_roc_curves.png"); plt.close(fig)

    print("[Phase 2] Baseline results:")
    for name, scores in results.items():
        print(f"  {name:15s}  Acc={scores['accuracy']:.3f}  F1={scores['f1']:.3f}  "
              f"AUC={scores['auc_roc']:.3f}")


if __name__ == "__main__":
    main()
