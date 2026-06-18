"""
Phase 8 (Upgrade 3) - in-processing fairness penalty.

Phase 7 showed that resampling cannot remove the gender gap, so here we put
fairness *inside* the objective. We reuse the Phase-2 MLP and add a
differentiable demographic-parity surrogate to its loss:

    loss = BCE(y, y_hat)  +  lambda * | mean(sigmoid(z) | s=1) - mean(sigmoid(z) | s=0) |

i.e. the absolute gap in mean sigmoid output between the two groups, estimated
per mini-batch. It is a smooth, differentiable stand-in for the hard
demographic-parity difference. Sweeping lambda over {0, 0.1, 0.5, 1, 2, 5, 10}
traces out the empirical accuracy vs parity-gap Pareto frontier.

Two subtleties worth knowing:
  - Missing-group batches: if a mini-batch happens to contain only one group
    the surrogate is undefined, so we simply skip the penalty for that batch
    (ordinary shuffling plus a presence check, not a stratified sampler).
  - Surrogate vs. reported metric: the penalty uses the soft sigmoid in [0, 1],
    but the reported gap uses hard 0/1 predictions. As lambda grows the soft
    surrogate decays smoothly while the hard gap can rebound, so the frontier
    is read off the empirical hard-decision gap, not the surrogate.

The headline (over the 10-seed study, not this single run): switching the
penalty on robustly cuts the demographic-parity gap (~60% at lambda = 1,
p = 0.002) at ~1.6 accuracy points; the exact operating point is
seed-sensitive, with lambda ~ 2-5 tracing the mean Pareto frontier - a single
knob trading accuracy against fairness, to be chosen against a target
disparity ceiling rather than fixed at one value.

Outputs
  results/phase8_fairness_inprocess.json
  figures/phase8_pareto_frontier.png
  figures/phase8_fairness_vs_lambda.png
"""
from __future__ import annotations
import numpy as np, matplotlib.pyplot as plt
import torch, torch.nn as nn

from common import (load_processed, predictive_metrics, fairness_metrics,
                    save_json, set_plot_style,
                    RESULTS_DIR, FIGURES_DIR, RANDOM_SEED)
from phase2_baselines import MLP


def train_fair_mlp(X, y, s, lam: float, epochs=60, batch=128, lr=1e-3,
                   class_weight: float | None = None, device="cpu"):
    """Train the baseline MLP with BCE + lambda * |demographic-parity surrogate|.

    The penalty is computed on the soft sigmoid outputs, so it stays bounded in
    [0, 1] and a large lambda will not by itself destabilise training.
    """
    torch.manual_seed(RANDOM_SEED)
    model = MLP(X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    pos_weight = torch.tensor([class_weight]).float() if class_weight else None
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y.astype(np.float32))
    s_tensor = torch.from_numpy(s.astype(np.float32))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tensor, y_tensor, s_tensor),
        batch_size=batch, shuffle=True, drop_last=False,
    )
    history = []
    for epoch in range(epochs):
        model.train()
        running_loss = running_bce = running_penalty = 0.0
        for x_batch, y_batch, s_batch in loader:
            optimizer.zero_grad()
            logits = model(x_batch)
            bce_term = bce_loss(logits, y_batch)
            soft_preds = torch.sigmoid(logits)
            group1_mask = (s_batch == 1); group0_mask = (s_batch == 0)
            # Only apply the parity penalty when both groups are in this batch.
            if group1_mask.any() and group0_mask.any():
                dp_gap = torch.abs(soft_preds[group1_mask].mean() - soft_preds[group0_mask].mean())
            else:
                dp_gap = torch.tensor(0.0)
            loss = bce_term + lam * dp_gap
            loss.backward(); optimizer.step()
            running_loss += loss.item(); running_bce += bce_term.item(); running_penalty += dp_gap.item()
        history.append({"epoch": epoch + 1, "loss": running_loss, "bce": running_bce,
                        "penalty": running_penalty})
    return model, history


def evaluate(model, X_test, y_test, s_test):
    """Predictive + fairness metrics for a trained model on the test set."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test).float()).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    return {**predictive_metrics(y_test, preds, probs),
            **fairness_metrics(y_test, preds, s_test)}


def pareto_frontier(points):
    """Mask of Pareto-optimal points, maximising accuracy (col 0), minimising gap (col 1).

    A point i is dominated when some j is at least as accurate AND at least as
    fair, and strictly better on at least one of the two.
    """
    pts = np.array(points)
    mask = np.ones(len(pts), dtype=bool)
    for i in range(len(pts)):
        for j in range(len(pts)):
            if i == j: continue
            if pts[j, 0] >= pts[i, 0] and pts[j, 1] <= pts[i, 1] and (
               pts[j, 0] >  pts[i, 0] or  pts[j, 1] <  pts[i, 1]):
                mask[i] = False; break
    return mask


def main():
    set_plot_style()
    data = load_processed()
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    s_train, s_test = data["s_train"], data["s_test"]

    lambdas = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    table = {}
    neg_pos_ratio = (y_train == 0).sum() / max(1, (y_train == 1).sum())

    for lam in lambdas:
        print(f"[Phase 8] training fair-MLP with lambda = {lam}")
        model, _ = train_fair_mlp(X_train, y_train, s_train, lam=lam,
                                  class_weight=float(neg_pos_ratio))
        table[f"lambda={lam}"] = evaluate(model, X_test, y_test, s_test)
        row = table[f"lambda={lam}"]
        print(f"  ACC={row['accuracy']:.3f}  F1={row['f1']:.3f}  "
              f"DPD={row['demographic_parity_diff']:.3f}  "
              f"EOD={row['equal_opportunity_diff']:.3f}")

    save_json({"sweep": table, "lambdas": lambdas},
              RESULTS_DIR / "phase8_fairness_inprocess.json")

    # Pareto plot: accuracy against demographic-parity gap, one point per lambda.
    accuracies = [table[f"lambda={l}"]["accuracy"] for l in lambdas]
    dp_diffs = [table[f"lambda={l}"]["demographic_parity_diff"] for l in lambdas]
    mask = pareto_frontier(list(zip(accuracies, dp_diffs)))

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(dp_diffs, accuracies, "o--", color="#7f8c8d", alpha=0.6)
    for i, (l, acc, dp) in enumerate(zip(lambdas, accuracies, dp_diffs)):
        colour = "#c0392b" if mask[i] else "#3498db"
        ax.scatter(dp, acc, s=90, color=colour, edgecolor="black", zorder=5)
        ax.annotate(f"λ={l}", (dp, acc),
                    xytext=(6, 4), textcoords="offset points", fontsize=8)
    # Connect the Pareto-optimal points into a frontier line.
    frontier = sorted([(dp, acc) for dp, acc, is_opt in zip(dp_diffs, accuracies, mask) if is_opt])
    if len(frontier) > 1:
        frontier_x, frontier_y = zip(*frontier)
        ax.plot(frontier_x, frontier_y, "-", color="#c0392b", lw=2, label="Pareto frontier")
    ax.set_xlabel("Demographic-Parity Difference (lower = fairer)")
    ax.set_ylabel("Accuracy (higher = better)")
    ax.set_title("Phase 8 - Accuracy / Fairness Pareto Frontier")
    ax.legend(loc="lower right")
    fig.savefig(FIGURES_DIR / "phase8_pareto_frontier.png"); plt.close(fig)

    # How each metric responds as the fairness weight lambda increases.
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(lambdas, [table[f"lambda={l}"]["accuracy"] for l in lambdas],
            "o-", color="#27ae60", label="Accuracy")
    ax.plot(lambdas, [table[f"lambda={l}"]["f1"] for l in lambdas],
            "s-", color="#2980b9", label="F1")
    ax.plot(lambdas, [table[f"lambda={l}"]["demographic_parity_diff"] for l in lambdas],
            "^-", color="#c0392b", label="Demographic-Parity Diff")
    ax.plot(lambdas, [table[f"lambda={l}"]["equal_opportunity_diff"] for l in lambdas],
            "v-", color="#e67e22", label="Equal-Opportunity Diff")
    ax.set_xlabel("λ (fairness regulariser)"); ax.set_ylabel("metric")
    ax.set_title("Phase 8 - Metrics vs Fairness Strength λ")
    ax.legend(fontsize=8)
    fig.savefig(FIGURES_DIR / "phase8_fairness_vs_lambda.png"); plt.close(fig)
    print("[Phase 8] done")


if __name__ == "__main__":
    main()
