"""
Phase 3 - VAE oversampling of the minority class (the original proposal).

Instead of duplicating dropout rows (naive oversampling), we train a small
variational autoencoder on the minority class ALONE and sample fresh
synthetic dropouts from its prior. Those samples top the training set up to
roughly 50/50, the three baseline classifiers are retrained, and we compare
back to Phase 2.

The catch is the feature layout. Each encoded row is a mix of:
  - `n_numerical` standardised numerical features, then
  - 18 one-hot categorical blocks (their cardinalities live in the metadata).

A single Gaussian-output decoder regresses every one-hot block toward its
mean and emits values like 0.7 where the data is strictly 0 or 1 - so the
synthetic "categories" never actually occur, and the tree models choke on
them. This is categorical mode-collapse, and it motivates Upgrade 1
(Phase 4). We work around it here with a *hybrid decoder*:
  - a Gaussian (MSE) head over the numerical block, and
  - one softmax (cross-entropy) head per categorical block.
At sampling time each categorical block is realised as a hard one-hot by
Gumbel-Softmax sampling of the decoder's logits - a tempered draw from the
per-block categorical distribution. (Sampling runs under no_grad; no gradient
ever flows through it. Training itself uses the per-block cross-entropy, and
it is this hybrid likelihood that fixes the mode collapse.) A linear KL
warm-up over the first 40 epochs keeps the latent from collapsing early in
training (posterior collapse).

A note on what to expect: because the minority class is already ~32% of the
data, synthetic oversampling barely moves the predictive metrics. That is the
honest result - the real contribution of this phase is the decoder fix that
makes a VAE usable on categorical tabular data at all, and it carries straight
over to the diffusion model in Phase 4.

Outputs
  results/phase3_vae.json
  figures/phase3_vae_training_curve.png
  figures/phase3_vae_real_vs_synth_distributions.png
  figures/phase3_metrics_vs_baseline.png
  data/vae_synth.npz
"""
from __future__ import annotations
import json, numpy as np, matplotlib.pyplot as plt, seaborn as sns
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from common import (
    load_processed, predictive_metrics, save_json, set_plot_style,
    RESULTS_DIR, FIGURES_DIR, DATA_DIR, RANDOM_SEED,
)
from phase2_baselines import train_mlp, mlp_predict


class TabVAE(nn.Module):
    """Tabular VAE with a hybrid decoder (Gaussian numericals + softmax one-hots).

    The encoder/decoder share a plain MLP trunk; the decoder then splits into
    one Gaussian head for the numerical block and one softmax head per
    categorical column, which is what lets the model emit valid one-hots
    rather than blurred fractional categories.
    """
    def __init__(self, n_numerical: int, cat_cardinalities: list[int],
                 latent_dim: int = 16, hidden_dim: int = 128):
        super().__init__()
        self.n_numerical = n_numerical
        self.cat_cardinalities = cat_cardinalities
        self.input_dim = n_numerical + sum(cat_cardinalities)
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),     nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder_trunk = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.decoder_numerical = nn.Linear(hidden_dim, n_numerical)
        # One head per categorical column, each producing logits over its categories.
        self.decoder_categorical = nn.ModuleList(
            [nn.Linear(hidden_dim, card) for card in cat_cardinalities])

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterise(self, mu, logvar):
        # Sample z = mu + sigma * eps so the draw stays differentiable w.r.t. mu/logvar.
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        trunk = self.decoder_trunk(z)
        numerical_out = self.decoder_numerical(trunk)
        cat_logits = [head(trunk) for head in self.decoder_categorical]
        return numerical_out, cat_logits

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        numerical_out, cat_logits = self.decode(z)
        return numerical_out, cat_logits, mu, logvar


def vae_loss(numerical_pred, cat_logits, x, n_numerical, cat_cardinalities,
             mu, logvar, beta: float = 1.0):
    """ELBO with a hybrid likelihood: Gaussian on numericals, categorical on one-hots.

    Reconstruction = MSE over the numerical block + summed cross-entropy over
    each categorical block (each block's target is the index of its hot
    column). The KL term is weighted by `beta` so the caller can warm it up.
    """
    rec_num = F.mse_loss(numerical_pred, x[:, :n_numerical], reduction="sum")
    rec_cat, offset = 0.0, n_numerical
    for card, block_logits in zip(cat_cardinalities, cat_logits):
        target = x[:, offset:offset + card].argmax(dim=1)
        rec_cat = rec_cat + F.cross_entropy(block_logits, target, reduction="sum")
        offset += card
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return rec_num + rec_cat + beta * kl, rec_num.item(), rec_cat, kl.item()


def gumbel_softmax_sample(logits, tau: float = 0.5, hard: bool = True):
    """Draw a hard one-hot category from `logits` (Gumbel-max sampling).

    With hard=True this returns a valid one-hot drawn from the tempered
    softmax distribution over the block's categories. We only call it inside
    no_grad sampling, so it is purely a *sampling* device here - the
    differentiable straight-through path exists but is unused; training relies
    on the per-block cross-entropy instead.
    """
    return F.gumbel_softmax(logits, tau=tau, hard=hard)


def train_vae(X, n_numerical, cat_cardinalities, epochs: int = 200,
              batch: int = 64, lr: float = 1e-3,
              beta_max: float = 1.0, kl_warmup: int = 40, verbose=False):
    """Train the VAE on `X`, ramping the KL weight to avoid posterior collapse.

    On a small minority set the KL term can swamp reconstruction early and push
    the decoder to ignore the latent entirely, so `beta` rises linearly from 0
    to `beta_max` over the first `kl_warmup` epochs.
    """
    torch.manual_seed(RANDOM_SEED)
    model = TabVAE(n_numerical, cat_cardinalities)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_tensor = torch.from_numpy(X).float()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tensor),
        batch_size=batch, shuffle=True, drop_last=True,
    )

    history = []
    for epoch in range(epochs):
        beta = beta_max * min(1.0, epoch / max(1, kl_warmup))
        model.train()
        total = rec_num_sum = rec_cat_sum = kl_sum = 0
        for (x_batch,) in loader:
            optimizer.zero_grad()
            numerical_out, cat_logits, mu, logvar = model(x_batch)
            loss, rec_num, rec_cat, kl_val = vae_loss(
                numerical_out, cat_logits, x_batch, n_numerical,
                cat_cardinalities, mu, logvar, beta)
            loss.backward()
            optimizer.step()
            total += loss.item()
            rec_num_sum += rec_num
            rec_cat_sum += rec_cat
            kl_sum += kl_val
        history.append(dict(epoch=epoch + 1, total=total, rec_num=rec_num_sum,
                            rec_cat=rec_cat_sum, kl=kl_sum, beta=beta))
        if verbose and (epoch + 1) % 20 == 0:
            print(f"  ep {epoch+1:3d}  total={total:.0f}  KL={kl_sum:.0f}  beta={beta:.2f}")
    return model, history


@torch.no_grad()
def vae_sample(model: TabVAE, n_samples: int, cat_cardinalities, n_numerical):
    """Sample `n_samples` synthetic rows in the same encoded layout as X_train.

    Draw z from the standard-normal prior, decode, and snap each categorical
    block to a hard one-hot so the result is interchangeable with real rows.
    """
    z = torch.randn(n_samples, model.latent_dim)
    numerical_out, cat_logits = model.decode(z)
    parts = [numerical_out.numpy()]
    for block_logits in cat_logits:
        one_hot = gumbel_softmax_sample(block_logits, tau=0.5, hard=True).numpy()
        parts.append(one_hot)
    return np.concatenate(parts, axis=1).astype(np.float32)


def evaluate_augmented(X_aug, y_aug, X_test, y_test):
    """Train RF / XGB / MLP on an augmented training set; return their metrics."""
    metrics = {}
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_aug, y_aug)
    metrics["RandomForest"] = predictive_metrics(
        y_test, rf.predict(X_test), rf.predict_proba(X_test)[:, 1])

    neg_pos_ratio = (y_aug == 0).sum() / max(1, (y_aug == 1).sum())
    xgb = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9,
        eval_metric="logloss", scale_pos_weight=neg_pos_ratio,
        random_state=RANDOM_SEED, n_jobs=-1, verbosity=0,
    )
    xgb.fit(X_aug, y_aug)
    metrics["XGBoost"] = predictive_metrics(
        y_test, xgb.predict(X_test), xgb.predict_proba(X_test)[:, 1])

    mlp = train_mlp(X_aug, y_aug, class_weight=float(neg_pos_ratio))
    pred, score = mlp_predict(mlp, X_test)
    metrics["MLP"] = predictive_metrics(y_test, pred, score)
    return metrics


def main():
    set_plot_style()
    data = load_processed()
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    n_numerical = data["num_numerical"]
    cat_cardinalities = data["cat_card"]

    # Train the VAE on dropout rows only - we want it to model the minority.
    X_minority = X_train[y_train == 1]
    print(f"[Phase 3] Minority training set: {X_minority.shape}")
    vae, history = train_vae(X_minority, n_numerical, cat_cardinalities,
                             epochs=200, batch=64, verbose=False)
    print(f"[Phase 3] VAE final loss={history[-1]['total']:.1f}  "
          f"KL={history[-1]['kl']:.1f}")

    # Sample enough synthetic dropouts to roughly balance the two classes.
    n_majority = int((y_train == 0).sum())
    n_minority = int((y_train == 1).sum())
    n_synth = n_majority - n_minority
    X_synth = vae_sample(vae, n_synth, cat_cardinalities, n_numerical)
    y_synth = np.ones(n_synth, dtype=np.int64)
    np.savez_compressed(DATA_DIR / "vae_synth.npz", X=X_synth, y=y_synth)

    X_aug = np.concatenate([X_train, X_synth], axis=0)
    y_aug = np.concatenate([y_train, y_synth], axis=0)
    print(f"[Phase 3] Augmented training set: {X_aug.shape} "
          f"(added {n_synth} synthetic dropouts)")

    # Retrain the same three classifiers on the augmented data.
    print("[Phase 3] Training classifiers on VAE-augmented data ...")
    results = evaluate_augmented(X_aug, y_aug, X_test, y_test)
    for name, scores in results.items():
        print(f"  {name:15s}  Acc={scores['accuracy']:.3f}  F1={scores['f1']:.3f}  "
              f"AUC={scores['auc_roc']:.3f}")

    # Difference vs the Phase-2 baseline, metric by metric.
    with open(RESULTS_DIR / "phase2_baselines.json") as f:
        base = json.load(f)
    delta = {model: {metric: results[model][metric] - base[model][metric]
                     for metric in results[model]}
             for model in results}

    save_json({"vae_augmented": results,
               "baseline_phase2": base,
               "delta": delta,
               "n_synth_added": int(n_synth),
               "vae_final_loss": history[-1]["total"]},
              RESULTS_DIR / "phase3_vae.json")

    # Training curves: total loss (log scale) and the warmed-up KL term.
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(epochs, [h["total"] for h in history], color="#34495e")
    ax[0].set_yscale("log"); ax[0].set_title("VAE Total Loss")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss (log)")
    ax[1].plot(epochs, [h["kl"] for h in history], color="#c0392b")
    ax[1].set_title("VAE KL term (with warm-up)"); ax[1].set_xlabel("epoch")
    fig.savefig(FIGURES_DIR / "phase3_vae_training_curve.png"); plt.close(fig)

    # Real vs synthetic densities for the first four numerical features -
    # a quick eyeball check that the synthetic minority sits where the real one does.
    n_plot = min(4, n_numerical)
    feature_names = data["feature_names"][:n_plot]
    fig, axes = plt.subplots(1, n_plot, figsize=(3.25 * n_plot, 3), squeeze=False)
    axes = axes[0]
    for i, ax in enumerate(axes):
        sns.kdeplot(X_minority[:, i], ax=ax, label="real (dropouts)",
                    fill=True, alpha=0.4, color="#3498db")
        sns.kdeplot(X_synth[:, i], ax=ax, label="VAE synth",
                    fill=True, alpha=0.4, color="#e67e22")
        ax.set_title(feature_names[i], fontsize=9); ax.set_xlabel("")
    axes[0].legend(fontsize=8)
    fig.suptitle("Phase 3 - Real vs VAE Synthetic Distributions (numericals)",
                 fontweight="bold")
    fig.savefig(FIGURES_DIR / "phase3_vae_real_vs_synth_distributions.png")
    plt.close(fig)

    # Baseline vs augmented, per model: faint bar = baseline, solid bar = +VAE.
    metric_names = ["accuracy", "f1", "auc_roc"]
    x = np.arange(len(metric_names)); bar_width = 0.27
    palette = ["#3498db", "#e67e22", "#2ecc71"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, name in enumerate(results):
        base_vals = [base[name][m] for m in metric_names]
        aug_vals = [results[name][m] for m in metric_names]
        ax.bar(x + i * bar_width - bar_width, base_vals, width=bar_width * 0.45,
               color=palette[i], alpha=0.45,
               label=f"{name} baseline" if i == 0 else None)
        ax.bar(x + i * bar_width - bar_width * 0.55, aug_vals, width=bar_width * 0.45,
               color=palette[i], label=f"{name} +VAE")
    ax.set_xticks(x); ax.set_xticklabels([m.upper() for m in metric_names])
    ax.set_ylim(0, 1); ax.set_title("Phase 3 - Baseline vs VAE-Augmented")
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    fig.savefig(FIGURES_DIR / "phase3_metrics_vs_baseline.png"); plt.close(fig)
    print("[Phase 3] done")


if __name__ == "__main__":
    main()
