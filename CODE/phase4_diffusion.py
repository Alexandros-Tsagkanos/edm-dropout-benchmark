"""
Phase 4 (Upgrade 1) - TabDDPM-style diffusion for minority oversampling.

This swaps the Phase-3 VAE for a small diffusion model. The motivation is
distributional fidelity: a VAE tends to smear the feature spread, whereas a
diffusion model reconstructs it more faithfully, which is what we want from a
synthetic minority that downstream classifiers will train on.

How it works:
  - Forward process: a linear beta schedule over T = 200 steps, so any noised
    sample has the closed form x_t = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*eps.
  - Denoiser: a small residual MLP that predicts the noise eps from (x_t, t),
    with the timestep fed in through an additive sinusoidal embedding. It is
    trained with the standard DDPM eps-prediction MSE loss (Ho et al. 2020).
  - Sampling: ancestral denoising from t=T-1 down to 0, then each categorical
    block is snapped to a valid one-hot by per-block argmax. The denoised
    coordinates approximate a one-hot, they are NOT logits: pushing them
    through a Gumbel-Softmax (as an earlier revision did) draws categories
    almost uniformly on high-cardinality blocks, because at tau=0.5 the hot
    coordinate only carries weight e^2 against (K-1) others. Argmax keeps the
    dominant category deterministically - the same projection the SMOTE
    baseline applies. (The VAE is different: its decoder emits true logits,
    so sampling them with Gumbel-Softmax is a legitimate categorical draw.)

We score distributional fidelity directly on both blocks: the mean absolute
gap in per-feature mean and std for the numerical block, and the mean
per-block total-variation distance between real and synthetic categorical
marginals (the diagnostic that exposes a broken categorical sampler). On the
numericals the diffusion model roughly halves the VAE's std-mismatch at the
cost of a comparably worse mean error - a familiar signature of an
under-trained reverse process. Downstream predictive metrics stay on par with
the VAE, because, as in Phase 3, a 32%-minority dataset has little to gain
from more synthetic minorities. The transferable result is the architecture.

Outputs
  results/phase4_diffusion.json
  figures/phase4_diffusion_training_curve.png
  figures/phase4_diffusion_real_vs_synth.png
  figures/phase4_vae_vs_diffusion.png
  data/diffusion_synth.npz
"""
from __future__ import annotations
import json, math, numpy as np, matplotlib.pyplot as plt, seaborn as sns
import torch, torch.nn as nn, torch.nn.functional as F

from common import (
    load_processed, save_json, set_plot_style,
    RESULTS_DIR, FIGURES_DIR, DATA_DIR, RANDOM_SEED,
)
from phase3_vae import evaluate_augmented


def make_schedule(T: int = 200, beta_start: float = 1e-4, beta_end: float = 0.02):
    """Precompute the linear beta schedule and its cumulative-product buffers.

    alpha_bar[t] is the fraction of the original signal that survives to step t,
    which is what makes the closed-form forward noising in q_sample possible.
    """
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return dict(betas=betas, alphas=alphas, alpha_bar=alpha_bar)


def sinusoidal_emb(t: torch.Tensor, dim: int = 64):
    """Transformer-style sinusoidal embedding of the integer timestep t.

    Gives the network a smooth, continuous representation of t rather than a
    bare integer, which stabilises training across the 200 noise levels.
    """
    half_dim = dim // 2
    freqs = torch.exp(-math.log(10_000) * torch.arange(half_dim) / half_dim).to(t.device)
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class Denoiser(nn.Module):
    """The noise predictor eps_theta(x_t, t): a residual MLP with two blocks."""
    def __init__(self, input_dim: int, time_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.time_dim = time_dim
        self.time_proj = nn.Sequential(nn.Linear(time_dim, hidden_dim), nn.SiLU())
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.out = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, t):
        # Inject the timestep additively, then refine through two residual blocks.
        emb = self.time_proj(sinusoidal_emb(t, self.time_dim))
        h = self.input_proj(x) + emb
        h = h + self.block1(h)
        h = h + self.block2(h)
        return self.out(h)


def q_sample(x0, t, alpha_bar, noise=None):
    """Forward noising in closed form: x_t = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*eps."""
    if noise is None:
        noise = torch.randn_like(x0)
    abar_t = alpha_bar[t].unsqueeze(1)
    return torch.sqrt(abar_t) * x0 + torch.sqrt(1.0 - abar_t) * noise, noise


def train_diffusion(X, T: int = 200, epochs: int = 200, batch: int = 64,
                    lr: float = 2e-3, verbose=False):
    """Train the denoiser to predict the noise added at a random timestep."""
    torch.manual_seed(RANDOM_SEED)
    sched = make_schedule(T)
    model = Denoiser(X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_tensor = torch.from_numpy(X).float()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tensor),
        batch_size=batch, shuffle=True, drop_last=True,
    )
    history = []
    for epoch in range(epochs):
        model.train(); total = 0
        for (x_batch,) in loader:
            # Sample a random timestep per row, noise to it, predict that noise.
            t = torch.randint(0, T, (x_batch.size(0),))
            x_noisy, noise = q_sample(x_batch, t, sched["alpha_bar"])
            noise_pred = model(x_noisy, t)
            loss = F.mse_loss(noise_pred, noise)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item() * x_batch.size(0)
        history.append({"epoch": epoch + 1, "mse": total / len(X_tensor)})
        if verbose and (epoch + 1) % 25 == 0:
            print(f"  ep {epoch+1:3d}  mse={history[-1]['mse']:.4f}")
    return model, sched, history


@torch.no_grad()
def diffusion_sample(model, sched, n_samples, input_dim,
                     n_numerical: int, cat_cardinalities: list[int]):
    """Generate synthetic rows by reverse (ancestral) sampling, then fix one-hots.

    Start from pure noise and denoise step by step down to x_0; once we have
    x_0, each categorical block is snapped to a valid one-hot by argmax (the
    denoised block approximates a one-hot, not logits - see the module
    docstring), so the result matches the encoded training layout.
    """
    betas = sched["betas"]; alphas = sched["alphas"]; alpha_bar = sched["alpha_bar"]
    T = len(betas)
    x = torch.randn(n_samples, input_dim)
    for t in reversed(range(T)):
        t_batch = torch.full((n_samples,), t, dtype=torch.long)
        noise_pred = model(x, t_batch)
        alpha_t = alphas[t]; alpha_bar_t = alpha_bar[t]
        coef = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
        mean = (1.0 / torch.sqrt(alpha_t)) * (x - coef * noise_pred)
        if t > 0:
            noise = torch.randn_like(x); sigma = torch.sqrt(betas[t])
            x = mean + sigma * noise
        else:
            x = mean                                  # last step is deterministic

    # Guard against a diverged reverse process - it can happen on feature spaces
    # with very few numerical columns (e.g. mostly one-hot data), where the
    # continuous diffusion is poorly conditioned. Keep the output finite and in a
    # sane standardised range so downstream models never see inf/NaN. On
    # well-behaved data (numericals already within a few sigma) this is a no-op.
    x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
    x[:, :n_numerical] = torch.clamp(x[:, :n_numerical], -10.0, 10.0)

    # Snap each categorical block to a hard one-hot via per-block argmax.
    # The denoised coordinates are approximate one-hots, NOT logits, so a
    # stochastic Gumbel-Softmax here would near-uniformly randomise
    # high-cardinality blocks; the deterministic argmax keeps the dominant
    # category (same projection as the SMOTE baseline's snap_categoricals).
    parts = [x[:, :n_numerical]]
    offset = n_numerical
    for card in cat_cardinalities:
        block = x[:, offset:offset + card]
        one_hot = torch.zeros_like(block)
        one_hot[torch.arange(block.size(0)), block.argmax(dim=1)] = 1.0
        parts.append(one_hot); offset += card
    return torch.cat(parts, dim=1).numpy().astype(np.float32)


def utility_dist_score(X_real, X_synth, n_numerical: int):
    """Lightweight distributional-fidelity proxy on the numerical block.

    Mean absolute gap between real and synthetic per-column mean and std.
    Lower means the synthetic numericals match the real distribution better.
    """
    real_mean, real_std = X_real[:, :n_numerical].mean(0), X_real[:, :n_numerical].std(0)
    synth_mean, synth_std = X_synth[:, :n_numerical].mean(0), X_synth[:, :n_numerical].std(0)
    return {
        "mean_abs_diff_means": float(np.mean(np.abs(real_mean - synth_mean))),
        "mean_abs_diff_stds":  float(np.mean(np.abs(real_std - synth_std))),
    }


def categorical_tv_score(X_real, X_synth, n_numerical: int, cat_cardinalities):
    """Mean per-block total-variation distance between categorical marginals.

    For each one-hot block, TV = 0.5 * sum_k |p_k - q_k| between the real and
    synthetic category frequencies; we report the mean over blocks. 0 means
    identical marginals, 1 means disjoint support - a broken categorical
    sampler (e.g. near-uniform draws) shows up here even when the numerical
    moments look fine.
    """
    tvs, offset = [], n_numerical
    for card in cat_cardinalities:
        p = X_real[:, offset:offset + card].mean(0)
        q = X_synth[:, offset:offset + card].mean(0)
        tvs.append(0.5 * float(np.abs(p - q).sum()))
        offset += card
    return float(np.mean(tvs))


def main():
    set_plot_style()
    data = load_processed()
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    n_numerical = data["num_numerical"]
    cat_cardinalities = data["cat_card"]

    # Train the diffusion model on the dropout rows only.
    X_minority = X_train[y_train == 1]
    print(f"[Phase 4] Training diffusion on minority: {X_minority.shape}")
    model, sched, history = train_diffusion(X_minority, T=200, epochs=200,
                                            batch=64, verbose=False)
    print(f"[Phase 4] final eps-MSE = {history[-1]['mse']:.4f}")

    n_synth = int((y_train == 0).sum() - (y_train == 1).sum())
    X_synth = diffusion_sample(model, sched, n_synth, X_minority.shape[1],
                               n_numerical, cat_cardinalities)
    y_synth = np.ones(n_synth, dtype=np.int64)
    np.savez_compressed(DATA_DIR / "diffusion_synth.npz", X=X_synth, y=y_synth)

    X_aug = np.concatenate([X_train, X_synth], axis=0)
    y_aug = np.concatenate([y_train, y_synth], axis=0)
    print(f"[Phase 4] Augmented training set: {X_aug.shape}")

    print("[Phase 4] Training classifiers on diffusion-augmented data ...")
    results = evaluate_augmented(X_aug, y_aug, X_test, y_test)
    for name, scores in results.items():
        print(f"  {name:15s}  Acc={scores['accuracy']:.3f}  F1={scores['f1']:.3f}  "
              f"AUC={scores['auc_roc']:.3f}")

    # Distributional-fidelity comparison: this diffusion run vs the Phase-3 VAE.
    # Numerical moments (mean/std gaps) plus the categorical-marginal TV check.
    utility_diffusion = utility_dist_score(X_minority, X_synth, n_numerical)
    utility_diffusion["categorical_tv_mean"] = categorical_tv_score(
        X_minority, X_synth, n_numerical, cat_cardinalities)
    vae_synth = np.load(DATA_DIR / "vae_synth.npz")["X"]
    utility_vae = utility_dist_score(X_minority, vae_synth, n_numerical)
    utility_vae["categorical_tv_mean"] = categorical_tv_score(
        X_minority, vae_synth, n_numerical, cat_cardinalities)

    with open(RESULTS_DIR / "phase3_vae.json") as f:
        phase3 = json.load(f)
    with open(RESULTS_DIR / "phase2_baselines.json") as f:
        phase2 = json.load(f)
    save_json({
        "diffusion_augmented": results,
        "vae_augmented": phase3["vae_augmented"],
        "baseline_phase2": phase2,
        "utility_vae":       utility_vae,
        "utility_diffusion": utility_diffusion,
        "n_synth_added": int(n_synth),
        "final_eps_mse": history[-1]["mse"],
    }, RESULTS_DIR / "phase4_diffusion.json")

    # Training curve: eps-prediction MSE over epochs.
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot([h["epoch"] for h in history], [h["mse"] for h in history],
            color="#16a085")
    ax.set_title("Phase 4 - Diffusion eps-MSE Training Loss")
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE")
    fig.savefig(FIGURES_DIR / "phase4_diffusion_training_curve.png")
    plt.close(fig)

    # Real vs diffusion-synthetic densities for the first few numerical features
    # (capped at four; some datasets have fewer than four numericals).
    n_plot = min(4, n_numerical)
    feature_names = data["feature_names"][:n_plot]
    fig, axes = plt.subplots(1, n_plot, figsize=(3.25 * n_plot, 3), squeeze=False)
    axes = axes[0]
    for i, ax in enumerate(axes):
        sns.kdeplot(X_minority[:, i], ax=ax, label="real",
                    fill=True, alpha=0.4, color="#3498db")
        sns.kdeplot(X_synth[:, i], ax=ax, label="diffusion",
                    fill=True, alpha=0.4, color="#16a085")
        ax.set_title(feature_names[i], fontsize=9); ax.set_xlabel("")
    axes[0].legend(fontsize=8)
    fig.suptitle("Phase 4 - Real vs Diffusion Synthetic (numericals)",
                 fontweight="bold")
    fig.savefig(FIGURES_DIR / "phase4_diffusion_real_vs_synth.png")
    plt.close(fig)

    # Side-by-side: distributional fidelity (left) and classifier metrics (right).
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    # Left: distribution gap, lower is better.
    labels = ["|mean diff|", "|std diff|"]
    x = np.arange(len(labels)); bar_width = 0.35
    axes[0].bar(x - bar_width / 2, [utility_vae["mean_abs_diff_means"],
                                    utility_vae["mean_abs_diff_stds"]],
                width=bar_width, label="VAE", color="#e67e22")
    axes[0].bar(x + bar_width / 2, [utility_diffusion["mean_abs_diff_means"],
                                    utility_diffusion["mean_abs_diff_stds"]],
                width=bar_width, label="Diffusion", color="#16a085")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
    axes[0].set_title("Distribution gap (numerical) - lower is better")
    axes[0].legend()

    # Right: test-set metrics, VAE-augmented vs diffusion-augmented per model.
    metric_names = ["accuracy", "f1", "auc_roc"]
    rows = []
    for name in ["RandomForest", "XGBoost", "MLP"]:
        rows.append((f"{name} VAE",  [phase3["vae_augmented"][name][m] for m in metric_names]))
        rows.append((f"{name} Diff", [results[name][m]                for m in metric_names]))
    ind = np.arange(len(metric_names)); bar_width = 0.13
    colors = ["#e67e22", "#16a085"] * 3
    for j, (label, values) in enumerate(rows):
        axes[1].bar(ind + (j - len(rows) / 2) * bar_width, values, width=bar_width,
                    label=label, color=colors[j], alpha=0.85 if "Diff" in label else 0.55)
    axes[1].set_xticks(ind); axes[1].set_xticklabels([m.upper() for m in metric_names])
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Classifier metrics: VAE vs Diffusion augmentation")
    axes[1].legend(fontsize=6, ncol=3, loc="lower right")
    fig.savefig(FIGURES_DIR / "phase4_vae_vs_diffusion.png"); plt.close(fig)

    print("[Phase 4] utility scores:", utility_vae, utility_diffusion)
    print("[Phase 4] done")


if __name__ == "__main__":
    main()
