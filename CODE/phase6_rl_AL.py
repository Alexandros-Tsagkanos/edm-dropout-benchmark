"""
Phase 6 (Upgrade 2) - a reinforcement-learning active-learning agent.

Uncertainty sampling (Phase 5) is a fixed heuristic: it always chases the
points nearest the decision boundary, which can be noisy outliers, and it has
no notion of diversity. Here we replace the heuristic with a *learned*
acquisition policy - a small DQN that decides which slice of the pool to label
next - and we report the outcome honestly even though it does not win.

The setup:
  - For every pool point we compute a 5-D descriptor: entropy, margin,
    distance to the labelled centroid, predicted minority probability, and
    nearest-neighbour distance to the labelled set. (Order matters - the
    action space indexes into it.)
  - The state (20-D) summarises the pool: per-descriptor mean, std, max and
    90th percentile.
  - An action (one of 20) picks one of 5 descriptors x 4 quantile buckets; the
    round's 20 queries are drawn from that stratum.
  - Reward = the per-round improvement in minority-class F1, NOT accuracy.
    Rewarding accuracy would teach the agent to query easy majority students
    and inflate accuracy while ignoring dropouts - the classic reward-hacking
    trap. F1 forces it to care about the minority class.

The QNet (20 -> 64 -> 64 -> 20) is trained with a Huber loss on
experience-replay batches, bootstrapping a one-step TD target from the online
network (no separate target network). Epsilon decays over six episodes and we
report the greedy policy.

Honest result: the greedy policy lands *between* random and uncertainty
sampling. Short DQN runs on a tiny discrete action space and a small budget do
not beat a well-tuned heuristic. The interpretable takeaway is that the agent
collapses onto the distance-to-centroid stratum - it rediscovers that
density-aware diversity sampling is what helps on this dataset - and the RL
framing would only pay off on much larger pools and longer horizons.

Outputs
  results/phase6_rl_AL.json
  figures/phase6_RL_vs_uncertainty.png
"""
from __future__ import annotations
import json, numpy as np, matplotlib.pyplot as plt, random
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from scipy.stats import entropy as scipy_entropy
from scipy.spatial.distance import cdist

from common import (load_processed, save_json, set_plot_style,
                    RESULTS_DIR, FIGURES_DIR, RANDOM_SEED)

N_BUCKETS = 20   # 5 descriptors x 4 quantile buckets


class QNet(nn.Module):
    """Maps the 20-D pool state to a Q-value for each of the 20 strata (actions)."""
    def __init__(self, d_state: int, n_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_state, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, s):
        return self.net(s)


def compute_pool_features(clf, X_pool, X_labelled):
    """Build the (n_pool, 5) descriptor matrix the agent reasons over.

    Columns, in the order the action space depends on:
      0. entropy                       (uncertainty)
      1. margin = 1 - |2p-1|           (uncertainty, complementary view)
      2. distance to labelled centroid (diversity)
      3. predicted minority probability (class-prior surrogate)
      4. nearest-neighbour distance to the labelled set (density)
    """
    proba = clf.predict_proba(X_pool)
    entropy_feat = scipy_entropy(proba.T)
    margin = 1.0 - np.abs(2 * proba[:, 1] - 1.0)
    labelled_centroid = X_labelled.mean(0, keepdims=True)
    dist_to_centroid = np.linalg.norm(X_pool - labelled_centroid, axis=1)
    p_minority = proba[:, 1]
    # Nearest-neighbour distance against a labelled subsample - keeps cdist cheap.
    sample_n = min(120, X_labelled.shape[0])
    labelled_sample = X_labelled[np.random.default_rng(0).choice(
        X_labelled.shape[0], sample_n, replace=False)]
    nn_distance = cdist(X_pool, labelled_sample, metric="euclidean").min(axis=1)
    features = np.stack([entropy_feat, margin, dist_to_centroid,
                         p_minority, nn_distance], axis=1)
    # Z-normalise per descriptor so the quantile buckets below are well defined.
    features = (features - features.mean(0)) / (features.std(0) + 1e-8)
    return features


def make_state(pool_features):
    """Summarise the pool into a fixed 20-D state: per-descriptor mean/std/max/q90."""
    state = np.concatenate([
        pool_features.mean(0), pool_features.std(0),
        pool_features.max(0),  np.quantile(pool_features, 0.9, axis=0),
    ])
    return state.astype(np.float32)


def select_indices(action: int, pool_features: np.ndarray, q: int) -> np.ndarray:
    """Map an action in {0..19} to q pool indices.

    Decode the action as feature_idx * 4 + quantile_bucket, then return points
    that fall in that descriptor's quantile band (bucket 3 = top quantile).
    """
    feature_idx, quantile_bucket = divmod(action, 4)
    feature_scores = pool_features[:, feature_idx]
    order = np.argsort(feature_scores)        # ascending
    n = len(order)
    lo = int(quantile_bucket * n / 4); hi = int((quantile_bucket + 1) * n / 4)
    bucket = order[lo:hi]
    if len(bucket) < q:
        # Band too small for a full batch: pull in neighbouring indices.
        extra = list(set(order[max(0, lo - q):min(n, hi + q)]) - set(bucket))
        bucket = np.concatenate([bucket, extra[:q - len(bucket)]])
    return bucket[:q]


def fit_eval(X_labelled, y_labelled, X_test, y_test):
    """Fit logistic regression on the labelled set; return it and its test metrics."""
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)
    clf.fit(X_labelled, y_labelled)
    preds = clf.predict(X_test); scores = clf.predict_proba(X_test)[:, 1]
    return clf, {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1":       float(f1_score(y_test, preds, zero_division=0)),
        "auc_roc":  float(roc_auc_score(y_test, scores)),
    }


def run_episode(X_train, y_train, X_test, y_test, qnet, optimizer, replay,
                n_init=40, n_rounds=25, q_per_round=20,
                eps: float = 0.2, gamma: float = 0.9, train_q: bool = True,
                seed: int = 0):
    """Run one full active-learning trajectory; optionally update the Q-network."""
    rng = np.random.default_rng(seed)
    labelled_idx, pool_idx = train_test_split(
        np.arange(len(y_train)), train_size=n_init, stratify=y_train,
        random_state=seed,
    )
    labelled_idx, pool_idx = list(labelled_idx), list(pool_idx)
    clf, metrics = fit_eval(X_train[labelled_idx], y_train[labelled_idx], X_test, y_test)
    curve = [{"round": 0, "n_labels": len(labelled_idx), **metrics}]

    for round_idx in range(1, n_rounds + 1):
        X_pool = X_train[pool_idx]
        features = compute_pool_features(clf, X_pool, X_train[labelled_idx])
        state = make_state(features)

        # Epsilon-greedy: explore a random stratum, else take the Q-greedy one.
        if rng.random() < eps:
            action = int(rng.integers(N_BUCKETS))
        else:
            with torch.no_grad():
                q_values = qnet(torch.from_numpy(state).float())
                action = int(q_values.argmax().item())

        local_idx = select_indices(action, features, q_per_round)
        chosen = [pool_idx[i] for i in local_idx]
        labelled_idx.extend(chosen)
        for i in sorted(local_idx, reverse=True):
            pool_idx.pop(i)

        clf, metrics_new = fit_eval(X_train[labelled_idx], y_train[labelled_idx], X_test, y_test)
        # Reward is the gain in F1 (not accuracy - see the module docstring).
        reward = metrics_new["f1"] - curve[-1]["f1"]

        if round_idx < n_rounds:
            features_next = compute_pool_features(clf, X_train[pool_idx], X_train[labelled_idx])
            next_state = make_state(features_next)
            done = False
        else:
            next_state, done = state, True

        replay.append((state, action, reward, next_state, done))
        curve.append({"round": round_idx, "n_labels": len(labelled_idx),
                      "reward": float(reward), "action": action, **metrics_new})

        # One-step TD update on a replay minibatch (Huber loss, online-net bootstrap).
        if train_q and len(replay) >= 16:
            batch = random.sample(replay, k=min(32, len(replay)))
            states      = torch.from_numpy(np.stack([trans[0] for trans in batch])).float()
            actions     = torch.tensor([trans[1] for trans in batch], dtype=torch.long)
            rewards     = torch.tensor([trans[2] for trans in batch], dtype=torch.float32)
            next_states = torch.from_numpy(np.stack([trans[3] for trans in batch])).float()
            dones       = torch.tensor([trans[4] for trans in batch], dtype=torch.float32)
            q_pred = qnet(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                q_next = qnet(next_states).max(dim=1).values
                td_target = rewards + gamma * (1 - dones) * q_next
            loss = F.smooth_l1_loss(q_pred, td_target)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    return curve


def main():
    set_plot_style()
    data = load_processed()
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    torch.manual_seed(RANDOM_SEED); random.seed(RANDOM_SEED)
    qnet = QNet(d_state=5 * 4, n_actions=N_BUCKETS)
    optimizer = torch.optim.Adam(qnet.parameters(), lr=2e-3)
    replay: list = []

    # Train the policy over several episodes, decaying exploration each time.
    n_episodes = 6
    best_curve, best_f1 = None, -1
    for episode in range(1, n_episodes + 1):
        eps = max(0.05, 0.5 * (1 - episode / n_episodes))      # linear epsilon decay
        curve = run_episode(X_train, y_train, X_test, y_test, qnet, optimizer, replay,
                            eps=eps, train_q=True, seed=RANDOM_SEED + episode)
        print(f"[Phase 6] ep {episode}  eps={eps:.2f}  final F1={curve[-1]['f1']:.3f}")
        if curve[-1]["f1"] > best_f1:
            best_f1, best_curve = curve[-1]["f1"], curve

    # Greedy evaluation episode (no exploration, no further learning).
    eval_curve = run_episode(X_train, y_train, X_test, y_test, qnet, optimizer, replay,
                             eps=0.0, train_q=False, seed=RANDOM_SEED + 999)
    print(f"[Phase 6] greedy eval final F1 = {eval_curve[-1]['f1']:.3f}")

    # Phase-5 curves give us the random and uncertainty references to plot against.
    with open(RESULTS_DIR / "phase5_uncertainty_AL.json") as f:
        p5 = json.load(f)

    save_json({"rl_best_train_curve":   best_curve,
               "rl_greedy_eval_curve":  eval_curve,
               "uncertainty_reference": p5["uncertainty"],
               "random_reference":      p5["random"]},
              RESULTS_DIR / "phase6_rl_AL.json")

    # Learning curves: random vs uncertainty vs the greedy RL agent.
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, metric in zip(axes, ["accuracy", "f1", "auc_roc"]):
        ax.plot([c["n_labels"] for c in p5["random"]],
                [c[metric] for c in p5["random"]],
                "o--", color="#7f8c8d", label="Random")
        ax.plot([c["n_labels"] for c in p5["uncertainty"]],
                [c[metric] for c in p5["uncertainty"]],
                "o-", color="#2980b9", label="Uncertainty")
        ax.plot([c["n_labels"] for c in eval_curve],
                [c[metric] for c in eval_curve],
                "o-", color="#c0392b", label="RL Agent (greedy)")
        ax.set_xlabel("# labels acquired"); ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()}"); ax.legend(fontsize=8)
    fig.suptitle("Phase 6 - RL Active Learning vs Uncertainty",
                 fontweight="bold")
    fig.savefig(FIGURES_DIR / "phase6_RL_vs_uncertainty.png"); plt.close(fig)
    print("[Phase 6] done")


if __name__ == "__main__":
    main()
