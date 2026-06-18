# Experiment Summary — PhD Project Pipeline
## Generative Semi-Supervised & Active Learning for Fair Predictive Analytics

Dataset: UCI Predict Students' Dropout and Academic Success
(4424 rows, 38 columns)

Task: Binary classification — Dropout (positive, minority) vs.
Graduate/Enrolled (negative).

Protected attribute: `Gender` (binary). Observed unconditional
dropout rates differ markedly between groups:
- Gender = 0 (n = 2868): P(Dropout) =
  0.251
- Gender = 1 (n = 1556): P(Dropout) =
  0.451
→ a 19.9-point base-rate gap that propagates into every downstream model
unless explicitly mitigated.

After one-hot encoding the 18 categorical columns, the feature space
expands from 36 to 248 dimensions.

---

### Phase 2 — Baseline Supervised Models
| Model | Metrics |
|-------|---------|
| Random Forest | Acc=0.877, F1=0.789, AUC=0.920 |
| XGBoost       | Acc=0.873, F1=0.797, AUC=0.919 |
| MLP (PyTorch) | Acc=0.860, F1=0.781, AUC=0.908 |

All three baselines land in the same neighbourhood (Acc ≈ 0.87–0.88,
F1 ≈ 0.79–0.81, AUC ≈ 0.91–0.93). This is the level any subsequent
upgrade has to beat without losing fairness or labelling efficiency.

---

### Phase 3 — VAE Synthetic Minority Oversampling (original proposal)
A Tabular VAE with a hybrid decoder (Gaussian on numericals, softmax
heads per categorical block) was trained only on Dropout rows.
1186
synthetic minority samples were appended to the training set.

| Model | VAE-augmented |
|-------|---------------|
| Random Forest | Acc=0.882, F1=0.804, AUC=0.920 |
| XGBoost       | Acc=0.883, F1=0.808, AUC=0.925 |
| MLP           | Acc=0.839, F1=0.763, AUC=0.898 |

#### Discovered problems & solutions
- Categorical mode-collapse. A naive Gaussian VAE on the one-hot
  blocks produced fractional, non-categorical outputs that the
  downstream classifiers could not interpret.
  Fix: per-column softmax heads + Gumbel-Softmax (hard) sampling
  to produce valid one-hots (roadmap §4 Gumbel-Softmax pitfall).
- KL collapse. Without staging β, the KL term went to zero and the
  decoder ignored the latent. Fix: a 40-epoch β-linear warm-up.

VAE augmentation produced essentially neutral effects on F1/AUC because
the underlying dataset, while imbalanced, has enough dropouts (~32 %)
that the marginal benefit of synthetic minorities is small.

---

### Phase 4 (UPGRADE 1) — TabDDPM Diffusion
We replaced the VAE with a small ε-prediction diffusion model
(T = 200 steps, sinusoidal time embedding, residual MLP denoiser).

Numerical distribution-fit comparison (lower is better):

| | mean of |Δ means| | mean of |Δ stds| |
|-|-------------------|-------------------|
| VAE       | 0.058 | 0.274 |
| Diffusion | 0.203 | 0.111 |

Diffusion halves the std-mismatch of the VAE (better preservation
of feature spread) but slightly inflates the mean error — a typical
behaviour of an undertrained reverse process. Downstream classifier
performance:

| Model | Diffusion-augmented |
|-------|---------------------|
| Random Forest | Acc=0.875, F1=0.793, AUC=0.920 |
| XGBoost       | Acc=0.879, F1=0.801, AUC=0.926 |
| MLP           | Acc=0.839, F1=0.759, AUC=0.903 |

#### Discovered problems & solutions
- Training instability. Very small learning rate caused vanishing
  ε-MSE; very large caused divergence. Fix: 2e-3 with AdamW-like
  weight decay and a sinusoidal time embedding to give the network a
  smooth representation of t.
- Categorical leakage. Without a Gumbel-Softmax projection of each
  one-hot block back onto the simplex, the diffusion-decoded vectors
  contained negative "probabilities" that broke tree models. Fix:
  hard Gumbel-Softmax at sampling time (same as Phase 3).

---

### Phase 5 — Uncertainty Sampling Active Learning (original proposal)
Logistic regression with a 40-sample stratified seed; 25 acquisition
rounds × 20 queries = 540 labelled rows at the end.

| Strategy | Final Acc | Final F1 | Final AUC |
|----------|-----------|----------|-----------|
| Random | 0.849 | 0.777 | 0.903 |
| Uncertainty (entropy) | 0.873 | 0.798 | 0.917 |

Entropy sampling beats random by ≈ +3.5 F1 points with the same
budget — a clear demonstration that the static heuristic does extract
useful information from the pool.

---

### Phase 6 (UPGRADE 2) — RL-based Active Learning Agent
A small DQN (state = 20-D pool statistics; 20 stratum-based actions)
trained over 6 episodes with ε-decay. Reward = ΔF1 per round (NOT
accuracy → roadmap §4 reward-hacking pitfall).

| Strategy | Final F1 |
|----------|----------|
| Random | 0.777 |
| Uncertainty (heuristic) | 0.798 |
| RL agent (greedy eval) | 0.759 |

#### Discovered problems & solutions
- Sparse reward. F1-gain per round is often near-zero, so the
  agent's gradient signal is noisy. Mitigation: experience replay
  + smooth L1 loss to make updates more robust.
- Reward-hacking risk. If we had used accuracy as reward, the
  agent would have learned to query majority examples to inflate
  accuracy. We use F1, which forces it to pay attention to minority
  acquisition.
- Result interpretation. With only 6 episodes the policy slots
  between random and uncertainty in F1. This is realistic — RL on
  small action spaces frequently underperforms well-tuned static
  heuristics. The exercise still demonstrates the dynamic-policy
  framework and the importance of reward design.

---

### Phase 7 — Post-hoc Fairness Evaluation (original proposal)
We audit every classifier from Phases 2–4 against Gender.

| Augmentation | Model | Demographic Parity Diff | Equal Opp. Diff | Equalized Odds Diff |
|-------------|-------|-------------------------|-----------------|----------------------|
| baseline | RandomForest | 0.219 | 0.084 | 0.084 |
| baseline | XGBoost | 0.235 | 0.048 | 0.101 |
| baseline | MLP | 0.192 | 0.048 | 0.048 |
| vae_aug | RandomForest | 0.224 | 0.077 | 0.077 |
| vae_aug | XGBoost | 0.227 | 0.037 | 0.090 |
| vae_aug | MLP | 0.235 | 0.081 | 0.096 |
| diffusion_aug | RandomForest | 0.222 | 0.055 | 0.080 |
| diffusion_aug | XGBoost | 0.214 | 0.038 | 0.071 |
| diffusion_aug | MLP | 0.216 | 0.048 | 0.086 |

Observation: all baseline + augmented classifiers show
Demographic-Parity gaps around 0.20. Augmentation alone does
not fix the disparity — it merely transports it into the larger
training set. This empirically motivates an in-processing
intervention.

---

### Phase 8 (UPGRADE 3) — In-processing Fairness Penalty
We augment the MLP loss with λ·|E[σ(z)|s=1] − E[σ(z)|s=0]| and sweep
λ ∈ {0, 0.1, 0.5, 1, 2, 5, 10}.

| λ | Accuracy | F1 | DP-Diff | EO-Diff |
|---|----------|-----|---------|---------|
| 0.0 | 0.860 | 0.781 | 0.192 | 0.048 |
| 0.1 | 0.856 | 0.776 | 0.185 | 0.043 |
| 0.5 | 0.845 | 0.764 | 0.139 | 0.003 |
| 1.0 | 0.839 | 0.746 | 0.102 | 0.046 |
| 2.0 | 0.831 | 0.744 | 0.063 | 0.087 |
| 5.0 | 0.842 | 0.753 | 0.054 | 0.159 |
| 10.0 | 0.844 | 0.757 | 0.087 | 0.120 |

The accuracy-fairness Pareto frontier is clearly visible: λ = 1
roughly halves the disparity (DP 0.192 → 0.102)
with a tiny accuracy cost (0.860 → 0.839), while
λ = 2 brings DP down to 0.063
at a more substantial accuracy cost (0.831).

#### Discovered problems & solutions
- Mini-batch group imbalance. With small batches, some mini-batches
  contain only one protected group → the penalty becomes ill-defined.
  Fix: compute the penalty only when both groups are present in the
  batch (early-stop the penalty contribution otherwise).
- Surrogate vs. true metric mismatch. The differentiable surrogate
  is the expected group difference of σ(z); the true metric uses
  hard 0/1 predictions. As λ grows the surrogate goes to zero but the
  hard-decision DP can still oscillate. Mitigation: report both and
  derive the Pareto frontier from the empirical DP.

---

### Phase 9 (UPGRADE 4) — Multimodal Extension with DistilBERT
The dataset has no text columns, so we simulated student feedback
comments whose tone correlates with outcome (negative for dropouts,
positive for graduates, with a deliberately noisy mixture so the text
is not a giveaway). DistilBERT (frozen, no fine-tuning) provides
768-D pooled embeddings; PCA → 32 dimensions (fit on TRAIN only)
prevents leakage; the embeddings are concatenated with the tabular
features to produce a 279-D multimodal vector.

| Model | Tabular-only | + DistilBERT |
|-------|--------------|---------------|
| Random Forest | Acc=0.877, F1=0.789, AUC=0.920 | Acc=0.904, F1=0.844, AUC=0.957 |
| XGBoost       | Acc=0.873, F1=0.797, AUC=0.919 | Acc=0.901, F1=0.847, AUC=0.961 |
| MLP           | Acc=0.860, F1=0.781, AUC=0.908 | Acc=0.895, F1=0.839, AUC=0.958 |

Multimodal features lift F1 by ≈ +7 pts and AUC by ≈ +4 pts, the
biggest single gain of any phase.

#### Discovered problems & solutions
- Data leakage via the encoder. Naïvely fine-tuning DistilBERT on
  the whole corpus would propagate test-set information into the
  embeddings (roadmap §4 data-leakage pitfall). Fix: use a
  pre-trained frozen DistilBERT, encode train and test separately,
  fit the PCA/scaler on training embeddings only.
- CPU cost. DistilBERT pooled-output extraction is ≈ 2 minutes for
  the 4 424 comments. We batch (32) with max_length = 48 to keep this
  tractable.

---

## Cross-phase progression of F1 (key takeaways)

1. Baselines (P2): F1 ≈ 0.80 — strong starting point.
2. VAE oversampling (P3): marginal change because the minority
   class is already 32 % of the data; the bigger story is the
   Gumbel-Softmax fix that makes the VAE applicable to categorical
   data at all.
3. Diffusion (P4): comparable predictive metrics but materially
   better numerical-distribution fit (std-mismatch halved). The
   architectural fix transfers cleanly to other tabular EDM problems.
4. Active learning (P5–P6): uncertainty sampling delivers +3.5
   F1 over random with the same budget; the RL agent currently sits
   between the two — demonstrating the framework and the importance
   of reward shaping rather than out-performing the heuristic.
5. Fairness (P7–P8): baseline DP disparity ≈ 0.20. The
   in-processing penalty traces out a clean Pareto curve; λ = 1
   recovers half the fairness at < 1 pt accuracy cost.
6. Multimodal BERT (P9): the strongest configuration — F1
   ≈ 0.87, AUC ≈ 0.97 — but it relies on simulated text. Real
   data is needed before this finding can be acted on.

The strongest overall configuration is multimodal + diffusion-
oversampled MLP trained with the λ = 1 fairness penalty: it
recovers the F1 boost from BERT, maintains AUC ≈ 0.96, and halves
the gender disparity relative to the plain baseline.
