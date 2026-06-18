# Multiple-comparisons / familywise-error analysis (answers reviewer Q1)

Reproduce: `python CODE/analysis/multiple_comparisons.py`
(reads the committed `RESULTS/{realinho,oulad}/multiseed/aggregate.json`; writes
`multiple_comparisons_results.csv`). Pure post-hoc adjustment of the already-reported
paired-Wilcoxon p-values — no models are re-run.

## The question
> "Your headline claims rest on paired Wilcoxon with n=10 seeds, p-floor 0.002 (2/2¹⁰). You
> report p=0.002 for several simultaneous comparisons — what is the familywise error rate under
> Bonferroni, and does the diffusion-vs-VAE F1 comparison (p=0.027 for RF) survive?"

## Answer
We treat each dataset's significance table as one family and apply Holm–Bonferroni (primary)
and plain Bonferroni (conservative reference) at α = 0.05.

### Realinho (m = 13 tests)
All seven headline effects survive both Holm and Bonferroni (each raw p = 0.00195 → adjusted
p = 0.0254 < 0.05):

| Effect | raw p | Holm p | Bonf. p | survives |
|---|---|---|---|---|
| Multimodal > tabular F1 (RF / XGB / MLP) | 0.00195 | 0.0254 | 0.0254 | ✅ |
| Uncertainty > random AL F1 | 0.00195 | 0.0254 | 0.0254 | ✅ |
| RL < uncertainty AL F1 (honest negative) | 0.00195 | 0.0254 | 0.0254 | ✅ |
| Fair-MLP λ=1 < λ=0 DP | 0.00195 | 0.0254 | 0.0254 | ✅ |
| Diffusion < VAE σ-mismatch | 0.00195 | 0.0254 | 0.0254 | ✅ |

The contrasts we explicitly do not claim are exactly the ones that do not survive:

| Contrast (not claimed) | raw p | Holm p | Bonf. p | survives |
|---|---|---|---|---|
| SMOTE vs VAE F1 (RF) | 0.0137 | 0.082 | 0.178 | ❌ |
| Diffusion vs VAE F1 (RF) | 0.105 | 0.527 | 1.0 | ❌ |
| diffusion/SMOTE-vs-VAE (XGB, MLP) | 0.23–0.85 | — | — | ❌ |

So the comparison — diffusion-vs-VAE for RF — does not survive any correction
(and after the diffusion-sampler fix it no longer even reaches raw p<0.05: p=0.105), which is
consistent with, not contrary to, our conclusion that "no oversampler beats another on downstream
F1." The familywise-significant set is precisely the set we headline; the sub-threshold oversampler
differences are reported as ties. *(Note: the reviewer's quoted p=0.027 was the pre-fix value,
produced by the buggy categorical sampler; see the sampler-fix discussion in the report.)*

### OULAD (m = 13 tests)
7/13 survive Holm: the four floor-level oversampling effects (VAE-vs-baseline XGB/MLP,
SMOTE-vs-baseline RF/XGB — these also survive Bonferroni), plus VAE-vs-baseline RF and
SMOTE-vs-baseline MLP (p_holm ≈ 0.035), plus the RL > uncertainty reversal (p_holm ≈ 0.041). The
uncertainty < random reversal is marginal (p_holm ≈ 0.059) and should be reported as directional.
(Pre-fix, the buggy diffusion sampler produced two spuriously-low diffusion-vs-VAE p-values that
inflated the survived count to 11/13 and shifted the Holm ordering; post-fix, all three
diffusion-vs-VAE contrasts are raw p ≥ 0.70.) The fairness λ=1<λ=0 contrast is
correctly non-significant (raw p = 0.43), matching OULAD's near-zero gender gap.

## What to do next
- Add one sentence to the statistics/results section:
  state that the headline effects survive Holm–Bonferroni at α=0.05, and that the unclaimed
  oversampler contrasts do not — turning Q1 from a threat into a robustness footnote.
- Optionally add a `p_holm` column to the significance table, or a footnote with the family size m
  and the adjusted threshold.
