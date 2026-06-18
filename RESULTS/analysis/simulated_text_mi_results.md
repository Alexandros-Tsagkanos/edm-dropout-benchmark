# Mutual information injected by the simulated Phase-9 text (answers reviewer Q2)

Reproduce: `python CODE/analysis/simulated_text_mi.py`
(reads the committed `RESULTS/realinho/single_seed/data/student_comments.csv`; prints the report below).

## The reviewer's question
> "If the comments are generated via a hash with three tone pools whose mixture depends on the
> outcome, what is I(text; label) by construction, and how do you defend that the ~0.06 F1 lift is
> not simply recovering the artificial signal you injected?"

## Answer — we quantify the injected signal exactly
The generative process (`make_comment`) draws a tone pool conditioned on the label
(y=1: neg ¾, neu ¼; y=0: pos ⅖, neu ⅖, neg ⅕) then a template uniformly within the pool. Because
the within-pool choice is label-independent and the 15 templates are distinct, the label
information collapses to the pool: I(text; y) ≈ I(pool; y) (the template draw shares the
y-dependent RNG stream, so a tiny residual dependence is possible — the empirical 0.285 below is the
operative number), and this upper-bounds anything a frozen DistilBERT + PCA can extract.

| Quantity | Value |
|---|---|
| P(y = 1) (dropout), H(y) | 0.321, 0.906 bits |
| I(text; y) — exact, from the mixture | 0.278 bits (30.7% of H(y)) |
| I(comment; y) — empirical cross-check from the 15 committed comments | 0.285 bits (agrees) |
| P(y=1 \| pool): pos / neu / neg | 0.00 / 0.23 / 0.64 |
| Text-ONLY Bayes classifier, dropout F1 on the 1106-row Phase-9 test split | 0.70 (acc 0.79) |
| Reference (committed single-seed): tabular baseline F1 / multimodal F1 | ~0.79 / ~0.85 |

## Reading
The synthetic channel injects ~0.28 bits — about 31% of the label's entropy — by construction,
enough that the text alone predicts dropout at F1 ≈ 0.70. The ~0.06 multimodal lift is therefore
a ceiling driven by injected signal, not evidence of a real-world text effect. This is not
train/test leakage (the encoder is frozen and PCA/scaler are fit on train only — verified); it is a
construct-validity limitation of synthetic, label-correlated text.

## What to do next
- Report I(text;y) ≈ 0.28 bits and the text-only F1 ≈ 0.70 explicitly where Phase 9 is introduced,
  and reframe Phase 9 as a leakage-safe-fusion protocol demonstration, not a dropout result.
- Point to the real-text protocol validation (`phase9b_realtext.py`, forum-post urgency, AUC ≈ 0.82)
  as the genuine evidence that the pipeline works on real student writing — while stating plainly
  that a real dropout-paired text corpus is required to claim a real multimodal gain — a gap that even
  TRIAD Drop does not truly fill (its comments are GPT-synthetic on the same UCI base; see
  `../../DATASETS/triad_drop/provenance_triad.md`).
