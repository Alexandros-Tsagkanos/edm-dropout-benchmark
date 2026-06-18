# Multi-label real-text validation on the Stanford MOOC Posts dataset

Reproduce: `python CODE/analysis/stanford_multilabel_protocol.py`
(reads the extracted `stanfordMOOCForumPostsSet.txt`; writes `stanford_multilabel_results.csv`
and `stanford_multilabel_auc.png`). Encoder, pooling, and the leakage-safe steps are imported from the
pipeline's own `phase9_multimodal_bert.encode_comments` + `common`, so the protocol is identical to
Phase 9.

## Why
The external review's sharpest critique is the simulated Phase-9 text. `phase9b_realtext.py` already
validates the leakage-safe encoding protocol on one real label (forum-post urgency, cross-corpus,
AUC≈0.82). The original Stanford dataset carries five further real annotation dimensions that the
project's reduced `Stanford.csv` discards. Running the same protocol across all of them tests
whether the encoding pipeline carries genuine signal on real student writing across constructs,
not just urgency.

## Protocol
Frozen DistilBERT (mean-pooled, 768-d) on all 29,604 real posts, encoded once; per label a
stratified 75/25 split, PCA(32) + StandardScaler fit on train only, then LogisticRegression /
RandomForest (`class_weight="balanced"`). Labels binarised "high vs low" (1–7 dims at ≥4, matching
phase9b; binary dims at ≥0.5). Seed 42. Disclosure: `encode_comments` is imported from phase 9
unchanged, so posts are truncated at its 48-token window (~ the first 35 words of each post) —
the AUCs below are achieved from post openings, which if anything understates the available signal.

## Results (held-out test AUC; n = 29,604)

| Construct | positive rate | LogReg AUC | RF AUC | best F1 |
|---|---|---|---|---|
| Urgency      | 0.22 | 0.867 | 0.885 | 0.61 |
| Confusion    | 0.67 | 0.814 | 0.840 | 0.85 |
| Sentiment    | 0.85 | 0.837 | 0.850 | 0.93 |
| Opinion      | 0.56 | 0.867 | 0.880 | 0.81 |
| Question     | 0.20 | 0.887 | 0.897 | 0.63 |
| Answer       | 0.20 | 0.815 | 0.838 | 0.53 |

All six real constructs are predicted well above chance — AUC ∈ [0.81, 0.90], mean ≈ 0.86 (RF).
The within-Stanford urgency AUC (0.87–0.89) is, as expected, a little higher than `phase9b`'s harder
cross-corpus urgency result (AUC≈0.82); the two are complementary protocol instances. F1 varies with
class balance (minority-positive labels like Answer/Question have lower F1 despite high AUC), so AUC —
threshold-independent — is the headline.

## Reading
The leakage-safe frozen-DistilBERT + train-only-PCA pipeline robustly encodes genuine student forum
text across six independent constructs, not just the single urgency task. This substantially
strengthens the claim that Phase 9's mechanics are sound on real text — the limitation is purely the
absence of a real dropout-paired corpus, not the encoding protocol.

Stays honest: the Stanford posts have no dropout label; this validates the protocol, not a
multimodal dropout result. See `../../DATASETS/stanford_mooc_posts/provenance_stanford.md` and `../../FINAL REPORT/REPORT.pdf`.
