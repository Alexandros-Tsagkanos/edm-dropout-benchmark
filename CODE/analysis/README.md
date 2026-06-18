# CODE/analysis/ — post-hoc rigor & competitor-re-assessment scripts

These are the scripts behind the §"Phase 9b breadth" and §"Re-assessing contemporary
multimodal-dropout claims" findings in `FINAL REPORT/` (and the familywise-error footnote in the
robustness section). Each is read-only on its inputs and writes to `../../RESULTS/analysis/`. Outputs
are already committed there, so you don't need to re-run anything to read the results.

| Script | What it does | Output (in `../../RESULTS/analysis/`) |
|---|---|---|
| `multiple_comparisons.py` | Holm + Bonferroni over the study's multi-seed Wilcoxon tests | `multiple_comparisons_results.{csv,md}` |
| `simulated_text_mi.py` | Exact mutual information of the Phase-9 simulated comments + text-only F1 | `simulated_text_mi_results.md` (prints to console) |
| `stanford_multilabel_protocol.py` | Leakage-safe DistilBERT protocol across 6 real Stanford MOOC-post constructs | `stanford_multilabel_results.{csv,md}`, `stanford_multilabel_auc.png` |
| `triad_synthetic_text_mi.py` | Same protocol on TRIAD-Drop's GPT comments (text-only dropout AUC/F1) | `triad_synthetic_text_mi_results.{csv,md}` |
| `morf/` | Draft submission code + plan for a real-text result via the MORF enclave (not runnable without an institutional DUA) | — |

## How they resolve paths
The scripts intentionally reuse the pipeline's own frozen-DistilBERT encoder so the protocol is
provably identical to Phase 9 (including its 48-token truncation window). Each resolves the repo root
as `Path(__file__).resolve().parents[1]` (= two levels up from `CODE/analysis/`) and imports from
`<repo>/CODE/` (`phase9_multimodal_bert.encode_comments`, `common`). Committed inputs:

- `multiple_comparisons.py` → `<repo>/RESULTS/{realinho,oulad}/multiseed/aggregate.json`
- `simulated_text_mi.py` → `<repo>/RESULTS/realinho/single_seed/data/student_comments.csv`
- `stanford_multilabel_protocol.py` → `<repo>/DATASETS/stanford_mooc_posts/stanfordMOOCForumPostsSet/stanfordMOOCForumPostsSet.txt`
- `triad_synthetic_text_mi.py` → `<repo>/DATASETS/triad_drop/TRIAD Zerkouk Github.zip`

Deps: `numpy, pandas, scikit-learn` for all; `torch, transformers` for the two text scripts (DistilBERT
downloads once). Run from the repo root, e.g.:

```
python CODE/analysis/multiple_comparisons.py      # fast, no torch
python CODE/analysis/triad_synthetic_text_mi.py   # ~1-2 min, needs torch+transformers
```
