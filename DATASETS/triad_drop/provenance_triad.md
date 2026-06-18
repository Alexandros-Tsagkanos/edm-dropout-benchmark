# Provenance and usage: the TRIAD-Drop repository

## What the file is
`TRIAD Zerkouk Github.zip` (SHA-256 `3957ff9d0f90ed65952e8bd6e280fcfed4a2cad9c1b174be88ad343c9ac37b15`)
is the GitHub repository for TRIAD-Drop (Mihoubi, Zerkouk, Chikhaoui, 2025; the paper is bundled here
as `TRIAD_paper.pdf` = arXiv:2507.05285). Contents of `TRIAD-Drop-main/`:

| File | Size | Notes |
|---|---|---|
| `README.md` | 2.8 KB | describes the method and the dataset |
| `LICENSE` | 1.1 KB | MIT (© 2025 Mihoubi Miloud) |
| `.gitignore` | 4.7 KB | — |
| `dataset_with_comments.csv` | 1.4 MB | the only data file (4,423 × 36) |

## The dataset is UCI Realinho plus GPT-4.5 synthetic comments
`dataset_with_comments.csv` is the UCI "Predict Students' Dropout and Academic Success" (Realinho)
dataset — the same tabular base this project uses — with one added column. It carries 34 of the 36
Realinho features (the two grade/admission-grade columns are dropped), the `Target` (Graduate 2208 /
Dropout 1421 / Enrolled 794), and a `student_comment`. Per their own README, the comment is generated
by GPT-4.5, conditioned on each student's attributes:

> Augmented: Synthetic student comments generated via GPT-4.5
> prompt = "As a {age}-year-old {gender} student with GPA {gpa}, write forum post about {course}"

The comments transparently restate each row's features and lean by outcome (e.g. a Dropout-row comment
opens "Despite attending all day classes, I'm truly struggling acadically … high unemployment rate …";
a Graduate-row one reads "… manageable … maintaining a decent grade average …"). The repeated GPT
artifact "acadically" is visible across rows.

## How it can and cannot be used
It cannot make a real multimodal dropout result — the text is synthetic, not collected from students.
Using it would be our Phase-9 approach with GPT prose instead of templates.

It can serve as a same-base, MIT-licensed comparator, and that is how we use it (see
`../../RESULTS/analysis/triad_synthetic_text_mi_results.md`). Running our leakage-safe text-only
protocol on TRIAD's comments predicts dropout from the text alone at AUC 0.99 / F1 0.94 — a text-only
AUC above even our tabular AUC (~0.92), so the comments carry label information beyond the student
attributes. The leak is even lexical: 89.9% of Graduate-row comments contain the string "graduat" (vs
0.7% of Dropout rows), and a three-line keyword rule reaches 77% three-class accuracy with no model at
all. Their paper abstract reports 89% accuracy / F1 0.88; the 0.85 in their repo README is a
three-class macro-F1, not directly comparable to our split, so we do not rank against it. Their
"multimodal gain" is therefore largely a recovery of injected label signal.

The point of the comparison is the methodological contrast, not a leaderboard: on the same UCI base,
the most-cited recent competitor's text is more leaky and is left undisclosed, whereas our own
simulated text injects less signal (0.28 bits), and we both disclose and quantify it.

## Honest framing
We report this as an empirical, reproducible property of their released dataset — what a text-only
classifier achieves on it — and make no claim about intent. The MIT licence permits reuse and
redistribution with attribution; cite `Mihoubi2025TRIADDrop`.

## Note on SentiDrop
SentiDrop (same author group; `SentiDrop_paper.pdf` here, arXiv:2507.10421) claims real student
comments, but on a private "partner" distance-learning platform with no downloadable dataset. We have
not verified its text provenance independently; given that TRIAD's released data is synthetic-on-UCI,
SentiDrop's claim of genuinely real text should be checked before relying on it.
