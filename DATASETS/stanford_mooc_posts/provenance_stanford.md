# Provenance & usage: the Stanford MOOC Posts dataset

## What the data is
`stanfordMOOCForumPostsSet/` (in this folder) holds the extracted contents of the original Stanford
MOOC Posts dataset (Agrawal & Paepcke, 2014), whose source tarball
`stanfordMOOCForumPostsSet.tar.gz` (13.2 MB) contains two equivalent files:

- `stanfordMOOCForumPostsSet.txt` — tab-separated, 29,604 posts + header (one line per post)
- `stanfordMOOCForumPostsSet.xlsx` — the same data as a spreadsheet

- SHA-256 (`.tar.gz`): `a7c1aef6178c80333122e4135fe6cd61329ffc616e5fc16e55c5bfe72117c6ae`
- Already cited in the paper's bibliography as `Agrawal2014Stanford`.
- Source URL: `http://infolab.stanford.edu/~paepcke/stanfordMOOCForumPostsSet.tar.gz`

> Kept locally for the analysis; do not redistribute publicly. The extracted folder sits here so
> `../../CODE/analysis/stanford_multilabel_protocol.py` can read the `.txt` directly; the original tarball is
> not kept (re-download from the source URL and verify with the SHA-256 if needed). For any public
> release of this package, drop the extracted folder and cite/link the source instead. Our committed
> outputs are in `../../RESULTS/analysis/stanford_multilabel_*`.

### Full schema (18 tab-separated columns)
```
Text | Opinion(1/0) | Question(1/0) | Answer(1/0) | Sentiment(1-7) | Confusion(1-7) |
Urgency(1-7) | CourseType | forum_post_id | course_display_name | forum_uid | created_at |
post_type | anonymous | anonymous_to_peers | up_count | comment_thread_id | reads
```
The six annotation dimensions are annotator means (e.g. `Sentiment` = 6.5); the `(1-7)` dims are
1–7 scales, the `(1/0)` dims are binary-ish.

## Relationship to the `forum_urgency` dataset (they are NOT the same)
- `datasets/forum_urgency/` is the Švábenský et al. 2023 (EDM) urgency-detection package. It
  trains on 9 Penn-coded courses (`All_Courses_REDACTED_CODED.csv`) and tests on the Stanford
  MOOC Posts.
- The project's `datasets/forum_urgency/Stanford.csv` (the test set used by `phase9b_realtext.py`) is a
  reduced derivative of this tarball — produced from the Stanford dataset and kept to only
  `post_text, Urgency Practice, Urgency_1_7, CourseType` (29,604 rows). The Švábenský `README.md`
  documents this chain and notes a minor redaction fix vs the paper's version.

```
stanfordMOOCForumPostsSet.tar.gz  (original, 18 cols, 6 annotated dimensions)
        │  (Švábenský redaction-fixed extraction; keeps Text + Urgency)
        ▼
datasets/forum_urgency/Stanford.csv  (post_text + Urgency_1_7 only)  ── test set for phase9b_realtext.py
```

So the tar is the original superset; the project currently uses only its Urgency dimension and
discards five real annotation dimensions (Opinion, Question, Answer, Sentiment, Confusion).

## How it can / cannot be used
Can (and now does — see `../../CODE/analysis/stanford_multilabel_protocol.py`):
- Broaden the real-text protocol validation: run the identical frozen-DistilBERT + train-only-PCA
  protocol across the five discarded real dimensions, showing the encoding pipeline carries genuine
  signal on real student writing across several constructs — not just urgency. Hardens the reviewer's
  "most exposed flank" (the simulated Phase-9 text).
- Provenance/reproducibility: the SHA-256 and chain above give a clean, citable lineage for the
  `phase9b` test set.

Cannot:
- Make the multimodal dropout result real. The Stanford posts carry no dropout label and cannot
  be paired at the student level with Realinho or OULAD. This remains real text, not real
  dropout-paired text — a gap that, notably, even TRIAD Drop does not truly fill (its comments are
  GPT-synthetic on the same UCI base; see `../triad_drop/provenance_triad.md` and `../../FINAL REPORT/REPORT.pdf`).

## Note
The extracted dataset (~24 MB) is third-party data under its own terms, kept in this folder only so
the committed analysis is re-runnable; for a public artifact, remove it and cite
`Agrawal2014Stanford` with the source URL + SHA-256 above instead of redistributing.
