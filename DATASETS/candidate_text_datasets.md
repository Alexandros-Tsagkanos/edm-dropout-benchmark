# Candidate datasets for a real text + dropout result (search log)

Goal of the search: find a corpus that pairs student-authored free text with a dropout/
completion label at the student level, so Phase 9 (multimodal) could become a genuine result
instead of a construction ceiling. This file records what exists so the search is not repeated.

## Headline finding
There is no large, English, openly-downloadable dataset that pairs student free text with a dropout
label. The gap is structural (forum text carries personally-identifying content → it is redacted to
counts or locked in privacy enclaves). It is why our Phase 9 used simulated text — and, tellingly,
why TRIAD Drop's own "multimodal" result actually uses GPT-4.5 synthetic comments on the public UCI
base rather than real student text (see `triad_drop/provenance_triad.md`). The point is worth stating in the
paper's limitations.

## Candidate table

| Dataset | Student free text? | Dropout label? | Access | Use for the text goal | Use as a tabular cohort |
|---|---|---|---|---|---|
| MOOCCubeX (Tsinghua/XuetangX) | comments + replies | derive from engagement | Free download (GitHub) |  only downloadable option — but Chinese (swap DistilBERT → Chinese/multilingual encoder) | large |
| MORF (Penn/Baker) | real English forum text | completion | enclave (submit code, no raw data) | ◐ gold-standard but gated; can't fuse locally | ◐ via enclave only |
| TRIAD Drop (Mihoubi/Zerkouk et al.) | ✗ GPT-4.5 synthetic | (UCI base) | public (MIT) | ✗ not real text | ◐ = our Realinho base (synthetic text only) |
| SentiDrop (same group, TÉLUQ) | real (private "partner" platform) | | private, unreleased (no repo/DOI) | ✗ inaccessible | ✗ |
| Tecnológico de Monterrey (Data 2022) | ✗ none | `retention` | request/agreement-gated (CC0 per descriptor; only 9-row teaser in hand) | ✗ no text | strong 3rd tabular cohort |
| edX HarvardX–MITx Person-Course; KDD Cup 2015; OULAD; Realinho; Moodle/Zenodo sets | ✗ counts only | | open | ✗ | (OULAD/Realinho already used) |

## The TecMonterrey descriptor (bundled: `tecnologico_monterrey/TecMonterrey_DataDescriptor_Data2022.pdf` + `tecnologico_monterrey/Student Dropout Dictionary_Data teaser.xlsx`)
It is NOT the TRIAD cohort, and it has no text. Verified by reading the full PDF report
data-dictionary rows:

- Identity: Student Dataset from Tecnológico de Monterrey in Mexico to Predict Dropout in Higher
  Education, Alvarado-Uribe et al., Data (MDPI) 2022, 7, 119, doi `10.3390/data7090119`; dataset
  DOI `10.57687/FK2/PWJRSJ`. 143,326 records / 121,584 students (Mexican HS + undergraduate,
  2014–2020), 50 structured variables, label `retention` (1 = retained, 0 = dropout).
- Not TRIAD/SentiDrop: different authors, institution, ~30× larger; the PDF contains zero hits for
  retrieval / cross-modal / comment / sentiment / "distance learning" / TRIAD / 4423.
- Tabular-only — no free-text column. Every field is integer/float/binary/categorical
  (sociodemographic, academic, financial, "student-life" activity flags). So it cannot serve the
  multimodal/text goal — same limitation as Realinho and OULAD.
- Access: the descriptor lists CC0, but the full dataset is request/agreement-gated — you must
  contact the team and complete a form; only the 9-row teaser is currently in hand.

### Why it's still worth keeping (a strong third tabular cohort)
This directly hardens the axis the reviewer praised most — external validity — which is more
valuable to this paper than chasing a real text cohort (TRIAD's "text cohort" turned out to be
GPT-synthetic comments on our own UCI base — see `triad_drop/provenance_triad.md`):
- A third, large, geographically distinct cohort (Mexico) alongside Realinho (Portugal) + OULAD
  (UK) → strongly answers the "single-dataset / kitchen-sink" critique.
- Enables richer fairness: `gender` plus `socioeconomic.level` and `social.lag` → SES /
  intersectional fairness, the gap the review flagged.
- Caveats for integration (if access is granted): records are enrollment-level (a student can
  recur across cohorts/levels — e.g., id 499 appears twice → dedup or treat as enrolments, OULAD-style),
  and the prep step would mirror `DATASETS/oulad/prep_oulad.py` (demographics/academic features, derive
  `y = 1 - retention`).

## Recommendation
- For the multimodal/text goal: the only downloadable path is MOOCCubeX (with a Chinese/
  multilingual encoder); MORF is the English gold-standard but gated. Both are future work — neither
  is a quick add.
- For this project now: we are keeping the honest framing (simulated text = protocol ceiling; real-text
  protocol validated on Stanford across 6 constructs) and cite the data-availability gap as a
  limitation. Raise Tecnológico de Monterrey with the supervisor as a candidate third tabular
  dataset (pending the access request) — a clean, high-value way to extend generalization + fairness
  without touching the text question.

### Ready-to-cite blurb (related-work / future-work)
> A genuine multimodal dropout result requires a corpus pairing student-authored text with outcomes at
> the student level; such data is scarce and typically private or access-restricted, and where
> ``multimodal'' results are reported on open data the text is often synthetic (e.g., TRIAD~Drop
> augments the UCI base with GPT-4.5-generated comments \cite{Mihoubi2025TRIADDrop}), while large open
> dropout datasets — Realinho, OULAD, and the Tecnológico de Monterrey set
> \cite{AlvaradoUribe2022TecMty} — are tabular and carry no free text.
> We therefore validate the encoding protocol on real educational text (Stanford MOOC posts) and
> leave a real text-paired dropout study to future work, contingent on obtaining such a corpus
> (e.g., MOOCCubeX \cite{Yu2021MOOCCubeX} with a multilingual encoder, or the MORF enclave
> \cite{Gardner2018MORF}).
