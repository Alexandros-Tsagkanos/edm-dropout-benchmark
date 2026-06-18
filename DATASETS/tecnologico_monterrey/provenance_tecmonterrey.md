# Provenance & usage: the Tecnológico de Monterrey dropout dataset

## What these files are
- `TecMonterrey_DataDescriptor_Data2022.pdf` — the Data Descriptor: *"Student Dataset from Tecnológico
  de Monterrey in Mexico to Predict Dropout in Higher Education"*, Alvarado-Uribe et al., Data (MDPI)
  2022, 7, 119, doi `10.3390/data7090119`. Dataset DOI `10.57687/FK2/PWJRSJ`. Bib key
  `AlvaradoUribe2022TecMty`.
- `Student Dropout Dictionary_Data teaser.xlsx` — the 9-row teaser + full 50-variable data
  dictionary (this is all that is openly available without a request).

## Key facts
- 143,326 records / 121,584 students (Mexican HS + undergraduate, 2014–2020), 50 structured
  variables, label `retention` (1 = retained, 0 = dropout). Records are enrollment-level (a student
  can recur across cohorts/levels — e.g. id 499 appears twice → dedup or treat as enrolments, OULAD-style).
- Tabular only — no free text. Cannot serve the multimodal/text goal (same limit as Realinho/OULAD).
- NOT the TRIAD cohort (different authors, institution, ~30× larger).
- Access: the descriptor lists CC0, but the full dataset is request/agreement-gated; only the
  9-row teaser is in hand. Obtaining the full set requires contacting the team / completing a form.

## Why it's kept here
A strong candidate third tabular cohort (large, geographically distinct, with `gender` +
`socioeconomic.level` + `social.lag` enabling richer/intersectional fairness) — a future
cross-dataset-generalization extension to raise with the supervisor, pending data access. See the
overview in `../README.md`.
