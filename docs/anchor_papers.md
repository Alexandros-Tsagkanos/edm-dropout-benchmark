# Anchor papers and key references

The four MDPI papers this project extends are **not redistributed** in this repository
(they are copyrighted by their publisher). They are cited below with DOIs and links so
the contribution of each phase is traceable. The mapping from each paper to the phase(s)
that extend it is also given. Full bibliographic details for every work cited by the
study are in the reference list of `FINAL REPORT/REPORT.pdf`.

## The four anchor papers (Kotsiantis line)

1. **Tabular VAE for synthetic data in semi-supervised EDM** — extended by **Phases 3–4**
   (the hybrid-decoder VAE and the TabDDPM diffusion oversampler; we also execute the
   paper's stated future work of diffusion + uncertainty selection).
   Kostopoulos, G., Fazakis, N., Kotsiantis, S., & Dimakopoulos, Y. (2025).
   *Enhancing Semi-Supervised Learning in Educational Data Mining Through Synthetic Data
   Generation Using Tabular Variational Autoencoder.* Algorithms, 18(10), 663.
   doi:[10.3390/a18100663](https://doi.org/10.3390/a18100663) ·
   <https://www.mdpi.com/1999-4893/18/10/663>

2. **Self-labelled SSL for MOOC dropout (Co-Forest / CoBC best)** — extended by the
   **self-labelled axis** (`CODE/coforest.py`, `CODE/ssl_baseline.py`) and the active-learning
   **Phases 5–6**.
   Raftopoulos, G., Kostopoulos, G., Davrazos, G., Panagiotakopoulos, T., Kotsiantis, S., &
   Kameas, A. (2025). *Comparative Analysis of Self-Labeled Algorithms for Predicting MOOC
   Dropout: A Case Study.* Applied Sciences, 15(22), 12025.
   doi:[10.3390/app152212025](https://doi.org/10.3390/app152212025) ·
   <https://www.mdpi.com/2076-3417/15/22/12025>

3. **Fairness-aware ML benchmark (recommends Pareto frontiers; default-hyperparameter
   limitation)** — answered by **Phase 8** (the λ-sweep and Pareto frontier).
   Raftopoulos, G., Fazakis, N., Davrazos, G., & Kotsiantis, S. (2025).
   *A Comprehensive Review and Benchmarking of Fairness-Aware Variants of Machine Learning
   Models.* Algorithms, 18(7), 435.
   doi:[10.3390/a18070435](https://doi.org/10.3390/a18070435) ·
   <https://www.mdpi.com/1999-4893/18/7/435>

4. **Human-in-the-Loop AI review (reward hacking)** — informs **Phase 6** (the DQN active
   learner uses an F1-gain reward, deliberately not accuracy, to avoid reward hacking).
   Lazaros, K., Vrahatis, A. G., & Kotsiantis, S. (2026).
   *Human-in-the-Loop Artificial Intelligence: A Systematic Review of Concepts, Methods,
   and Applications.* Entropy, 28(4), 377.
   doi:[10.3390/e28040377](https://doi.org/10.3390/e28040377) ·
   <https://www.mdpi.com/1099-4300/28/4/377>

## Datasets (cite the originators when you use the data)

- **UCI "Predict Students' Dropout and Academic Success" (Realinho)** — Realinho, V.,
  Machado, J., Baptista, L., & Martins, M. V. (2022). *Predicting Student Dropout and
  Academic Success.* Data, 7(11), 146. Distributed via the UCI ML Repository. See also
  Martins, M. V., Tolledo, D., Machado, J., Baptista, L. M. T., & Realinho, V. (2021).
  *Early Prediction of Student's Performance in Higher Education: A Case Study.* WorldCIST
  2021, AISC 1365, 166–175, Springer.
  doi:[10.1007/978-3-030-72657-7_16](https://doi.org/10.1007/978-3-030-72657-7_16)
- **OULAD (Open University Learning Analytics Dataset)** — Kuzilek, J., Hlosta, M., &
  Zdrahal, Z. (2017). *Open University Learning Analytics Dataset.* Scientific Data, 4,
  170171. doi:[10.1038/sdata.2017.171](https://doi.org/10.1038/sdata.2017.171)
- **Stanford MOOC Posts** — Agrawal, A., & Paepcke, A. (2014). *The Stanford MOOC Posts
  Dataset.* <http://infolab.stanford.edu/~paepcke/stanfordMOOCForumPostsSet.tar.gz>
- **Forum-urgency package (Penn)** — Švábenský, V., Baker, R. S., Zambrano, A., Zou, Y., &
  Slater, S. (2023). *Towards Generalizable Detection of Urgency of Discussion Forum Posts.*
  EDM 2023, 302–309. doi:[10.5281/zenodo.8115790](https://doi.org/10.5281/zenodo.8115790)
- **Tecnológico de Monterrey dropout dataset** (candidate third cohort, not used in results)
  — Alvarado-Uribe, J., Mejía-Almada, P., Masetto Herrera, A. L., Molontay, R., Hilliger, I.,
  Hegde, V., Montemayor Gallegos, J. E., Ramírez Díaz, R. A., & Ceballos, H. G. (2022).
  *Student Dataset from Tecnológico de Monterrey in Mexico to Predict Dropout in Higher
  Education.* Data, 7(9), 119. doi:[10.3390/data7090119](https://doi.org/10.3390/data7090119)

## Contemporary competitors (re-assessed in the study)

- **TRIAD-Drop** — Mihoubi, M., Zerkouk, M., & Chikhaoui, B. (2025). *Beyond Classical and
  Contemporary Models: A Transformative AI Framework for Student Dropout Prediction in
  Distance Learning using RAG, Prompt Engineering, and Cross-Modal Fusion.*
  arXiv:[2507.05285](https://arxiv.org/abs/2507.05285). Public MIT repo: UCI Realinho base
  + GPT-4.5 synthetic student comments. See `DATASETS/triad_drop/provenance_triad.md`.
- **SentiDrop** — Zerkouk, M., Mihoubi, M., & Chikhaoui, B. (2025). *SentiDrop: A Multi-Modal
  Machine Learning Model for Predicting Dropout in Distance Learning.*
  arXiv:[2507.10421](https://arxiv.org/abs/2507.10421).
