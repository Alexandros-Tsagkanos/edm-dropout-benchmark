# Generative, Semi-Supervised and Fair Learning for Student-Dropout Prediction

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/Alexandros-Tsagkanos/edm-dropout-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/Alexandros-Tsagkanos/edm-dropout-benchmark/actions/workflows/ci.yml)
[![Data: provenance-tracked](https://img.shields.io/badge/data-provenance--tracked-brightgreen.svg)](DATA.md)

A reproducible, multi-seed, significance-tested benchmark for **student-dropout prediction**,
and the code/results supplement to its technical report. The bar throughout is scientific
honesty and reproducibility: every claim is backed by a committed artefact, and
negative/fragile results are reported as plainly as positive ones.

Author: **Alexandros Tsagkanos**. Supervisor: **Prof. Sotiris Kotsiantis**.
University of Patras, Department of Mathematics.

> This repository accompanies a paper in preparation. The technical report is in
> [`FINAL REPORT/REPORT.pdf`](FINAL%20REPORT/REPORT.pdf). See [`DATA.md`](DATA.md) for data
> availability and [`docs/anchor_papers.md`](docs/anchor_papers.md) for the works it extends.

---

## 1. Overview

This project predicts student dropout on the UCI "Predict Students' Dropout and Academic Success"
dataset (Realinho; 4,424 students, binary Dropout vs. Graduate/Enrolled, ~32% positive). It is a
ten-phase pipeline built from a faithful re-implementation of the original proposal — three supervised
baselines, a VAE oversampler, uncertainty-based active learning, a post-hoc fairness audit — plus four
upgrades (TabDDPM diffusion, a DQN active learner, an in-processing fairness penalty, and a multimodal
DistilBERT extension), and a methodological-rigor layer on top: a classical SMOTE baseline, bootstrap
confidence intervals, a ten-seed robustness study with paired significance tests, a second dataset
(OULAD, also ten seeds) for cross-cohort generalization, a self-labelled SSL baseline (Co-Forest),
and a real-text validation of the multimodal protocol on a genuine educational corpus.

**Headline (mean over ten seeds).** The strongest configuration is multimodal XGBoost + DistilBERT at
F1 ≈ 0.85 / AUC ≈ 0.96 — a robust ~0.06 F1 lift over the tabular baseline (p = 0.002), though, because
the Phase-9 text is simulated, this is a ceiling rather than a deployable number (the simulated text
injects 0.28 bits of label information by construction, ~31% of the label entropy). An in-processing
fairness penalty cuts the gender demographic-parity gap from 0.21 to 0.08 (~60%, p = 0.002) at a real
~1.6-point accuracy cost. On the second dataset (OULAD, 32,593 enrolments, ten seeds) the oversampling
and active-learning findings partly reverse — a deliberate cross-cohort honesty check — and the
multimodal protocol, re-run on real forum text, reaches AUC ≈ 0.82 (urgency detection, not dropout),
confirming the encoding pipeline works on genuine student writing.

The recurring finding is that, on this moderate-imbalance tabular data, the more elaborate levers
(generative oversampling, RL active learning, self-labelling) do **not** beat a well-regularised
supervised baseline. The contribution is the integrated, honest, reproducible evaluation.

Full numbers are in [`FINAL REPORT/REPORT.pdf`](FINAL%20REPORT/REPORT.pdf). Every output from our own
runs — and the exact command that regenerates each — lives under [`RESULTS/`](RESULTS/) (see
[`RESULTS/REPRODUCE.md`](RESULTS/REPRODUCE.md)).

## 2. Quick start — run everything with one click

You need Python 3.9+. The dependencies install automatically on first run from `requirements.txt`
(or run `pip install -r requirements.txt` yourself). No GPU is required; a full `run_all.py` (all
three datasets) takes ~20 minutes on a modern CPU (Realinho alone is ~6 min; OULAD's larger table
makes it the slow part).

- Windows: double-click `run_all.bat`.
- macOS: double-click `run_all.command` (the first time you may need `chmod +x run_all.command`).
- Linux: `chmod +x run_all.sh` then `./run_all.sh`.
- Any OS, from a terminal: `python run_all.py`.

`python run_all.py` does a single-seed run of every dataset in turn — Realinho (phases 1–10 + SMOTE +
bootstrap CIs), OULAD (prep + phases 2–8 + SMOTE), and the real-text protocol check — installing any
missing dependencies first and printing per-phase progress. Outputs land under `RESULTS/`. No report is
produced; the technical report `FINAL REPORT/REPORT.pdf` is hand-written.

You can also run one dataset at a time — each `DATASETS/<dataset>/` folder carries its own run scripts:

- Single seed: `python DATASETS/realinho/run_single.py` (likewise `oulad`).
- Ten-seed study: `python DATASETS/realinho/run_multiseed.py 10` (~50 min). OULAD is resumable:
  `python DATASETS/oulad/run_multiseed.py next` (repeat until ten done), then `... aggregate 10`.
- Real-text check: `python DATASETS/forum_urgency/phase9b_realtext.py` (needs `Stanford.csv`, see below).
- Self-labelled SSL baseline: `python CODE/ssl_baseline.py` (Realinho; set `PROJECT_ROOT` to the OULAD
  single-seed folder for OULAD).
- Post-hoc analyses: the scripts in `CODE/analysis/` (see `CODE/analysis/README.md`).

## 3. Reproducing the results

Each phase writes a JSON file to the `results/` folder under `RESULTS/<dataset>/single_seed/`; the
final phase collates them into `master_comparison.csv` (one row per phase × method, every metric) and
auto-generates the narrative `experiment_summary.md` from those JSON values, so the summary can never
drift out of sync with the numbers. [`RESULTS/REPRODUCE.md`](RESULTS/REPRODUCE.md) lists the exact
command that regenerates each part of `RESULTS/`.

**Reproducibility note.** All randomness is seeded (`RANDOM_SEED=42` by default). Deterministic
components — the Random Forest, the logistic-regression active learner, the distributional-fidelity
scores — reproduce exactly. The neural components (the VAE, diffusion, the fairness-penalised MLP, the
DQN, and XGBoost) can drift by a few thousandths across different torch/xgboost versions; this is a
library effect, not a code bug. That is exactly why the substantive claims rest on the ten-seed study
rather than any single run: the per-phase tables in the report show mean ± std and the comparisons are
backed by paired Wilcoxon tests (with familywise Holm/Bonferroni correction — reproduce it with
`python CODE/analysis/multiple_comparisons.py`, the same check CI runs).

## 4. Data availability

This repository **bundles only the openly-licensed data** and **references the rest** with provenance
(source URL + SHA-256), following each dataset's terms. Bundled: UCI Realinho and the OULAD
demographics table (both CC BY 4.0), the MIT-licensed Penn forum-urgency package, and the MIT-licensed
TRIAD-Drop repository. Referenced (not redistributed): the Stanford MOOC Posts corpus and `Stanford.csv`
(reconstruct via [`DATASETS/forum_urgency/DATA.md`](DATASETS/forum_urgency/DATA.md)), and the
Tecnológico de Monterrey set. The four MDPI anchor PDFs are not redistributed; they are cited with DOIs
in [`docs/anchor_papers.md`](docs/anchor_papers.md). Full table: **[`DATA.md`](DATA.md)**.

## 5. Repository layout

```
edm-dropout-benchmark/
  README.md  LICENSE  CITATION.cff  DATA.md  requirements.txt
  run_all.{py,bat,command,sh}   one-click runner: single-seed run of every dataset
  .github/workflows/ci.yml      byte-compile + reproduce-one-artefact smoke test

  CODE/                         generic phases that run on ANY dataset:
    common.py                   shared paths, the split, encoding/scaling, metric helpers
    phase2..phase10, phase3b_smote.py, bootstrap_ci.py
    coforest.py, ssl_baseline.py    self-labelled SSL baseline (Co-Forest / Self-Training)
    analysis/                   post-hoc rigor + competitor re-assessment (see its README)

  DATASETS/                     raw data + each dataset's own preprocessing + run scripts:
    realinho/                   data.csv + phase1_preprocessing.py + run_single + run_multiseed
    oulad/                      studentInfo.csv + prep_oulad.py + run_single + run_multiseed
    forum_urgency/              Penn MIT package + phase9b_realtext.py (+ DATA.md for Stanford.csv)
    triad_drop/, stanford_mooc_posts/, tecnologico_monterrey/   provenance notes (+ MIT TRIAD zip)
    candidate_text_datasets.md  search log for a real text+dropout corpus

  RESULTS/                      curated outputs + how to reproduce them:
    REPRODUCE.md                exact command for every artefact
    realinho/, oulad/           single_seed/ (results + figures) and multiseed/ (aggregates,
                                per-seed results JSON, headline figures, Result_Report.pdf)
    forum_urgency/, analysis/   real-text + post-hoc analysis outputs

  FINAL REPORT/                 REPORT.pdf (the technical report) + figures/
  docs/                         anchor_papers.md (citations)
```

The top-level `CODE/`, `DATASETS/`, `RESULTS/` names are load-bearing — `CODE/common.py` resolves
paths relative to them — so they are kept as-is. Regenerable heavy artefacts (per-run model weights,
processed arrays, per-seed figure dumps) are **not** committed; reproduce them with `RESULTS/REPRODUCE.md`.

## 6. The pipeline, phase by phase

Every phase consumes only the artefacts produced by earlier phases, so all numbers are computed on one
fixed train/test split (frozen by Phase 1). The shared module `CODE/common.py` is the single source of
truth for paths, the stratified 75/25 split, the train-only one-hot encoding and scaling (the key
anti-leakage decision), and the predictive- and fairness-metric helpers.

- `phase1_preprocessing.py` (DATASETS/realinho) — EDA and the canonical split; computes class balance
  and the gender dropout base-rates (0.25 vs 0.45 — the ~20-point gap that drives the fairness work),
  one-hot encodes and standardises (fit on train only), and freezes the arrays every later phase reloads.
- `phase2_baselines.py` — Random Forest, XGBoost, and a small PyTorch MLP. The MLP is reused unchanged
  in Phases 3, 4, 8 and 9, so cross-phase comparisons are architecture-controlled.
- `phase3_vae.py` — a hybrid-decoder tabular VAE oversamples the minority class; the contribution is the
  per-block categorical likelihoods that make the VAE produce valid one-hot categories.
- `phase4_diffusion.py` (Upgrade 1) — a TabDDPM-style diffusion oversampler. It robustly halves the
  VAE's std mismatch (Δσ: 0.27 → 0.13, p = 0.002) but inflates the mean error by a matching amount, so
  net moment error is unchanged; downstream F1 is on par with the VAE.
- `phase3b_smote.py` — a classical SMOTE baseline through the identical evaluator. Shows that on this
  moderate-imbalance data no oversampler — VAE, diffusion, or SMOTE — gives a meaningful F1 advantage.
- `phase5_uncertainty_AL.py` — entropy-based uncertainty active learning vs. random (540 labels, ~16%
  of the pool). Uncertainty robustly wins (F1 0.795 vs 0.758, p = 0.002).
- `phase6_rl_AL.py` (Upgrade 2) — a DQN active learner with an F1-gain reward (chosen to avoid reward
  hacking). Honest negative result: it ties random and lands below uncertainty (p = 0.002).
- `coforest.py` + `ssl_baseline.py` — a faithful Co-Forest reimplementation (Li & Zhou, 2007) vs.
  Self-Training vs. a supervised seed under label scarcity; recovers most of the supervised F1 from
  1–10% of labels and beats a supervised seed only on OULAD — a cohort-dependent reversal.
- `phase7_fairness_posthoc.py` — audits gender disparity for every model × augmentation cell;
  augmentation alone does not fix unfairness (it carries the base-rate gap into a larger training set).
- `phase8_fairness_inprocess.py` (Upgrade 3) — adds a differentiable demographic-parity penalty to the
  MLP loss and sweeps λ; robustly reduces disparity (0.21 → 0.08 at λ = 1, p = 0.002) at ~1.6 accuracy
  points; λ best chosen as a region (~2–5).
- `phase9_multimodal_bert.py` (Upgrade 4) — fuses a frozen DistilBERT encoding of (simulated) comments
  with the tabular features via a leakage-safe protocol; the biggest single gain (F1 + ~0.06,
  p = 0.002), but a ceiling, since the text is simulated.
- `phase10_master_comparison.py` — collates every JSON into `master_comparison.csv` and the
  auto-generated `experiment_summary.md`.
- `bootstrap_ci.py` — 95% percentile bootstrap CIs for the headline metrics.
- `CODE/analysis/` — familywise-error correction, the exact mutual-information quantification of the
  simulated-text ceiling, a six-construct real-text validation on the Stanford MOOC posts, and a
  text-only re-assessment of the TRIAD-Drop competitor. See `CODE/analysis/README.md`.

## 7. Papers this work extends

Four MDPI papers from the Kotsiantis line anchor the study (tabular-VAE synthetic data; self-labelled
SSL for MOOC dropout; a fairness-aware ML benchmark; a Human-in-the-Loop review). Their PDFs are not
redistributed here; full citations, DOIs, and the phase-by-phase mapping are in
[`docs/anchor_papers.md`](docs/anchor_papers.md).

## 8. How to cite

If you use this code or its results, please cite the repository (GitHub's "Cite this repository"
button reads [`CITATION.cff`](CITATION.cff)) and the accompanying technical report
`FINAL REPORT/REPORT.pdf`. When you use a bundled dataset, also cite its originators
(see `docs/anchor_papers.md` → *Datasets*).

## 9. License

Code and the authors' own generated artefacts: **MIT** (see [`LICENSE`](LICENSE)). Bundled third-party
datasets retain their own licences (UCI Realinho and OULAD under CC BY 4.0 with attribution; the Penn
package and TRIAD-Drop under MIT). See [`DATA.md`](DATA.md).

## 10. Acknowledgements

Supervised by Prof. Sotiris Kotsiantis (University of Patras, Department of Mathematics). Builds on the
open-access MDPI educational-data-mining literature of the Kotsiantis group and on the UCI Realinho,
OULAD, Stanford MOOC Posts, and Penn forum-urgency datasets — with thanks to their authors.
