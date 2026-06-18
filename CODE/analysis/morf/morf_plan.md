# MORF workstream — plan to obtain a real text + dropout result

Goal. Get the project's first genuinely real-text multimodal dropout result by running our
leakage-safe text→outcome protocol on real MOOC discussion-forum text inside the MOOC Replication
Framework (MORF) enclave — the one English source that pairs real student text with completion.

> Reality check up front (honest gating). MORF is a privacy enclave: you do not download
> data. You submit code (a Docker image + a controller script); MORF runs it on privacy-restricted
> Coursera data and returns only aggregate results. Therefore we cannot run this locally now — the
> hard prerequisite is an institutional Data Use Agreement (DUA) and an access grant from the MORF
> team. This folder contains the draft submission code + an access checklist so the experiment is
> ready to go the moment access is granted. The code here is a template validated against MORF's
> public docs (May 2026) — re-check it against the live `mwe/` example before submitting.

## What MORF provides (verified from the MORF site + papers, May 2026)
- Data: Coursera Spark + Phoenix exports — *"209 unique sessions across 77 courses
  (Spark) and 173 courses (Phoenix)"*, from 2 institutions. Raw exports are mounted at
  `/input/<course>/<session>/` inside your container: clickstream, SQL exports, demographics
  summaries. Discussion-forum text is in the Coursera SQL exports (forum tables), and MORF's own
  MORF-ENA tooling analyses "threaded forum posts" — so real forum text + a completion/dropout label
  per learner are available in-enclave. (Confirm exact forum-table availability with the MORF team.)
- Access model: *"available for collaborative projects; an institutional data use agreement is
  required."* Limited alpha. Contact: `morf-info@umich.edu`. Maintained by the
  educational-technology-collective (U. Michigan; Gardner/Brooks/Baker), `Gardner2018MORF`.

## The experiment we would submit (mirrors our pipeline)
A leakage-safe, text-only + fused completion predictor, identical in spirit to Phase 9 / our Stanford
protocol — but on real forum text:
1. `extract_session` (max parallelism): for each `/input/course/session/`, parse the forum SQL
   export → concatenate each learner's posts → produce per-learner `(text, tabular features,
   completion_label)`. Completion/dropout derived from the session's grade/certification export.
2. `train_course` / `test_course`: per course, frozen DistilBERT mean-pool of the forum text →
   PCA fit on train only → LogisticRegression / RF (and a tabular+text fusion), exactly our
   leakage-safe recipe. (MORF handles the train/holdout session split; we keep PCA/scaler train-only.)
3. `evaluate_course`: AUC / F1 for completion, text-only vs tabular vs fused — the real-text
   analogue of our Stanford AUCs and our Phase-9 fusion.

Why this matters: this is the one path to a real multimodal dropout result (vs our disclosed
synthetic ceiling and the synthetic/private competitors). Even a modest real-text lift, honestly
reported, would convert Phase 9 from "protocol validated on a proxy task" into a real finding.

## Submission workflow (morf-py)
1. `pip install morf-api` (client side — any modern Python; this only submits the job).
2. Build a Docker image whose `ENTRYPOINT ["python3","mwe.py"]` accepts `--mode
   {extract|extract-holdout|train|test}` and reads `/input/<course>/<session>/` (see
   `mwe_dropout_text.py` + `Dockerfile`). The image is our environment, so it can use a modern
   Python + torch/transformers base.
3. Write the controller with the MORF API (`extract_session`, `train_course`, `test_course`,
   `evaluate_course`) — see `morf_controller.py`.
4. Create a `config.properties` / `client.config` with identifiers + public (HTTPS/S3) URLs to the
   controller and the Docker image.
5. Submit: `from morf.utils.submit import easy_submit; easy_submit(client_config_url=..., email_to=...)`.
   Smoke-test first with `submit_mwe(email_to=...)`. Status emails come from `morf-alerts@umich.edu`.

## Honest risks / unknowns (resolve with the MORF team before committing effort)
- DUA latency — institutional agreements take weeks; needs the supervisor + university research/legal office.
- Alpha availability — confirm MORF is still accepting new external projects.
- Forum-text access — confirm the forum SQL tables are exposed to feature-extraction (vs clickstream only).
- Compute/runtime limits in the enclave for a DistilBERT encode over many sessions.
- API drift — the exact `mwe/` controller API may differ from these drafts; validate against the live repo.

## Next actions
See `ACCESS_CHECKLIST.md`. Step 1 (email `morf-info@umich.edu`) and step 2 (start the DUA with your
supervisor) are the unblockers; the code here is ready to refine once access is in motion.
