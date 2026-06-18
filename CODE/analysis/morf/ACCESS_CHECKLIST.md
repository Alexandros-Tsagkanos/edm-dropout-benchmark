# MORF access checklist (do these in order)

MORF runs your code on privacy-restricted data; access is the long pole. Start 1–2 now.

- [ ] 1. Inquire (unblocks everything). Email `morf-info@umich.edu`: who you are, supervisor +
      institution, the project (multimodal dropout/completion with leakage-safe text encoding), and ask
      (a) is MORF currently accepting new external/collaborative projects, (b) the DUA template, and
      (c) whether discussion-forum text (not just clickstream) is exposed to feature-extraction.
- [ ] 2. Start the institutional DUA (slow — start in parallel). Loop in your supervisor and your
      university's research/legal/contracts office. Expect weeks. This is the hard prerequisite.
- [ ] 3. Confirm scope with MORF. Forum-table availability + completion/grade label per learner;
      enclave compute limits (DistilBERT encode over many sessions); permitted output (aggregate metrics).
- [ ] 4. Set up the client. `pip install morf-api` on any modern Python; get an account/credentials
      per MORF's instructions; smoke-test with `submit_mwe(email_to=...)` to confirm the pipeline works
      end-to-end on their sample job before building ours.
- [ ] 5. Finalize our job artifacts (drafted in this folder — validate against the live `mwe/`):
      - [ ] `mwe_dropout_text.py` — match the real Coursera forum SQL schema + grade/certification export.
      - [ ] `Dockerfile` — build + push the image to a public registry; host `config.properties` +
            controller at public HTTPS/S3 URLs.
      - [ ] `morf_controller.py` — confirm the `extract_session/train_course/test_course/evaluate_course`
            calls match the current MORF API.
- [ ] 6. Submit. `from morf.utils.submit import easy_submit; easy_submit(client_config_url=..., email_to=...)`.
      Watch for `morf-alerts@umich.edu` status emails.
- [ ] 7. Report honestly. Whatever the real-text lift is (even if small/null), report it as the
      genuine multimodal result and update Phase 9's framing accordingly.

Fallback if MORF access stalls: MOOCCubeX (downloadable, real comments + derivable dropout, but
Chinese → multilingual encoder) is the only no-DUA path to real text+dropout (see `../../DATASETS/README.md`).
