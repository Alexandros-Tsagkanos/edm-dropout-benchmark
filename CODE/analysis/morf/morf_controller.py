#!/usr/bin/env python3
"""
MORF controller (client-side) — DRAFT TEMPLATE.

This is the script MORF runs to orchestrate our job. It calls the MORF workflow API, which pulls our
Docker image (see Dockerfile + mwe_dropout_text.py) and invokes it once per session/course with the
appropriate --mode. It does NOT touch raw data directly (that happens inside the container, in the
enclave).

DRAFT STATUS: the function names below follow MORF's documented API (May 2026); validate against the
live `mwe/` controller in github.com/educational-technology-collective/morf before submitting.

Run order:
  extract_session   -> per-session feature/text extraction (max parallelism)
  train_course      -> one model per course (frozen DistilBERT + train-only PCA + classifier)
  test_course       -> predict on held-out sessions of each course
  evaluate_course   -> AUC / F1 for completion, per course

Submit (from a machine with morf-api installed and MORF access):
  from morf.utils.submit import easy_submit
  easy_submit(client_config_url="https://.../config.properties",
              email_to="you@university.edu")
"""
# from morf.workflow.extract  import extract_session
# from morf.workflow.train    import train_course
# from morf.workflow.test     import test_course
# from morf.workflow.evaluate import evaluate_course

# MORF passes these through to our Docker image; keep the label/lib choices identical to our
# Phase-9 / Stanford protocol so the real-text result is comparable to our synthetic-text numbers.


def main():
    # Each call runs our container with the matching --mode over the mounted /input data.
    # The docker image URL + resource config live in config.properties (see ACCESS_CHECKLIST.md).
    extract_session()      # noqa: F821  -> writes per-learner (text, tabular, completion) to /output
    train_course()         # noqa: F821  -> frozen DistilBERT mean-pool + train-only PCA + LR/RF + fusion
    test_course()          # noqa: F821  -> scores held-out sessions
    evaluate_course()      # noqa: F821  -> AUC/F1 for completion (text-only / tabular / fused)


if __name__ == "__main__":
    main()
