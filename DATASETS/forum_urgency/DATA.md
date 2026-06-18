# Obtaining `Stanford.csv` (not bundled)

`phase9b_realtext.py` evaluates the multimodal text protocol on a redacted derivative of
the **Stanford MOOC Posts** dataset, kept locally as `Stanford.csv`. That file is **not
redistributed in this repository** (the original Stanford data has its own access terms;
see `../stanford_mooc_posts/provenance_stanford.md`). The other files in this folder
(`All_Courses_REDACTED_CODED.csv`, `README.md`, `LICENSE`) are the Penn EDM-2023 package
and are MIT-licensed, so they are included.

To run `phase9b_realtext.py`, recreate `Stanford.csv` here by one of:

1. **From the Švábenský et al. (2023) EDM package** (recommended — this is exactly the
   chain documented in `../stanford_mooc_posts/provenance_stanford.md`): obtain their
   redistributed `Stanford.csv` and place it in this folder unchanged.

2. **From the original Stanford tarball:** download
   `http://infolab.stanford.edu/~paepcke/stanfordMOOCForumPostsSet.tar.gz`
   (verify SHA-256 `a7c1aef6178c80333122e4135fe6cd61329ffc616e5fc16e55c5bfe72117c6ae`),
   extract `stanfordMOOCForumPostsSet.txt`, and reduce it to the four columns the script
   reads: `post_text, Urgency Practice, Urgency_1_7, CourseType` (29,604 rows).

The committed outputs of this step are already in `../../RESULTS/forum_urgency/`, so the
reported numbers can be inspected without re-downloading anything.
