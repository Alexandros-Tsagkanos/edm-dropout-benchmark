#!/usr/bin/env python3
"""
Multiple-comparisons / familywise-error analysis for the multi-seed significance tests.

Answers the reviewer's Q1: "Given p=0.002 for several simultaneous paired-Wilcoxon
comparisons (n=10 seeds, floor 2/2^10=0.00195), what survives a multiple-comparison
adjustment, and does the diffusion-vs-VAE comparison (p=0.027 for RF) survive?"

We apply Holm-Bonferroni (primary; uniformly more powerful than plain Bonferroni and
still controls the familywise error rate) and plain Bonferroni (conservative reference)
*within each dataset's family of tests*. Inputs are the committed multi-seed artifacts;
nothing is recomputed, so this is a pure post-hoc adjustment of the reported p-values.

Reads  (read-only, the repository's canonical multi-seed aggregates):
  ../../RESULTS/realinho/multiseed/aggregate.json  ("significance": [...], key "wilcoxon_p")
  ../../RESULTS/oulad/multiseed/aggregate.json      ("significance": [...], key "p")
Writes:
  ../../RESULTS/analysis/multiple_comparisons_results.csv
"""
from __future__ import annotations
import json
import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ALPHA = 0.05

FAMILIES = {
    "realinho": ROOT / "RESULTS" / "realinho" / "multiseed" / "aggregate.json",
    "oulad":    ROOT / "RESULTS" / "oulad"    / "multiseed" / "aggregate.json",
}


def holm_adjusted(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values, order-preserving with monotonicity."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])   # (m-rank) since rank is 0-based
        adj[idx] = min(running, 1.0)
    return adj


def bonferroni_adjusted(pvals: list[float]) -> list[float]:
    m = len(pvals)
    return [min(m * p, 1.0) for p in pvals]


def load_family(path: Path) -> list[dict]:
    sig = json.loads(path.read_text())["significance"]
    rows = []
    for s in sig:
        # realinho uses wilcoxon_p / mean_diff; oulad uses p / delta
        p = s.get("wilcoxon_p", s.get("p"))
        delta = s.get("mean_diff", s.get("delta"))
        rows.append({"comparison": s["comparison"], "p_raw": float(p),
                     "delta": float(delta), "n": int(s["n"])})
    return rows


def main():
    all_out = []
    for fam, path in FAMILIES.items():
        rows = load_family(path)
        m = len(rows)
        pvals = [r["p_raw"] for r in rows]
        holm = holm_adjusted(pvals)
        bonf = bonferroni_adjusted(pvals)
        print(f"\n=== Family: {fam}  (m = {m} tests, Bonferroni alpha/m = {ALPHA/m:.5f}) ===")
        print(f"{'comparison':46s} {'p_raw':>9s} {'p_holm':>9s} {'p_bonf':>9s}  holm? bonf?")
        for r, ph, pb in sorted(zip(rows, holm, bonf), key=lambda t: t[0]["p_raw"]):
            sh = "yes" if ph <= ALPHA else "no "
            sb = "yes" if pb <= ALPHA else "no "
            print(f"{r['comparison']:46s} {r['p_raw']:9.5f} {ph:9.5f} {pb:9.5f}  {sh:4s} {sb}")
            all_out.append({
                "family": fam, "comparison": r["comparison"], "n": r["n"],
                "delta": round(r["delta"], 5), "p_raw": round(r["p_raw"], 6),
                "p_holm": round(ph, 6), "p_bonferroni": round(pb, 6),
                "survives_holm_0.05": ph <= ALPHA,
                "survives_bonferroni_0.05": pb <= ALPHA,
                "m_family": m,
            })
        n_holm = sum(1 for r, ph in zip(rows, holm) if ph <= ALPHA)
        print(f"  -> {n_holm}/{m} survive Holm at alpha={ALPHA}")

    out_csv = ROOT / "RESULTS" / "analysis" / "multiple_comparisons_results.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_out[0].keys()))
        w.writeheader()
        w.writerows(all_out)
    print(f"\n[wrote] {out_csv}")


if __name__ == "__main__":
    main()
