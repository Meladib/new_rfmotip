#!/usr/bin/env python3
"""
diag_d_new_density_train.py  —  D-NEW-DENSITY-TRAIN
======================================================
Answers: Is the Case B failure rate driven by training distribution
underexposure, or is it purely architectural (feature discriminability)?

Method
------
1. For each val sequence, compute:
     - Case B rate       (from diag_234 output — already have this)
     - Training coverage = fraction of training windows whose
                           max_concurrent is within ±3 of this
                           sequence's mean_concurrent

2. Compute: r(case_b_rate, inverse_training_coverage)
   r > 0.4 → Case B is driven by underexposure → density sampler CONFIRMED
   r < 0.2 → Case B is architecture-driven   → density sampler DROPPED

3. Additional analysis:
   - Stratify val sequences by density tier and compare Case B rates
   - Plot Case B rate vs mean_concurrent with training coverage overlay

Scientific justification
------------------------
If density sampler is the right fix, then sequences whose density is
underrepresented in training should have systematically higher Case B
rates than sequences of similar density that ARE well-represented.
If Case B rate is uniform across coverage levels, the failure is
architectural — more training data at that density won't fix it.

Run from repo root:
  python diagnostics/diag_d_new_density_train.py \\
    --diag234_json  diagnostics/diag234_results/diag234_results.json \\
    --d5_json       diagnostics/diag5_results/d5_results.json \\
    --d6_json       diagnostics/diag6_results/d6_results.json \\
    --output_dir    diagnostics/diag_density_train_results/
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--diag234_json",
                   default="diagnostics/diag234_results/diag234_results.json")
    p.add_argument("--d5_json",
                   default="diagnostics/diag5_results/d5_results.json")
    p.add_argument("--d6_json",
                   default="diagnostics/diag6_results/d6_results.json")
    p.add_argument("--output_dir",
                   default="diagnostics/diag_density_train_results/")
    p.add_argument("--coverage_window", type=float, default=3.0,
                   help="±N objects window for coverage matching")
    return p.parse_args()


def main():
    args = get_args()
    od = Path(args.output_dir)
    od.mkdir(parents=True, exist_ok=True)

    # ── Load DIAG 234 per-sequence Case B rates ───────────────────────
    with open(args.diag234_json) as f:
        diag234 = json.load(f)

    per_seq_d4 = {r["sequence"]: r for r in diag234["diag4"]["per_sequence"]}
    print(f"Loaded DIAG 4 data for {len(per_seq_d4)} sequences")

    # ── Load D-NEW-5 per-sequence HOTA + density ──────────────────────
    with open(args.d5_json) as f:
        d5 = json.load(f)

    per_seq_d5 = {r["sequence"]: r for r in d5["per_sequence"]}
    print(f"Loaded D-NEW-5 data for {len(per_seq_d5)} sequences")

    # ── Load D-NEW-6 training window density distribution ─────────────
    with open(args.d6_json) as f:
        d6 = json.load(f)

    train_seqs    = d6["per_train_seq"]
    n_windows_total = d6["n_windows"]

    # Build training window density distribution
    # We need the full window distribution — recompute from training seq stats
    # Approximate: for each training sequence, approximate window distribution
    # as uniform between max_concurrent ± stdev (simplified for this analysis)
    # More accurately: use the per-sequence mean and max to weight
    print(f"Training sequences: {len(train_seqs)}, total windows: {n_windows_total}")

    # Build continuous training density distribution as list of (mean_concurrent, n_windows)
    train_density_dist = []
    for s in train_seqs:
        # Estimate number of windows from sequence length and window size
        n_frames = s["n_frames"]
        n_wins   = max(0, n_frames - 30 + 1)  # stride-1 windows of size 30
        mean_c   = s["mean_concurrent"]
        # Weight by approximate window count (each window has ~mean_concurrent objects)
        train_density_dist.extend([mean_c] * n_wins)

    train_density_arr = np.array(train_density_dist, dtype=np.float32)
    print(f"Reconstructed {len(train_density_arr)} training window density estimates\n")

    def training_coverage(target_density: float, window: float) -> float:
        """Fraction of training windows within ±window of target_density."""
        mask = (train_density_arr >= target_density - window) & \
               (train_density_arr <= target_density + window)
        return float(mask.mean())

    # ── Build analysis table ──────────────────────────────────────────
    rows = []
    for seq_name, d4_data in per_seq_d4.items():
        if seq_name not in per_seq_d5:
            continue

        d5_data = per_seq_d5[seq_name]
        total_nb = d4_data["breakdown"]["total"]
        case_b_n = d4_data["breakdown"]["case_B"]
        case_b_pct = d4_data["breakdown"]["case_B_pct"]

        mean_conc = d5_data["mean_concurrent"]
        max_conc  = d5_data["max_concurrent"]
        hota      = d5_data["HOTA"]
        assa      = d5_data["AssA"]

        coverage = training_coverage(mean_conc, args.coverage_window)
        inv_cov  = 1.0 / (coverage + 1e-6)   # inverse coverage as proxy for underexposure

        rows.append({
            "sequence":       seq_name,
            "mean_concurrent": mean_conc,
            "max_concurrent":  max_conc,
            "HOTA":            hota,
            "AssA":            assa,
            "total_spurious":  total_nb,
            "case_B_n":        case_b_n,
            "case_B_pct":      case_b_pct,
            "train_coverage":  coverage,
            "inv_coverage":    inv_cov,
        })

    rows.sort(key=lambda r: r["mean_concurrent"])

    # ── Correlation analysis ──────────────────────────────────────────
    cb_rates  = np.array([r["case_B_pct"]    for r in rows])
    inv_covs  = np.array([r["inv_coverage"]  for r in rows])
    mean_concs = np.array([r["mean_concurrent"] for r in rows])
    hotas      = np.array([r["HOTA"]          for r in rows])

    valid = ~np.isnan(cb_rates) & ~np.isnan(inv_covs)

    r_cb_invcov  = float(np.corrcoef(cb_rates[valid], inv_covs[valid])[0, 1])
    r_hota_cov   = float(np.corrcoef(hotas[valid],    inv_covs[valid])[0, 1])
    r_cb_density = float(np.corrcoef(cb_rates[valid], mean_concs[valid])[0, 1])

    # ── Print report ──────────────────────────────────────────────────
    print("=" * 65)
    print("D-NEW-DENSITY-TRAIN — Results")
    print("=" * 65)
    print(f"\n  r(Case_B_pct, 1/train_coverage):  {r_cb_invcov:.3f}")
    print(f"  r(HOTA,       1/train_coverage):  {r_hota_cov:.3f}")
    print(f"  r(Case_B_pct, mean_concurrent):   {r_cb_density:.3f}")

    print(f"\n  {'Sequence':<24} {'mean_c':>7} {'cov%':>6} "
          f"{'CaseB%':>8} {'HOTA':>7}")
    print("  " + "-" * 60)
    for r in rows:
        flag = " ←" if r["mean_concurrent"] >= 12 else ""
        print(f"  {r['sequence']:<24} {r['mean_concurrent']:>7.1f} "
              f"{r['train_coverage']*100:>5.1f}% "
              f"{r['case_B_pct']:>7.1f}% "
              f"{r['HOTA']:>7.3f}{flag}")

    # Density tier analysis
    print(f"\n  Density tier analysis:")
    tiers = [(0, 6, "low"), (6, 12, "medium"), (12, 100, "high")]
    for lo, hi, name in tiers:
        tier_rows = [r for r in rows if lo <= r["mean_concurrent"] < hi]
        if not tier_rows:
            continue
        avg_cb  = np.mean([r["case_B_pct"]   for r in tier_rows])
        avg_cov = np.mean([r["train_coverage"] for r in tier_rows]) * 100
        avg_hota = np.mean([r["HOTA"]          for r in tier_rows])
        print(f"    {name:<8} (mean_c {lo}-{hi}): "
              f"n={len(tier_rows):>2}  "
              f"CaseB={avg_cb:.1f}%  "
              f"coverage={avg_cov:.1f}%  "
              f"HOTA={avg_hota:.3f}")

    # ── Verdict ───────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("D-NEW-DENSITY-TRAIN VERDICT")
    print(f"{'='*65}")

    if r_cb_invcov > 0.4:
        verdict = "PASS"
        conclusion = (
            f"r(CaseB, 1/coverage) = {r_cb_invcov:.3f} > 0.4. "
            "Case B rate IS driven by training underexposure. "
            "Sequences with low training coverage have systematically "
            "higher Case B rates. Density-adaptive sampler (V4b) CONFIRMED."
        )
    elif r_cb_invcov > 0.2:
        verdict = "MARGINAL"
        conclusion = (
            f"r(CaseB, 1/coverage) = {r_cb_invcov:.3f} — moderate. "
            "Training underexposure partially explains Case B rates. "
            "Density sampler may help but effect will be modest. "
            "Consider combining with projection head (V4c) for full fix."
        )
    else:
        verdict = "FAIL"
        conclusion = (
            f"r(CaseB, 1/coverage) = {r_cb_invcov:.3f} < 0.2. "
            "Case B rate is NOT driven by training underexposure. "
            "The failure is architectural — feature discriminability. "
            "DROP density sampler (V4b). Focus on re-ID projection head (V4c)."
        )

    print(f"\n  r(CaseB, 1/coverage):  {r_cb_invcov:.3f}")
    print(f"  RESULT: {verdict}")
    print(f"  {conclusion}")

    # ── Plots ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    sc = ax.scatter(inv_covs[valid], cb_rates[valid],
                    c=mean_concs[valid], cmap="RdYlGn_r", s=80,
                    edgecolors="k", linewidths=0.5)
    plt.colorbar(sc, ax=ax, label="mean_concurrent")
    if valid.sum() > 2:
        z = np.polyfit(inv_covs[valid], cb_rates[valid], 1)
        xr = np.linspace(inv_covs[valid].min(), inv_covs[valid].max(), 50)
        ax.plot(xr, np.polyval(z, xr), "r--", lw=1.5, alpha=0.7)
    ax.set_xlabel("1 / training coverage (underexposure)")
    ax.set_ylabel("Case B %")
    ax.set_title(f"Case B vs Underexposure\nr={r_cb_invcov:.3f}")
    for r in rows:
        ax.annotate(r["sequence"][-4:],
                    (r["inv_coverage"], r["case_B_pct"]),
                    fontsize=6, alpha=0.7)

    ax = axes[1]
    sc2 = ax.scatter(mean_concs[valid], cb_rates[valid],
                     c=[r["train_coverage"] for r in rows],
                     cmap="RdYlGn", s=80, edgecolors="k", linewidths=0.5)
    plt.colorbar(sc2, ax=ax, label="Training coverage")
    ax.set_xlabel("Val sequence mean_concurrent")
    ax.set_ylabel("Case B %")
    ax.set_title(f"Case B vs Scene Density\nr={r_cb_density:.3f}")

    ax = axes[2]
    tier_names  = ["Low\n(0-6)", "Medium\n(6-12)", "High\n(12+)"]
    tier_cb     = []
    tier_hota   = []
    tier_cov    = []
    for lo, hi, _ in tiers:
        tr = [r for r in rows if lo <= r["mean_concurrent"] < hi]
        tier_cb.append(np.mean([r["case_B_pct"]    for r in tr]) if tr else 0)
        tier_hota.append(np.mean([r["HOTA"]         for r in tr]) if tr else 0)
        tier_cov.append(np.mean([r["train_coverage"] for r in tr]) * 100 if tr else 0)

    x = np.arange(3)
    ax.bar(x - 0.2, tier_cb,   0.35, label="Case B %",       color="orange", alpha=0.8)
    ax.bar(x + 0.2, tier_cov,  0.35, label="Train cov %",    color="steelblue", alpha=0.8)
    ax2 = ax.twinx()
    ax2.plot(x, tier_hota, "rD--", ms=8, lw=2, label="HOTA")
    ax.set_xticks(x)
    ax.set_xticklabels(tier_names)
    ax.set_ylabel("% value")
    ax2.set_ylabel("HOTA", color="red")
    ax.set_title("Density Tier: CaseB / Coverage / HOTA")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)

    plt.suptitle(f"D-NEW-DENSITY-TRAIN: {verdict}\n"
                 f"r(CaseB, 1/coverage)={r_cb_invcov:.3f}  "
                 f"r(CaseB, density)={r_cb_density:.3f}", fontsize=12)
    plt.tight_layout()
    plt.savefig(od / "d_new_density_train.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Save JSON ──────────────────────────────────────────────────────
    result = {
        "r_caseb_inv_coverage":    round(r_cb_invcov, 4),
        "r_hota_inv_coverage":     round(r_hota_cov,  4),
        "r_caseb_mean_concurrent": round(r_cb_density, 4),
        "pass_threshold":          0.4,
        "marginal_threshold":      0.2,
        "verdict":                 verdict,
        "conclusion":              conclusion,
        "per_sequence":            rows,
    }
    with open(od / "d_new_density_train_results.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved to {od}/")
    print("  d_new_density_train.png")
    print("  d_new_density_train_results.json")


if __name__ == "__main__":
    main()