#!/usr/bin/env python3
"""
diag_d6_train_density.py
=========================
D-NEW-6: Training Density Distribution

Pure GT analysis — no model, no GPU required.

Measures distribution of N_concurrent across every 30-frame window
the dataloader could sample during training, and compares against
val sequence density.

Answers: Does the training set lack high-density scenarios?
  → If windows with max_concurrent ≥ 15 are rare (<5%) → training
    underexposure is confirmed → weighted sampler is justified.

Run from repo root:
  python diagnostics/diag_d6_train_density.py \
    --train_dir /data/pos+mot/Datadir/DanceTrack/train \
    --val_dir   /data/pos+mot/Datadir/DanceTrack/val \
    --output_dir diagnostics/diag6_results/
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

WINDOW_SIZE = 30   # matches SAMPLE_LENGTHS in config


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir",  default="/data/pos+mot/Datadir/DanceTrack/train")
    p.add_argument("--val_dir",    default="/data/pos+mot/Datadir/DanceTrack/val")
    p.add_argument("--output_dir", default="diagnostics/diag6_results/")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# GT LOADING
# ─────────────────────────────────────────────────────────────
def load_concurrent(seq_dir: str) -> dict:
    """Returns {frame_id: n_active_objects}. Skips conf=0 rows."""
    gt_path = os.path.join(seq_dir, "gt", "gt.txt")
    counts = defaultdict(set)
    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue
            if int(float(parts[6])) == 0:
                continue
            counts[int(parts[0])].add(int(parts[1]))
    return {fid: len(ids) for fid, ids in counts.items()}


def seq_stats(seq_dir: str) -> dict:
    concurrent = load_concurrent(seq_dir)
    if not concurrent:
        return None
    vals = list(concurrent.values())
    return {
        "mean_concurrent": float(np.mean(vals)),
        "max_concurrent":  int(np.max(vals)),
        "n_frames":        len(vals),
    }


# ─────────────────────────────────────────────────────────────
# WINDOW DENSITY
# ─────────────────────────────────────────────────────────────
def window_density(seq_dir: str) -> list:
    """Returns list of max_concurrent for every 30-frame window."""
    concurrent = load_concurrent(seq_dir)
    if not concurrent:
        return []
    frames = sorted(concurrent.keys())
    if len(frames) < WINDOW_SIZE:
        return []
    return [
        max(concurrent[frames[i + k]] for k in range(WINDOW_SIZE))
        for i in range(len(frames) - WINDOW_SIZE + 1)
    ]


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    args = get_args()
    od = Path(args.output_dir)
    od.mkdir(parents=True, exist_ok=True)

    # ── Load train sequences ──────────────────────────────────────────
    train_seqs = sorted(p for p in Path(args.train_dir).iterdir() if p.is_dir())
    val_seqs   = sorted(p for p in Path(args.val_dir).iterdir()   if p.is_dir())

    train_stats, val_stats = [], []
    all_windows = []

    for seq in train_seqs:
        s = seq_stats(str(seq))
        if s:
            s["sequence"] = seq.name
            train_stats.append(s)
            all_windows.extend(window_density(str(seq)))

    for seq in val_seqs:
        s = seq_stats(str(seq))
        if s:
            s["sequence"] = seq.name
            val_stats.append(s)

    wt = np.array(all_windows)
    train_max  = [s["max_concurrent"]  for s in train_stats]
    train_mean = [s["mean_concurrent"] for s in train_stats]
    val_max    = [s["max_concurrent"]  for s in val_stats]
    val_mean   = [s["mean_concurrent"] for s in val_stats]

    # ── Print report ─────────────────────────────────────────────────
    print("=" * 65)
    print("D-NEW-6 — Training Density Distribution")
    print("=" * 65)
    print(f"\n  Train sequences : {len(train_stats)}")
    print(f"  Val sequences   : {len(val_stats)}")
    print(f"  Total 30-frame windows: {len(wt):,}")

    print(f"\n  Sequence-level max_concurrent:")
    print(f"  {'Split':<8}  {'Mean':>6}  {'Max':>5}  {'≥10':>5}  {'≥15':>5}  {'≥20':>5}")
    print("  " + "-" * 42)
    for split, vals in [("Train", train_max), ("Val", val_max)]:
        n = len(vals)
        print(f"  {split:<8}  {np.mean(vals):>6.1f}  {np.max(vals):>5}  "
              f"{sum(v>=10 for v in vals):>4}/{n}  "
              f"{sum(v>=15 for v in vals):>4}/{n}  "
              f"{sum(v>=20 for v in vals):>4}/{n}")

    print(f"\n  Training window (30-frame) max_concurrent thresholds:")
    print(f"  {'Threshold':<12}  {'Windows':>8}  {'%':>7}")
    print("  " + "-" * 32)
    for t in [5, 8, 10, 12, 15, 18, 20, 25]:
        cnt = int((wt >= t).sum())
        pct = cnt / len(wt) * 100
        print(f"  max >= {t:>3}:    {cnt:>8,}  {pct:>6.2f}%")

    print(f"\n  Window max_concurrent statistics:")
    for label, val in [("Mean",   np.mean(wt)),
                        ("Median", np.median(wt)),
                        ("P90",    np.percentile(wt, 90)),
                        ("P95",    np.percentile(wt, 95)),
                        ("Max",    wt.max())]:
        print(f"    {label:<8}: {val:.1f}")

    # ── Interpretation ────────────────────────────────────────────────
    pct15 = float((wt >= 15).mean() * 100)
    pct20 = float((wt >= 20).mean() * 100)
    print(f"\n  Windows ≥ 15 objects: {pct15:.1f}%")
    print(f"  Windows ≥ 20 objects: {pct20:.1f}%")
    print()
    if pct15 < 5.0:
        print("  FINDING: STRONG training underexposure (Hypothesis C confirmed).")
        print("  IDDecoder trained on <5% high-density windows.")
        print("  → Weighted sequence sampler is strongly justified.")
    elif pct15 < 15.0:
        print("  FINDING: MODERATE underexposure — high-density windows rare.")
        print("  → Weighted sampler would increase exposure, likely to help.")
    else:
        print("  FINDING: High-density windows appear frequently.")
        print("  → Training distribution not the primary bottleneck.")
        print("  → Focus on feature discriminability (projection head).")

    # ── Per-sequence table ────────────────────────────────────────────
    print(f"\n  {'Sequence':<24}  {'max_c':>6}  {'mean_c':>7}")
    print("  " + "-" * 42)
    for s in sorted(train_stats, key=lambda x: -x["max_concurrent"]):
        flag = " ← HIGH" if s["max_concurrent"] >= 15 else ""
        print(f"  {s['sequence']:<24}  {s['max_concurrent']:>6}  "
              f"{s['mean_concurrent']:>7.1f}{flag}")

    # ── Plots ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    bins = range(1, int(wt.max()) + 3)
    ax.hist(wt, bins=bins, color="steelblue", alpha=0.8, edgecolor="black")
    for t, c, lbl in [(10, "red", "N=10"), (15, "orange", "N=15"),
                       (20, "darkred", "N=20")]:
        ax.axvline(t, color=c, ls="--", lw=1.5, label=lbl)
    ax.set_xlabel("max_concurrent in 30-frame window")
    ax.set_ylabel("Count")
    ax.set_title(f"Training window density\n"
                 f"(≥15: {pct15:.1f}%,  ≥20: {pct20:.1f}%)")
    ax.legend()

    ax = axes[1]
    sorted_w = np.sort(wt)
    cdf = np.arange(1, len(sorted_w) + 1) / len(sorted_w)
    ax.plot(sorted_w, 1 - cdf, color="steelblue", lw=2)
    for t, c, lbl in [(10, "red", f"N=10 ({(wt>=10).mean()*100:.1f}%)"),
                       (15, "orange", f"N=15 ({pct15:.1f}%)"),
                       (20, "darkred", f"N=20 ({pct20:.1f}%)")]:
        ax.axvline(t, color=c, ls="--", lw=1.5, label=lbl)
    ax.set_xlabel("max_concurrent threshold")
    ax.set_ylabel("Fraction of windows above threshold")
    ax.set_title("Training window CDF")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    max_both = max(max(train_max), max(val_max))
    bins2 = range(0, max_both + 3, 2)
    ax.hist(train_max, bins=bins2, alpha=0.6, color="steelblue", label="Train")
    ax.hist(val_max,   bins=bins2, alpha=0.6, color="orange",    label="Val")
    ax.set_xlabel("Sequence max_concurrent")
    ax.set_ylabel("Count")
    ax.set_title("Sequence-level density: Train vs Val")
    ax.legend()

    plt.suptitle("D-NEW-6: Training Density Distribution")
    plt.tight_layout()
    plt.savefig(od / "d6_train_density.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved to {od}/d6_train_density.png")

    # ── Save JSON ──────────────────────────────────────────────────────
    result = {
        "n_train_seqs":               len(train_stats),
        "n_val_seqs":                 len(val_stats),
        "n_windows":                  len(wt),
        "window_mean":                float(np.mean(wt)),
        "window_p90":                 float(np.percentile(wt, 90)),
        "window_p95":                 float(np.percentile(wt, 95)),
        "windows_above_10_pct":       float((wt >= 10).mean() * 100),
        "windows_above_15_pct":       pct15,
        "windows_above_20_pct":       pct20,
        "train_max_mean":             float(np.mean(train_max)),
        "val_max_mean":               float(np.mean(val_max)),
        "hypothesis_C_strength":      "strong" if pct15 < 5 else
                                      "moderate" if pct15 < 15 else "weak",
        "per_train_seq":              train_stats,
        "per_val_seq":                val_stats,
    }
    with open(od / "d6_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Results saved to {od}/d6_results.json")


if __name__ == "__main__":
    main()