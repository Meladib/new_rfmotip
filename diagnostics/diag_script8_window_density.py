"""
DIAG 8 — Training Window Density Distribution
==============================================
Pure GT analysis. No model. No GPU. Run from repo root.

Measures the distribution of N_concurrent across every 30-frame window
that the dataloader could sample during training.

The key question: what N_concurrent values does the IDDecoder actually
train on at the window level — not just at the sequence level?

Confirmed context:
- 28/40 train sequences have sequence-level max < 10 (CLAUDE.md)
- But sequence-level max ≠ window-level max
- dancetrack0020: mean=34.6, max=40 — but does it contribute high-N windows?
- If almost no windows have N≥15 → Hypothesis C (training underexposure) is strong

Output:
- Per-threshold window counts (how many windows have max N >= 5, 10, 15, 20, ...)
- Per-sequence breakdown
- Two plots: histogram of window max N and window mean N
- Saved to diagnostics/diag8_outputs/
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

# --- Config ---
TRAIN_DIR = "/data/pos+mot/Datadir/DanceTrack/train"
WINDOW_SIZE = 30          # matches SAMPLE_LENGTHS: [30] in config
OUTPUT_DIR = "diagnostics/diag8_outputs"
STRIDE = 1                # check every possible window start


# ---------------------------------------------------------------------------
# GT loading
# ---------------------------------------------------------------------------

def load_gt(seq_dir: str) -> dict:
    """
    Returns: {frame_id (int): {track_id (int): [x, y, w, h]}}
    DanceTrack GT format: frame, id, left, top, w, h, conf, class, vis
    Only loads rows where conf=1 (active objects).
    """
    gt_path = os.path.join(seq_dir, "gt", "gt.txt")
    gt = defaultdict(dict)
    with open(gt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 7:
                continue
            frame_id = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            conf = int(float(parts[6]))
            if conf == 0:
                continue  # ignore inactive annotations
            gt[frame_id][track_id] = [x, y, w, h]
    return gt


def get_n_concurrent_per_frame(gt: dict) -> dict:
    """Returns {frame_id: n_active_tracks}."""
    return {frame: len(tracks) for frame, tracks in gt.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.isdir(TRAIN_DIR):
        print(f"ERROR: TRAIN_DIR not found: {TRAIN_DIR}")
        sys.exit(1)

    seq_names = sorted([
        s for s in os.listdir(TRAIN_DIR)
        if os.path.isdir(os.path.join(TRAIN_DIR, s))
    ])
    print(f"Found {len(seq_names)} train sequences.")

    all_window_max_n = []    # one entry per window
    all_window_mean_n = []
    per_seq_stats = []

    for seq_name in seq_names:
        seq_dir = os.path.join(TRAIN_DIR, seq_name)
        gt = load_gt(seq_dir)
        if not gt:
            print(f"  WARNING: empty GT for {seq_name}, skipping.")
            continue

        n_per_frame = get_n_concurrent_per_frame(gt)
        frames = sorted(n_per_frame.keys())
        n_frames = len(frames)

        if n_frames < WINDOW_SIZE:
            print(f"  WARNING: {seq_name} has only {n_frames} frames < {WINDOW_SIZE}, skipping.")
            continue

        seq_win_max = []
        seq_win_mean = []

        for start_idx in range(0, n_frames - WINDOW_SIZE + 1, STRIDE):
            window_frames = frames[start_idx: start_idx + WINDOW_SIZE]
            counts = [n_per_frame[f] for f in window_frames]
            seq_win_max.append(max(counts))
            seq_win_mean.append(float(np.mean(counts)))

        all_window_max_n.extend(seq_win_max)
        all_window_mean_n.extend(seq_win_mean)

        per_seq_stats.append({
            "seq":             seq_name,
            "n_windows":       len(seq_win_max),
            "win_max_mean":    float(np.mean(seq_win_max)),
            "win_max_p50":     float(np.percentile(seq_win_max, 50)),
            "win_max_p90":     float(np.percentile(seq_win_max, 90)),
            "win_max_p95":     float(np.percentile(seq_win_max, 95)),
            "seq_global_max":  max(n_per_frame.values()),
            "seq_global_mean": float(np.mean(list(n_per_frame.values()))),
        })

    total_windows = len(all_window_max_n)
    assert total_windows > 0, "No windows collected."

    # ---------------------------------------------------------------------------
    # Print report
    # ---------------------------------------------------------------------------
    print()
    print("=" * 72)
    print("DIAG 8 — Training Window Density Distribution")
    print("=" * 72)
    print(f"Total 30-frame windows across all {len(per_seq_stats)} sequences: {total_windows:,}")

    print()
    print("Window MAX N_concurrent — cumulative >= thresholds:")
    print(f"  {'Threshold':>10}  {'Count':>8}  {'Pct of windows':>16}")
    for thr in [5, 8, 10, 12, 15, 18, 20, 25, 30]:
        cnt = sum(1 for x in all_window_max_n if x >= thr)
        pct = 100.0 * cnt / total_windows
        print(f"  max >= {thr:2d}    {cnt:>8,}  {pct:>15.2f}%")

    print()
    print("Window MEAN N_concurrent — cumulative >= thresholds:")
    print(f"  {'Threshold':>10}  {'Count':>8}  {'Pct of windows':>16}")
    for thr in [5, 8, 10, 12, 15, 20]:
        cnt = sum(1 for x in all_window_mean_n if x >= thr)
        pct = 100.0 * cnt / total_windows
        print(f"  mean >= {thr:2d}   {cnt:>8,}  {pct:>15.2f}%")

    print()
    print("Summary statistics (window max N):")
    for label, val in [
        ("Mean",   np.mean(all_window_max_n)),
        ("Median", np.median(all_window_max_n)),
        ("P75",    np.percentile(all_window_max_n, 75)),
        ("P90",    np.percentile(all_window_max_n, 90)),
        ("P95",    np.percentile(all_window_max_n, 95)),
        ("P99",    np.percentile(all_window_max_n, 99)),
        ("Max",    max(all_window_max_n)),
    ]:
        print(f"  {label:<8}: {val:.2f}")

    print()
    print(f"{'Sequence':<22} {'N_wins':>7} {'WinMaxMean':>11} {'WinMaxP90':>10} "
          f"{'WinMaxP95':>10} {'SeqGlobalMax':>13} {'SeqGlobalMean':>14}")
    print("-" * 90)
    for s in sorted(per_seq_stats, key=lambda x: -x["win_max_mean"]):
        print(f"{s['seq']:<22} {s['n_windows']:>7} {s['win_max_mean']:>11.1f} "
              f"{s['win_max_p90']:>10.1f} {s['win_max_p95']:>10.1f} "
              f"{s['seq_global_max']:>13} {s['seq_global_mean']:>14.1f}")

    # ---------------------------------------------------------------------------
    # Key interpretive question
    # ---------------------------------------------------------------------------
    pct_gte_15 = 100.0 * sum(1 for x in all_window_max_n if x >= 15) / total_windows
    pct_gte_20 = 100.0 * sum(1 for x in all_window_max_n if x >= 20) / total_windows
    print()
    print("INTERPRETATION GUIDE:")
    print(f"  Windows with max N >= 15: {pct_gte_15:.1f}%")
    print(f"  Windows with max N >= 20: {pct_gte_20:.1f}%")
    print()
    if pct_gte_15 < 5.0:
        print("  >> STRONG signal for Hypothesis C (training underexposure).")
        print("  >> IDDecoder has almost never seen N>=15 during training.")
        print("  >> Weighted sampler is strongly motivated.")
    elif pct_gte_15 < 15.0:
        print("  >> MODERATE signal for Hypothesis C.")
        print("  >> High-density windows exist but are rare.")
        print("  >> Weighted sampler would increase their frequency.")
    else:
        print("  >> WEAK signal for Hypothesis C alone.")
        print("  >> High-density windows appear frequently enough.")
        print("  >> Mechanism B (attention dilution) or A (softmax dilution) more likely.")
        print("  >> Check DIAG 5 results.")

    # ---------------------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------------------
    max_n_observed = max(all_window_max_n)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("DIAG 8 — Training Window Density Distribution", fontsize=13)

    # Plot 1: histogram of window max N
    ax = axes[0]
    bins = range(1, max_n_observed + 2)
    ax.hist(all_window_max_n, bins=bins, edgecolor="black", color="steelblue", alpha=0.8)
    for thr, color, label in [(10, "red", "N=10"), (15, "orange", "N=15"), (20, "darkred", "N=20")]:
        ax.axvline(x=thr, color=color, linestyle="--", linewidth=1.5, label=label)
    ax.set_xlabel("Max N_concurrent in 30-frame window", fontsize=11)
    ax.set_ylabel("Number of windows", fontsize=11)
    ax.set_title("Window max N distribution")
    ax.legend()

    # Plot 2: cumulative distribution of window max N
    ax2 = axes[1]
    sorted_max = np.sort(all_window_max_n)
    cdf = np.arange(1, len(sorted_max) + 1) / len(sorted_max)
    # Plot 1 - CDF = fraction of windows with max >= threshold
    ax2.plot(sorted_max, 1 - cdf, color="steelblue", linewidth=2)
    for thr, color, label in [(10, "red", "N=10"), (15, "orange", "N=15"), (20, "darkred", "N=20")]:
        frac = sum(1 for x in all_window_max_n if x >= thr) / total_windows
        ax2.axvline(x=thr, color=color, linestyle="--", linewidth=1.5,
                    label=f"N={thr}: {frac*100:.1f}% of windows")
    ax2.set_xlabel("Max N_concurrent threshold", fontsize=11)
    ax2.set_ylabel("Fraction of windows with max N >= threshold", fontsize=11)
    ax2.set_title("Cumulative: fraction of windows above threshold")
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "window_density_distribution.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved: {out_path}")

    # Save raw data for reference
    np.save(os.path.join(OUTPUT_DIR, "all_window_max_n.npy"), np.array(all_window_max_n))
    np.save(os.path.join(OUTPUT_DIR, "all_window_mean_n.npy"), np.array(all_window_mean_n))
    print(f"Raw arrays saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
