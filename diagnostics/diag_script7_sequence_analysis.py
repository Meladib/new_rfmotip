"""
DIAG 7 — Sequence Characteristic Analysis (GT only)
=====================================================
Pure GT analysis. No model. No GPU. Run from repo root.

Confirmed anomaly (FULL_DIAGNOSTIC_REPORT.md Part 11):
  dancetrack0081: mean_concurrent=19.1, correct_mean=0.967  ← outperforms density expectation
  dancetrack0041: mean_concurrent=16.7, correct_mean=0.882  ← underperforms density expectation
  Motion is NOT the differentiator: 0081=3.7px, 0041=4.6px (similar)

The open question: what structural GT property explains why 0081 is an outlier?

This script computes per-sequence:
  1. Trajectory crossing frequency  — how often do GT boxes overlap (IoU > 0)?
  2. Mean minimum inter-object distance — how spatially close are objects on average?
  3. Track length distribution       — short tracks = more true newborns = harder ID task
  4. Occlusion proxy                 — fraction of frames where one box center is inside another box
  5. Identity ambiguity score        — mean fraction of frames where an object has a near-duplicate
                                       neighbor (bbox overlap > 0 with another object)

Runs on all 25 val sequences so we can rank them and check whether 0081 and 0041
are outliers on any of these dimensions relative to their density cohort.

Output:
  - Full per-sequence table printed to stdout
  - Scatter plots: crossing_freq vs correct_mean, distance vs correct_mean
  - Saved to diagnostics/diag7_outputs/
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

# Sequences to highlight in diagnostic text
HIGHLIGHT_SEQS = {
    "dancetrack0041": {"correct_mean": 0.882, "mean_concurrent": 16.7},
    "dancetrack0081": {"correct_mean": 0.967, "mean_concurrent": 19.1},
    "dancetrack0026": {"correct_mean": 0.854, "mean_concurrent": 13.2},
    "dancetrack0097": {"correct_mean": 0.994, "mean_concurrent": 3.9},
    "dancetrack0079": {"correct_mean": 0.980, "mean_concurrent": 12.1},
    "dancetrack0094": {"correct_mean": 0.941, "mean_concurrent": 19.5},
}

# From diag_script4_score_distribution.py results (FULL_DIAGNOSTIC_REPORT Part 10)
CORRECT_MEAN_FROM_DIAG4 = {
    "dancetrack0026": 0.884,
    "dancetrack0041": 0.887,
    "dancetrack0090": 0.933,
    "dancetrack0043": 0.935,
    "dancetrack0094": 0.941,
    "dancetrack0034": 0.944,
    "dancetrack0063": 0.948,
    "dancetrack0014": 0.950,
    "dancetrack0047": 0.954,
    "dancetrack0073": 0.958,
    "dancetrack0081": 0.967,
    "dancetrack0004": 0.968,
    "dancetrack0035": 0.972,
    "dancetrack0058": 0.976,
    "dancetrack0079": 0.980,
    "dancetrack0025": 0.980,
    "dancetrack0019": 0.980,
    "dancetrack0007": 0.987,
    "dancetrack0065": 0.989,
    "dancetrack0010": 0.989,
    "dancetrack0077": 0.991,
    "dancetrack0030": 0.994,
    "dancetrack0097": 0.995,
    "dancetrack0005": 0.996,
    "dancetrack0018": 0.998,
}

VAL_DIR = "/data/pos+mot/Datadir/DanceTrack/val"
OUTPUT_DIR = "diagnostics/diag7_outputs"


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def iou(box_a, box_b):
    """box_a, box_b: [x1, y1, x2, y2]"""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def center(box_xywh):
    x, y, w, h = box_xywh
    return (x + w / 2.0, y + h / 2.0)


def euclidean(c1, c2):
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5


def center_inside_box(point, box_xyxy):
    px, py = point
    x1, y1, x2, y2 = box_xyxy
    return x1 <= px <= x2 and y1 <= py <= y2


# ---------------------------------------------------------------------------
# GT loading
# ---------------------------------------------------------------------------

def load_gt(seq_dir: str):
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
                continue
            gt[frame_id][track_id] = [x, y, w, h]
    return gt


# ---------------------------------------------------------------------------
# Per-sequence metrics
# ---------------------------------------------------------------------------

def compute_seq_metrics(gt: dict, seq_name: str) -> dict:
    frames = sorted(gt.keys())
    n_frames = len(frames)

    if n_frames == 0:
        return None

    # --- 1. N_concurrent per frame ---
    n_concurrent = [len(gt[f]) for f in frames]
    mean_concurrent = float(np.mean(n_concurrent))

    # --- 2. Crossing frequency (IoU > 0 between any pair) ---
    # Per frame: how many pairs of objects have overlapping boxes?
    crossing_events_per_frame = []
    for f in frames:
        tracks = list(gt[f].values())
        n = len(tracks)
        if n < 2:
            crossing_events_per_frame.append(0)
            continue
        pairs_with_overlap = 0
        for i in range(n):
            for j in range(i + 1, n):
                box_a = xywh_to_xyxy(tracks[i])
                box_b = xywh_to_xyxy(tracks[j])
                if iou(box_a, box_b) > 0.0:
                    pairs_with_overlap += 1
        crossing_events_per_frame.append(pairs_with_overlap)

    total_pairs_per_frame = [
        len(gt[f]) * (len(gt[f]) - 1) / 2 for f in frames if len(gt[f]) >= 2
    ]
    mean_crossing_pairs = float(np.mean(crossing_events_per_frame))
    # Normalize by mean number of possible pairs (controls for density)
    norm_crossing = (
        float(np.mean(crossing_events_per_frame) / np.mean(total_pairs_per_frame))
        if np.mean(total_pairs_per_frame) > 0 else 0.0
    )

    # --- 3. Mean minimum inter-object distance (pixel centers) ---
    min_dists_per_frame = []
    for f in frames:
        tracks = list(gt[f].values())
        n = len(tracks)
        if n < 2:
            continue
        centers = [center(t) for t in tracks]
        frame_min_dists = []
        for i in range(n):
            dists = [euclidean(centers[i], centers[j]) for j in range(n) if j != i]
            frame_min_dists.append(min(dists))
        min_dists_per_frame.append(float(np.mean(frame_min_dists)))

    mean_min_dist = float(np.mean(min_dists_per_frame)) if min_dists_per_frame else float("inf")

    # --- 4. Occlusion proxy ---
    # Fraction of (object, frame) instances where the object's center
    # is inside another object's box.
    occluded_count = 0
    total_object_frames = 0
    for f in frames:
        track_ids = list(gt[f].keys())
        boxes_xyxy = {tid: xywh_to_xyxy(gt[f][tid]) for tid in track_ids}
        centers_dict = {tid: center(gt[f][tid]) for tid in track_ids}
        for tid in track_ids:
            total_object_frames += 1
            c = centers_dict[tid]
            for other_tid in track_ids:
                if other_tid == tid:
                    continue
                if center_inside_box(c, boxes_xyxy[other_tid]):
                    occluded_count += 1
                    break  # only count once per object per frame

    occlusion_rate = occluded_count / total_object_frames if total_object_frames > 0 else 0.0

    # --- 5. Track length distribution ---
    # Build per-track frame lists
    track_frames = defaultdict(list)
    for f in frames:
        for tid in gt[f]:
            track_frames[tid].append(f)
    track_lengths = [len(v) for v in track_frames.values()]
    mean_track_len = float(np.mean(track_lengths)) if track_lengths else 0.0
    short_track_frac = float(np.mean([1.0 if l <= 30 else 0.0 for l in track_lengths]))
    n_unique_tracks = len(track_frames)

    # --- 6. Identity ambiguity score ---
    # Per frame, for each object, count whether it has at least one neighbor
    # with IoU > 0. Average this binary indicator across all object-frames.
    ambiguous_count = 0
    total_obj_frames = 0
    for f in frames:
        tracks = list(gt[f].items())  # [(tid, xywh), ...]
        n = len(tracks)
        for i in range(n):
            total_obj_frames += 1
            box_i = xywh_to_xyxy(tracks[i][1])
            has_overlap = False
            for j in range(n):
                if j == i:
                    continue
                box_j = xywh_to_xyxy(tracks[j][1])
                if iou(box_i, box_j) > 0.0:
                    has_overlap = True
                    break
            if has_overlap:
                ambiguous_count += 1

    ambiguity_rate = ambiguous_count / total_obj_frames if total_obj_frames > 0 else 0.0

    return {
        "seq":                seq_name,
        "n_frames":           n_frames,
        "n_unique_tracks":    n_unique_tracks,
        "mean_concurrent":    mean_concurrent,
        "mean_crossing_pairs": mean_crossing_pairs,
        "norm_crossing":      norm_crossing,       # normalized by possible pairs
        "mean_min_dist_px":   mean_min_dist,
        "occlusion_rate":     occlusion_rate,
        "mean_track_len":     mean_track_len,
        "short_track_frac":   short_track_frac,    # fraction with len <= 30
        "ambiguity_rate":     ambiguity_rate,
        "correct_mean":       CORRECT_MEAN_FROM_DIAG4.get(seq_name, float("nan")),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.isdir(VAL_DIR):
        print(f"ERROR: VAL_DIR not found: {VAL_DIR}")
        sys.exit(1)

    seq_names = sorted([
        s for s in os.listdir(VAL_DIR)
        if os.path.isdir(os.path.join(VAL_DIR, s))
    ])
    print(f"Found {len(seq_names)} val sequences.")

    results = []
    for seq_name in seq_names:
        seq_dir = os.path.join(VAL_DIR, seq_name)
        gt = load_gt(seq_dir)
        if not gt:
            print(f"  WARNING: empty GT for {seq_name}")
            continue
        print(f"  Processing {seq_name} ({len(gt)} frames)...")
        metrics = compute_seq_metrics(gt, seq_name)
        if metrics:
            results.append(metrics)

    if not results:
        print("No results collected.")
        sys.exit(1)

    # Sort by correct_mean ascending (worst sequences first)
    results.sort(key=lambda x: x["correct_mean"])

    # ---------------------------------------------------------------------------
    # Print report
    # ---------------------------------------------------------------------------
    print()
    print("=" * 100)
    print("DIAG 7 — Sequence Characteristic Analysis")
    print("=" * 100)
    print()
    header = (f"{'Sequence':<22} {'CorrMean':>8} {'MeanConc':>9} {'CrossFrq':>9} "
              f"{'NormCross':>10} {'MinDist':>8} {'OcclRate':>9} "
              f"{'TrkLen':>7} {'ShortFrac':>10} {'AmbigRate':>10}")
    print(header)
    print("-" * 105)

    for r in results:
        marker = " *" if r["seq"] in HIGHLIGHT_SEQS else "  "
        print(
            f"{r['seq']:<22}{marker} {r['correct_mean']:>6.3f}   {r['mean_concurrent']:>7.1f}   "
            f"{r['mean_crossing_pairs']:>7.2f}   {r['norm_crossing']:>8.4f}   "
            f"{r['mean_min_dist_px']:>6.1f}   {r['occlusion_rate']:>7.4f}   "
            f"{r['mean_track_len']:>5.1f}   {r['short_track_frac']:>8.3f}   "
            f"{r['ambiguity_rate']:>8.4f}"
        )

    print()
    print("* = highlighted sequence (0041, 0081, 0026, 0097, 0079, 0094)")
    print()

    # ---------------------------------------------------------------------------
    # 0041 vs 0081 focused comparison
    # ---------------------------------------------------------------------------
    r0041 = next((r for r in results if r["seq"] == "dancetrack0041"), None)
    r0081 = next((r for r in results if r["seq"] == "dancetrack0081"), None)

    if r0041 and r0081:
        print("=" * 60)
        print("FOCUSED COMPARISON: dancetrack0041 vs dancetrack0081")
        print("=" * 60)
        metrics_to_compare = [
            ("correct_mean",       "IDDecoder correct score"),
            ("mean_concurrent",    "Mean concurrent objects"),
            ("mean_crossing_pairs","Mean crossing pairs/frame"),
            ("norm_crossing",      "Normalized crossing rate"),
            ("mean_min_dist_px",   "Mean min inter-obj dist (px)"),
            ("occlusion_rate",     "Occlusion rate"),
            ("mean_track_len",     "Mean track length (frames)"),
            ("short_track_frac",   "Fraction of short tracks (<=30fr)"),
            ("ambiguity_rate",     "Ambiguity rate (any IoU>0 neighbor)"),
            ("n_unique_tracks",    "Unique tracks"),
        ]
        print(f"  {'Metric':<40} {'0041':>10} {'0081':>10}  {'Diff (0081-0041)':>18}")
        print("-" * 82)
        for key, label in metrics_to_compare:
            v0041 = r0041[key]
            v0081 = r0081[key]
            diff = v0081 - v0041 if isinstance(v0041, float) else v0081 - v0041
            print(f"  {label:<40} {v0041:>10.4f} {v0081:>10.4f}  {diff:>+18.4f}")

        print()
        print("INTERPRETATION:")
        # Crossing frequency
        if r0081["norm_crossing"] < r0041["norm_crossing"] * 0.7:
            print("  >> 0081 has significantly LOWER normalized crossing rate than 0041.")
            print("     This supports: 0081 objects are spatially segregated despite high count.")
            print("     0041 has more trajectory intersections → harder identity discrimination.")
        elif r0041["norm_crossing"] < r0081["norm_crossing"] * 0.7:
            print("  >> 0041 has significantly LOWER crossing rate — crossing is NOT the differentiator.")
        else:
            print("  >> Crossing frequency is similar — not the primary differentiator.")

        # Min distance
        if r0081["mean_min_dist_px"] > r0041["mean_min_dist_px"] * 1.2:
            print("  >> 0081 objects are MORE spatially separated (higher min dist).")
            print("     Consistent with spatial segregation hypothesis.")
        elif r0041["mean_min_dist_px"] > r0081["mean_min_dist_px"] * 1.2:
            print("  >> 0041 objects are more spread out — distance is NOT the differentiator.")

        # Ambiguity rate
        if r0041["ambiguity_rate"] > r0081["ambiguity_rate"] * 1.2:
            print("  >> 0041 has higher ambiguity rate (more objects with overlapping neighbors).")
            print("     This is the likely primary cause of the 0041 degradation.")
        elif r0081["ambiguity_rate"] > r0041["ambiguity_rate"] * 1.2:
            print("  >> 0081 has higher ambiguity rate — ambiguity is NOT the differentiator.")

    # ---------------------------------------------------------------------------
    # Correlation analysis
    # ---------------------------------------------------------------------------
    seqs_with_data = [r for r in results if not np.isnan(r["correct_mean"])]
    if len(seqs_with_data) >= 5:
        correct_means = np.array([r["correct_mean"] for r in seqs_with_data])
        print()
        print("Pearson r (correct_mean vs metric) across all val sequences with data:")
        for key, label in [
            ("mean_concurrent",    "mean_concurrent"),
            ("norm_crossing",      "norm_crossing"),
            ("mean_min_dist_px",   "mean_min_dist_px"),
            ("occlusion_rate",     "occlusion_rate"),
            ("ambiguity_rate",     "ambiguity_rate"),
            ("short_track_frac",   "short_track_frac"),
            ("mean_track_len",     "mean_track_len"),
        ]:
            vals = np.array([r[key] for r in seqs_with_data])
            if np.std(vals) < 1e-9:
                r_val = float("nan")
            else:
                r_val = float(np.corrcoef(vals, correct_means)[0, 1])
            print(f"  {label:<30}: r = {r_val:+.3f}")
        print()
        print("  (r > 0 → metric correlates with better performance)")
        print("  (r < 0 → metric correlates with worse performance)")
        print("  Compare: density vs correct_mean is already confirmed at r=-0.636")

    # ---------------------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("DIAG 7 — Sequence GT Characteristics vs IDDecoder Performance", fontsize=13)

    plot_pairs = [
        ("mean_concurrent",    "Mean concurrent objects",    axes[0, 0]),
        ("norm_crossing",      "Normalized crossing rate",   axes[0, 1]),
        ("mean_min_dist_px",   "Mean min inter-obj dist (px)", axes[0, 2]),
        ("occlusion_rate",     "Occlusion rate",             axes[1, 0]),
        ("ambiguity_rate",     "Ambiguity rate",             axes[1, 1]),
        ("short_track_frac",   "Short track fraction",       axes[1, 2]),
    ]

    highlight_colors = {
        "dancetrack0041": "red",
        "dancetrack0081": "green",
        "dancetrack0026": "orange",
        "dancetrack0097": "blue",
        "dancetrack0079": "purple",
        "dancetrack0094": "brown",
    }

    seqs_with_data = [r for r in results if not np.isnan(r["correct_mean"])]

    for key, xlabel, ax in plot_pairs:
        x = [r[key] for r in seqs_with_data]
        y = [r["correct_mean"] for r in seqs_with_data]
        colors = [highlight_colors.get(r["seq"], "gray") for r in seqs_with_data]

        ax.scatter(x, y, c=colors, alpha=0.7, s=60)

        for r in seqs_with_data:
            if r["seq"] in highlight_colors:
                ax.annotate(
                    r["seq"].replace("dancetrack", "dt"),
                    (r[key], r["correct_mean"]),
                    fontsize=7, ha="left", va="bottom"
                )

        # Trend line
        if np.std(x) > 1e-9:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_sorted = sorted(x)
            ax.plot(x_sorted, p(x_sorted), "k--", linewidth=1, alpha=0.5)
            r_val = np.corrcoef(x, y)[0, 1]
            ax.set_title(f"{xlabel}\n(r={r_val:+.3f})", fontsize=9)
        else:
            ax.set_title(xlabel, fontsize=9)

        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("IDDecoder correct_mean", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "sequence_characteristics.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved: {out_path}")
    print("Colors: red=0041, green=0081, orange=0026, blue=0097, purple=0079, brown=0094")


if __name__ == "__main__":
    main()
