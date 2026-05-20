#!/usr/bin/env python3
"""
diag_d5_density_vs_hota.py
============================
D-NEW-5: Per-Sequence HOTA vs GT Density Correlation

Answers: Is the model systematically worse on dense sequences?
  r(HOTA, max_concurrent) < -0.5 → density is a confirmed bottleneck
  r(HOTA, max_concurrent) > -0.3 → density is not the primary cause

Requires:
  - A per-sequence HOTA results file (from TrackEval detailed output)
    OR passing --eval_dir pointing to the eval_during_train folder
  - Val GT annotations for density computation

Run from repo root:
  python diagnostics/diag_d5_density_vs_hota.py \
    --eval_dir outputs/rfmotip_dancetrack_ctsv/eval_during_train/ \
    --val_dir  /data/pos+mot/Datadir/DanceTrack/val \
    --epoch    7 \
    --output_dir diagnostics/diag5_results/

  OR provide a per-sequence JSON directly:
  python diagnostics/diag_d5_density_vs_hota.py \
    --per_seq_json diagnostics/per_seq_hota.json \
    --val_dir      /data/pos+mot/Datadir/DanceTrack/val \
    --output_dir   diagnostics/diag5_results/
"""

import os
import sys
import json
import argparse
import csv
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def get_args():
    p = argparse.ArgumentParser()
    # Two input options:
    p.add_argument("--eval_dir",     default=None,
                   help="eval_during_train dir containing epoch subdirs")
    p.add_argument("--epoch",        type=int, default=None,
                   help="Epoch number to use from eval_dir")
    p.add_argument("--per_seq_json", default=None,
                   help="JSON: {seq_name: {HOTA, AssA, DetA}}")
    # Required:
    p.add_argument("--val_dir",      default="/data/pos+mot/Datadir/DanceTrack/val")
    p.add_argument("--output_dir",   default="diagnostics/diag5_results/")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# GT DENSITY
# ─────────────────────────────────────────────────────────────
def load_density(seq_dir: str) -> dict:
    counts = defaultdict(set)
    gt_path = os.path.join(seq_dir, "gt", "gt.txt")
    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 7 or int(float(parts[6])) == 0:
                continue
            counts[int(parts[0])].add(int(parts[1]))
    vals = [len(v) for v in counts.values()]
    if not vals:
        return {}
    return {
        "mean_concurrent": float(np.mean(vals)),
        "max_concurrent":  int(np.max(vals)),
        "n_frames":        len(vals),
    }


# ─────────────────────────────────────────────────────────────
# HOTA LOADING — from eval_during_train or JSON
# ─────────────────────────────────────────────────────────────
def load_from_eval_dir(eval_dir: str, epoch: int) -> dict:
    """
    Parses per-sequence HOTA from eval_during_train directory.
    Tries pedestrian_detailed.csv first, then any per-sequence txt files.
    Returns {seq_name: {HOTA, AssA, DetA}}.
    """
    # Find epoch directory — try several naming conventions
    epoch_dir = None
    for candidate in [
        os.path.join(eval_dir, f"epoch_{epoch}"),
        os.path.join(eval_dir, f"epoch_{epoch:02d}"),
        os.path.join(eval_dir, str(epoch)),
    ]:
        if os.path.isdir(candidate):
            epoch_dir = candidate
            break
    if epoch_dir is None:
        candidates = glob.glob(os.path.join(eval_dir, f"*{epoch}*"))
        if candidates:
            epoch_dir = sorted(candidates)[0]
    if epoch_dir is None:
        # Try root eval_dir directly
        epoch_dir = eval_dir

    print(f"  Using eval dir: {epoch_dir}")

    # ── Strategy 1: pedestrian_detailed.csv ──────────────────────────
    csv_files = glob.glob(
        os.path.join(epoch_dir, "**", "pedestrian_detailed.csv"), recursive=True)
    if csv_files:
        print(f"  Found: {csv_files[0]}")
        per_seq = {}
        with open(csv_files[0]) as f:
            # Sniff delimiter and header
            sample = f.read(2048); f.seek(0)
            dialect = csv.Sniffer().sniff(sample)
            reader = csv.DictReader(f, dialect=dialect)
            for row in reader:
                seq = (row.get("seq") or row.get("Sequence") or
                       row.get("sequence") or "").strip()
                if not seq or seq.lower() in ("", "combined", "average",
                                               "pedestrian"):
                    continue
                try:
                    # TrackEval detailed CSV uses HOTA___50 (IoU=0.5 threshold)
                    # Fall back to HOTA if present, then HOTA___50
                    def _get(row, *keys):
                        for k in keys:
                            v = row.get(k, "")
                            if v.strip() not in ("", "nan"):
                                try: return float(v)
                                except ValueError: pass
                        return float("nan")
                    per_seq[seq] = {
                        "HOTA": _get(row, "HOTA", "HOTA___50"),
                        "AssA": _get(row, "AssA", "AssA___50"),
                        "DetA": _get(row, "DetA", "DetA___50"),
                    }
                except (ValueError, KeyError):
                    pass
        if per_seq:
            print(f"  Parsed {len(per_seq)} sequences from CSV.")
            return per_seq

    # ── Strategy 2: one txt file per sequence (TrackEval plain output) ─
    # Some setups write per-sequence scores as individual .txt files
    per_seq_txts = glob.glob(
        os.path.join(epoch_dir, "**", "dancetrack*.txt"), recursive=True)
    if per_seq_txts:
        print(f"  Found {len(per_seq_txts)} per-sequence txt files.")
        per_seq = {}
        for path in per_seq_txts:
            seq_name = os.path.basename(path).replace(".txt", "")
            try:
                with open(path) as f:
                    lines = [l.strip() for l in f if l.strip()]
                # Format: "HOTA: 52.477" or tab-separated header+values
                metrics = {}
                for line in lines:
                    for key in ["HOTA", "AssA", "DetA"]:
                        if key in line:
                            parts = line.replace(":", " ").split()
                            for i, p in enumerate(parts):
                                if p == key and i + 1 < len(parts):
                                    try:
                                        metrics[key] = float(parts[i + 1])
                                    except ValueError:
                                        pass
                if "HOTA" in metrics:
                    per_seq[seq_name] = metrics
            except Exception:
                pass
        if per_seq:
            return per_seq

    # ── Strategy 3: print available files for manual inspection ───────
    print(f"  Could not parse per-sequence HOTA from {epoch_dir}.")
    print(f"  Available files:")
    for root, dirs, files in os.walk(epoch_dir):
        for fname in files[:5]:
            print(f"    {os.path.join(root, fname)}")
        break
    print()
    print("  Use --per_seq_json instead. Format:")
    print('  {"dancetrack0004": {"HOTA": 55.1, "AssA": 40.2, "DetA": 71.5}, ...}')
    return {}


def load_from_json(json_path: str) -> dict:
    with open(json_path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    args = get_args()
    od = Path(args.output_dir)
    od.mkdir(parents=True, exist_ok=True)

    # ── Load per-seq HOTA ─────────────────────────────────────────────
    if args.per_seq_json:
        per_seq_hota = load_from_json(args.per_seq_json)
        print(f"Loaded per-seq HOTA from {args.per_seq_json}: "
              f"{len(per_seq_hota)} sequences")
    elif args.eval_dir and args.epoch is not None:
        per_seq_hota = load_from_eval_dir(args.eval_dir, args.epoch)
        print(f"Loaded per-seq HOTA from eval_dir epoch {args.epoch}: "
              f"{len(per_seq_hota)} sequences")
    else:
        print("ERROR: provide either --per_seq_json or both --eval_dir and --epoch")
        sys.exit(1)

    if not per_seq_hota:
        print("No per-sequence HOTA data found. Exiting.")
        sys.exit(1)

    # ── Load GT density ───────────────────────────────────────────────
    val_seqs = sorted(p for p in Path(args.val_dir).iterdir() if p.is_dir())
    rows = []
    for seq_dir in val_seqs:
        seq_name = seq_dir.name
        if seq_name not in per_seq_hota:
            continue
        density = load_density(str(seq_dir))
        if not density:
            continue
        rows.append({
            "sequence":       seq_name,
            "HOTA":           per_seq_hota[seq_name].get("HOTA", float("nan")),
            "AssA":           per_seq_hota[seq_name].get("AssA", float("nan")),
            "DetA":           per_seq_hota[seq_name].get("DetA", float("nan")),
            "mean_concurrent": density["mean_concurrent"],
            "max_concurrent":  density["max_concurrent"],
        })

    rows.sort(key=lambda r: r["max_concurrent"])
    print(f"Matched {len(rows)} sequences with density data.\n")

    if len(rows) < 3:
        print("Too few sequences for correlation. Exiting.")
        sys.exit(1)

    hotas     = np.array([r["HOTA"]            for r in rows])
    assas     = np.array([r["AssA"]            for r in rows])
    max_conc  = np.array([r["max_concurrent"]  for r in rows])
    mean_conc = np.array([r["mean_concurrent"] for r in rows])

    valid = ~(np.isnan(hotas) | np.isnan(max_conc))

    if valid.sum() < 3:
        print(f"  Only {valid.sum()} sequences have non-nan HOTA values.")
        print("  Cannot compute correlation. Check --eval_dir / --per_seq_json.")
        print("  Sequences with nan HOTA:")
        for r in rows:
            if np.isnan(r["HOTA"]):
                print(f"    {r['sequence']}")
        sys.exit(1)
    r_hota_max  = float(np.corrcoef(hotas[valid],  max_conc[valid])[0, 1])
    r_assa_max  = float(np.corrcoef(assas[valid],  max_conc[valid])[0, 1])
    r_hota_mean = float(np.corrcoef(hotas[valid], mean_conc[valid])[0, 1])

    # ── Print report ──────────────────────────────────────────────────
    print("=" * 65)
    print("D-NEW-5 — Per-Sequence HOTA vs Density Correlation")
    print("=" * 65)
    print(f"\n  Pearson r(HOTA,  max_concurrent):  {r_hota_max:.3f}")
    print(f"  Pearson r(AssA,  max_concurrent):  {r_assa_max:.3f}")
    print(f"  Pearson r(HOTA, mean_concurrent):  {r_hota_mean:.3f}")
    print()

    if r_hota_max < -0.5:
        print("  FINDING: Strong negative correlation (r < -0.5).")
        print("  Dense sequences systematically underperform.")
        print("  → Training density mismatch is a CONFIRMED bottleneck.")
        print("  → Weighted sequence sampler is strongly justified (D-NEW-6).")
    elif r_hota_max < -0.3:
        print("  FINDING: Moderate negative correlation (-0.5 < r < -0.3).")
        print("  Density has partial influence on performance.")
        print("  → Weighted sampler would likely help alongside other fixes.")
    else:
        print("  FINDING: Weak or no correlation (r > -0.3).")
        print("  HOTA degradation is not density-specific.")
        print("  → Density mismatch is NOT the primary bottleneck.")
        print("  → Focus on feature discriminability (projection head).")

    print(f"\n  {'Sequence':<24}  {'max_c':>6}  {'mean_c':>7}  "
          f"{'HOTA':>7}  {'AssA':>7}")
    print("  " + "-" * 60)
    for r in rows:
        flag = " ←" if r["max_concurrent"] >= 15 else ""
        print(f"  {r['sequence']:<24}  {r['max_concurrent']:>6}  "
              f"{r['mean_concurrent']:>7.1f}  "
              f"{r['HOTA']:>7.3f}  {r['AssA']:>7.3f}{flag}")

    # ── Plots ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    sc = ax.scatter(max_conc[valid], hotas[valid],
                    c=assas[valid], cmap="RdYlGn", s=90,
                    edgecolors="k", linewidths=0.5)
    plt.colorbar(sc, ax=ax, label="AssA")
    z = np.polyfit(max_conc[valid], hotas[valid], 1)
    xr = np.linspace(max_conc[valid].min(), max_conc[valid].max(), 50)
    ax.plot(xr, np.polyval(z, xr), "r--", lw=1.5, alpha=0.7)
    for r in rows:
        ax.annotate(r["sequence"][-4:],
                    (r["max_concurrent"], r["HOTA"]),
                    fontsize=6, alpha=0.7)
    ax.set_xlabel("max_concurrent (GT)")
    ax.set_ylabel("HOTA")
    ax.set_title(f"HOTA vs max density  r={r_hota_max:.3f}")

    ax = axes[1]
    sc2 = ax.scatter(mean_conc[valid], hotas[valid],
                     c=assas[valid], cmap="RdYlGn", s=90,
                     edgecolors="k", linewidths=0.5)
    plt.colorbar(sc2, ax=ax, label="AssA")
    z2 = np.polyfit(mean_conc[valid], hotas[valid], 1)
    xr2 = np.linspace(mean_conc[valid].min(), mean_conc[valid].max(), 50)
    ax.plot(xr2, np.polyval(z2, xr2), "r--", lw=1.5, alpha=0.7)
    ax.set_xlabel("mean_concurrent (GT)")
    ax.set_ylabel("HOTA")
    ax.set_title(f"HOTA vs mean density  r={r_hota_mean:.3f}")

    ax = axes[2]
    colors = ["red" if r["max_concurrent"] >= 15 else "steelblue" for r in rows]
    ax.bar(range(len(rows)), hotas, color=colors, alpha=0.85)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r["sequence"][-4:] for r in rows],
                       rotation=45, fontsize=7)
    ax.set_ylabel("HOTA")
    ax.axhline(np.nanmean(hotas), color="black", ls="--", lw=1.5,
               label=f"mean={np.nanmean(hotas):.2f}")
    ax.set_title("HOTA per sequence (red = max_concurrent ≥ 15)")
    ax.legend(fontsize=8)

    plt.suptitle(f"D-NEW-5: HOTA vs Density  r(HOTA,max)={r_hota_max:.3f}  "
                 f"r(HOTA,mean)={r_hota_mean:.3f}")
    plt.tight_layout()
    plt.savefig(od / "d5_hota_vs_density.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved to {od}/d5_hota_vs_density.png")

    # ── Save JSON ─────────────────────────────────────────────────────
    result = {
        "r_hota_max_concurrent":  round(r_hota_max,  4),
        "r_assa_max_concurrent":  round(r_assa_max,  4),
        "r_hota_mean_concurrent": round(r_hota_mean, 4),
        "n_sequences":            len(rows),
        "finding": ("strong_density_bottleneck" if r_hota_max < -0.5
                    else "moderate" if r_hota_max < -0.3
                    else "weak_density_effect"),
        "per_sequence":           rows,
    }
    with open(od / "d5_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Results saved to {od}/d5_results.json")


if __name__ == "__main__":
    main()