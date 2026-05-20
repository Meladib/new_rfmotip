#!/usr/bin/env python3
"""
diag_d8_temperature_sweep.py
=============================
D-NEW-8: Softmax temperature sweep at inference on V3 checkpoint.

Runs the val set at T ∈ {0.3, 0.5, 0.7, 1.0, 1.5}.
No training. No weight changes. Inference-only.

T < 1.0  →  sharpens score distribution
T = 1.0  →  baseline (current behaviour)
T > 1.0  →  softens score distribution

If T < 1.0 improves HOTA: margins are too soft → projection head justified.
If T = 1.0 is optimal:   margins are fine → competition/density is the driver.

Run from repo root:
  python diagnostics/diag_d8_temperature_sweep.py \
    --config     configs/rf_detrV3_motip_dancetrack.yaml \
    --checkpoint outputsV3/rfmotip_dancetrack/train/checkpoint_7.pth \
    --output_dir diagnostics/diag8_results/
"""

import os
import sys
import json
import argparse
import shutil
import subprocess
import tempfile
import glob

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# ARGS
# ─────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",      required=True)
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--data_root",   default="/data/pos+mot/Datadir/")
    p.add_argument("--temps",       type=float, nargs="+",
                   default=[0.3, 0.5, 0.7, 1.0, 1.5])
    p.add_argument("--max_frames",  type=int, default=0,
                   help="Limit frames per sequence for quick test (0=all)")
    p.add_argument("--max_seqs",    type=int, default=0,
                   help="Limit number of sequences for quick test (0=all)")
    p.add_argument("--output_dir",  default="diagnostics/diag8_results/")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# TRACKEVAL
# ─────────────────────────────────────────────────────────────
def run_trackeval(tracker_dir: str, data_root: str, tmp_dir: str) -> dict:
    """
    Exact mirror of submit_and_evaluate.py for DanceTrack val.
    tracker_dir: directory where seq txt files are written directly.
    With --TRACKERS_TO_EVAL "" and --TRACKER_SUB_FOLDER "",
    TrackEval treats files in TRACKERS_FOLDER as tracker "" directly.
    pedestrian_summary.txt lands in tracker_dir itself.
    """
    gt_dir = os.path.join(data_root, "DanceTrack", "val")

    # Always generate seqmap from actual tracker files (handles --max_seqs correctly)
    seq_names   = sorted(f[:-4] for f in os.listdir(tracker_dir) if f.endswith(".txt"))
    seqmap_file = os.path.join(tmp_dir, "val_seqmap.txt")
    with open(seqmap_file, "w") as f:
        f.write("name\n")
        for s in seq_names:
            f.write(s + "\n")

    candidates = ["TrackEval/scripts/run_mot_challenge.py"]
    candidates += glob.glob("**/run_mot_challenge.py", recursive=True)
    script = next((c for c in candidates if os.path.exists(c)), None)
    if script is None:
        print("  WARNING: run_mot_challenge.py not found.")
        return {}

    cmd = [
        sys.executable, script,
        "--SPLIT_TO_EVAL",      "val",
        "--METRICS",            "HOTA", "CLEAR", "Identity",
        "--GT_FOLDER",          gt_dir,
        "--SEQMAP_FILE",        seqmap_file,
        "--SKIP_SPLIT_FOL",     "True",
        "--TRACKERS_TO_EVAL",   "",
        "--TRACKER_SUB_FOLDER", "",
        "--TRACKERS_FOLDER",    tracker_dir,
        "--USE_PARALLEL",       "False",
        "--PLOT_CURVES",        "False",
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, timeout=600, check=False)
        if res.returncode != 0:
            print("  TrackEval stderr:", res.stderr.decode()[-500:])
    except Exception as e:
        print(f"  TrackEval error: {e}")
        return {}

    # Summary lands directly in tracker_dir (same as submit_and_evaluate.py)
    summary = os.path.join(tracker_dir, "pedestrian_summary.txt")
    if not os.path.exists(summary):
        # Fallback search
        found = glob.glob(os.path.join(tracker_dir, "**", "pedestrian_summary.txt"), recursive=True)
        if found:
            summary = found[0]
        else:
            print("  pedestrian_summary.txt not found. Files in tracker_dir:")
            for root, _, files in os.walk(tracker_dir):
                for fn in files[:10]:
                    print(f"    {os.path.join(root, fn)}")
            return {}

    metrics = {}
    with open(summary) as f:
        lines = [l.strip() for l in f if l.strip()]
    if len(lines) >= 2:
        for k, v in zip(lines[0].split(), lines[1].split()):
            try:
                metrics[k] = float(v)
            except ValueError:
                pass
    return metrics


# ─────────────────────────────────────────────────────────────
# ONE TEMPERATURE RUN
# ─────────────────────────────────────────────────────────────
def run_one_temp(T: float, model, cfg: dict, data_root: str,
                 device: torch.device, max_frames: int = 0,
                 max_seqs: int = 0) -> dict:
    from models.runtime_tracker import RuntimeTracker
    from data.dancetrack import DanceTrack
    from data.seq_dataset import SeqDataset

    tmp_dir = tempfile.mkdtemp(prefix=f"d8_T{T:.2f}_")
    trk_dir = os.path.join(tmp_dir, "tracker")
    os.makedirs(trk_dir, exist_ok=True)

    dt = DanceTrack(data_root=data_root, split="val", load_annotation=False)

    seq_names = sorted(dt.sequence_infos.keys())
    if max_seqs > 0:
        seq_names = seq_names[:max_seqs]
    for seq_idx, seq_name in enumerate(seq_names):
        print(f"  [{seq_idx+1}/{len(seq_names)}] {seq_name}", flush=True)
        seq_ds = SeqDataset(
            seq_info=dt.sequence_infos[seq_name],
            image_paths=dt.image_paths[seq_name],
            max_shorter=800,
            max_longer=cfg.get("INFERENCE_MAX_LONGER", 1440),
            size_divisibility=cfg.get("SIZE_DIVISIBILITY", 32),
            dtype=torch.float32,
        )
        loader = DataLoader(seq_ds, batch_size=1, shuffle=False,
                            num_workers=0, collate_fn=lambda x: x[0])

        tracker = RuntimeTracker(
            model=model,
            sequence_hw=seq_ds.seq_hw(),
            use_sigmoid=cfg.get("USE_FOCAL_LOSS", False),
            assignment_protocol=cfg.get("ASSIGNMENT_PROTOCOL", "object-max"),
            miss_tolerance=cfg.get("MISS_TOLERANCE", 30),
            det_thresh=cfg.get("DET_THRESH", 0.5),
            newborn_thresh=cfg.get("NEWBORN_THRESH", 0.5),
            id_thresh=cfg.get("ID_THRESH", 0.2),
            area_thresh=cfg.get("AREA_THRESH", 0),
            only_detr=False,
            dtype=torch.float32,
        )

        # Inject temperature by patching assignment methods
        if T != 1.0:
            def make_patch(orig, temperature):
                def patched(id_scores=None):
                    scaled = (id_scores.clamp(min=1e-9).log()
                              / temperature).softmax(dim=-1)
                    return orig(id_scores=scaled)
                return patched

            tracker._object_max_assignment = make_patch(
                tracker._object_max_assignment, T)
            tracker._hungarian_assignment  = make_patch(
                tracker._hungarian_assignment,  T)
            tracker._id_max_assignment     = make_patch(
                tracker._id_max_assignment,     T)

        n_frames = len(loader)
        lines = []
        with torch.no_grad():
            for t, (image, _) in enumerate(loader):
                if max_frames > 0 and t >= max_frames:
                    break
                image.tensors = image.tensors.to(device)
                image.mask    = image.mask.to(device)
                tracker.update(image=image)
                res = tracker.get_track_results()
                for obj_id, bbox in zip(res["id"], res["bbox"]):
                    x = bbox[0].item(); y = bbox[1].item()
                    w = bbox[2].item(); h = bbox[3].item()
                    lines.append(
                        f"{t+1},{obj_id.item()},"
                        f"{x:.3f},{y:.3f},{w:.3f},{h:.3f},"
                        f"1,-1,-1,-1\n"
                    )
                if (t + 1) % 100 == 0 or (t + 1) == n_frames:
                    print(f"    frame {t+1}/{n_frames}", flush=True)

        with open(os.path.join(trk_dir, f"{seq_name}.txt"), "w") as f:
            f.writelines(lines)

    print(f"  Tracking done. Running TrackEval...", flush=True)
    metrics = run_trackeval(trk_dir, data_root, tmp_dir)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    h = metrics.get("HOTA", float("nan"))
    print(f"  TrackEval done. HOTA={h:.3f}", flush=True)
    return metrics


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    args = get_args()
    od = Path(args.output_dir)
    od.mkdir(parents=True, exist_ok=True)

    from utils.misc import yaml_to_dict
    from configs.util import load_super_config
    from models.motip import build as build_motip
    from models.misc import load_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = yaml_to_dict(args.config)
    cfg = load_super_config(cfg, cfg.get("SUPER_CONFIG_PATH"))

    print("Loading model...")
    model, _ = build_motip(config=cfg)
    load_checkpoint(model, path=args.checkpoint)
    model.eval().to(device)

    print(f"Temperature sweep: T ∈ {args.temps}\n")

    results = {}
    for T in sorted(args.temps):
        print(f"  T={T:.1f} ...", end=" ", flush=True)
        metrics = run_one_temp(T, model, cfg, args.data_root, device, args.max_frames, args.max_seqs)
        results[T] = metrics
        h = metrics.get("HOTA", float("nan"))
        a = metrics.get("AssA", float("nan"))
        print(f"HOTA={h:.3f}  AssA={a:.3f}")

    # ── Summary ──────────────────────────────────────────────────────
    base = results.get(1.0, {}).get("HOTA", float("nan"))
    print(f"\n{'='*60}")
    print("D-NEW-8 — Temperature Sweep Results")
    print(f"{'='*60}")
    print(f"  {'T':>6}  {'HOTA':>8}  {'AssA':>8}  {'DetA':>8}  {'vs T=1.0':>10}")
    for T in sorted(results):
        h = results[T].get("HOTA", float("nan"))
        a = results[T].get("AssA",  float("nan"))
        d = results[T].get("DetA",  float("nan"))
        delta = f"{h-base:+.3f}" if not (np.isnan(h) or np.isnan(base)) else "baseline"
        mark  = " ← baseline" if T == 1.0 else ""
        print(f"  {T:>6.1f}  {h:>8.3f}  {a:>8.3f}  {d:>8.3f}  {delta:>10}{mark}")

    valid = [(T, results[T].get("HOTA", float("nan")))
             for T in sorted(results)
             if not np.isnan(results[T].get("HOTA", float("nan")))]
    best_T, best_H, gain = float("nan"), float("nan"), float("nan")
    if valid:
        best_T, best_H = max(valid, key=lambda x: x[1])
        gain = best_H - base if not np.isnan(base) else float("nan")
        print(f"\n  Best: T={best_T}  HOTA={best_H:.3f}  gain={gain:+.3f}")
        print()
        if not np.isnan(gain):
            if best_T < 1.0 and gain > 0.1:
                print("  FINDING: Sharpening (T<1) improves HOTA.")
                print("  → Score margins are too soft. Separability gap is the bottleneck.")
                print(f"  → Projection head intervention is justified.")
                print(f"  → Also: set inference T={best_T} for immediate gain (zero training cost).")
            elif abs(gain) <= 0.1:
                print("  FINDING: T=1.0 is optimal — temperature has no effect.")
                print("  → Margins are already sharp. Case B is driven by competition frequency.")
                print("  → Density-weighted sampler / projection head are the correct next steps.")
            elif best_T > 1.0 and gain > 0.1:
                print("  FINDING: Softening improves HOTA (unexpected).")
                print("  → id_thresh may be too high. Scores are over-sharp.")

    # ── Plot ─────────────────────────────────────────────────────────
    temps_sorted = sorted(results.keys())
    hota_vals = [results[T].get("HOTA", float("nan")) for T in temps_sorted]
    assa_vals = [results[T].get("AssA", float("nan")) for T in temps_sorted]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(temps_sorted, hota_vals, "bo-", label="HOTA", lw=2, ms=8)
    ax.plot(temps_sorted, assa_vals, "rs--", label="AssA", lw=1.5, ms=6)
    ax.axvline(1.0, color="grey", ls="--", alpha=0.5, label="T=1.0 (baseline)")
    if not np.isnan(base):
        ax.axhline(base, color="blue", ls=":", alpha=0.4)
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Metric")
    ax.set_title("D-NEW-8: Softmax Temperature Sweep\n"
                 "(T<1 = sharpen, T>1 = soften)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(od / "d8_temperature_sweep.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Save ─────────────────────────────────────────────────────────
    out = {
        "checkpoint": args.checkpoint,
        "temps_tested": sorted(results.keys()),
        "best_T": best_T if valid else None,
        "best_HOTA": best_H if valid else None,
        "baseline_HOTA": base,
        "gain": float(gain) if not np.isnan(gain) else None,
        "results": {str(T): results[T] for T in sorted(results)},
    }
    with open(od / "d8_results.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved to {od}/")
    print("  d8_temperature_sweep.png")
    print("  d8_results.json")


if __name__ == "__main__":
    main()