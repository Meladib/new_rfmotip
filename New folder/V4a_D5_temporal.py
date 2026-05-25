#!/usr/bin/env python3
"""
D5_temporal.py
==============
D5-TEMPORAL: Does the V4a AssA gap come from temporal accumulation of ID errors?

Hypothesis: reid_proj introduces small per-frame assignment errors that
compound over long sequences — invisible in per-frame score statistics
(D4 confirmed) but measurable in IDSW and Frag per sequence.

Prediction (if hypothesis confirmed):
  1. V4a has more IDSW than V3 globally
  2. (IDSW_V4a - IDSW_V3) per sequence CORRELATES with sequence length
     and number of GT IDs — longer / more complex sequences show larger gap
  3. Frag counts show same pattern (track breaks compound over time)

If not confirmed:
  IDSW is similar between V3 and V4a → temporal accumulation is NOT the cause
  → generalization gap root cause is unidentified by this diagnostic chain

Run from RF-MOTIPV4 repo root:
    python "New folder/D5_temporal.py" \
        --config    configs/rf_detrV4_motip_dancetrack.yaml \
        --ckpt_v3   /data/adib/new/github/RF-MOTIP/outputsV3/rfmotip_dancetrack/train/checkpoint_7.pth \
        --ckpt_v4_6 outputs/rfmotip_dancetrack_V3_full/checkpoint_6.pth \
        --data_root /data/pos+mot/Datadir/ \
        --output_dir "New folder/D5_output/"
"""

import os, sys, json, argparse, shutil, subprocess, tempfile, csv
import configparser
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Args ──────────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",      required=True)
    p.add_argument("--ckpt_v3",     required=True)
    p.add_argument("--ckpt_v4_6",   required=True)
    p.add_argument("--data_root",   default="/data/pos+mot/Datadir/")
    p.add_argument("--output_dir",  default="New folder/D5_output/")
    return p.parse_args()


# ── Model loader (confirmed pattern from D3/D4) ───────────────────────────────
def load_model(config_path, ckpt_path, device, use_reid_proj=None):
    from utils.misc import yaml_to_dict
    from configs.util import load_super_config
    from models.motip import build as build_motip
    from models.misc import load_checkpoint

    config = yaml_to_dict(config_path)
    config = load_super_config(config, config.get("SUPER_CONFIG_PATH"))
    config["RESUME_MODEL"] = None
    if use_reid_proj is not None:
        config["USE_REID_PROJ"] = use_reid_proj
    model, _ = build_motip(config=config)
    load_checkpoint(model, path=ckpt_path)
    model.eval().to(device)
    return model, config


# ── Run tracker on all val sequences, write MOT txt files ─────────────────────
@torch.no_grad()
def run_tracker(model, config, data_root, tracker_dir, device):
    """
    Runs RuntimeTracker on all 25 val sequences.
    Writes one MOT-format txt file per sequence to tracker_dir.
    Returns {seq_name: num_frames} for sequence length info.
    Confirmed API: RuntimeTracker(model, sequence_hw, ...) + rt.update(image)
    from diag_d8_temperature_sweep.py and D3/D4 scripts.
    """
    from models.runtime_tracker import RuntimeTracker
    from data.dancetrack import DanceTrack
    from data.seq_dataset import SeqDataset

    os.makedirs(tracker_dir, exist_ok=True)
    dt = DanceTrack(data_root=data_root, split="val", load_annotation=False)
    seq_names = sorted(dt.sequence_infos.keys())
    seq_lengths = {}

    for seq_idx, seq_name in enumerate(seq_names):
        seq_ds = SeqDataset(
            seq_info=dt.sequence_infos[seq_name],
            image_paths=dt.image_paths[seq_name],
            max_shorter=800,
            max_longer=config.get("INFERENCE_MAX_LONGER", 1440),
            size_divisibility=config.get("SIZE_DIVISIBILITY", 32),
            dtype=torch.float32,
        )
        loader = DataLoader(
            seq_ds, batch_size=1, shuffle=False,
            num_workers=0, collate_fn=lambda x: x[0]
        )

        rt = RuntimeTracker(
            model=model,
            sequence_hw=seq_ds.seq_hw(),
            use_sigmoid=config.get("USE_FOCAL_LOSS", False),
            assignment_protocol=config.get("ASSIGNMENT_PROTOCOL", "object-max"),
            miss_tolerance=config.get("MISS_TOLERANCE", 30),
            det_thresh=config.get("DET_THRESH", 0.3),
            newborn_thresh=config.get("NEWBORN_THRESH", 0.6),
            id_thresh=config.get("ID_THRESH", 0.2),
            area_thresh=config.get("AREA_THRESH", 0),
            only_detr=False,
            dtype=torch.float32,
        )

        results = []
        for frame_idx, (image, _) in enumerate(loader):
            image.tensors = image.tensors.to(device)
            image.mask    = image.mask.to(device)
            rt.update(image)
            frame_res = rt.get_track_results()
            for obj_id, score, bbox in zip(
                frame_res["id"],
                frame_res["score"],
                frame_res["bbox"],
            ):
                results.append(
                    f"{frame_idx + 1},{obj_id.item()},"
                    f"{bbox[0].item():.2f},{bbox[1].item():.2f},"
                    f"{bbox[2].item():.2f},{bbox[3].item():.2f},"
                    f"1,-1,-1,-1\n"
                )

        txt_path = os.path.join(tracker_dir, f"{seq_name}.txt")
        with open(txt_path, "w") as f:
            f.writelines(results)

        seq_lengths[seq_name] = frame_idx + 1
        print(f"  [{seq_idx+1}/{len(seq_names)}] {seq_name}: "
              f"{frame_idx+1} frames, {len(results)} detections")

    return seq_lengths


# ── Run TrackEval with CLEAR + HOTA, OUTPUT_DETAILED=True ────────────────────
def run_trackeval_detailed(tracker_dir, data_root):
    # Confirmed from submit_and_evaluate.py: GT_FOLDER must include split,
    # TRACKER_SUB_FOLDER="" with TRACKERS_TO_EVAL="" means txt files
    # are read directly from tracker_dir.
    gt_dir  = os.path.join(data_root, "DanceTrack", "val")
    seq_map = os.path.join(data_root, "DanceTrack", "val_seqmap.txt")

    cmd = [
        sys.executable,
        "TrackEval/scripts/run_mot_challenge.py",
        "--GT_FOLDER",          gt_dir,
        "--TRACKERS_FOLDER",    tracker_dir,
        "--SEQMAP_FILE",        seq_map,
        "--SPLIT_TO_EVAL",      "val",
        "--METRICS",            "HOTA", "CLEAR", "Identity",
        "--SKIP_SPLIT_FOL",     "True",
        "--TRACKERS_TO_EVAL",   "",
        "--TRACKER_SUB_FOLDER", "",
        "--USE_PARALLEL",       "False",
        "--NUM_PARALLEL_CORES", "4",
        "--PLOT_CURVES",        "False",
        "--OUTPUT_SUMMARY",     "True",
        "--OUTPUT_DETAILED",    "True",
        "--PRINT_RESULTS",      "False",
        "--PRINT_CONFIG",       "False",
    ]

    ret = subprocess.run(cmd, capture_output=True, text=True)
    if ret.returncode != 0:
        print("TrackEval stderr:", ret.stderr[-2000:])
        raise RuntimeError("TrackEval failed")

    # With TRACKERS_TO_EVAL="" and TRACKER_SUB_FOLDER="",
    # output lands directly in tracker_dir/
    detailed_csv = os.path.join(tracker_dir, "pedestrian_detailed.csv")
    if not os.path.exists(detailed_csv):
        # fallback search
        for root, _, files in os.walk(tracker_dir):
            for f in files:
                if "detailed" in f and f.endswith(".csv"):
                    return os.path.join(root, f)
        raise FileNotFoundError(
            f"pedestrian_detailed.csv not found under {tracker_dir}")
    return detailed_csv


# ── Parse pedestrian_detailed.csv → per-sequence metrics ─────────────────────
def parse_detailed_csv(csv_path):
    """
    Returns {seq_name: {metric: value}} for all sequences.
    Skips the COMBINED row.
    """
    results = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq = row.get("seq", row.get("sequence", "")).strip()
            if not seq or seq.lower() == "combined":
                continue
            results[seq] = {}
            for k, v in row.items():
                if k in ("seq", "sequence"):
                    continue
                try:
                    results[seq][k.strip()] = float(v)
                except (ValueError, TypeError):
                    pass
    return results


# ── Load sequence properties from GT ─────────────────────────────────────────
def load_seq_properties(data_root):
    """
    Returns {seq_name: {length, num_gt_ids}} from seqinfo.ini + gt.txt.
    """
    val_dir = Path(data_root) / "DanceTrack" / "val"
    props = {}
    for seq_dir in sorted(val_dir.iterdir()):
        if not seq_dir.is_dir():
            continue
        seq_name = seq_dir.name
        ini = configparser.ConfigParser()
        ini.read(seq_dir / "seqinfo.ini")
        length = int(ini["Sequence"]["seqLength"])

        # Count unique GT IDs
        gt_ids = set()
        gt_path = seq_dir / "gt" / "gt.txt"
        if gt_path.exists():
            with open(gt_path) as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 2:
                        try:
                            gt_ids.add(int(parts[1]))
                        except ValueError:
                            pass
        props[seq_name] = {"length": length, "num_gt_ids": len(gt_ids)}
    return props


# ── Compute per-sequence IDSW delta and correlation ───────────────────────────
def analyze(v3_seq, v4_seq, seq_props):
    """
    v3_seq, v4_seq: {seq_name: {metric: value}}
    seq_props: {seq_name: {length, num_gt_ids}}
    Returns analysis dict.
    """
    common = sorted(set(v3_seq) & set(v4_seq) & set(seq_props))

    idsw_v3  = np.array([v3_seq[s].get("IDSW", 0)    for s in common])
    idsw_v4  = np.array([v4_seq[s].get("IDSW", 0)    for s in common])
    frag_v3  = np.array([v3_seq[s].get("Frag", 0)    for s in common])
    frag_v4  = np.array([v4_seq[s].get("Frag", 0)    for s in common])
    lengths  = np.array([seq_props[s]["length"]       for s in common])
    n_ids    = np.array([seq_props[s]["num_gt_ids"]   for s in common])

    delta_idsw = idsw_v4 - idsw_v3   # positive = V4a has more IDSW
    delta_frag = frag_v4 - frag_v3

    # Pearson correlation: delta_idsw vs sequence length
    r_length = float(np.corrcoef(delta_idsw, lengths)[0, 1])
    r_n_ids  = float(np.corrcoef(delta_idsw, n_ids)[0, 1])

    # Normalized IDSW rate per frame
    idsw_rate_v3 = idsw_v3 / np.maximum(lengths, 1)
    idsw_rate_v4 = idsw_v4 / np.maximum(lengths, 1)

    return {
        "sequences":     common,
        "idsw_v3":       idsw_v3.tolist(),
        "idsw_v4":       idsw_v4.tolist(),
        "frag_v3":       frag_v3.tolist(),
        "frag_v4":       frag_v4.tolist(),
        "delta_idsw":    delta_idsw.tolist(),
        "delta_frag":    delta_frag.tolist(),
        "lengths":       lengths.tolist(),
        "n_ids":         n_ids.tolist(),
        "total_idsw_v3": int(idsw_v3.sum()),
        "total_idsw_v4": int(idsw_v4.sum()),
        "total_frag_v3": int(frag_v3.sum()),
        "total_frag_v4": int(frag_v4.sum()),
        "mean_delta_idsw": float(delta_idsw.mean()),
        "r_delta_idsw_vs_length": r_length,
        "r_delta_idsw_vs_n_ids":  r_n_ids,
        "mean_idsw_rate_v3": float(idsw_rate_v3.mean()),
        "mean_idsw_rate_v4": float(idsw_rate_v4.mean()),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args   = get_args()
    od     = Path(args.output_dir)
    od.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    seq_props = load_seq_properties(args.data_root)
    print(f"Loaded properties for {len(seq_props)} val sequences.\n")

    # ── V3 ────────────────────────────────────────────────────────────────────
    print("=" * 65)
    print("STEP 1: Running V3 on all val sequences")
    print("=" * 65)
    v3_tmp       = tempfile.mkdtemp(prefix="d5_v3_")
    v3_trk_dir   = os.path.join(v3_tmp, "tracker")
    model_v3, config = load_model(
        args.config, args.ckpt_v3, device, use_reid_proj=False)
    run_tracker(model_v3, config, args.data_root, v3_trk_dir, device)
    del model_v3; torch.cuda.empty_cache()

    print("\nRunning TrackEval for V3...")
    v3_csv = run_trackeval_detailed(v3_trk_dir, args.data_root)
    v3_seq     = parse_detailed_csv(v3_csv)
    print(f"  Parsed {len(v3_seq)} sequences from {v3_csv}")

    # ── V4a_6 ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 2: Running V4a_6 on all val sequences")
    print("=" * 65)
    v4_tmp       = tempfile.mkdtemp(prefix="d5_v4_")
    v4_trk_dir   = os.path.join(v4_tmp, "tracker")
    model_v4, _  = load_model(
        args.config, args.ckpt_v4_6, device, use_reid_proj=None)
    run_tracker(model_v4, config, args.data_root, v4_trk_dir, device)
    del model_v4; torch.cuda.empty_cache()

    print("\nRunning TrackEval for V4a_6...")
    v4_csv = run_trackeval_detailed(v4_trk_dir, args.data_root)
    v4_seq     = parse_detailed_csv(v4_csv)
    print(f"  Parsed {len(v4_seq)} sequences from {v4_csv}")

    # ── Analysis ──────────────────────────────────────────────────────────────
    stats = analyze(v3_seq, v4_seq, seq_props)

    # ── Verdict ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("D5-TEMPORAL VERDICT")
    print("=" * 65)
    print(f"\n  Total IDSW — V3: {stats['total_idsw_v3']}  "
          f"V4a: {stats['total_idsw_v4']}  "
          f"Δ: {stats['total_idsw_v4'] - stats['total_idsw_v3']:+d}")
    print(f"  Total Frag — V3: {stats['total_frag_v3']}  "
          f"V4a: {stats['total_frag_v4']}  "
          f"Δ: {stats['total_frag_v4'] - stats['total_frag_v3']:+d}")
    print(f"\n  Mean IDSW rate/frame — V3: {stats['mean_idsw_rate_v3']:.5f}  "
          f"V4a: {stats['mean_idsw_rate_v4']:.5f}")
    print(f"\n  Correlation (ΔIDSW vs sequence length):   "
          f"r = {stats['r_delta_idsw_vs_length']:+.3f}")
    print(f"  Correlation (ΔIDSW vs num GT IDs):        "
          f"r = {stats['r_delta_idsw_vs_n_ids']:+.3f}")

    # Per-sequence table
    print(f"\n  {'Sequence':<25} {'Len':>6} {'n_IDs':>6} "
          f"{'IDSW_V3':>8} {'IDSW_V4':>8} {'ΔIDSW':>7} "
          f"{'Frag_V3':>8} {'Frag_V4':>8} {'ΔFrag':>7}")
    print(f"  {'-'*85}")
    for i, seq in enumerate(stats["sequences"]):
        print(f"  {seq:<25} "
              f"{stats['lengths'][i]:>6.0f} "
              f"{stats['n_ids'][i]:>6.0f} "
              f"{stats['idsw_v3'][i]:>8.0f} "
              f"{stats['idsw_v4'][i]:>8.0f} "
              f"{stats['delta_idsw'][i]:>+7.0f} "
              f"{stats['frag_v3'][i]:>8.0f} "
              f"{stats['frag_v4'][i]:>8.0f} "
              f"{stats['delta_frag'][i]:>+7.0f}")

    # Verdict logic
    more_idsw    = stats["total_idsw_v4"] > stats["total_idsw_v3"]
    length_corr  = stats["r_delta_idsw_vs_length"] > 0.3
    temporal_confirmed = more_idsw and length_corr

    print(f"\n  V4a has more IDSW than V3:                {more_idsw}")
    print(f"  ΔIDSW correlates with sequence length:    {length_corr} "
          f"(r={stats['r_delta_idsw_vs_length']:+.3f})")
    print(f"\n  TEMPORAL ACCUMULATION HYPOTHESIS: "
          f"{'CONFIRMED' if temporal_confirmed else 'NOT CONFIRMED'}")

    if temporal_confirmed:
        print("""
  V4a has more ID switches than V3, and the gap grows with
  sequence length. Small per-frame assignment errors compound
  over time — invisible in per-frame score statistics (D4)
  but measurable in IDSW across full sequences.
  → root cause: reid_proj introduces marginal feature perturbation
    that accumulates into ID switches in long sequences.
  → implication for V4b: need to constrain feature space change
    or use staged training to stabilize IDDecoder.""")
    elif more_idsw and not length_corr:
        print("""
  V4a has more IDSW than V3, but the gap does NOT correlate
  with sequence length. The errors are not cumulative — they
  occur uniformly regardless of sequence duration.
  → suggests a scene-complexity factor, not temporal accumulation.
  → check correlation with n_gt_ids and scene density.""")
    else:
        print("""
  IDSW counts are similar between V3 and V4a.
  Temporal accumulation is NOT the cause of the AssA gap.
  → The full diagnostic chain D1-D5 has not identified the
    root cause. The difference may be too small to isolate
    with sequence-level statistics and requires frame-level
    GT-matched analysis.""")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: IDSW per sequence V3 vs V4a
    ax = axes[0]
    x  = np.arange(len(stats["sequences"]))
    ax.bar(x - 0.2, stats["idsw_v3"], 0.4, label="V3",   color="steelblue", alpha=0.8)
    ax.bar(x + 0.2, stats["idsw_v4"], 0.4, label="V4a_6", color="orange",   alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [s.replace("dancetrack", "dt") for s in stats["sequences"]],
        rotation=90, fontsize=7)
    ax.set_ylabel("IDSW count")
    ax.set_title("IDSW per sequence\nV3 vs V4a_6")
    ax.legend()

    # Panel 2: ΔIDSW vs sequence length
    ax = axes[1]
    ax.scatter(stats["lengths"], stats["delta_idsw"],
               color="orange", alpha=0.8, s=60)
    # Trend line
    z = np.polyfit(stats["lengths"], stats["delta_idsw"], 1)
    x_line = np.linspace(min(stats["lengths"]), max(stats["lengths"]), 100)
    ax.plot(x_line, np.polyval(z, x_line), "r--", lw=1.5,
            label=f"r={stats['r_delta_idsw_vs_length']:+.3f}")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Sequence length (frames)")
    ax.set_ylabel("ΔIDSW (V4a − V3)")
    ax.set_title("ΔIDSW vs sequence length\n(positive = V4a worse)")
    ax.legend()

    # Panel 3: ΔIDSW vs num GT IDs
    ax = axes[2]
    ax.scatter(stats["n_ids"], stats["delta_idsw"],
               color="orange", alpha=0.8, s=60)
    z2 = np.polyfit(stats["n_ids"], stats["delta_idsw"], 1)
    x2 = np.linspace(min(stats["n_ids"]), max(stats["n_ids"]), 100)
    ax.plot(x2, np.polyval(z2, x2), "r--", lw=1.5,
            label=f"r={stats['r_delta_idsw_vs_n_ids']:+.3f}")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Number of GT IDs in sequence")
    ax.set_ylabel("ΔIDSW (V4a − V3)")
    ax.set_title("ΔIDSW vs number of objects\n(positive = V4a worse)")
    ax.legend()

    plt.suptitle("D5-TEMPORAL: ID switch temporal accumulation analysis",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(od / "D5_temporal.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(od / "D5_result.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Saved: {od}/D5_result.json")
    print(f"  Saved: {od}/D5_temporal.png")

    # Cleanup temp dirs
    shutil.rmtree(v3_tmp, ignore_errors=True)
    shutil.rmtree(v4_tmp, ignore_errors=True)


if __name__ == "__main__":
    main()