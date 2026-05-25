#!/usr/bin/env python3
"""
D3_assign.py
============
D3-ASSIGN: Does reid_proj distort assignment scores, causing threshold miscalibration?

Hypothesis from D2-GEN:
  reid_proj shifts similarity distribution DOWN (intra 0.74→0.55, inter 0.66→0.40).
  Inference thresholds (ID_THRESH=0.2, NEWBORN_THRESH=0.6) were calibrated on V3's
  distribution. With V4a, id_scores may cluster lower → more objects fall below
  id_thresh → false newborns → higher Case B rate → lower AssA.

Measurements:
  1. Max id_score distribution: V3 vs V4a (correctly tracked objects)
  2. False newborn rate: objects classified as newborn despite having valid tracked ID
  3. id_thresh sweep: V4a_3 at [0.05, 0.10, 0.15, 0.20] → find threshold that
     reduces false newborn rate to V3 level

Run from RF-MOTIPV4 repo root:
    python "New folder/D3_assign.py" \
        --config    configs/rf_detrV4_motip_dancetrack.yaml \
        --ckpt_v3   /data/adib/new/github/RF-MOTIP/outputsV3/rfmotip_dancetrack/train/checkpoint_7.pth \
        --ckpt_v4_3 outputs/rfmotip_dancetrack_V3_full/checkpoint_3.pth \
        --ckpt_v4_6 outputs/rfmotip_dancetrack_V3_full/checkpoint_6.pth \
        --data_root /data/pos+mot/Datadir/ \
        --output_dir "New folder/D3_output/" \
        --num_seqs 10
"""

import os, sys, json, argparse
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Args ─────────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     required=True)
    p.add_argument("--ckpt_v3",    required=True)
    p.add_argument("--ckpt_v4_3",  required=True)
    p.add_argument("--ckpt_v4_6",  required=True)
    p.add_argument("--data_root",  default="/data/pos+mot/Datadir/")
    p.add_argument("--output_dir", default="New folder/D3_output/")
    p.add_argument("--num_seqs",   type=int, default=10,
                   help="Number of val sequences to run (all 25 for full diagnostic)")
    p.add_argument("--id_thresh_sweep", type=float, nargs="+",
                   default=[0.05, 0.10, 0.15, 0.20])
    return p.parse_args()


# ── Model loader ──────────────────────────────────────────────────────────────
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

# ── Run one checkpoint on N sequences, collect id_score statistics ────────────
@torch.no_grad()
def run_checkpoint(model, config, data_root, num_seqs, id_thresh,
                   newborn_thresh, device):
    """
    Returns dict with:
      - max_scores_correct: list of max id_scores for correctly assigned objects
      - max_scores_newborn: list of max id_scores for objects assigned as newborn
      - false_newborn_count: objects assigned newborn despite trajectory not empty
      - total_tracked_objects: objects that had a valid trajectory to match
    """
    from models.runtime_tracker import RuntimeTracker
    from models.misc import get_model
    from data.dancetrack import DanceTrack
    from data.seq_dataset import SeqDataset

    dt = DanceTrack(data_root=data_root, split="val", load_annotation=False)
    seq_names = sorted(dt.sequence_infos.keys())[:num_seqs]

    scores_correct  = []   # max id_score when correctly assigned
    scores_newborn  = []   # max id_score when assigned as newborn
    false_newborns  = 0    # newborn despite having valid trajectory
    total_tracked   = 0    # times an object had non-empty trajectory
    all_max_scores  = []   # every max id_score regardless of assignment

    for seq_name in seq_names:
        seq_ds = SeqDataset(
            seq_info=dt.sequence_infos[seq_name],
            image_paths=dt.image_paths[seq_name],
            max_shorter=800,
            max_longer=config.get("INFERENCE_MAX_LONGER", 1440),
            size_divisibility=config.get("SIZE_DIVISIBILITY", 32),
            dtype=torch.float32,
        )
        loader = DataLoader(seq_ds, batch_size=1, shuffle=False,
                            num_workers=0, collate_fn=lambda x: x[0])

        rt = RuntimeTracker(
            model=model,
            sequence_hw=seq_ds.seq_hw(),
            use_sigmoid=config.get("USE_FOCAL_LOSS", False),
            assignment_protocol=config.get("ASSIGNMENT_PROTOCOL", "object-max"),
            miss_tolerance=config.get("MISS_TOLERANCE", 30),
            det_thresh=config.get("DET_THRESH", 0.3),
            newborn_thresh=newborn_thresh,
            id_thresh=id_thresh,
            area_thresh=config.get("AREA_THRESH", 0),
            only_detr=False,
            dtype=torch.float32,
        )
        num_vocab = rt.num_id_vocabulary

        # ── Patch _object_max_assignment to capture id_scores ─────────────
        captured = {"id_scores": None, "id_labels": None,
                    "traj_labels": None}

        orig_oma = rt._object_max_assignment

        def patched_oma(id_scores):
            captured["id_scores"]  = id_scores.detach().cpu()
            captured["traj_labels"] = (
                rt.trajectory_id_labels[0].cpu()
                if rt.trajectory_id_labels is not None
                   and rt.trajectory_id_labels.numel() > 0
                else torch.tensor([])
            )
            result = orig_oma(id_scores=id_scores)
            captured["id_labels"] = result
            return result

        rt._object_max_assignment = patched_oma

        for frame_data, _ in loader:
            frame_data = frame_data.to(device)
            rt.update(frame_data)

            id_scores  = captured.get("id_scores")
            id_labels  = captured.get("id_labels")
            traj_labels = captured.get("traj_labels")

            if id_scores is None or id_labels is None:
                continue

            has_trajectory = (traj_labels is not None and
                              len(traj_labels) > 0 and
                              traj_labels.numel() > 0)

            for obj_idx in range(len(id_labels)):
                max_score = float(id_scores[obj_idx].max())
                all_max_scores.append(max_score)
                assigned  = id_labels[obj_idx]

                if assigned != num_vocab:
                    scores_correct.append(max_score)
                else:
                    scores_newborn.append(max_score)
                    if has_trajectory:
                        total_tracked += 1
                        false_newborns += 1

        print(f"    {seq_name}: "
              f"correct={len(scores_correct)}  "
              f"newborn={len(scores_newborn)}  "
              f"false_nb={false_newborns}")

    return {
        "max_scores_correct": scores_correct,
        "max_scores_newborn": scores_newborn,
        "false_newborns":     false_newborns,
        "total_tracked":      total_tracked,
        "all_max_scores":     all_max_scores,
        "mean_correct_score": float(np.mean(scores_correct)) if scores_correct else 0.0,
        "mean_newborn_score": float(np.mean(scores_newborn)) if scores_newborn else 0.0,
        "false_newborn_rate": false_newborns / max(total_tracked, 1),
        "newborn_rate":       len(scores_newborn) / max(len(scores_correct) + len(scores_newborn), 1),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args   = get_args()
    od     = Path(args.output_dir)
    od.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Sequences: {args.num_seqs}\n")

    DEFAULT_ID_THRESH      = 0.2
    DEFAULT_NEWBORN_THRESH = 0.6

    results = {}

    # ── Step 1: V3 at default thresholds ─────────────────────────────────────
    print("=" * 65)
    print("STEP 1: V3 (no reid_proj) — default thresholds")
    print("=" * 65)
    model_v3, config = load_model(args.config, args.ckpt_v3, device, use_reid_proj=False)
    results["V3_default"] = run_checkpoint(
        model_v3, config, args.data_root, args.num_seqs,
        id_thresh=DEFAULT_ID_THRESH,
        newborn_thresh=DEFAULT_NEWBORN_THRESH, device=device)
    del model_v3; torch.cuda.empty_cache()

    # ── Step 2: V4a_3 at default thresholds ──────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 2: V4a checkpoint_3 (peak) — default thresholds")
    print("=" * 65)
    model_v4_3, _ = load_model(args.config, args.ckpt_v4_3, device)
    results["V4a_3_default"] = run_checkpoint(
        model_v4_3, config, args.data_root, args.num_seqs,
        id_thresh=DEFAULT_ID_THRESH,
        newborn_thresh=DEFAULT_NEWBORN_THRESH, device=device)

    # ── Step 3: V4a_3 threshold sweep ────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 3: V4a checkpoint_3 — id_thresh sweep")
    print("=" * 65)
    for thresh in args.id_thresh_sweep:
        key = f"V4a_3_thresh_{thresh:.2f}"
        print(f"\n  id_thresh={thresh:.2f}")
        results[key] = run_checkpoint(
            model_v4_3, config, args.data_root, args.num_seqs,
            id_thresh=thresh,
            newborn_thresh=DEFAULT_NEWBORN_THRESH, device=device)
    del model_v4_3; torch.cuda.empty_cache()

    # ── Step 4: V4a_6 at default thresholds ──────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 4: V4a checkpoint_6 (post-LR-drop) — default thresholds")
    print("=" * 65)
    model_v4_6, _ = load_model(args.config, args.ckpt_v4_6, device)
    results["V4a_6_default"] = run_checkpoint(
        model_v4_6, config, args.data_root, args.num_seqs,
        id_thresh=DEFAULT_ID_THRESH,
        newborn_thresh=DEFAULT_NEWBORN_THRESH, device=device)
    del model_v4_6; torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("D3-ASSIGN VERDICT")
    print("=" * 65)
    print(f"\n  {'Condition':<35} {'Mean correct':>13} {'Mean newborn':>13} "
          f"{'Newborn rate':>13} {'False NB rate':>14}")
    print(f"  {'-'*90}")

    v3_newborn_rate = results["V3_default"]["newborn_rate"]
    v3_false_rate   = results["V3_default"]["false_newborn_rate"]

    for key, r in results.items():
        marker = " ← baseline" if key == "V3_default" else ""
        print(f"  {key:<35} "
              f"{r['mean_correct_score']:>13.4f} "
              f"{r['mean_newborn_score']:>13.4f} "
              f"{r['newborn_rate']:>13.4f} "
              f"{r['false_newborn_rate']:>14.4f}{marker}")

    # Confirm threshold miscalibration
    v4_newborn_rate = results["V4a_3_default"]["newborn_rate"]
    thresh_confirmed = v4_newborn_rate > v3_newborn_rate * 1.1  # 10% higher

    # Find best threshold for V4a_3
    best_thresh, best_diff = DEFAULT_ID_THRESH, float("inf")
    for thresh in args.id_thresh_sweep:
        key  = f"V4a_3_thresh_{thresh:.2f}"
        diff = abs(results[key]["newborn_rate"] - v3_newborn_rate)
        if diff < best_diff:
            best_diff, best_thresh = diff, thresh

    print(f"\n  Threshold miscalibration: {'CONFIRMED' if thresh_confirmed else 'NOT CONFIRMED'}")
    print(f"  V3 newborn rate:          {v3_newborn_rate:.4f}")
    print(f"  V4a_3 newborn rate:       {v4_newborn_rate:.4f} "
          f"({'higher ↑' if v4_newborn_rate > v3_newborn_rate else 'lower ↓'})")

    if thresh_confirmed:
        print(f"\n  Best id_thresh for V4a to match V3 newborn rate: {best_thresh:.2f}")
        print(f"""
  → reid_proj shifts id_score distribution DOWN, making objects fall
    below id_thresh=0.2 more often → false newborns → lower AssA.
  → EVIDENCE: V4a requires lower id_thresh to match V3 performance.
  → NEXT ACTION: retrain V4b with recalibrated id_thresh={best_thresh:.2f},
    OR keep thresholds and accept the performance cost as a known
    limitation to report in the paper.""")
    else:
        print("""
  → Threshold miscalibration NOT confirmed.
    Newborn rates are similar between V3 and V4a.
  → The AssA gap has a different root cause.
  → Investigate: gradient interaction between reid_proj and IDDecoder.""")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: mean correct score comparison
    ax = axes[0]
    keys   = ["V3_default", "V4a_3_default", "V4a_6_default"]
    labels = ["V3", "V4a_3\n(peak)", "V4a_6\n(post-LR)"]
    vals   = [results[k]["mean_correct_score"] for k in keys]
    ax.bar(labels, vals, color=["steelblue", "orange", "red"], alpha=0.8)
    ax.axhline(DEFAULT_ID_THRESH, ls="--", color="black",
               label=f"id_thresh={DEFAULT_ID_THRESH}")
    ax.set_ylabel("Mean max id_score (correctly assigned objects)")
    ax.set_title("Mean correct id_score\n(should stay well above id_thresh)")
    ax.legend()
    for i, v in enumerate(vals):
        ax.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=9)

    # Panel 2: newborn rate comparison
    ax = axes[1]
    vals = [results[k]["newborn_rate"] for k in keys]
    ax.bar(labels, vals, color=["steelblue", "orange", "red"], alpha=0.8)
    ax.axhline(v3_newborn_rate, ls="--", color="steelblue", alpha=0.6,
               label="V3 baseline")
    ax.set_ylabel("Newborn rate (newborns / total detections)")
    ax.set_title("Newborn rate\n(higher = more false newborns)")
    ax.legend()
    for i, v in enumerate(vals):
        ax.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=9)

    # Panel 3: id_thresh sweep
    ax = axes[2]
    sweep_keys = [f"V4a_3_thresh_{t:.2f}" for t in args.id_thresh_sweep]
    sweep_vals = [results[k]["newborn_rate"] for k in sweep_keys if k in results]
    sweep_x    = [t for t in args.id_thresh_sweep if f"V4a_3_thresh_{t:.2f}" in results]
    ax.plot(sweep_x, sweep_vals, marker="o", color="orange",
            label="V4a_3 newborn rate")
    ax.axhline(v3_newborn_rate, ls="--", color="steelblue",
               label=f"V3 baseline ({v3_newborn_rate:.4f})")
    ax.set_xlabel("id_thresh")
    ax.set_ylabel("Newborn rate")
    ax.set_title("id_thresh sweep for V4a_3\n(find threshold matching V3)")
    ax.legend()

    plt.suptitle("D3-ASSIGN: Threshold Miscalibration Analysis", fontsize=13)
    plt.tight_layout()
    plt.savefig(od / "D3_assign.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Save ──────────────────────────────────────────────────────────────────
    save = {k: {kk: vv for kk, vv in v.items()
                if not isinstance(vv, list)}   # skip large score lists
            for k, v in results.items()}
    save["threshold_miscalibration_confirmed"] = bool(thresh_confirmed)
    save["best_id_thresh_for_v4a"] = float(best_thresh)
    save["v3_newborn_rate"] = float(v3_newborn_rate)

    with open(od / "D3_result.json", "w") as f:
        json.dump(save, f, indent=2)
    print(f"\n  Saved: {od}/D3_result.json")
    print(f"  Saved: {od}/D3_assign.png")


if __name__ == "__main__":
    main()