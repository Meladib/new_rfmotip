#!/usr/bin/env python3
"""
diag_script3_newborn_rate.py
=============================
Diagnostic 3 — Newborn Rate vs Ground-Truth Newborn Rate

Hypothesis to verify
--------------------
The model predicts spurious newborns in crowded scenes because
hard_neg_sim=0.921 causes objects to score below id_thresh=0.2
for ALL existing trajectories simultaneously, triggering the
newborn token assignment for objects that already exist.

If confirmed:
  predicted_newborns_per_frame >> gt_newborns_per_frame
  AND spurious newborn rate correlates with local crowd density
  → identity fragmentation confirmed
  → Phase 3 (appearance-conditioned newborn init) justified

If refuted:
  predicted_newborns ≈ gt_newborns
  → fragmentation is not from spurious newborns but from
    incorrect trajectory re-association

Metrics computed
----------------
Per frame:
  - gt_newborns:          objects appearing for the first time in this frame
  - pred_newborns:        objects the model assigns the newborn token
  - spurious_newborns:    pred_newborns that correspond to already-tracked GT objects
  - missed_reids:         gt objects that should match existing tracks but were made newborn
  - crowd_density:        mean pairwise IoU in that frame (proxy for crowding)

Aggregate:
  - newborn_inflation_ratio: mean(pred_newborns) / mean(gt_newborns)
  - spurious_rate:           spurious_newborns / pred_newborns
  - correlation(spurious_rate, crowd_density)

Run from inside MOTIP/:
  python diagnostics/diag_script3_newborn_rate.py \\
    --config configs/rf_detr_motip_dancetrack.yaml \\
    --checkpoint outputs/rfmotip_dancetrack/checkpoint_3.pth \\
    --sequence_dir /data/DanceTrack/val/dancetrack0004 \\
    --output_dir diagnostics/diag3_newborn_rate/ \\
    --num_frames 200
"""

import sys
import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",       required=True)
    p.add_argument("--checkpoint",   required=True)
    p.add_argument("--sequence_dir", required=True,
                   help="Single DanceTrack sequence dir with img1/ and gt/gt.txt")
    p.add_argument("--output_dir",   default="diagnostics/diag3_newborn_rate/")
    p.add_argument("--num_frames",   type=int, default=200)
    p.add_argument("--device",       default=None)
    return p.parse_args()


def load_sequence(sequence_dir):
    import torch
    from configparser import ConfigParser

    seq = Path(sequence_dir)
    ini = ConfigParser()
    ini.read(seq / "seqinfo.ini")
    img_w   = int(ini["Sequence"]["imWidth"])
    img_h   = int(ini["Sequence"]["imHeight"])
    seq_len = int(ini["Sequence"]["seqLength"])

    image_paths = [str(seq / "img1" / f"{i + 1:08d}.jpg")
                   for i in range(seq_len)]

    fd = defaultdict(lambda: {"ids": [], "bboxes": []})
    with open(seq / "gt" / "gt.txt") as f:
        for line in f:
            parts = line.strip().split(",")
            fid  = int(parts[0])
            oid  = int(parts[1])
            xywh = [float(parts[2]), float(parts[3]),
                    float(parts[4]), float(parts[5])]
            fd[fid]["ids"].append(oid)
            fd[fid]["bboxes"].append(xywh)

    annotations = []
    for i in range(seq_len):
        fid = i + 1
        if fd[fid]["ids"]:
            ann = {
                "id":   torch.tensor(fd[fid]["ids"],    dtype=torch.int64),
                "bbox": torch.tensor(fd[fid]["bboxes"], dtype=torch.float32),
                "is_legal": True,
            }
        else:
            ann = {
                "id":   torch.zeros(0, dtype=torch.int64),
                "bbox": torch.zeros((0, 4), dtype=torch.float32),
                "is_legal": True,
            }
        annotations.append(ann)

    return image_paths, annotations, img_w, img_h, seq_len


def compute_gt_newborns(annotations):
    """
    For each frame, determine which GT track IDs appear for the first time.
    Returns List[Set[int]] — set of new track IDs per frame.
    """
    seen_ids = set()
    gt_newborns = []
    for ann in annotations:
        ids = ann["id"].tolist()
        new_this_frame = set(ids) - seen_ids
        gt_newborns.append(new_this_frame)
        seen_ids.update(ids)
    return gt_newborns


def compute_crowd_density(bboxes_xywh, img_w, img_h):
    """
    Mean pairwise IoU of GT boxes in a frame.
    Returns float in [0, 1]. 0 = no overlap, 1 = full overlap.
    """
    import torch

    if len(bboxes_xywh) < 2:
        return 0.0

    x, y, w, h = bboxes_xywh.unbind(-1)
    x1 = x / img_w
    y1 = y / img_h
    x2 = (x + w) / img_w
    y2 = (y + h) / img_h
    boxes = torch.stack([x1, y1, x2, y2], -1)   # (N, 4) xyxy

    inter_x1 = torch.max(boxes[:, None, 0], boxes[None, :, 0])
    inter_y1 = torch.max(boxes[:, None, 1], boxes[None, :, 1])
    inter_x2 = torch.min(boxes[:, None, 2], boxes[None, :, 2])
    inter_y2 = torch.min(boxes[:, None, 3], boxes[None, :, 3])
    inter = ((inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0))

    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area[:, None] + area[None, :] - inter
    iou = inter / (union + 1e-6)
    iou.fill_diagonal_(0.0)

    n = len(boxes)
    if n < 2:
        return 0.0
    return float(iou.sum() / (n * (n - 1)))


def instrument_runtime_tracker(rt, num_id_vocabulary):
    """
    Monkey-patch RuntimeTracker to log per-frame newborn counts and
    the set of GT-matched track IDs that were mistakenly made newborn.

    Returns container that accumulates per-frame stats.
    """
    container = {
        "per_frame": [],   # list of dicts
    }

    original_assign_newborn = rt._assign_newborn_id_labels
    original_update_track   = rt._update_trajectory_infos

    # Track what labels exist before assignment
    def patched_assign_newborn(pred_id_labels):
        n_pred_newborns = int((pred_id_labels == num_id_vocabulary).sum().item())
        existing_ids    = set(rt.trajectory_id_labels[0].tolist()) \
                          if rt.trajectory_id_labels.shape[0] > 0 else set()

        result = original_assign_newborn(pred_id_labels)

        container["_last_n_pred_newborns"] = n_pred_newborns
        container["_last_existing_ids"]    = existing_ids
        return result

    rt._assign_newborn_id_labels = patched_assign_newborn

    return container


def main():
    args = get_args()

    import torch
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    import torchvision.transforms.functional as TF
    from utils.misc import yaml_to_dict
    from configs.util import load_super_config
    from models.motip import build as build_motip
    from models.misc import load_checkpoint
    from models.runtime_tracker import RuntimeTracker
    from utils.nested_tensor import nested_tensor_from_tensor_list

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    od = Path(args.output_dir)
    od.mkdir(parents=True, exist_ok=True)

    config = yaml_to_dict(args.config)
    config = load_super_config(config, config.get("SUPER_CONFIG_PATH"))

    # ── Load model ────────────────────────────────────────────────────
    print("Loading model...")
    model, _ = build_motip(config=config)
    load_checkpoint(model, path=args.checkpoint)
    model.eval().to(device)

    num_id_vocab = config.get("NUM_ID_VOCABULARY", 50)

    # ── Load sequence ─────────────────────────────────────────────────
    image_paths, annotations, img_w, img_h, seq_len = load_sequence(args.sequence_dir)
    n_frames = min(args.num_frames, seq_len)
    print(f"Processing {n_frames} frames from {Path(args.sequence_dir).name}")

    # ── Compute GT newborns per frame ─────────────────────────────────
    gt_newborns_per_frame = compute_gt_newborns(annotations[:n_frames])

    # ── Set up RuntimeTracker ─────────────────────────────────────────
    rt = RuntimeTracker(
        model=model,
        sequence_hw=(img_h, img_w),
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

    MEANS     = [0.485, 0.456, 0.406]
    STDS      = [0.229, 0.224, 0.225]
    SIZE_DIV  = config.get("SIZE_DIVISIBILITY", 32)
    MAX_LONGER = config.get("INFERENCE_MAX_LONGER", 1440)

    # ── Per-frame tracking ────────────────────────────────────────────
    frame_stats = []
    # Track assigned IDs per frame to detect spurious newborns
    cumulative_gt_ids = set()

    with torch.no_grad():
        for t in range(n_frames):
            ann = annotations[t]
            gt_ids_this_frame = set(ann["id"].tolist()) if len(ann["id"]) > 0 else set()

            # Crowd density for this frame
            crowd = (compute_crowd_density(ann["bbox"], img_w, img_h)
                     if len(ann["bbox"]) >= 2 else 0.0)

            # GT newborns
            n_gt_new = len(gt_newborns_per_frame[t])

            # Run tracker
            img = Image.open(image_paths[t]).convert("RGB")
            img_t = TF.to_tensor(img)

            h_orig, w_orig = img_t.shape[-2:]
            shorter = min(h_orig, w_orig)
            longer  = max(h_orig, w_orig)
            scale   = 800.0 / shorter
            if longer * scale > MAX_LONGER:
                scale = MAX_LONGER / longer
            new_h = int(round(h_orig * scale))
            new_w = int(round(w_orig * scale))
            img_t = TF.resize(img_t, [new_h, new_w])
            img_t = TF.normalize(img_t, MEANS, STDS)

            frame = nested_tensor_from_tensor_list([img_t], SIZE_DIV).to(device)

            # Capture pred ID labels from tracker
            # Temporarily wrap the assignment to capture newborn count
            original_object_max = rt._object_max_assignment

            captured = {"n_pred_newborns": 0, "pred_labels": []}

            def patched_assignment(id_scores):
                labels = original_object_max(id_scores)
                n_new  = sum(1 for l in labels if l == num_id_vocab)
                captured["n_pred_newborns"] = n_new
                captured["pred_labels"]     = labels
                return labels

            rt._object_max_assignment = patched_assignment

            rt.update(image=frame)
            rt._object_max_assignment = original_object_max

            # Determine spurious newborns:
            # Objects that were assigned newborn but whose GT ID was already seen
            pred_labels = captured["pred_labels"]
            n_pred_new  = captured["n_pred_newborns"]

            # Track which GT IDs were previously seen
            cumulative_gt_ids.update(gt_newborns_per_frame[t])

            # Note: we can't perfectly match pred newborns to GT IDs without
            # full Hungarian matching, so we use the following proxy:
            # If pred_newborns > gt_newborns for this frame AND crowd > threshold,
            # those extra newborns are likely spurious.
            spurious_proxy = max(0, n_pred_new - n_gt_new)

            frame_stats.append({
                "frame":             t,
                "gt_newborns":       n_gt_new,
                "pred_newborns":     n_pred_new,
                "spurious_proxy":    spurious_proxy,
                "crowd_density":     crowd,
                "n_gt_objects":      len(gt_ids_this_frame),
            })

            if (t + 1) % 50 == 0:
                print(f"  Frame {t + 1}/{n_frames}: "
                      f"gt_new={n_gt_new} pred_new={n_pred_new} "
                      f"crowd={crowd:.3f}")

    # ── Aggregate ─────────────────────────────────────────────────────
    gt_news   = [s["gt_newborns"]   for s in frame_stats]
    pred_news = [s["pred_newborns"] for s in frame_stats]
    spurious  = [s["spurious_proxy"] for s in frame_stats]
    crowds    = [s["crowd_density"]  for s in frame_stats]

    mean_gt_new   = float(np.mean(gt_news))
    mean_pred_new = float(np.mean(pred_news))
    inflation     = mean_pred_new / (mean_gt_new + 1e-6)

    spurious_rate = (float(np.sum(spurious)) /
                     float(max(np.sum(pred_news), 1)))

    # Correlation between spurious newborns and crowd density
    if len(crowds) > 5 and np.std(crowds) > 1e-6:
        corr = float(np.corrcoef(crowds, [s["spurious_proxy"] for s in frame_stats])[0, 1])
    else:
        corr = 0.0

    # ── Plots ─────────────────────────────────────────────────────────
    frames = [s["frame"] for s in frame_stats]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Panel 1: GT vs pred newborns per frame
    ax = axes[0]
    ax.bar(frames, gt_news,   alpha=0.6, color="green", label=f"GT newborns (mean={mean_gt_new:.2f})")
    ax.bar(frames, pred_news, alpha=0.4, color="red",   label=f"Pred newborns (mean={mean_pred_new:.2f})")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Count")
    ax.set_title(f"GT vs Predicted Newborns per Frame\n"
                 f"Inflation ratio = {inflation:.2f}× "
                 f"({'FRAGMENTATION' if inflation > 1.5 else 'NORMAL'})")
    ax.legend()

    # Panel 2: Spurious newborns vs crowd density
    ax = axes[1]
    ax2 = ax.twinx()
    ax.bar(frames, spurious, alpha=0.6, color="orange", label="Spurious newborns (proxy)")
    ax2.plot(frames, crowds, color="purple", alpha=0.6, lw=1.5, label="Crowd density")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Spurious newborns", color="orange")
    ax2.set_ylabel("Crowd density (mean pairwise IoU)", color="purple")
    ax.set_title(f"Spurious Newborns vs Crowd Density\nCorrelation = {corr:.3f}")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    # Panel 3: Cumulative newborn inflation
    ax = axes[2]
    cum_gt   = np.cumsum(gt_news)
    cum_pred = np.cumsum(pred_news)
    ax.plot(frames, cum_gt,   color="green", lw=2, label="Cumulative GT newborns")
    ax.plot(frames, cum_pred, color="red",   lw=2, linestyle="--",
            label="Cumulative pred newborns")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Cumulative count")
    ax.set_title(f"Cumulative Newborn Inflation\n"
                 f"Spurious rate = {spurious_rate:.1%} of all predicted newborns")
    ax.legend()

    plt.suptitle(
        f"Newborn Rate Diagnostic — {Path(args.sequence_dir).name}\n"
        f"Inflation={inflation:.2f}×  Spurious={spurious_rate:.1%}  "
        f"Crowd-Spurious corr={corr:.3f}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(od / "diag3_newborn_rate.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {od}/diag3_newborn_rate.png")

    # ── Save JSON ──────────────────────────────────────────────────────
    if inflation > 1.5 and corr > 0.3:
        interpretation = (
            f"FRAGMENTATION CONFIRMED — inflation={inflation:.2f}×, "
            f"spurious rate={spurious_rate:.1%}, crowd correlation={corr:.3f}. "
            "Phase 3 (appearance-conditioned newborn init) is justified."
        )
    elif inflation > 1.5 and corr <= 0.3:
        interpretation = (
            f"FRAGMENTATION PRESENT but NOT crowd-driven — inflation={inflation:.2f}×. "
            "Spurious newborns occur uniformly, not in crowds. "
            "May be caused by id_thresh tuning rather than structural issue."
        )
    else:
        interpretation = (
            f"FRAGMENTATION ABSENT — inflation={inflation:.2f}×. "
            "Newborn rate matches GT closely. "
            "AssRe limitation comes from re-ID failure, not spurious newborns."
        )

    results = {
        "sequence":              Path(args.sequence_dir).name,
        "n_frames":              n_frames,
        "mean_gt_newborns":      mean_gt_new,
        "mean_pred_newborns":    mean_pred_new,
        "newborn_inflation_ratio": inflation,
        "spurious_newborn_rate": spurious_rate,
        "crowd_spurious_correlation": corr,
        "interpretation":        interpretation,
        "per_frame":             frame_stats,
    }

    with open(od / "diag3_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nRESULTS SUMMARY:")
    print(f"  GT newborns/frame:    {mean_gt_new:.2f}")
    print(f"  Pred newborns/frame:  {mean_pred_new:.2f}")
    print(f"  Inflation ratio:      {inflation:.2f}×")
    print(f"  Spurious rate:        {spurious_rate:.1%}")
    print(f"  Crowd correlation:    {corr:.3f}")
    print(f"\n  Interpretation: {interpretation}")
    print(f"\nOutputs → {od}/")


if __name__ == "__main__":
    main()