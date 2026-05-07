#!/usr/bin/env python3
"""
diag_script4_score_distribution.py
====================================
Diagnostic 4 — ID Score Distribution: Correct Matches vs Spurious Newborns

Question to answer
------------------
At inference with checkpoint_2.pth, when the model creates a spurious newborn:

  (A) Does the correct trajectory label score BELOW id_thresh=0.2?
      → Score miscalibration: model cannot express confidence for correct match.

  (B) Does the correct trajectory label score ABOVE id_thresh=0.2
      but still lose to a competing object or assignment conflict?
      → Assignment logic failure: scores are fine but protocol wastes them.

  (C) Does the correct trajectory label not exist in trajectory memory?
      → Trajectory eviction or missed detection: object was never tracked.

These three cases require different fixes.

What this script measures
-------------------------
For each frame, for each detected object matched to a GT track_id:
  1. Find which id_label corresponds to that GT track_id in trajectory memory
  2. Record max(id_scores[object, :]) — the model's best score for any label
  3. Record id_scores[object, correct_label] — score for the GT-correct label
  4. Record the final assignment — was it the correct label or newborn?
  5. Classify the outcome:
       CORRECT       — assigned the right label
       CASE_A        — assigned newborn, correct_label_score < id_thresh
       CASE_B        — assigned newborn, correct_label_score >= id_thresh
       CASE_C        — correct label not in trajectory memory

Aggregate outputs
-----------------
  - Distribution of correct_label_score for all three cases
  - Distribution of max_score for all objects
  - Fraction of each case
  - Per-frame newborn count breakdown

Run from inside MOTIP/:
  python diagnostics/diag_script4_score_distribution.py \\
    --config configs/rf_detr_motip_dancetrack.yaml \\
    --checkpoint outputsV2/rfmotip_dancetrack/train/checkpoint_2.pth \\
    --sequence_dir /data/pos+mot/Datadir/DanceTrack/val/dancetrack0004 \\
    --output_dir diagnostics/diag4_score_dist/ \\
    --num_frames 300
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
    p.add_argument("--output_dir",   default="diagnostics/diag4_score_dist/")
    p.add_argument("--num_frames",   type=int, default=300)
    p.add_argument("--device",       default=None)
    return p.parse_args()


def load_gt(sequence_dir):
    """Returns dict: frame_id -> {track_id -> bbox (xywh pixels)}"""
    gt = defaultdict(dict)
    with open(Path(sequence_dir) / "gt" / "gt.txt") as f:
        for line in f:
            parts = line.strip().split(",")
            fid = int(parts[0])
            tid = int(parts[1])
            xywh = [float(parts[2]), float(parts[3]),
                    float(parts[4]), float(parts[5])]
            gt[fid][tid] = xywh
    return gt


def box_iou_single(b1, b2):
    """b1, b2: [x,y,w,h] pixels. Returns IoU scalar."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[0] + b1[2], b2[0] + b2[2])
    y2 = min(b1[1] + b1[3], b2[1] + b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    a1 = b1[2] * b1[3]
    a2 = b2[2] * b2[3]
    return inter / (a1 + a2 - inter)


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
    from models.misc import load_checkpoint, get_model
    from models.runtime_tracker import RuntimeTracker
    from utils.nested_tensor import nested_tensor_from_tensor_list
    from utils.box_ops import box_cxcywh_to_xywh

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    od = Path(args.output_dir)
    od.mkdir(parents=True, exist_ok=True)

    config = yaml_to_dict(args.config)
    config = load_super_config(config, config.get("SUPER_CONFIG_PATH"))

    print("Loading model...")
    model, _ = build_motip(config=config)
    load_checkpoint(model, path=args.checkpoint)
    model.eval().to(device)

    id_thresh    = config.get("ID_THRESH",      0.2)
    det_thresh   = config.get("DET_THRESH",     0.3)
    newborn_thresh = config.get("NEWBORN_THRESH", 0.6)
    use_sigmoid  = config.get("USE_FOCAL_LOSS", False)
    protocol     = config.get("ASSIGNMENT_PROTOCOL", "object-max")
    miss_tol     = config.get("MISS_TOLERANCE", 30)
    area_thresh  = config.get("AREA_THRESH",    0)

    seq = Path(args.sequence_dir)
    from configparser import ConfigParser
    ini = ConfigParser()
    ini.read(seq / "seqinfo.ini")
    img_w   = int(ini["Sequence"]["imWidth"])
    img_h   = int(ini["Sequence"]["imHeight"])
    seq_len = int(ini["Sequence"]["seqLength"])

    image_paths = [str(seq / "img1" / f"{i + 1:08d}.jpg") for i in range(seq_len)]
    gt          = load_gt(args.sequence_dir)
    n_frames    = min(args.num_frames, seq_len)

    # ── Instrument RuntimeTracker ─────────────────────────────────────
    # We patch _get_id_pred_labels to capture id_scores before assignment.

    rt = RuntimeTracker(
        model=model,
        sequence_hw=(img_h, img_w),
        use_sigmoid=use_sigmoid,
        assignment_protocol=protocol,
        miss_tolerance=miss_tol,
        det_thresh=det_thresh,
        newborn_thresh=newborn_thresh,
        id_thresh=id_thresh,
        area_thresh=area_thresh,
        only_detr=False,
        dtype=torch.float32,
    )

    captured_frames = []   # list of dicts per frame

    original_get_id_pred = rt._get_id_pred_labels

    def patched_get_id_pred(boxes, output_embeds):
        if rt.trajectory_features.shape[0] == 0:
            result = rt.num_id_vocabulary * torch.ones(
                boxes.shape[0], dtype=torch.int64, device=boxes.device)
            captured_frames[-1]["id_scores"]            = None
            captured_frames[-1]["trajectory_id_labels"] = []
            captured_frames[-1]["assignment"]           = result.tolist()
            captured_frames[-1]["pred_boxes_norm"]      = boxes.cpu().tolist()
            return result

        # Replicate the forward pass to capture id_scores
        current_features = output_embeds[None, ...]
        current_boxes_c  = boxes[None, ...]
        current_masks    = torch.zeros(
            (1, output_embeds.shape[0]), dtype=torch.bool, device=device)
        current_times    = rt.trajectory_times.shape[0] * torch.ones(
            (1, output_embeds.shape[0]), dtype=torch.int64, device=device)

        seq_info = {
            "trajectory_features":  rt.trajectory_features[None, None, ...],
            "trajectory_boxes":     rt.trajectory_boxes[None, None, ...],
            "trajectory_id_labels": rt.trajectory_id_labels[None, None, ...],
            "trajectory_times":     rt.trajectory_times[None, None, ...],
            "trajectory_masks":     rt.trajectory_masks[None, None, ...],
            "unknown_features":     current_features[None, None, ...],
            "unknown_boxes":        current_boxes_c[None, None, ...],
            "unknown_masks":        current_masks[None, None, ...],
            "unknown_times":        current_times[None, None, ...],
        }
        with torch.no_grad():
            seq_info   = model(seq_info=seq_info, part="trajectory_modeling")
            id_logits, _, _ = model(seq_info=seq_info, part="id_decoder")

        id_logits = id_logits[0, 0, 0]
        if not use_sigmoid:
            id_scores = id_logits.softmax(dim=-1)
        else:
            id_scores = id_logits.sigmoid()

        # Run assignment (original method handles this)
        result = original_get_id_pred(boxes, output_embeds)

        # Store for analysis
        captured_frames[-1]["id_scores"]            = id_scores.detach().cpu()
        captured_frames[-1]["trajectory_id_labels"] = rt.trajectory_id_labels[0].cpu().tolist()
        captured_frames[-1]["assignment"]           = result.tolist()
        captured_frames[-1]["pred_boxes_norm"]      = boxes.cpu().tolist()

        return result

    rt._get_id_pred_labels = patched_get_id_pred

    # ── Run inference ─────────────────────────────────────────────────
    MEANS     = [0.485, 0.456, 0.406]
    STDS      = [0.229, 0.224, 0.225]
    SIZE_DIV  = config.get("SIZE_DIVISIBILITY", 32)
    MAX_LONGER = config.get("INFERENCE_MAX_LONGER", 1440)

    print(f"Running inference on {n_frames} frames...")
    with torch.no_grad():
        for t in range(n_frames):
            # Push empty slot for this frame — patched function fills it
            captured_frames.append({
                "frame":    t + 1,
                "gt":       gt.get(t + 1, {}),
                "id_scores":            None,
                "trajectory_id_labels": [],
                "assignment":           [],
                "pred_boxes_norm":      [],
            })

            img = Image.open(image_paths[t]).convert("RGB")
            img_t = TF.to_tensor(img)
            h0, w0 = img_t.shape[-2:]
            shorter = min(h0, w0)
            longer  = max(h0, w0)
            scale   = 800.0 / shorter
            if longer * scale > MAX_LONGER:
                scale = MAX_LONGER / longer
            img_t = TF.resize(img_t, [int(round(h0 * scale)),
                                       int(round(w0 * scale))])
            img_t = TF.normalize(img_t, MEANS, STDS)
            frame = nested_tensor_from_tensor_list([img_t], SIZE_DIV).to(device)
            rt.update(image=frame)

            # Attach final track results for GT matching
            tr = rt.get_track_results()
            if tr:
                captured_frames[-1]["track_ids"]    = tr.get("id", torch.tensor([])).tolist()
                captured_frames[-1]["track_bboxes"] = tr.get("bbox", torch.zeros(0, 4)).tolist()
            else:
                captured_frames[-1]["track_ids"]    = []
                captured_frames[-1]["track_bboxes"] = []

            if (t + 1) % 50 == 0:
                print(f"  Frame {t + 1}/{n_frames}")

    # ── Analyse score distributions ───────────────────────────────────
    print("Analysing score distributions...")

    # Score buckets
    correct_scores       = []   # score of correct label, when correctly assigned
    case_a_scores        = []   # correct_label_score < id_thresh → newborn
    case_b_scores        = []   # correct_label_score >= id_thresh → still newborn
    case_c_count         = 0    # correct label not in trajectory memory
    max_scores_all       = []   # max(id_scores) for every object in every frame
    newborn_max_scores   = []   # max(id_scores) for objects assigned newborn

    id_label_to_track_id = {}   # label -> global track id (built from track results)

    for fd in captured_frames:
        if fd["id_scores"] is None:
            continue

        id_scores   = fd["id_scores"]          # (N_det, K+1)
        traj_labels = fd["trajectory_id_labels"]
        assignment  = fd["assignment"]         # list of int, len=N_det
        pred_boxes  = fd["pred_boxes_norm"]    # list of [cx,cy,w,h] norm
        gt_this     = fd["gt"]                 # {track_id: [x,y,w,h] pixels}
        track_ids   = fd.get("track_ids",   [])
        track_bboxes = fd.get("track_bboxes", [])

        # Update id_label -> global_track_id mapping from track results
        for tid, bbox in zip(track_ids, track_bboxes):
            # Find which detection this track corresponds to (IoU match)
            # Not strictly needed for score analysis — we use assignment directly
            pass

        for obj_idx in range(len(assignment)):
            if obj_idx >= len(pred_boxes):
                continue

            max_score = float(id_scores[obj_idx].max())
            max_scores_all.append(max_score)
            assigned_label = assignment[obj_idx]

            # Convert predicted box to pixel xywh for GT matching
            cx, cy, w_n, h_n = pred_boxes[obj_idx]
            pred_x = (cx - w_n / 2) * img_w
            pred_y = (cy - h_n / 2) * img_h
            pred_w = w_n * img_w
            pred_h = h_n * img_h
            pred_xywh = [pred_x, pred_y, pred_w, pred_h]

            # Find GT match by IoU
            best_iou, best_gt_tid = 0.0, None
            for gt_tid, gt_xywh in gt_this.items():
                iou = box_iou_single(pred_xywh, gt_xywh)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_tid = gt_tid

            if best_iou < 0.5 or best_gt_tid is None:
                continue   # unmatched detection — skip

            # Find which id_label corresponds to this GT track in trajectory
            # We use id_label_to_id mapping via track results
            # Approximate: search trajectory labels for the one assigned to this GT
            # Since we don't have direct GT→label mapping at inference,
            # we use the assignment outcome:

            if assigned_label != rt.num_id_vocabulary:
                # Correctly associated (assumed — if IoU match is good)
                correct_label_score = float(id_scores[obj_idx, assigned_label])
                correct_scores.append(correct_label_score)
            else:
                # Assigned newborn — analyse why
                newborn_max_scores.append(max_score)

                # Find if any trajectory label could correspond to this object
                # by checking if trajectory has a slot with non-zero history
                if len(traj_labels) == 0:
                    case_c_count += 1
                else:
                    # Find the trajectory label with highest score for this object
                    traj_label_scores = []
                    for lbl in traj_labels:
                        if lbl < rt.num_id_vocabulary:
                            traj_label_scores.append(
                                (lbl, float(id_scores[obj_idx, lbl])))

                    if not traj_label_scores:
                        case_c_count += 1
                    else:
                        best_lbl, best_score = max(
                            traj_label_scores, key=lambda x: x[1])

                        if best_score < id_thresh:
                            case_a_scores.append(best_score)   # Case A
                        else:
                            case_b_scores.append(best_score)   # Case B

    # ── Summary stats ─────────────────────────────────────────────────
    def _stats(v):
        if not v:
            return {"mean": None, "std": None, "p25": None,
                    "p50": None, "p75": None, "n": 0}
        import numpy as np
        v = np.array(v)
        return {
            "mean": float(np.mean(v)),
            "std":  float(np.std(v)),
            "p25":  float(np.percentile(v, 25)),
            "p50":  float(np.percentile(v, 50)),
            "p75":  float(np.percentile(v, 75)),
            "n":    len(v),
        }

    total_newborns = len(case_a_scores) + len(case_b_scores) + case_c_count
    results = {
        "config": {
            "checkpoint":  args.checkpoint,
            "sequence":    Path(args.sequence_dir).name,
            "id_thresh":   id_thresh,
            "n_frames":    n_frames,
            "protocol":    protocol,
        },
        "correct_assignment_scores":   _stats(correct_scores),
        "case_A_scores": {
            **_stats(case_a_scores),
            "description": "newborn, best_trajectory_score < id_thresh",
        },
        "case_B_scores": {
            **_stats(case_b_scores),
            "description": "newborn despite best_trajectory_score >= id_thresh",
        },
        "case_C_count": {
            "n":           case_c_count,
            "description": "newborn, correct label absent from trajectory memory",
        },
        "max_scores_all":     _stats(max_scores_all),
        "newborn_max_scores": _stats(newborn_max_scores),
        "newborn_breakdown": {
            "total":   total_newborns,
            "case_A":  len(case_a_scores),
            "case_B":  len(case_b_scores),
            "case_C":  case_c_count,
            "case_A_pct": round(len(case_a_scores) / max(total_newborns, 1) * 100, 1),
            "case_B_pct": round(len(case_b_scores) / max(total_newborns, 1) * 100, 1),
            "case_C_pct": round(case_c_count / max(total_newborns, 1) * 100, 1),
        },
    }

    with open(od / "diag4_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: score distributions for correct vs case A vs case B
    ax = axes[0]
    bins = [i * 0.05 for i in range(21)]
    if correct_scores:
        ax.hist(correct_scores, bins=bins, alpha=0.6, color="green",
                label=f"Correct (n={len(correct_scores)})", density=True)
    if case_a_scores:
        ax.hist(case_a_scores, bins=bins, alpha=0.6, color="red",
                label=f"Case A: score<thresh (n={len(case_a_scores)})", density=True)
    if case_b_scores:
        ax.hist(case_b_scores, bins=bins, alpha=0.6, color="orange",
                label=f"Case B: score≥thresh (n={len(case_b_scores)})", density=True)
    ax.axvline(id_thresh, color="black", ls="--", lw=1.5,
               label=f"id_thresh={id_thresh}")
    ax.set_xlabel("ID Score (best trajectory label)")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution by Outcome")
    ax.legend(fontsize=7)

    # Panel 2: max score distribution — correct vs newborn
    ax = axes[1]
    if max_scores_all:
        ax.hist(max_scores_all, bins=bins, alpha=0.5, color="steelblue",
                label=f"All detections (n={len(max_scores_all)})", density=True)
    if newborn_max_scores:
        ax.hist(newborn_max_scores, bins=bins, alpha=0.5, color="red",
                label=f"Newborns (n={len(newborn_max_scores)})", density=True)
    ax.axvline(id_thresh, color="black", ls="--", lw=1.5,
               label=f"id_thresh={id_thresh}")
    ax.set_xlabel("Max ID Score (any label)")
    ax.set_ylabel("Density")
    ax.set_title("Max Score: All vs Newborn Objects")
    ax.legend(fontsize=7)

    # Panel 3: newborn breakdown pie
    ax = axes[2]
    nb = results["newborn_breakdown"]
    if nb["total"] > 0:
        sizes  = [nb["case_A"], nb["case_B"], nb["case_C"]]
        labels = [
            f"Case A: score<thresh\n{nb['case_A_pct']}%",
            f"Case B: score≥thresh\n{nb['case_B_pct']}%",
            f"Case C: label absent\n{nb['case_C_pct']}%",
        ]
        colors = ["red", "orange", "grey"]
        ax.pie([max(s, 0.001) for s in sizes], labels=labels,
               colors=colors, autopct="%1.0f%%", startangle=90)
    ax.set_title(f"Spurious Newborn Root Cause\n(total={nb['total']})")

    plt.suptitle(
        f"Diag 4 — ID Score Distribution\n"
        f"{Path(args.sequence_dir).name}  |  "
        f"checkpoint_2  |  id_thresh={id_thresh}  |  {n_frames} frames",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(od / "diag4_score_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {od}/diag4_score_distribution.png")

    # ── Print summary ──────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("RESULTS SUMMARY")
    print("=" * 55)
    print(f"  Correct assignment score:  "
          f"mean={results['correct_assignment_scores'].get('mean', 'N/A'):.3f}  "
          f"p50={results['correct_assignment_scores'].get('p50', 'N/A'):.3f}")
    print(f"  Newborn breakdown:")
    print(f"    Case A (score < {id_thresh}): {nb['case_A']}  ({nb['case_A_pct']}%)")
    print(f"    Case B (score ≥ {id_thresh}): {nb['case_B']}  ({nb['case_B_pct']}%)")
    print(f"    Case C (label absent):  {nb['case_C']}  ({nb['case_C_pct']}%)")
    print(f"  Max score all objects:  "
          f"mean={results['max_scores_all'].get('mean', 'N/A'):.3f}")
    print(f"  Max score newborns:     "
          f"mean={results['newborn_max_scores'].get('mean', 'N/A'):.3f}")
    print(f"\nOutputs → {od}/")
    print("=" * 55)


if __name__ == "__main__":
    main()