"""
diag_det_eval.py
================
Evaluates fine-tuned RF-DETR detection performance on DanceTrack val set.
Computes COCO-standard AP metrics: mAP, AP50, AP75.

Scientific purpose
------------------
Proves that the frozen detector decision is principled: if the fine-tuned
RF-DETR achieves strong AP on DanceTrack val, any remaining tracking gap
is attributable to the association module, not to detection quality.

Metrics reported
----------------
  mAP   — mean AP over IoU thresholds [0.50 : 0.05 : 0.95]  (COCO primary)
  AP50  — AP at IoU=0.50  (most permissive, standard MOT detection metric)
  AP75  — AP at IoU=0.75  (stricter localization)
  AR    — average recall at max 100 detections per image

Output
------
  Console: per-sequence AP50 + overall table
  File:    diagnostics/det_eval_results.json

Run from repo root:
  python diagnostics/diag_det_eval.py
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
import torchvision.transforms.functional as TF
from pathlib import Path
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# CONFIG — edit these paths if needed
# ---------------------------------------------------------------------------
DATA_ROOT   = "/data/pos+mot/Datadir/DanceTrack"
CKPT_PATH   = "rfdetr_dancetrack_motip/checkpoint_best_total.pth"
SPLIT       = "val"
OUTPUT_PATH = "diagnostics/det_eval_results.json"

# Detection threshold — predictions below this are discarded before AP eval.
# Keep low (0.05) so pycocotools can sweep the full precision-recall curve.
SCORE_THRESH = 0.05

# GT visibility threshold — GT boxes with visibility < this are ignored.
# DanceTrack standard is 0.0 (include all annotated objects).
VIS_THRESH   = 0.0

# ---------------------------------------------------------------------------


def load_gt_mot(gt_path: str) -> dict:
    """
    Parse a MOT-format gt.txt file.
    Format: frame_id, obj_id, x, y, w, h, conf, class, visibility
    Returns: {frame_id (int): list of [x, y, w, h, visibility]}
    Filters out conf=0 rows (DanceTrack marks ignore regions with conf=0).
    """
    gt = {}
    with open(gt_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            frame_id  = int(parts[0])
            conf      = float(parts[6]) if len(parts) > 6 else 1.0
            vis       = float(parts[8]) if len(parts) > 8 else 1.0
            if conf == 0:          # ignore region — skip
                continue
            if vis < VIS_THRESH:
                continue
            x, y, w, h = float(parts[2]), float(parts[3]), \
                          float(parts[4]), float(parts[5])
            if w <= 0 or h <= 0:
                continue
            gt.setdefault(frame_id, []).append([x, y, w, h, vis])
    return gt


def build_rf_detr(ckpt_path: str, device: torch.device):
    """
    Build RF-DETR detector only (no MOTIP wrapper).
    Loads architecture from checkpoint args, applies weights.
    """
    from models.rfdetr.models.lwdetr import build_model, PostProcess

    ckpt      = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args_ckpt = ckpt["args"]

    print(f"  RF-DETR config: num_classes={args_ckpt.num_classes}, "
          f"resolution={args_ckpt.resolution}, "
          f"dec_layers={args_ckpt.dec_layers}, "
          f"group_detr={getattr(args_ckpt, 'group_detr', 1)}, "
          f"num_queries={args_ckpt.num_queries}")

    # Replicate class-count adjustment used in models/motip/__init__.py
    args_ckpt.num_classes -= 1
    model = build_model(args=args_ckpt)
    args_ckpt.num_classes += 1

    # PostProcess: maps normalised cx,cy,w,h → absolute x1,y1,x2,y2
    postprocess = PostProcess(num_select=args_ckpt.num_select)

    # Clean state dict — strip "module." prefix from DDP-trained checkpoints
    state       = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
    own_state   = model.state_dict()
    filtered    = {k: v for k, v in state.items()
                   if k in own_state and v.shape == own_state[k].shape}
    skipped     = [k for k in state if k not in filtered]
    missing, _  = model.load_state_dict(filtered, strict=False)

    print(f"  Weights loaded: {len(filtered)} matched, "
          f"{len(skipped)} skipped (shape mismatch), "
          f"{len(missing)} missing")

    model.eval().to(device)
    return model, postprocess, args_ckpt.resolution


@torch.no_grad()
def run_detector(model, postprocess, img_path: str, resolution: int,
                 device: torch.device):
    """
    Run RF-DETR on one image.
    Returns list of [x, y, w, h, score] in original image coordinates.
    """
    img      = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size

    # Preprocessing — identical to existing inference scripts
    t = TF.to_tensor(img)
    t = TF.normalize(t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    t = TF.resize(t, [resolution, resolution])
    t = t.unsqueeze(0).to(device)

    out = model(t)

    # PostProcess expects target_sizes as (h, w)
    target_sizes = torch.tensor([[orig_h, orig_w]], device=device)
    results      = postprocess(out, target_sizes)[0]   # single image

    scores = results["scores"].cpu().numpy()  # (num_select,)
    boxes  = results["boxes"].cpu().numpy()   # (num_select, 4) x1,y1,x2,y2

    # Filter low-confidence predictions
    keep = scores >= SCORE_THRESH
    scores, boxes = scores[keep], boxes[keep]

    # Convert x1,y1,x2,y2 → x,y,w,h (COCO format)
    dets = []
    for (x1, y1, x2, y2), s in zip(boxes, scores):
        w = float(x2 - x1)
        h = float(y2 - y1)
        if w > 0 and h > 0:
            dets.append([float(x1), float(y1), w, h, float(s)])
    return dets


def build_coco_gt(seq_gt_list: list) -> COCO:
    """
    Build an in-memory COCO GT object from a list of
    (image_id, frame_boxes) pairs where frame_boxes = [[x,y,w,h,vis], ...].
    """
    images, annotations = [], []
    ann_id = 1

    for image_id, boxes in seq_gt_list:
        images.append({"id": image_id})
        for (x, y, w, h, vis) in boxes:
            annotations.append({
                "id":          ann_id,
                "image_id":    image_id,
                "category_id": 1,
                "bbox":        [x, y, w, h],
                "area":        w * h,
                "iscrowd":     0,
            })
            ann_id += 1

    coco_dict = {
        "info":        {"description": "DanceTrack val GT"},
        "licenses":    [],
        "images":     images,
        "annotations": annotations,
        "categories":  [{"id": 1, "name": "pedestrian"}],
    }
    coco_gt = COCO()
    coco_gt.dataset = coco_dict
    coco_gt.createIndex()
    return coco_gt


def eval_sequence(seq_name: str, seq_dir: Path, model, postprocess,
                  resolution: int, device: torch.device,
                  seq_offset: int) -> dict:
    """
    Evaluate one DanceTrack val sequence.
    Returns per-sequence results dict and (coco_gt, dt_results) for global eval.
    seq_offset: integer offset so image IDs are globally unique.
    """
    gt_path   = seq_dir / "gt" / "gt.txt"
    img_dir   = seq_dir / "img1"

    if not gt_path.exists():
        print(f"  WARNING: GT not found at {gt_path}, skipping.")
        return None, None, None

    gt_by_frame = load_gt_mot(str(gt_path))
    img_files   = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))

    seq_gt_list = []   # (image_id, boxes)
    dt_results  = []   # COCO detection results format

    for frame_idx, img_path in enumerate(img_files):
        frame_id = frame_idx + 1          # 1-indexed (MOT convention)
        image_id = seq_offset + frame_id  # globally unique

        # GT for this frame
        gt_boxes = gt_by_frame.get(frame_id, [])
        seq_gt_list.append((image_id, gt_boxes))

        # Predictions
        dets = run_detector(model, postprocess, str(img_path),
                            resolution, device)
        for (x, y, w, h, score) in dets:
            dt_results.append({
                "image_id":    image_id,
                "category_id": 1,
                "bbox":        [x, y, w, h],
                "score":       score,
            })

    # Per-sequence COCO eval
    coco_gt  = build_coco_gt(seq_gt_list)
    coco_dt  = coco_gt.loadRes(dt_results) if dt_results else None

    seq_metrics = {"sequence": seq_name, "n_frames": len(img_files),
                   "n_gt_boxes": sum(len(v) for v in gt_by_frame.values())}

    if coco_dt is not None:
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        seq_metrics["AP50"]  = float(ev.stats[1])   # AP @ IoU=0.50
        seq_metrics["mAP"]   = float(ev.stats[0])   # mAP 0.50:0.95
        seq_metrics["AP75"]  = float(ev.stats[2])   # AP @ IoU=0.75
        seq_metrics["AR"]    = float(ev.stats[8])   # AR maxDets=100
    else:
        seq_metrics.update({"AP50": 0.0, "mAP": 0.0, "AP75": 0.0, "AR": 0.0})

    return seq_metrics, seq_gt_list, dt_results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build detector
    print("\nBuilding RF-DETR detector...")
    model, postprocess, resolution = build_rf_detr(CKPT_PATH, device)
    print(f"  Input resolution: {resolution}×{resolution}")

    val_dir   = Path(DATA_ROOT) / SPLIT
    sequences = sorted([d for d in val_dir.iterdir() if d.is_dir()])
    print(f"\nFound {len(sequences)} sequences in {val_dir}")

    all_results   = []
    global_gt     = []   # flat list of (image_id, boxes)
    global_dt     = []   # flat list of detection dicts

    for seq_idx, seq_dir in enumerate(sequences):
        seq_name   = seq_dir.name
        seq_offset = seq_idx * 100000   # 100k gap ensures no image_id collision

        print(f"\n[{seq_idx+1}/{len(sequences)}] {seq_name}")
        seq_metrics, seq_gt_list, dt_results = eval_sequence(
            seq_name, seq_dir, model, postprocess,
            resolution, device, seq_offset
        )

        if seq_metrics is None:
            continue

        all_results.append(seq_metrics)
        global_gt.extend(seq_gt_list)
        global_dt.extend(dt_results)

        print(f"  AP50={seq_metrics['AP50']:.4f}  "
              f"mAP={seq_metrics['mAP']:.4f}  "
              f"AP75={seq_metrics['AP75']:.4f}  "
              f"AR={seq_metrics['AR']:.4f}  "
              f"frames={seq_metrics['n_frames']}  "
              f"gt_boxes={seq_metrics['n_gt_boxes']}")

    # Global eval across all sequences
    print("\n" + "=" * 60)
    print("GLOBAL EVALUATION — all val sequences")
    print("=" * 60)

    coco_gt_global = build_coco_gt(global_gt)
    coco_dt_global = coco_gt_global.loadRes(global_dt) if global_dt else None

    global_metrics = {}
    if coco_dt_global is not None:
        ev = COCOeval(coco_gt_global, coco_dt_global, iouType="bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        global_metrics = {
            "mAP":   float(ev.stats[0]),
            "AP50":  float(ev.stats[1]),
            "AP75":  float(ev.stats[2]),
            "AR_1":  float(ev.stats[6]),
            "AR_10": float(ev.stats[7]),
            "AR_100":float(ev.stats[8]),
        }
        print(f"\n  mAP  (IoU 0.50:0.95): {global_metrics['mAP']:.4f}")
        print(f"  AP50 (IoU 0.50)      : {global_metrics['AP50']:.4f}")
        print(f"  AP75 (IoU 0.75)      : {global_metrics['AP75']:.4f}")
        print(f"  AR@100               : {global_metrics['AR_100']:.4f}")

    # Per-sequence summary table
    print("\n" + "-" * 60)
    print(f"{'Sequence':<20} {'AP50':>8} {'mAP':>8} {'AP75':>8} {'AR':>8}")
    print("-" * 60)
    for r in all_results:
        print(f"  {r['sequence']:<18} {r['AP50']:>8.4f} {r['mAP']:>8.4f} "
              f"{r['AP75']:>8.4f} {r['AR']:>8.4f}")
    print("-" * 60)

    # Save results
    output = {
        "checkpoint":       CKPT_PATH,
        "split":            SPLIT,
        "score_thresh":     SCORE_THRESH,
        "vis_thresh":       VIS_THRESH,
        "global_metrics":   global_metrics,
        "per_sequence":     all_results,
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()