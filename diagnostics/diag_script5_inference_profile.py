#!/usr/bin/env python3
"""
diag_script5_inference_profile.py
===================================
Profile inference latency per component to find the actual bottleneck.

Measures wall time for each stage:
  1. Image preprocessing (CPU + GPU transfer)
  2. RF-DETR forward (detector only)
  3. TrajectoryModeling forward
  4. IDDecoder forward
  5. Assignment protocol
  6. Total per frame

Run from inside MOTIP/:
  python diagnostics/diag_script5_inference_profile.py \\
    --config configs/rf_detr_motip_dancetrack.yaml \\
    --checkpoint outputsV2/rfmotip_dancetrack/train/checkpoint_7.pth \\
    --sequence_dir /data/pos+mot/Datadir/DanceTrack/val/dancetrack0041 \\
    --output_dir diagnostics/diag5_profile/ \\
    --num_frames 200 \\
    --dtype FP16
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
    p.add_argument("--sequence_dir", required=True)
    p.add_argument("--output_dir",   default="diagnostics/diag5_profile/")
    p.add_argument("--num_frames",   type=int, default=200)
    p.add_argument("--dtype",        default="FP16", choices=["FP16", "FP32"])
    p.add_argument("--warmup",       type=int, default=10,
                   help="Frames to run before timing starts")
    return p.parse_args()


def main():
    args = get_args()

    import torch
    import numpy as np
    import time
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    import torchvision.transforms.functional as TF
    from configparser import ConfigParser
    from utils.misc import yaml_to_dict
    from configs.util import load_super_config
    from models.motip import build as build_motip
    from models.misc import load_checkpoint, get_model
    from utils.nested_tensor import nested_tensor_from_tensor_list

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    od = Path(args.output_dir)
    od.mkdir(parents=True, exist_ok=True)

    config = yaml_to_dict(args.config)
    config = load_super_config(config, config.get("SUPER_CONFIG_PATH"))

    dtype = torch.float16 if args.dtype == "FP16" else torch.float32

    # ── Load model ────────────────────────────────────────────────────
    print("Loading model...")
    model, _ = build_motip(config=config)
    load_checkpoint(model, path=args.checkpoint)
    model.eval().to(device)
    if dtype == torch.float16:
        model.half()

    inner     = get_model(model)
    detr      = inner.detr
    traj_mod  = inner.trajectory_modeling
    id_dec    = inner.id_decoder

    # ── Load sequence ─────────────────────────────────────────────────
    seq = Path(args.sequence_dir)
    ini = ConfigParser()
    ini.read(seq / "seqinfo.ini")
    img_w   = int(ini["Sequence"]["imWidth"])
    img_h   = int(ini["Sequence"]["imHeight"])
    seq_len = int(ini["Sequence"]["seqLength"])

    image_paths = [str(seq / "img1" / f"{i+1:08d}.jpg") for i in range(seq_len)]
    n_frames    = min(args.num_frames + args.warmup, seq_len)

    MEANS      = [0.485, 0.456, 0.406]
    STDS       = [0.229, 0.224, 0.225]
    SIZE_DIV   = config.get("SIZE_DIVISIBILITY", 32)
    MAX_LONGER = config.get("INFERENCE_MAX_LONGER", 1440)
    MAX_SHORTER = 800

    def preprocess(image_path):
        img = Image.open(image_path).convert("RGB")
        t   = TF.to_tensor(img)
        h, w = t.shape[-2:]
        scale = MAX_SHORTER / min(h, w)
        if max(h, w) * scale > MAX_LONGER:
            scale = MAX_LONGER / max(h, w)
        t = TF.resize(t, [int(round(h*scale)), int(round(w*scale))])
        t = TF.normalize(t, MEANS, STDS)
        if dtype == torch.float16:
            t = t.half()
        return nested_tensor_from_tensor_list([t], SIZE_DIV).to(device)

    # ── Timing containers ─────────────────────────────────────────────
    times = {
        "preprocess":         [],
        "detr":               [],
        "trajectory_modeling": [],
        "id_decoder":         [],
        "assignment":         [],
        "total":              [],
    }

    # Minimal tracker state
    miss_tolerance  = config.get("MISS_TOLERANCE", 30)
    num_id_vocab    = inner.num_id_vocabulary
    use_sigmoid     = config.get("USE_FOCAL_LOSS", False)
    det_thresh      = config.get("DET_THRESH", 0.3)
    id_thresh       = config.get("ID_THRESH", 0.2)

    trajectory_features  = torch.zeros(0, device=device, dtype=dtype)
    trajectory_boxes     = torch.zeros(0, device=device, dtype=dtype)
    trajectory_id_labels = torch.zeros(0, device=device, dtype=torch.int64)
    trajectory_times     = torch.zeros(0, device=device, dtype=torch.int64)
    trajectory_masks     = torch.zeros(0, device=device, dtype=torch.bool)

    def cuda_time(fn):
        """Run fn, return (result, elapsed_ms) with proper CUDA sync."""
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        return result, (time.perf_counter() - t0) * 1000

    print(f"Running {n_frames} frames (first {args.warmup} are warmup)...")

    for t_idx in range(n_frames):
        is_warmup = t_idx < args.warmup

        # 1. Preprocess
        frame, t_pre = cuda_time(lambda: preprocess(image_paths[t_idx]))

        # 2. DETR forward
        with torch.no_grad():
            detr_out, t_detr = cuda_time(lambda: detr(samples=frame))

        # Extract active detections
        logits  = detr_out["pred_logits"][0]
        boxes   = detr_out["pred_boxes"][0]
        embeds  = detr_out["outputs"][0]
        scores  = logits.sigmoid().max(-1).values
        active  = scores > det_thresh
        boxes   = boxes[active]
        embeds  = embeds[active]

        if len(boxes) == 0 or trajectory_features.shape[0] == 0:
            # No tracking needed this frame
            if not is_warmup:
                times["preprocess"].append(t_pre)
                times["detr"].append(t_detr)
                times["trajectory_modeling"].append(0.0)
                times["id_decoder"].append(0.0)
                times["assignment"].append(0.0)
                times["total"].append(t_pre + t_detr)
            continue

        # 3. Trajectory modeling
        seq_info = {
            "trajectory_features":  trajectory_features[None, None, ...],
            "trajectory_boxes":     trajectory_boxes[None, None, ...],
            "trajectory_id_labels": trajectory_id_labels[None, None, ...],
            "trajectory_times":     trajectory_times[None, None, ...],
            "trajectory_masks":     trajectory_masks[None, None, ...],
            "unknown_features":     embeds[None, None, ...],
            "unknown_boxes":        boxes[None, None, ...],
            "unknown_masks":        torch.zeros(
                (1, len(embeds)), dtype=torch.bool, device=device)[None, None, ...],
            "unknown_times":        (trajectory_times.shape[0] * torch.ones(
                (1, len(embeds)), dtype=torch.int64, device=device))[None, None, ...],
        }

        with torch.no_grad():
            seq_info_out, t_traj = cuda_time(
                lambda: traj_mod(seq_info))

        # 4. ID decoder
        with torch.no_grad():
            id_out, t_id = cuda_time(
                lambda: id_dec(seq_info_out, use_decoder_checkpoint=False))

        id_logits = id_out[0][0, 0, 0]

        # 5. Assignment (CPU — no CUDA sync needed but we time it)
        t0 = time.perf_counter()
        if not use_sigmoid:
            id_scores = id_logits.softmax(dim=-1)
        else:
            id_scores = id_logits.sigmoid()
        # object-max assignment
        id_labels = []
        id_max_confs = {}
        for obj_i in range(len(boxes)):
            conf, label = id_scores[obj_i].max(dim=0)
            conf  = conf.item()
            label = label.item()
            if conf < id_thresh:
                id_labels.append(num_id_vocab)
            elif label in id_max_confs and conf < id_max_confs[label]:
                id_labels.append(num_id_vocab)
            else:
                id_max_confs[label] = conf
                id_labels.append(label)
        t_assign = (time.perf_counter() - t0) * 1000

        # Update minimal trajectory state (append current frame)
        trajectory_features  = torch.cat([
            trajectory_features[-miss_tolerance+2:],
            embeds.unsqueeze(0)], dim=0) if trajectory_features.shape[0] > 0 else embeds.unsqueeze(0)
        trajectory_times     = torch.cat([
            trajectory_times[-miss_tolerance+2:],
            torch.full((1,), t_idx, dtype=torch.int64, device=device)], dim=0) \
            if trajectory_times.shape[0] > 0 else torch.tensor([t_idx], device=device)
        trajectory_boxes     = boxes.unsqueeze(0) if trajectory_boxes.shape[0] == 0 \
            else torch.cat([trajectory_boxes[-miss_tolerance+2:], boxes.unsqueeze(0)], dim=0)
        trajectory_id_labels = torch.tensor(id_labels, device=device, dtype=torch.int64).unsqueeze(0) \
            if trajectory_id_labels.shape[0] == 0 \
            else torch.cat([trajectory_id_labels[-miss_tolerance+2:],
                            torch.tensor(id_labels, device=device, dtype=torch.int64).unsqueeze(0)], dim=0)
        trajectory_masks     = torch.zeros(
            (1, len(boxes)), dtype=torch.bool, device=device) \
            if trajectory_masks.shape[0] == 0 \
            else torch.cat([trajectory_masks[-miss_tolerance+2:],
                            torch.zeros((1, len(boxes)), dtype=torch.bool, device=device)], dim=0)

        if not is_warmup:
            t_total = t_pre + t_detr + t_traj + t_id + t_assign
            times["preprocess"].append(t_pre)
            times["detr"].append(t_detr)
            times["trajectory_modeling"].append(t_traj)
            times["id_decoder"].append(t_id)
            times["assignment"].append(t_assign)
            times["total"].append(t_total)

        if (t_idx + 1) % 50 == 0:
            print(f"  Frame {t_idx+1}/{n_frames}")

    # ── Aggregate ─────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("INFERENCE PROFILE RESULTS")
    print("=" * 55)

    results = {}
    for key, vals in times.items():
        if not vals:
            continue
        mean_ms = float(np.mean(vals))
        std_ms  = float(np.std(vals))
        pct     = mean_ms / float(np.mean(times["total"])) * 100 if times["total"] else 0
        results[key] = {
            "mean_ms": round(mean_ms, 2),
            "std_ms":  round(std_ms, 2),
            "pct_of_total": round(pct, 1),
        }
        print(f"  {key:<25} {mean_ms:>7.2f} ms  ±{std_ms:>5.2f}  ({pct:>5.1f}%)")

    total_mean = float(np.mean(times["total"])) if times["total"] else 0
    fps = 1000 / total_mean if total_mean > 0 else 0
    results["fps"] = round(fps, 2)
    results["dtype"] = args.dtype
    results["n_frames_timed"] = len(times["total"])
    print(f"\n  {'Total':<25} {total_mean:>7.2f} ms  →  {fps:.1f} FPS")
    print("=" * 55)

    # ── Plot ──────────────────────────────────────────────────────────
    keys  = ["preprocess", "detr", "trajectory_modeling", "id_decoder", "assignment"]
    means = [results[k]["mean_ms"] for k in keys if k in results]
    pcts  = [results[k]["pct_of_total"] for k in keys if k in results]
    keys  = [k for k in keys if k in results]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Bar chart: mean ms per component
    colors = ["steelblue", "tomato", "orange", "green", "grey"]
    axes[0].bar(keys, means, color=colors[:len(keys)], alpha=0.8)
    for i, (k, m) in enumerate(zip(keys, means)):
        axes[0].text(i, m + 0.3, f"{m:.1f}ms", ha="center", fontsize=9)
    axes[0].set_ylabel("Mean latency (ms)")
    axes[0].set_title(f"Latency per Component\n"
                      f"Total={total_mean:.1f}ms  FPS={fps:.1f}  dtype={args.dtype}")
    axes[0].tick_params(axis="x", rotation=20)

    # Pie: % of total
    axes[1].pie(pcts, labels=[f"{k}\n{p:.1f}%" for k, p in zip(keys, pcts)],
                colors=colors[:len(keys)], autopct="%1.0f%%", startangle=90)
    axes[1].set_title("% of Total Inference Time")

    plt.tight_layout()
    plt.savefig(od / "diag5_inference_profile.png", dpi=150, bbox_inches="tight")
    plt.close()

    with open(od / "diag5_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nOutputs → {od}/")


if __name__ == "__main__":
    main()