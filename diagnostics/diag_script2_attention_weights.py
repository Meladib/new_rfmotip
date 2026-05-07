#!/usr/bin/env python3
"""
diag_script2_attention_weights.py
===================================
Diagnostic 2 — Temporal Attention Weight Distribution in IDDecoder

Hypothesis to verify
--------------------
The IDDecoder cross-attention treats all 28 trajectory frames equally
(flat temporal weighting), wasting context and ignoring the high quality
of the frozen RF-DETR features.

If confirmed: attention weights are approximately uniform across frame ages
→ the model is not learning temporal recency bias.
If refuted:   attention weights peak at recent frames (low age)
→ the model is already leveraging temporal structure.

What this script measures
-------------------------
Hooks into IDDecoder.cross_attn_layers[layer] (all layers).
Captures the attention weight matrix returned by PyTorch MultiheadAttention:
    attn_weights shape: (B*G*n_heads, T_unknown*N_unknown, T_traj*N_traj)

For each (unknown_object, trajectory_slot) pair, records the attention weight
as a function of the temporal distance:
    frame_age = unknown_time - trajectory_time  (0 = same frame, 27 = oldest)

Plots: mean attention weight vs frame age, per decoder layer.

Key insight from code
---------------------
cross_attn_layers[layer] call in _forward_a_layer:
    cross_out, _ = self.cross_attn_layers[layer](
        query=cross_unknown_embeds,
        key=cross_trajectory_embeds,
        value=cross_trajectory_embeds,
        key_padding_mask=cross_attn_key_padding_mask,
        attn_mask=cross_attn_mask_with_rel_pe,
    )
The second return value `_` is discarded — we capture it via hook.
IMPORTANT: need_weights=True must be set. PyTorch MHA returns None for
attn_weights when need_weights=False. The hook patches this.

Run from inside MOTIP/:
  python diagnostics/diag_script2_attention_weights.py \\
    --config configs/rf_detr_motip_dancetrack.yaml \\
    --checkpoint outputs/rfmotip_dancetrack/checkpoint_3.pth \\
    --sequence_dir /data/DanceTrack/val/dancetrack0004 \\
    --output_dir diagnostics/diag2_attention_weights/ \\
    --num_frames 100
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
                   help="Single DanceTrack sequence dir (has img1/ and gt/gt.txt)")
    p.add_argument("--output_dir",   default="diagnostics/diag2_attention_weights/")
    p.add_argument("--num_frames",   type=int, default=100)
    p.add_argument("--device",       default=None)
    return p.parse_args()


def load_sequence(sequence_dir):
    """Load image paths and GT annotations from a DanceTrack sequence dir."""
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


def install_attention_hooks(model):
    """
    Patch IDDecoder.cross_attn_layers to capture attention weights.

    PyTorch MultiheadAttention returns (output, attn_weights).
    The existing code discards attn_weights via `cross_out, _ = self.cross_attn_layers[layer](...)`.
    We wrap each cross_attn layer's forward to capture the weights.

    Returns a container that accumulates (frame_age -> [weight]) mappings.
    """
    import torch
    from models.misc import get_model

    inner = get_model(model)
    id_decoder = inner.id_decoder
    num_layers = id_decoder.num_layers

    # Container: layer -> frame_age -> list of mean attention weights
    container = {
        "weights_by_age": {li: defaultdict(list) for li in range(num_layers)},
        "trajectory_times": None,
        "unknown_times":    None,
    }

    # We hook the IDDecoder.forward to capture times
    original_forward = id_decoder.forward

    def patched_forward(seq_info, use_decoder_checkpoint=False):
        # Store times for later use in attention hooks
        container["trajectory_times"] = seq_info["trajectory_times"].clone()
        container["unknown_times"]    = seq_info["unknown_times"].clone()
        return original_forward(seq_info, use_decoder_checkpoint=False)

    id_decoder.forward = patched_forward

    # Patch each cross_attn layer to return weights
    handles = []
    for li in range(num_layers):
        original_cross_attn_forward = id_decoder.cross_attn_layers[li].forward

        def make_patched(layer_idx, orig_fwd):
            def patched_cross_attn(query, key, value,
                                   key_padding_mask=None,
                                   attn_mask=None,
                                   need_weights=False,
                                   **kwargs):
                # Force need_weights=True to get attention weights
                out, attn_w = orig_fwd(
                    query, key, value,
                    key_padding_mask=key_padding_mask,
                    attn_mask=attn_mask,
                    need_weights=True,       # ← key change
                    average_attn_weights=True,
                    **kwargs,
                )
                # attn_w: (B*G, T_curr*N_curr, T_traj*N_traj) averaged over heads

                if (attn_w is not None and
                        container["trajectory_times"] is not None and
                        container["unknown_times"] is not None):
                    try:
                        traj_times = container["trajectory_times"]
                        unk_times  = container["unknown_times"]

                        B, G, T_t, N_t = traj_times.shape
                        B2, G2, T_u, N_u = unk_times.shape

                        # Flatten to match attn_w shape
                        traj_t_flat = traj_times.reshape(B * G, T_t * N_t)  # (BG, T_t*N_t)
                        unk_t_flat  = unk_times.reshape(B * G, T_u * N_u)   # (BG, T_u*N_u)

                        # attn_w: (BG, T_u*N_u, T_t*N_t)
                        if attn_w.shape[0] == B * G:
                            for bg in range(B * G):
                                for ui in range(T_u * N_u):
                                    u_time = unk_t_flat[bg, ui].item()
                                    for ti in range(T_t * N_t):
                                        t_time = traj_t_flat[bg, ti].item()
                                        age    = u_time - t_time
                                        w      = attn_w[bg, ui, ti].item()
                                        container["weights_by_age"][layer_idx][int(age)].append(w)
                    except Exception:
                        pass  # Don't crash training if hook fails

                return out, attn_w
            return patched_cross_attn

        id_decoder.cross_attn_layers[li].forward = make_patched(li, original_cross_attn_forward)

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
    from models.misc import load_checkpoint, get_model
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

    # ── Install hooks ─────────────────────────────────────────────────
    print("Installing attention hooks...")
    container = install_attention_hooks(model)

    # ── Load sequence ─────────────────────────────────────────────────
    image_paths, annotations, img_w, img_h, seq_len = load_sequence(args.sequence_dir)
    n_frames = min(args.num_frames, seq_len)
    print(f"Processing {n_frames} frames from {Path(args.sequence_dir).name}")

    # ── Run RuntimeTracker-style inference ────────────────────────────
    # We use RuntimeTracker directly so trajectory history builds up
    # and the cross-attention has meaningful temporal context
    from models.runtime_tracker import RuntimeTracker

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

    MEANS = [0.485, 0.456, 0.406]
    STDS  = [0.229, 0.224, 0.225]
    SIZE_DIV = config.get("SIZE_DIVISIBILITY", 32)
    MAX_LONGER = config.get("INFERENCE_MAX_LONGER", 1440)

    with torch.no_grad():
        for t in range(n_frames):
            img = Image.open(image_paths[t]).convert("RGB")
            img_t = TF.to_tensor(img)

            # Resize maintaining aspect ratio with max_shorter=800
            h, w = img_t.shape[-2:]
            shorter = min(h, w)
            longer  = max(h, w)
            scale   = 800.0 / shorter
            if longer * scale > MAX_LONGER:
                scale = MAX_LONGER / longer
            new_h = int(round(h * scale))
            new_w = int(round(w * scale))
            img_t = TF.resize(img_t, [new_h, new_w])
            img_t = TF.normalize(img_t, MEANS, STDS)

            frame = nested_tensor_from_tensor_list([img_t], SIZE_DIV).to(device)
            rt.update(image=frame)

            if (t + 1) % 20 == 0:
                # Report how many weights we have
                n_total = sum(
                    sum(len(v) for v in container["weights_by_age"][li].values())
                    for li in container["weights_by_age"]
                )
                print(f"  Frame {t + 1}/{n_frames} — accumulated {n_total} weight samples")

    # ── Aggregate and plot ────────────────────────────────────────────
    inner = get_model(model)
    num_layers = inner.id_decoder.num_layers

    fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 5))
    if num_layers == 1:
        axes = [axes]

    results = {}
    for li in range(num_layers):
        age_data = container["weights_by_age"][li]
        if not age_data:
            axes[li].text(0.5, 0.5, "no data", transform=axes[li].transAxes,
                          ha="center", va="center")
            results[f"layer_{li}"] = {}
            continue

        ages        = sorted(age_data.keys())
        mean_weights = [float(np.mean(age_data[a])) for a in ages]
        std_weights  = [float(np.std(age_data[a]))  for a in ages]

        results[f"layer_{li}"] = {
            "ages":         ages,
            "mean_weights": mean_weights,
            "std_weights":  std_weights,
        }

        # Uniformity test: if weights are uniform, std of mean_weights should be ~0
        mean_w_arr   = np.array(mean_weights)
        uniformity   = float(np.std(mean_w_arr) / (np.mean(mean_w_arr) + 1e-8))
        results[f"layer_{li}"]["uniformity_cv"] = uniformity

        ax = axes[li]
        ax.bar(ages, mean_weights, alpha=0.7, color="steelblue",
               yerr=std_weights, capsize=2, error_kw={"alpha": 0.3})
        ax.axhline(1.0 / (len(ages) + 1e-8), color="red", ls="--",
                   alpha=0.6, label=f"Uniform ({1/(len(ages)+1e-8):.4f})")
        ax.set_xlabel("Frame Age (0 = current, 27 = oldest)")
        ax.set_ylabel("Mean Attention Weight")
        ax.set_title(f"Layer {li}\nCV={uniformity:.3f}"
                     f" ({'FLAT' if uniformity < 0.1 else 'RECENCY BIAS'})")
        ax.legend(fontsize=7)

    plt.suptitle(
        "IDDecoder Cross-Attention Weight vs Trajectory Frame Age\n"
        "CV < 0.1 → flat (model not learning temporal recency)\n"
        "CV > 0.3 → recency bias present",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(od / "diag2_attention_weights.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {od}/diag2_attention_weights.png")

    # Interpretation
    cvs = [results[f"layer_{li}"].get("uniformity_cv", 0)
           for li in range(num_layers)
           if f"layer_{li}" in results]
    mean_cv = float(np.mean(cvs)) if cvs else 0.0

    interpretation = (
        "FLAT ATTENTION CONFIRMED — model not using temporal structure. "
        "Confidence-weighted attention (Phase 2) is justified."
        if mean_cv < 0.15
        else "RECENCY BIAS PRESENT — model is already learning temporal structure. "
             "Phase 2 may have lower impact than expected."
    )
    results["mean_cv_across_layers"] = mean_cv
    results["interpretation"]        = interpretation

    with open(od / "diag2_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nMean CV across layers: {mean_cv:.4f}")
    print(f"Interpretation: {interpretation}")
    print(f"\nOutputs → {od}/")


if __name__ == "__main__":
    main()