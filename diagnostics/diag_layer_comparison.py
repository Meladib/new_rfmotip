#!/usr/bin/env python3
"""
diag_layer_comparison.py
========================
Compares embedding quality and temporal stability across ALL decoder layers.

Motivation (from diag_temporal_stability.md §4 and diag_query_divergence.md)
-----------------------------------------------------------------------------
The DIAG_TEMPORAL_STABILITY script ran only on layer_index=-1 (final layer)
and produced mean cosine similarity = 0.531 across ALL 300 query slots.
Two open questions remain:

  (a) Is there a per-layer stability gradient (layer 0 vs 1 vs 2)?
      If layer 0 > layer 2: NAS weight sharing artifact — first layer is
      more regularized because it appears in every NAS sub-network.
      If layer 2 > layer 0: depth refinement adds stability — 3-layer
      inference config is appropriate.
      If all similar: encoder top-K re-ranking dominates, not depth.

  (b) How different are the embeddings at the stale layer (dec_layers-2)
      vs the final layer?  High similarity → stale-indices bug has low
      impact.  Low similarity → embeddings fed to MOTIP are wrong region
      of feature space.

What this script adds beyond diag_temporal_stability_script.py
-------------------------------------------------------------
  * Runs slot-level temporal stability on EVERY decoder layer (not just -1)
  * Adds OBJECT-MATCHED stability (requires GT + Hungarian matching)
    for each layer — this is the number that actually matters for MOTIP
  * Computes cross-layer embedding similarity:
      cos_sim( hs[layer_A][slot_k], hs[layer_B][slot_k] )
    Tells you how much refining across layers changes the representation.
  * Outputs a single comparison figure with all layers side by side.

Run from inside MOTIP/:
  python diagnostics/diag_layer_comparison.py \\
    --checkpoint rfdetr_dancetrack_motip/checkpoint_best_total.pth \\
    --sequence_dir /data/DanceTrack/val/dancetrack0004 \\
    --output_dir diagnostics/layers/

  # process more frames for better statistics
  python diagnostics/diag_layer_comparison.py \\
    --checkpoint rfdetr_dancetrack_motip/checkpoint_best_total.pth \\
    --sequence_dir /data/DanceTrack/val/dancetrack0004 \\
    --num_frames 100 \\
    --output_dir diagnostics/layers/
"""

import sys
import os
import json
import argparse
from pathlib import Path
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description="Per-layer decoder embedding comparison"
    )
    p.add_argument("--checkpoint",    required=True)
    p.add_argument("--sequence_dir",  default=None,
                   help="Single DanceTrack sequence dir")
    p.add_argument("--data_root",     default=None)
    p.add_argument("--split",         default="val")
    p.add_argument("--num_frames",    type=int, default=50,
                   help="Frames to process (default 50)")
    p.add_argument("--output_dir",    default="diagnostics/layers/")
    p.add_argument("--device",        default=None)
    p.add_argument("--resolution",    type=int, default=None)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# PATH SETUP
# ─────────────────────────────────────────────────────────────────────────────

def setup_paths():
    here       = os.path.dirname(os.path.abspath(__file__))
    motip_root = os.path.dirname(here)
    if motip_root not in sys.path:
        sys.path.insert(0, motip_root)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_matcher(checkpoint_path, device):
    import torch
    from models.rfdetr.models.lwdetr import (
        build_model,
        build_criterion_and_postprocessors,
    )

    ckpt      = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args_ckpt = ckpt["args"]

    args_ckpt.num_classes -= 1
    model = build_model(args=args_ckpt)
    args_ckpt.num_classes += 1

    criterion, _ = build_criterion_and_postprocessors(args=args_ckpt)

    ckpt_model = ckpt.get("model", None)
    if ckpt_model is not None:
        model_state = model.state_dict()
        filtered = {}
        for k, v in ckpt_model.items():
            bare_k = k[5:] if k.startswith("detr.") else k
            if bare_k in model_state and v.shape == model_state[bare_k].shape:
                filtered[bare_k] = v
        model.load_state_dict(filtered, strict=False)
    else:
        print("  WARNING: no 'model' key — random decoder weights")

    model.eval().to(device)

    # Capture full hs tensor: (dec_layers, B, N, D)
    container = {"hs_all": None}

    def _hook(module, inputs, output):
        hs = output[0]
        if isinstance(hs, (list, tuple)):
            hs = hs[0]
        container["hs_all"] = hs.detach()

    model.transformer.register_forward_hook(_hook)

    return model, criterion.matcher, args_ckpt, container


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

MEANS = [0.485, 0.456, 0.406]
STDS  = [0.229, 0.224, 0.225]


def load_and_preprocess(image_path, resolution, device):
    import torch
    from PIL import Image
    import torchvision.transforms.functional as TF
    img = Image.open(image_path).convert("RGB")
    t   = TF.to_tensor(img)
    t   = TF.resize(t, [resolution, resolution])
    t   = TF.normalize(t, MEANS, STDS)
    return t.unsqueeze(0).to(device)


def xywh_pixel_to_cxcywh_norm(boxes_xywh, img_w, img_h):
    import torch
    x, y, w, h = boxes_xywh.unbind(-1)
    return torch.stack([
        (x + w / 2) / img_w,
        (y + h / 2) / img_h,
        w / img_w,
        h / img_h,
    ], -1)


def load_sequence_single(sequence_dir):
    import torch
    from configparser import ConfigParser

    seq = Path(sequence_dir)
    ini = ConfigParser()
    ini.read(seq / "seqinfo.ini")
    img_w   = int(ini["Sequence"]["imWidth"])
    img_h   = int(ini["Sequence"]["imHeight"])
    seq_len = int(ini["Sequence"]["seqLength"])

    image_paths = [str(seq / "img1" / f"{i + 1:08d}.jpg") for i in range(seq_len)]

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
            }
        else:
            ann = {"id":   torch.zeros(0, dtype=torch.int64),
                   "bbox": torch.zeros((0, 4), dtype=torch.float32)}
        annotations.append(ann)

    return image_paths, annotations, img_w, img_h


# ─────────────────────────────────────────────────────────────────────────────
# PER-PAIR MEASUREMENTS
# ─────────────────────────────────────────────────────────────────────────────

def cosine_sim_batch(a, b):
    """a, b: (M, D)  →  (M,) cosine similarity."""
    import torch.nn.functional as F
    return (F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum(-1)


def measure_frame_pair(hs_t, hs_t1, annotation_t, annotation_t1,
                       matcher, pred_boxes_t, pred_logits_t,
                       pred_boxes_t1, pred_logits_t1,
                       gt_bboxes_t, gt_ids_t,
                       gt_bboxes_t1, gt_ids_t1,
                       img_w, img_h, device, dec_layers):
    """
    Given hs_t and hs_t1 (each shape dec_layers × N × D),
    for each decoder layer compute:

      1. SLOT-LEVEL temporal stability (same slot index across frames)
         — replicates diag_temporal_stability for each layer

      2. OBJECT-MATCHED temporal stability (same GT track_id across frames)
         — the number that actually matters for MOTIP

      3. CROSS-LAYER embedding similarity within a single frame
         cos_sim( hs[layer_i][slot], hs[layer_j][slot] ) for i < j

    Returns dict keyed by layer index (0..dec_layers-1) and layer pairs.
    """
    import torch
    import numpy as np

    # ── Slot-level stability (all 300 slots) ──────────────────────────
    slot_sims = {}
    for li in range(dec_layers):
        emb_t  = hs_t[li]      # (N, D)
        emb_t1 = hs_t1[li]
        sims   = cosine_sim_batch(emb_t, emb_t1).cpu().numpy()
        slot_sims[li] = {
            "mean":       float(np.mean(sims)),
            "std":        float(np.std(sims)),
            "n_unstable": int((sims < 0.5).sum()),
            "n_stable":   int((sims > 0.9).sum()),
        }

    # ── Object-matched stability ──────────────────────────────────────
    obj_sims = {li: [] for li in range(dec_layers)}

    if len(gt_ids_t) > 0 and len(gt_ids_t1) > 0:
        gt_boxes_t_norm  = xywh_pixel_to_cxcywh_norm(
            gt_bboxes_t, img_w, img_h).to(device)
        gt_labels_t      = torch.zeros(len(gt_ids_t), dtype=torch.long, device=device)

        # Match on final layer (correct)
        with torch.no_grad():
            idx_t = matcher(
                {"pred_logits": pred_logits_t.unsqueeze(0),
                 "pred_boxes":  pred_boxes_t.unsqueeze(0)},
                [{"boxes": gt_boxes_t_norm, "labels": gt_labels_t}],
            )
            idx_t1 = matcher(
                {"pred_logits": pred_logits_t1.unsqueeze(0),
                 "pred_boxes":  pred_boxes_t1.unsqueeze(0)},
                [{"boxes": xywh_pixel_to_cxcywh_norm(
                    gt_bboxes_t1, img_w, img_h).to(device),
                  "labels": torch.zeros(len(gt_ids_t1), dtype=torch.long,
                                        device=device)}],
            )

        # Build id -> slot mapping for each frame
        def id_to_slot(q_idx, gt_idx, gt_ids):
            return {int(gt_ids[gi]): int(qi)
                    for qi, gi in zip(q_idx.tolist(), gt_idx.tolist())}

        map_t  = id_to_slot(*idx_t[0],  gt_ids_t)
        map_t1 = id_to_slot(*idx_t1[0], gt_ids_t1)

        # For each track_id present in both frames, measure per-layer similarity
        common_ids = set(map_t.keys()) & set(map_t1.keys())
        for tid in common_ids:
            slot_t  = map_t[tid]
            slot_t1 = map_t1[tid]
            for li in range(dec_layers):
                sim = float(cosine_sim_batch(
                    hs_t[li][slot_t].unsqueeze(0),
                    hs_t1[li][slot_t1].unsqueeze(0),
                )[0])
                obj_sims[li].append(sim)

    obj_stats = {}
    for li in range(dec_layers):
        v = obj_sims[li]
        obj_stats[li] = {
            "mean": float(np.mean(v)) if v else None,
            "std":  float(np.std(v))  if v else None,
            "n":    len(v),
        }

    # ── Cross-layer embedding similarity (same frame, same slot) ──────
    cross_layer = {}
    for li in range(dec_layers):
        for lj in range(li + 1, dec_layers):
            emb_i = hs_t[li]    # (N, D)
            emb_j = hs_t[lj]
            sims  = cosine_sim_batch(emb_i, emb_j).cpu().numpy()
            key   = f"{li}_vs_{lj}"
            cross_layer[key] = {
                "mean": float(np.mean(sims)),
                "std":  float(np.std(sims)),
            }

    return {
        "slot_sims":    slot_sims,
        "obj_sims":     obj_stats,
        "cross_layer":  cross_layer,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(od, dec_layers, agg_slot, agg_obj, agg_cross, ckpt_args):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = list(range(dec_layers))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── Panel 1: slot-level stability per layer ────────────────────────
    ax = axes[0]
    means_slot = [agg_slot[li]["mean"] for li in layers]
    stds_slot  = [agg_slot[li]["std"]  for li in layers]
    ax.bar(layers, means_slot, color="steelblue", alpha=0.7,
           yerr=stds_slot, capsize=5, label="All 300 slots (slot-level)")
    ax.set_xlabel("Decoder Layer Index")
    ax.set_ylabel("Mean Cosine Similarity (T→T+1)")
    ax.set_title("Slot-Level Temporal Stability\n(all slots — same as diag_temporal_stability)")
    ax.axhline(0.9, color="green",  ls="--", alpha=0.6, label="stable (0.9)")
    ax.axhline(0.7, color="orange", ls="--", alpha=0.6, label="high-instability (0.7)")
    ax.axhline(0.5, color="red",    ls="--", alpha=0.6, label="instability (0.5)")
    ax.set_ylim(0, 1.1)
    ax.set_xticks(layers)
    ax.set_xticklabels([f"L{l}" + (" (final)" if l == dec_layers - 1 else
                                    " (stale)" if l == dec_layers - 2 else "")
                        for l in layers])
    ax.legend(fontsize=7)

    # ── Panel 2: object-matched stability per layer ────────────────────
    ax = axes[1]
    means_obj = [agg_obj[li]["mean"] if agg_obj[li]["mean"] else 0.0
                 for li in layers]
    stds_obj  = [agg_obj[li]["std"]  if agg_obj[li]["std"]  else 0.0
                 for li in layers]
    ns_obj    = [agg_obj[li]["n"] for li in layers]

    bars = ax.bar(layers, means_obj, color="darkorange", alpha=0.7,
                  yerr=stds_obj, capsize=5)
    for bar, n in zip(bars, ns_obj):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(stds_obj) + 0.02,
                f"n={n}", ha="center", va="bottom", fontsize=7)

    ax.axhline(0.9, color="green",  ls="--", alpha=0.6, label="stable (0.9)")
    ax.axhline(0.7, color="orange", ls="--", alpha=0.6, label="high-instability (0.7)")
    ax.axhline(0.5, color="red",    ls="--", alpha=0.6, label="instability (0.5)")
    ax.set_xlabel("Decoder Layer Index")
    ax.set_ylabel("Mean Cosine Similarity (object-matched, T→T+1)")
    ax.set_title("Object-Matched Temporal Stability\n"
                 "(same GT track_id — what MOTIP actually needs)")
    ax.set_ylim(0, 1.15)
    ax.set_xticks(layers)
    ax.set_xticklabels([f"L{l}" + (" (final)" if l == dec_layers - 1 else
                                    " (stale)" if l == dec_layers - 2 else "")
                        for l in layers])
    ax.legend(fontsize=7)

    # ── Panel 3: cross-layer similarity ───────────────────────────────
    ax = axes[2]
    pairs = sorted(agg_cross.keys())
    means_cross = [agg_cross[p]["mean"] for p in pairs]
    stds_cross  = [agg_cross[p]["std"]  for p in pairs]

    ax.bar(range(len(pairs)), means_cross, color="purple", alpha=0.7,
           yerr=stds_cross, capsize=5)
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels([p.replace("_vs_", " vs ") for p in pairs],
                       rotation=15, fontsize=8)
    ax.set_xlabel("Layer Pair")
    ax.set_ylabel("Mean Cosine Similarity (same slot, same frame)")
    ax.set_title("Cross-Layer Embedding Similarity\n"
                 "(how much refinement changes the representation)")
    ax.axhline(0.9, color="green",  ls="--", alpha=0.5, label="> 0.9: layers similar")
    ax.axhline(0.5, color="red",    ls="--", alpha=0.5, label="< 0.5: layers diverge")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=7)

    dec_l  = getattr(ckpt_args, "dec_layers",  "?")
    aux_l  = getattr(ckpt_args, "aux_loss",    "?")
    hd     = getattr(ckpt_args, "hidden_dim",  "?")
    plt.suptitle(
        f"Decoder Layer Comparison — dec_layers={dec_l}  aux_loss={aux_l}  "
        f"hidden_dim={hd}",
        fontsize=11,
    )
    plt.tight_layout()
    path = od / "diag_layer_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    setup_paths()

    import torch
    import numpy as np

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    od = Path(args.output_dir)
    od.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────
    print("Loading model...")
    model, matcher, ckpt_args, container = load_model_and_matcher(
        args.checkpoint, device)

    dec_layers = int(getattr(ckpt_args, "dec_layers", 3))
    resolution = args.resolution or getattr(ckpt_args, "resolution", 640)

    print(f"  dec_layers={dec_layers}  hidden_dim="
          f"{getattr(ckpt_args, 'hidden_dim', '?')}  "
          f"aux_loss={getattr(ckpt_args, 'aux_loss', '?')}")

    # ── Load sequence ─────────────────────────────────────────────────
    if args.sequence_dir:
        img_paths, annotations, img_w, img_h = load_sequence_single(
            args.sequence_dir)
        seq_name = Path(args.sequence_dir).name
    else:
        if args.data_root is None:
            print("ERROR: provide --sequence_dir or --data_root")
            sys.exit(1)
        from data.dancetrack import DanceTrack
        ds    = DanceTrack(data_root=args.data_root, split=args.split,
                           load_annotation=True)
        seq_name = sorted(ds.image_paths.keys())[0]
        info     = ds.sequence_infos[seq_name]
        img_paths   = [ds.image_paths[seq_name][t]
                       for t in range(info["length"])]
        annotations = ds.annotations[seq_name]
        img_w, img_h = info["width"], info["height"]
        print(f"Using first sequence: {seq_name}")

    n_frames = min(args.num_frames, len(img_paths))
    print(f"Processing {n_frames} frames  ({seq_name})")

    # ── Forward passes — collect hs_all per frame ─────────────────────
    hs_frames    = []    # list of (dec_layers, N, D) tensors
    pred_boxes_f = []
    pred_logits_f = []
    annotations_f = []

    for t in range(n_frames):
        ann = annotations[t]
        if isinstance(ann, dict) and not ann.get("is_legal", True):
            continue

        img_t = load_and_preprocess(img_paths[t], resolution, device)
        with torch.no_grad():
            out = model(img_t)

        if container["hs_all"] is None:
            continue

        hs_frames.append(container["hs_all"][:, 0].cpu())   # (L, N, D)
        pred_boxes_f.append(out["pred_boxes"][0].cpu())
        pred_logits_f.append(out["pred_logits"][0].cpu())
        annotations_f.append(ann)

    print(f"Collected {len(hs_frames)} valid frames")

    # ── Per-pair measurements ─────────────────────────────────────────
    # Accumulators per layer
    acc_slot = {li: {"means": [], "stds": [], "n_unstable": [], "n_stable": []}
                for li in range(dec_layers)}
    acc_obj  = {li: [] for li in range(dec_layers)}
    acc_cross = defaultdict(list)

    for t in range(len(hs_frames) - 1):
        hs_t  = hs_frames[t].to(device)
        hs_t1 = hs_frames[t + 1].to(device)
        ann_t  = annotations_f[t]
        ann_t1 = annotations_f[t + 1]

        result = measure_frame_pair(
            hs_t, hs_t1,
            ann_t, ann_t1,
            matcher,
            pred_boxes_f[t].to(device),   pred_logits_f[t].to(device),
            pred_boxes_f[t + 1].to(device), pred_logits_f[t + 1].to(device),
            ann_t["bbox"],  ann_t["id"],
            ann_t1["bbox"], ann_t1["id"],
            img_w, img_h, device, dec_layers,
        )

        for li in range(dec_layers):
            s = result["slot_sims"][li]
            acc_slot[li]["means"].append(s["mean"])
            acc_slot[li]["stds"].append(s["std"])
            acc_slot[li]["n_unstable"].append(s["n_unstable"])
            acc_slot[li]["n_stable"].append(s["n_stable"])

            o = result["obj_sims"][li]
            if o["n"] > 0 and o["mean"] is not None:
                acc_obj[li].extend(
                    [o["mean"]] * o["n"]   # approximate — expand for agg
                )

        for pair_key, v in result["cross_layer"].items():
            acc_cross[pair_key].append(v["mean"])

    # ── Aggregate ─────────────────────────────────────────────────────
    agg_slot = {}
    for li in range(dec_layers):
        m = acc_slot[li]["means"]
        agg_slot[li] = {
            "mean": float(np.mean(m)) if m else None,
            "std":  float(np.std(m))  if m else None,
            "mean_n_unstable": float(np.mean(acc_slot[li]["n_unstable"])) if m else None,
            "mean_n_stable":   float(np.mean(acc_slot[li]["n_stable"]))   if m else None,
        }

    agg_obj = {}
    for li in range(dec_layers):
        v = acc_obj[li]
        agg_obj[li] = {
            "mean": float(np.mean(v)) if v else None,
            "std":  float(np.std(v))  if v else None,
            "n":    len(v),
        }

    agg_cross = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                 for k, v in acc_cross.items()}

    # ── Interpretation ────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("PER-LAYER RESULTS")
    print("=" * 55)
    print(f"  {'Layer':<8}  {'Slot-level mean':<20}  {'Object-matched mean':<22}  n_obj")
    for li in range(dec_layers):
        tag = " (final)" if li == dec_layers - 1 else \
              " (stale)" if li == dec_layers - 2 else ""
        slot_m = agg_slot[li]["mean"]
        obj_m  = agg_obj[li]["mean"]
        obj_n  = agg_obj[li]["n"]
        obj_m_str = f"{obj_m:.4f}" if obj_m is not None else "N/A"
        print(f"  L{li}{tag:<10}  "
              f"{slot_m:.4f}{'':>14}  "
              f"{obj_m_str:<22}  {obj_n}")

    print("\nCross-layer similarity (same frame, same slot):")
    for pair, v in sorted(agg_cross.items()):
        print(f"  {pair}: mean={v['mean']:.4f}  std={v['std']:.4f}")

    # Hypothesis evaluation (from diag_query_divergence.md §5)
    final_slot_mean = agg_slot.get(dec_layers - 1, {}).get("mean", 0) or 0
    first_slot_mean = agg_slot.get(0, {}).get("mean", 0) or 0
    print("\nHypothesis evaluation:")
    if first_slot_mean > final_slot_mean + 0.05:
        h = ("#1 CONFIRMED: layer 0 > layer 2 (by "
             f"{first_slot_mean - final_slot_mean:.3f}) — "
             "NAS weight sharing artifact; first layer more regularized")
    elif final_slot_mean > first_slot_mean + 0.05:
        h = ("#2 CONFIRMED: layer 2 > layer 0 (by "
             f"{final_slot_mean - first_slot_mean:.3f}) — "
             "depth refinement adds stability")
    else:
        h = ("#3 CONFIRMED: all layers similar — "
             "encoder top-K re-ranking is the dominant source of instability, "
             "not depth")
    print(f"  {h}")

    # ── Plot ──────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_results(od, dec_layers, agg_slot, agg_obj, agg_cross, ckpt_args)

    # ── Save JSON ──────────────────────────────────────────────────────
    results = {
        "checkpoint":     args.checkpoint,
        "sequence":       seq_name,
        "n_pairs":        len(hs_frames) - 1,
        "dec_layers":     dec_layers,
        "model_info": {
            "hidden_dim":  int(getattr(ckpt_args, "hidden_dim", 0)),
            "num_queries": int(getattr(ckpt_args, "num_queries", 0)),
            "dec_layers":  dec_layers,
            "aux_loss":    bool(getattr(ckpt_args, "aux_loss", False)),
            "two_stage":   bool(getattr(ckpt_args, "two_stage", False)),
        },
        "per_layer_slot_stability":    {str(k): v for k, v in agg_slot.items()},
        "per_layer_object_stability":  {str(k): v for k, v in agg_obj.items()},
        "cross_layer_similarity":      agg_cross,
        "hypothesis": h,
    }

    path = od / "layer_comparison_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nJSON  → {path}")
    print(f"Plot  → {od}/diag_layer_comparison.png")
    print("=" * 55)


if __name__ == "__main__":
    main()