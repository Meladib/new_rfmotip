#!/usr/bin/env python3
"""
diag_object_matched.py
======================
Object-matched embedding diagnostics for RF-DETR + MOTIP.

What this adds vs diag_temporal_stability_script.py
----------------------------------------------------
The temporal stability script measures slot-level similarity (same slot k,
frames T→T+1) across ALL 300 query slots, including ~270+ inactive background
queries.  The mean of 0.531 it produced is therefore contaminated by inactive
slot behaviour and does not answer whether the same *physical object* produces
consistent embeddings across frames.

This script uses Hungarian matching + GT track IDs to measure only the
embeddings that actually correspond to real objects.

Diagnostics covered
-------------------
  Diag 1 — Similarity Gap (Hard-Aware)
            positive (same track_id, T→T+1) vs unmatched cross-frame
            vs within-frame, split by difficulty (easy/medium/hard)
  Diag 2 — Intrinsic Dimensionality
            PCA on object-matched embeddings; n_components for 90/99% variance
  Diag 4 — Active vs Inactive Separation
            L2 norm distributions: matched queries vs background queries
  Diag 5 — Object-Matched Temporal Decay
            cosine sim at gaps 1, 2, 5, 10, 20 frames (object-matched)
  Diag 6 — Identity Separability
            nearest-neighbour accuracy + t-SNE
  Diag 7 — Hard Negative Analysis  [CRITICAL]
            cosine sim between spatially-close different-ID objects (IoU > 0.3)

Does NOT use the out["outputs"] patch.
Uses a register_forward_hook on model.transformer (same strategy as
diag_temporal_stability_script.py) so no source file changes are needed.

Run from inside MOTIP/:
  python diagnostics/diag_object_matched.py \\
    --checkpoint rfdetr_dancetrack_motip/checkpoint_best_total.pth \\
    --sequence_dir /data/DanceTrack/val/dancetrack0004 \\
    --output_dir diagnostics/object_matched/

Multi-sequence mode:
  python diagnostics/diag_object_matched.py \\
    --checkpoint rfdetr_dancetrack_motip/checkpoint_best_total.pth \\
    --data_root ./datasets/ \\
    --split val \\
    --num_sequences 5 \\
    --output_dir diagnostics/object_matched/
"""

import sys
import os
import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description="Object-matched embedding diagnostics (Diags 1,2,4,5,6,7)"
    )
    p.add_argument("--checkpoint", required=True,
                   help="RF-DETR or MOTIP checkpoint (.pth)")
    # sequence input — one of these two modes
    p.add_argument("--sequence_dir", default=None,
                   help="Single DanceTrack sequence dir (has img1/ and gt/gt.txt)")
    p.add_argument("--data_root", default=None,
                   help="DanceTrack data root, e.g. ./datasets/  "
                        "(alternative to --sequence_dir)")
    p.add_argument("--split", default="val")
    p.add_argument("--num_sequences", type=int, default=3,
                   help="Sequences to process when using --data_root  (0=all)")
    # output
    p.add_argument("--output_dir", default="diagnostics/object_matched/")
    # model / runtime
    p.add_argument("--device", default=None,
                   help="cuda or cpu  (auto-detected if not set)")
    p.add_argument("--resolution", type=int, default=None,
                   help="Override inference resolution (default: from ckpt args)")
    p.add_argument("--score_thresh", type=float, default=0.3,
                   help="Confidence threshold (only used for reporting; "
                        "matcher uses all queries)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# PATH SETUP
# ─────────────────────────────────────────────────────────────────────────────

def setup_paths():
    """Ensure MOTIP root is on sys.path so all internal imports work."""
    here      = os.path.dirname(os.path.abspath(__file__))
    motip_root = os.path.dirname(here)          # one level up from diagnostics/
    if motip_root not in sys.path:
        sys.path.insert(0, motip_root)
    return motip_root


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING  (mirrors models/motip/__init__.py  case "rf_detr":)
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_matcher(checkpoint_path, device):
    """
    Mirrors models/motip/__init__.py case "rf_detr": exactly.
    Returns (model, matcher, args_ckpt).
    matcher = criterion.matcher  (HungarianMatcher already built).
    """
    import torch
    from models.rfdetr.models.lwdetr import (
        build_model,
        build_criterion_and_postprocessors,
    )

    ckpt      = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args_ckpt = ckpt["args"]

    # build_model does num_classes+1 internally, so subtract 1 first
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
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        print(f"  [load] Matched: {len(filtered)}  "
              f"Missing: {len(missing)}  Unexpected: {len(unexpected)}")
    else:
        print("  [load] WARNING: no 'model' key — random decoder weights")

    model.eval().to(device)
    matcher = criterion.matcher   # HungarianMatcher, already built

    return model, matcher, args_ckpt


# ─────────────────────────────────────────────────────────────────────────────
# FORWARD HOOK  (captures hs[-1] without patching lwdetr.py)
# ─────────────────────────────────────────────────────────────────────────────

def install_transformer_hook(model):
    """
    Register a forward hook on model.transformer.
    Transformer.forward() returns (hs, references, hs_enc, ref_enc).
    When return_intermediate=True and bbox_embed is not None:
        hs = torch.stack(intermediate)  shape (dec_layers, B, N, D)
    We capture hs[-1] — the final decoder layer output.

    Returns a container dict; container["hs_last"] is a (B, N, D) tensor
    set after each forward pass.
    """
    container = {"hs_last": None}

    def _hook(module, inputs, output):
        # output[0] = hs  shape (dec_layers, B, N, D)
        hs = output[0]
        if isinstance(hs, (list, tuple)):
            hs = hs[0]                  # unpack if nested list
        container["hs_last"] = hs[-1].detach()   # (B, N, D)

    model.transformer.register_forward_hook(_hook)
    return container


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
    return t.unsqueeze(0).to(device)         # (1, 3, H, W)


def xywh_pixel_to_cxcywh_norm(boxes_xywh, img_w, img_h):
    """DanceTrack GT: pixel x,y,w,h  →  cx,cy,w,h normalized [0,1]."""
    import torch
    x, y, w, h = boxes_xywh.unbind(-1)
    return torch.stack([
        (x + w / 2) / img_w,
        (y + h / 2) / img_h,
        w / img_w,
        h / img_h,
    ], -1)


def load_sequence_single(sequence_dir):
    """
    Load one DanceTrack sequence from its directory.
    Returns:
        image_paths : List[str]        (0-indexed; path to frame t = index t)
        annotations : List[dict]       keys "id" Tensor[M], "bbox" Tensor[M,4]
        img_w, img_h: int
    """
    import torch
    from configparser import ConfigParser

    seq = Path(sequence_dir)
    ini = ConfigParser()
    ini.read(seq / "seqinfo.ini")
    img_w   = int(ini["Sequence"]["imWidth"])
    img_h   = int(ini["Sequence"]["imHeight"])
    seq_len = int(ini["Sequence"]["seqLength"])

    image_paths = [str(seq / "img1" / f"{i + 1:08d}.jpg") for i in range(seq_len)]

    # Read GT
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
            ann = {
                "id":   torch.zeros(0, dtype=torch.int64),
                "bbox": torch.zeros((0, 4), dtype=torch.float32),
            }
        annotations.append(ann)

    return image_paths, annotations, img_w, img_h


# ─────────────────────────────────────────────────────────────────────────────
# PER-FRAME EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_frame(model, matcher, hook_container,
                  image_path, annotation,
                  img_w, img_h, resolution, device):
    """
    Run model on one frame, Hungarian-match to GT, return per-frame dict.
    Returns None if the frame has no GT objects.

    Output dict keys:
        active_embeds  Tensor(M, D)   embeddings of matched queries
        active_boxes   Tensor(M, 4)   predicted boxes cx,cy,w,h norm
        active_scores  Tensor(M,)     max sigmoid score of matched query
        query_indices  Tensor(M,)     which query slot was matched (0..N-1)
        gt_ids         Tensor(M,)     GT track IDs, aligned to above
        all_embeds     Tensor(N, D)   all N query embeddings
        inactive_mask  Tensor(N,)     True = not matched to any GT object
    """
    import torch

    gt_ids    = annotation["id"]
    gt_bboxes = annotation["bbox"]
    if len(gt_ids) == 0:
        return None

    img_t = load_and_preprocess(image_path, resolution, device)
    with torch.no_grad():
        out = model(img_t)

    # hs[-1] from hook  —  shape (B, N, D), B=1
    hs_last = hook_container["hs_last"]
    if hs_last is None:
        return None
    hs_last = hs_last[0]                       # (N, D)

    pred_boxes  = out["pred_boxes"][0]          # (N, 4)
    pred_logits = out["pred_logits"][0]         # (N, C)

    # Convert GT boxes to normalized cx,cy,w,h for the matcher
    gt_boxes_norm = xywh_pixel_to_cxcywh_norm(
        gt_bboxes, img_w, img_h).to(device)
    gt_labels = torch.zeros(len(gt_ids), dtype=torch.long, device=device)

    with torch.no_grad():
        indices = matcher(
            {
                "pred_logits": pred_logits.unsqueeze(0),
                "pred_boxes":  pred_boxes.unsqueeze(0),
            },
            [{"boxes": gt_boxes_norm, "labels": gt_labels}],
        )

    q_idx, gt_idx = indices[0]                 # both Tensor(M,)

    # Sort by GT index so rows of active_embeds align with gt_ids order
    order      = torch.argsort(gt_idx)
    q_sorted   = q_idx[order]
    gt_sorted  = gt_idx[order]

    inactive   = torch.ones(hs_last.shape[0], dtype=torch.bool, device=device)
    inactive[q_idx] = False

    return {
        "active_embeds":  hs_last[q_sorted],
        "active_boxes":   pred_boxes[q_sorted],
        "active_scores":  pred_logits.sigmoid().max(-1).values[q_sorted],
        "query_indices":  q_sorted,
        "gt_ids":         gt_ids[gt_sorted.cpu()],
        "all_embeds":     hs_last,
        "inactive_mask":  inactive,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SHARED UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def cosine_sim(a, b):
    import torch.nn.functional as F
    return (F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum(-1)


def pairwise_iou(boxes_cxcywh, box_iou_fn, box_cxcywh_to_xyxy_fn):
    """Returns (N, N) IoU matrix with diagonal set to 0."""
    if len(boxes_cxcywh) < 2:
        import torch
        return torch.zeros(len(boxes_cxcywh), len(boxes_cxcywh))
    bxyxy     = box_cxcywh_to_xyxy_fn(boxes_cxcywh)
    iou, _    = box_iou_fn(bxyxy, bxyxy)
    iou.fill_diagonal_(0.0)
    return iou


def difficulty_labels(boxes_cxcywh, box_iou_fn, box_cxcywh_to_xyxy_fn):
    """
    Returns List[str] of "easy"/"medium"/"hard" per box.
    Based on max IoU of each box with any other box in the same frame.
    """
    iou     = pairwise_iou(boxes_cxcywh, box_iou_fn, box_cxcywh_to_xyxy_fn)
    max_iou = iou.max(dim=1).values
    labels  = []
    for v in max_iou.tolist():
        labels.append("hard" if v > 0.7 else "medium" if v > 0.3 else "easy")
    return labels


def _stats(vals):
    import numpy as np
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean":   float(np.mean(vals)),
        "std":    float(np.std(vals)),
        "n":      len(vals),
        "values": vals,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DIAG 1 — SIMILARITY GAP  (Hard-Aware)
# ─────────────────────────────────────────────────────────────────────────────

def diag1_similarity_gap(frames, box_iou_fn, box_cxcywh_to_xyxy_fn):
    """
    For each consecutive frame pair (t, t+1):
      positive        : same track_id cross-frame
      unmatched_cross : different track_id cross-frame
      within_frame    : different track_id same frame

    Each pair is bucketed by difficulty of the frame-t object:
      easy   : max pairwise IoU with any other box <= 0.3
      medium : 0.3 < IoU <= 0.7
      hard   : IoU > 0.7

    Gap = positive.mean - unmatched_cross.mean per bucket.

    Interpretation (from GAP_DESCRIPTION.md):
      gap < 0.05          → no identity signal → must fine-tune decoder
      0.05 – 0.10         → weak signal → nonlinear model required
      0.10 – 0.20         → moderate → projection head viable
      > 0.20              → strong signal
      gap only in easy    → NOT usable for tracking
    """
    import numpy as np

    results = {k: defaultdict(list)
               for k in ["positive", "unmatched_cross", "within_frame"]}

    for t in range(len(frames) - 1):
        ft, ft1 = frames[t], frames[t + 1]
        if ft is None or ft1 is None:
            continue

        diff    = difficulty_labels(ft["active_boxes"], box_iou_fn,
                                    box_cxcywh_to_xyxy_fn)
        ids_t   = ft["gt_ids"]
        ids_t1  = ft1["gt_ids"]
        embs_t  = ft["active_embeds"]
        embs_t1 = ft1["active_embeds"]
        id_map  = {int(ids_t1[j]): j for j in range(len(ids_t1))}

        for i, id_i in enumerate(ids_t):
            b      = diff[i]
            id_int = int(id_i)

            # positive: same ID in next frame
            if id_int in id_map:
                j = id_map[id_int]
                results["positive"][b].append(
                    float(cosine_sim(embs_t[i], embs_t1[j])))

            # unmatched cross-frame: different ID
            for j, id_j in enumerate(ids_t1):
                if int(id_j) != id_int:
                    results["unmatched_cross"][b].append(
                        float(cosine_sim(embs_t[i], embs_t1[j])))

            # within frame: different ID same frame
            for k in range(len(ids_t)):
                if k != i:
                    results["within_frame"][b].append(
                        float(cosine_sim(embs_t[i], embs_t[k])))

    stats = {key: {b: _stats(results[key][b])
                   for b in ["easy", "medium", "hard"]}
             for key in results}

    gap = {}
    for b in ["easy", "medium", "hard"]:
        pos = stats["positive"].get(b, {})
        neg = stats["unmatched_cross"].get(b, {})
        if pos.get("mean") is not None and neg.get("mean") is not None:
            gap[b] = round(pos["mean"] - neg["mean"], 6)

    return stats, gap


# ─────────────────────────────────────────────────────────────────────────────
# DIAG 2 — INTRINSIC DIMENSIONALITY
# ─────────────────────────────────────────────────────────────────────────────

def diag2_dimensionality(all_embeds_np):
    """
    PCA on all object-matched embeddings.
    Returns n_components for 90% and 99% explained variance.

    Interpretation:
      n_components_90 < 32   → low-rank → small projection head
      32 – 80                → moderate → standard MLP
      > 80                   → high → avoid aggressive compression
    """
    import numpy as np

    E = all_embeds_np
    if len(E) > 50_000:
        E = E[np.random.choice(len(E), 50_000, replace=False)]

    # PCA via numpy SVD — no sklearn needed
    E_centered = E - E.mean(axis=0)
    _, s, _    = np.linalg.svd(E_centered, full_matrices=False)
    var_ratio  = (s ** 2) / (s ** 2).sum()
    cumvar     = np.cumsum(var_ratio)

    return {
        "n_components_90":          int(np.searchsorted(cumvar, 0.90)) + 1,
        "n_components_99":          int(np.searchsorted(cumvar, 0.99)) + 1,
        "explained_variance_ratio": var_ratio.tolist(),
        "cumulative_variance":      cumvar.tolist(),
        "n_samples_used":           int(len(E)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# DIAG 4 — ACTIVE vs INACTIVE SEPARATION
# ─────────────────────────────────────────────────────────────────────────────

def diag4_active_inactive(frames):
    """
    L2 norm of embeddings for matched (active) vs unmatched (inactive) queries.

    If norms are clearly separated, a simple norm threshold can filter
    background queries before passing to the ID decoder.
    If they overlap, the head must handle noisy background embeddings.
    """
    import torch
    import numpy as np

    active_norms, inactive_norms = [], []
    for f in frames:
        if f is None:
            continue
        active_norms.extend(
            torch.norm(f["all_embeds"][~f["inactive_mask"]], dim=-1).cpu().tolist())
        inactive_norms.extend(
            torch.norm(f["all_embeds"][f["inactive_mask"]],  dim=-1).cpu().tolist())

    if not active_norms:
        return {}

    return {
        "active_norm_mean":   float(np.mean(active_norms)),
        "active_norm_std":    float(np.std(active_norms)),
        "inactive_norm_mean": float(np.mean(inactive_norms)),
        "inactive_norm_std":  float(np.std(inactive_norms)),
        "n_active":           len(active_norms),
        "n_inactive":         len(inactive_norms),
        # raw lists kept for plotting; stripped from JSON summary
        "_active_norms":      active_norms,
        "_inactive_norms":    inactive_norms,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DIAG 5 — OBJECT-MATCHED TEMPORAL DECAY
# ─────────────────────────────────────────────────────────────────────────────

def diag5_temporal_decay(frames):
    """
    For each matched object, compute cosine sim between its embedding at
    frame t and frame t+gap, for gap in [1, 2, 5, 10, 20].

    IMPORTANT: call this per-sequence only.
    Track IDs are not globally unique across sequences.

    Contrast with diag_temporal_stability: that script measured slot k
    (position) across frames for ALL slots.  This function measures the
    same object (matched by GT track_id) across frames.

    Interpretation:
      Sharp drop at gap=1  → frame-sensitive, encoder re-ranking dominates
      Gradual decay        → stable identity representation
      High variance        → inconsistent; hard to learn trajectory model
    """
    import numpy as np

    GAPS = [1, 2, 5, 10, 20]

    # Build frame lookup: t -> {track_id: embedding_tensor}
    fe = {}
    for t, f in enumerate(frames):
        if f is None:
            continue
        fe[t] = {int(tid): f["active_embeds"][i]
                 for i, tid in enumerate(f["gt_ids"])}

    gap_sims = {g: [] for g in GAPS}
    for t in sorted(fe):
        for gap in GAPS:
            t2 = t + gap
            if t2 not in fe:
                continue
            for tid, emb_t in fe[t].items():
                if tid in fe[t2]:
                    gap_sims[gap].append(
                        float(cosine_sim(emb_t, fe[t2][tid])))

    return {g: _stats(v) for g, v in gap_sims.items() if v}


# ─────────────────────────────────────────────────────────────────────────────
# DIAG 6 — IDENTITY SEPARABILITY
# ─────────────────────────────────────────────────────────────────────────────

def diag6_nn_accuracy(all_embeds_np, all_ids_np):
    """
    For each active embedding, find its nearest neighbour by cosine distance
    across the entire collected set.  Measure % where NN has same track_id.

    A high accuracy means embeddings cluster by identity in feature space.
    A chance-level accuracy means no identity structure.

    Interpretation:
      > 0.7          → strong identity structure
      0.3 – 0.7      → partial separability
      <= chance * 3  → no identity structure
    """
    import numpy as np

    norms  = np.linalg.norm(all_embeds_np, axis=1, keepdims=True) + 1e-8
    E_norm = all_embeds_np / norms

    # Nearest neighbour via cosine similarity matrix — no sklearn needed
    # Subsample to 5000 max to keep memory manageable
    N = len(E_norm)
    if N > 5000:
        sample_idx = np.random.choice(N, 5000, replace=False)
        E_sub = E_norm[sample_idx]
        ids_sub = all_ids_np[sample_idx]
    else:
        E_sub = E_norm
        ids_sub = all_ids_np

    # cosine similarity matrix (5000×5000 max = 200MB float32 — acceptable)
    sim_matrix = E_sub @ E_sub.T                      # (M, M)
    np.fill_diagonal(sim_matrix, -2.0)                # exclude self
    nearest_idx = np.argmax(sim_matrix, axis=1)       # (M,)
    nearest_ids = ids_sub[nearest_idx]
    nn_acc      = float(np.mean(nearest_ids == ids_sub))
    n_ids       = len(np.unique(all_ids_np))

    return {
        "nn_accuracy":  round(nn_acc, 6),
        "chance_level": round(1.0 / n_ids, 6) if n_ids > 0 else 0.0,
        "n_unique_ids": int(n_ids),
        "n_total":      int(len(all_ids_np)),
        "n_sampled":    int(len(E_sub)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# DIAG 7 — HARD NEGATIVE ANALYSIS  [CRITICAL]
# ─────────────────────────────────────────────────────────────────────────────

def diag7_hard_negatives(frames, box_iou_fn, box_cxcywh_to_xyxy_fn):
    """
    For each pair of objects in the same frame with different track IDs:
      hard negative : IoU > 0.3  (spatially close, different identity)
      easy negative : IoU <= 0.3

    Measures cosine similarity for each pair.

    Critical interpretation:
      hard_neg_mean > 0.7              → identity confusion in crowds
                                         → decoder fine-tuning required
      hard_neg_mean - easy_neg_mean > 0.15
                                       → crowded cases are harder than
                                         expected → contrastive loss needed
      low hard_neg_mean                → good crowd discrimination
    """
    hard, easy = [], []

    for f in frames:
        if f is None or len(f["gt_ids"]) < 2:
            continue

        embs = f["active_embeds"]
        ids  = f["gt_ids"]
        iou  = pairwise_iou(f["active_boxes"], box_iou_fn,
                             box_cxcywh_to_xyxy_fn)

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if int(ids[i]) == int(ids[j]):
                    continue          # same track — skip
                sim = float(cosine_sim(embs[i], embs[j]))
                if float(iou[i, j]) > 0.3:
                    hard.append(sim)
                else:
                    easy.append(sim)

    return {
        "hard_negatives": _stats(hard),
        "easy_negatives": _stats(easy),
    }


# ─────────────────────────────────────────────────────────────────────────────
# DECISION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def make_decision(d1_gap, d2, d4, d5, d6, d7):
    """
    Apply decision rules from GAP_DESCRIPTION.md and DIAGNOSTICS.md.
    Returns a flat dict of human-readable verdicts.
    """
    import numpy as np

    out = {}

    # ── Diag 1: identity signal ──────────────────────────────────────
    hg = d1_gap.get("hard")
    eg = d1_gap.get("easy")
    mg = d1_gap.get("medium")

    if hg is None:
        sig = "INSUFFICIENT_DATA"
    elif hg < 0.05:
        sig = "NO_SIGNAL — MUST FINE-TUNE DECODER"
    elif hg < 0.10:
        sig = "WEAK — NONLINEAR MODEL REQUIRED"
    elif hg < 0.20:
        sig = "MODERATE — PROJECTION HEAD VIABLE"
    else:
        sig = "STRONG — DIRECT PROJECTION VIABLE"

    if eg is not None and hg is not None and eg > 0.20 and hg < 0.05:
        sig += "  [WARNING: gap only in easy cases — NOT usable for tracking]"

    out["identity_signal"] = {
        "verdict":    sig,
        "gap_easy":   eg,
        "gap_medium": mg,
        "gap_hard":   hg,
    }

    # ── Diag 2: dimensionality ───────────────────────────────────────
    if d2:
        n90 = d2["n_components_90"]
        out["dimensionality"] = {
            "verdict": ("LOW-RANK (<32)  — small projection head"
                        if n90 < 32 else
                        "MODERATE (32–80) — standard MLP"
                        if n90 < 80 else
                        "HIGH (>80)  — avoid aggressive compression"),
            "n_components_90": n90,
            "n_components_99": d2["n_components_99"],
        }

    # ── Diag 4: active/inactive ──────────────────────────────────────
    if d4:
        sep = abs(d4.get("active_norm_mean", 0) -
                  d4.get("inactive_norm_mean", 0))
        out["active_inactive_separation"] = {
            "verdict": ("CLEAR SEPARATION — norm filter viable"
                        if sep > 1.0 else
                        "OVERLAP — head must handle background noise"),
            "active_mean":   d4.get("active_norm_mean"),
            "inactive_mean": d4.get("inactive_norm_mean"),
            "delta":         round(sep, 4),
        }

    # ── Diag 5: temporal decay ───────────────────────────────────────
    if d5:
        gap1 = d5.get(1, {}).get("mean")
        gap5 = d5.get(5, {}).get("mean")
        if gap1 is not None:
            if gap1 < 0.5:
                decay = "SEVERE — embeddings effectively random at gap=1"
            elif gap1 < 0.7:
                decay = "HIGH INSTABILITY — significant drift at gap=1"
            elif gap1 < 0.9:
                decay = "MODERATE — some consistency but insufficient for robust re-ID"
            else:
                decay = "STABLE"
            out["temporal_decay"] = {
                "verdict":     decay,
                "mean_gap_1":  round(gap1, 4),
                "mean_gap_5":  round(gap5, 4) if gap5 else None,
                "mean_gap_20": round(d5.get(20, {}).get("mean", 0), 4),
            }

    # ── Diag 6: identity separability ───────────────────────────────
    if d6:
        nn  = d6["nn_accuracy"]
        ch  = d6["chance_level"]
        out["identity_separability"] = {
            "verdict": ("STRONG — >0.7 NN accuracy"
                        if nn > 0.7 else
                        "PARTIAL — 0.3–0.7"
                        if nn > 0.3 else
                        "WEAK — above chance"
                        if nn > ch * 3 else
                        "CHANCE LEVEL — no identity structure"),
            "nn_accuracy":  round(nn, 4),
            "chance_level": round(ch, 4),
        }

    # ── Diag 7: crowd discrimination ────────────────────────────────
    if d7:
        hn = d7["hard_negatives"].get("mean")
        en = d7["easy_negatives"].get("mean")
        if hn is None:
            crowd = "NO_HARD_NEGATIVES_FOUND"
        elif hn > 0.7:
            crowd = "IDENTITY CONFUSION IN CROWDS — decoder fine-tuning required"
        elif en is not None and (hn - en) > 0.15:
            crowd = "CROWD CONFUSION — contrastive loss recommended"
        else:
            crowd = "ACCEPTABLE — crowd discrimination present"
        out["crowd_discrimination"] = {
            "verdict":          crowd,
            "hard_neg_mean":    round(hn, 4) if hn else None,
            "easy_neg_mean":    round(en, 4) if en else None,
            "hard_minus_easy":  round(hn - en, 4) if (hn and en) else None,
        }

    # ── Final verdict ────────────────────────────────────────────────
    nn_acc  = d6.get("nn_accuracy", 0) if d6 else 0
    nn_ch   = d6.get("chance_level", 0) if d6 else 0
    hn_mean = d7["hard_negatives"].get("mean") if d7 else None

    needs_finetune = (
        (hg is not None and hg < 0.05) or
        (hn_mean is not None and hn_mean > 0.7) or
        (d6 is not None and nn_acc < nn_ch * 3)
    )
    proj_viable = (
        hg is not None and hg >= 0.05 and
        (hn_mean is None or hn_mean < 0.7)
    )

    if needs_finetune:
        final = "DECODER FINE-TUNING REQUIRED"
    elif proj_viable and hg >= 0.10:
        final = "PROJECTION HEAD VIABLE — proceed with temporal module"
    else:
        final = "MARGINAL — consider light fine-tuning + projection head"

    out["final_verdict"] = final
    return out


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_all(output_dir, d1_stats, d1_gap,
             d2, d4, d5, d7,
             all_embeds_np=None, all_ids_np=None):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    od = Path(output_dir)

    # ── Diag 1: similarity distributions ──────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, bucket in enumerate(["easy", "medium", "hard"]):
        ax = axes[i]
        has_data = False
        for key, color, label in [
            ("positive",        "green", "Positive (same ID)"),
            ("unmatched_cross", "red",   "Unmatched cross-frame"),
            ("within_frame",    "blue",  "Within-frame neg"),
        ]:
            vals = (d1_stats.get(key, {})
                            .get(bucket, {})
                            .get("values", []))
            if vals:
                ax.hist(vals, bins=50, alpha=0.4, color=color,
                        label=label, density=True)
                has_data = True
        g = d1_gap.get(bucket)
        title = (f"{bucket.capitalize()}  (gap = {g:+.4f})"
                 if g is not None else f"{bucket.capitalize()}  (no data)")
        if not has_data:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=11, color="grey")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)

    plt.suptitle(
        "Diag 1 — Object-Matched Similarity Gap  "
        "(positive vs unmatched × difficulty)",
        fontsize=11,
    )
    plt.tight_layout()
    path = od / "diag1_object_similarity_gap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {path.name}")

    # ── Diag 2: PCA cumulative variance ───────────────────────────────
    if d2 and d2.get("cumulative_variance"):
        fig, ax = plt.subplots(figsize=(10, 5))
        cv = d2["cumulative_variance"]
        ax.plot(range(1, len(cv) + 1), cv, lw=2, color="steelblue")
        ax.axhline(0.90, color="orange", ls="--",
                   label=f"90% @ k = {d2['n_components_90']}")
        ax.axhline(0.99, color="red",    ls="--",
                   label=f"99% @ k = {d2['n_components_99']}")
        ax.set_xlabel("Number of PCA Components")
        ax.set_ylabel("Cumulative Explained Variance")
        ax.set_title("Diag 2 — Intrinsic Dimensionality of Active Embeddings")
        ax.legend()
        path = od / "diag2_dimensionality.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  saved {path.name}")

    # ── Diag 4: active vs inactive L2 norms ───────────────────────────
    if d4 and d4.get("_active_norms"):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(d4["_active_norms"],   bins=60, alpha=0.5, color="green",
                label=f"Active   μ={d4['active_norm_mean']:.3f}  "
                      f"σ={d4['active_norm_std']:.3f}",
                density=True)
        ax.hist(d4["_inactive_norms"], bins=60, alpha=0.5, color="red",
                label=f"Inactive μ={d4['inactive_norm_mean']:.3f}  "
                      f"σ={d4['inactive_norm_std']:.3f}",
                density=True)
        ax.set_xlabel("L2 Norm")
        ax.set_ylabel("Density")
        ax.set_title("Diag 4 — Active vs Inactive Query L2 Norms")
        ax.legend()
        path = od / "diag4_active_vs_inactive.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  saved {path.name}")

    # ── Diag 5: object-matched temporal decay ─────────────────────────
    if d5:
        GAPS  = sorted(d5.keys())
        means = [d5[g]["mean"] for g in GAPS]
        stds  = [d5[g]["std"]  for g in GAPS]
        ns    = [d5[g]["n"]    for g in GAPS]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(GAPS, means, marker="o", lw=2, color="steelblue",
                label="Mean cosine sim (object-matched)")
        ax.fill_between(GAPS,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.2, color="steelblue", label="±1 std")
        for g, m, n in zip(GAPS, means, ns):
            ax.annotate(f"n={n}", (g, m), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=7)
        ax.axhline(0.90, color="green",  ls="--", alpha=0.6, label="stable (0.9)")
        ax.axhline(0.70, color="orange", ls="--", alpha=0.6, label="high-instability (0.7)")
        ax.axhline(0.50, color="red",    ls="--", alpha=0.6, label="instability (0.5)")
        ax.set_xlabel("Frame Gap")
        ax.set_ylabel("Cosine Similarity (object-matched by track_id)")
        ax.set_title("Diag 5 — Object-Matched Temporal Decay\n"
                     "(same physical object, different frames)")
        ax.set_ylim(-0.1, 1.05)
        ax.legend(fontsize=8)
        path = od / "diag5_object_temporal_decay.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  saved {path.name}")

    # ── Diag 7: hard negatives ─────────────────────────────────────────
    if d7:
        fig, ax = plt.subplots(figsize=(9, 5))
        hv = d7["hard_negatives"].get("values", [])
        ev = d7["easy_negatives"].get("values", [])
        if hv:
            hm = d7["hard_negatives"]["mean"]
            hs = d7["hard_negatives"]["std"]
            ax.hist(hv, bins=60, alpha=0.5, color="red",
                    label=f"Hard neg (IoU > 0.3)  "
                          f"μ={hm:.3f}  σ={hs:.3f}  n={len(hv)}",
                    density=True)
        if ev:
            em = d7["easy_negatives"]["mean"]
            es = d7["easy_negatives"]["std"]
            ax.hist(ev, bins=60, alpha=0.5, color="blue",
                    label=f"Easy neg (IoU ≤ 0.3)  "
                          f"μ={em:.3f}  σ={es:.3f}  n={len(ev)}",
                    density=True)
        if not hv and not ev:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=11, color="grey")
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Density")
        ax.set_title("Diag 7 — Hard Negative Analysis  "
                     "(spatially close objects, different IDs)")
        ax.legend(fontsize=8)
        path = od / "diag7_hard_negatives.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  saved {path.name}")

    # ── Diag 6: 2D projection via PCA (sklearn-free fallback for t-SNE) ─
    if all_embeds_np is not None and len(all_embeds_np) >= 10:
        try:
            N   = min(2000, len(all_embeds_np))
            idx = np.random.choice(len(all_embeds_np), N, replace=False)
            E   = all_embeds_np[idx]
            ids = all_ids_np[idx]

            # PCA 2D via numpy SVD (no sklearn/TSNE needed)
            E_c = E - E.mean(axis=0)
            _, _, Vt = np.linalg.svd(E_c, full_matrices=False)
            coords = E_c @ Vt[:2].T    # (N, 2)

            cmap   = plt.cm.get_cmap("tab20", 20)
            colors = [cmap(int(uid) % 20) for uid in ids]

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.scatter(coords[:, 0], coords[:, 1],
                       c=colors, s=5, alpha=0.6)
            ax.set_title(
                f"Diag 6 — PCA 2D Projection of Active Embeddings\n"
                f"n={N}, colored by track_id (mod 20)")
            ax.set_xlabel("PC 1")
            ax.set_ylabel("PC 2")
            path = od / "diag6_pca2d.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  saved {path.name}")
        except Exception as e:
            print(f"  PCA 2D plot skipped: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    setup_paths()

    import torch
    import numpy as np
    from models.rfdetr.util.box_ops import box_iou, box_cxcywh_to_xyxy

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    od = Path(args.output_dir)
    od.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────
    print("Loading model...")
    model, matcher, ckpt_args = load_model_and_matcher(args.checkpoint, device)
    hook = install_transformer_hook(model)

    resolution = args.resolution or getattr(ckpt_args, "resolution", 640)
    print(f"  hidden_dim   = {getattr(ckpt_args, 'hidden_dim', '?')}")
    print(f"  num_queries  = {getattr(ckpt_args, 'num_queries', '?')}")
    print(f"  dec_layers   = {getattr(ckpt_args, 'dec_layers', '?')}")
    print(f"  aux_loss     = {getattr(ckpt_args, 'aux_loss', '?')}")
    print(f"  two_stage    = {getattr(ckpt_args, 'two_stage', '?')}")
    print(f"  resolution   = {resolution}")

    # ── Load sequences ────────────────────────────────────────────────
    sequences = []   # (name, image_paths, annotations, img_w, img_h)

    if args.sequence_dir:
        name = Path(args.sequence_dir).name
        print(f"\nLoading sequence: {name}")
        ip, ann, iw, ih = load_sequence_single(args.sequence_dir)
        sequences.append((name, ip, ann, iw, ih))
    else:
        if args.data_root is None:
            print("ERROR: provide --sequence_dir or --data_root")
            sys.exit(1)
        from data.dancetrack import DanceTrack
        ds    = DanceTrack(data_root=args.data_root, split=args.split,
                           load_annotation=True)
        names = sorted(ds.image_paths.keys())
        if args.num_sequences > 0:
            names = names[:args.num_sequences]
        print(f"\nLoading {len(names)} sequences from {args.data_root}/{args.split}")
        for name in names:
            info = ds.sequence_infos[name]
            sequences.append((
                name,
                [ds.image_paths[name][t] for t in range(info["length"])],
                ds.annotations[name],
                info["width"],
                info["height"],
            ))

    # ── Extract per-frame embeddings ──────────────────────────────────
    all_frame_data  = []       # flat list (all sequences)
    per_seq_frames  = {}       # seq_name -> List[fd or None]

    for seq_name, img_paths, annotations, img_w, img_h in sequences:
        print(f"\nExtracting: {seq_name}  ({len(img_paths)} frames)")
        seq_fds = []

        for t, (ip, ann) in enumerate(zip(img_paths, annotations)):
            # Skip frames the DanceTrack class marks as illegal
            if isinstance(ann, dict) and not ann.get("is_legal", True):
                seq_fds.append(None)
                continue

            try:
                fd = extract_frame(
                    model, matcher, hook,
                    ip, ann, img_w, img_h, resolution, device,
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  OOM at t={t}, switching to CPU")
                    model  = model.cpu()
                    device = torch.device("cpu")
                    fd = extract_frame(
                        model, matcher, hook,
                        ip, ann, img_w, img_h, resolution, device,
                    )
                else:
                    raise

            seq_fds.append(fd)
            if fd is not None:
                all_frame_data.append(fd)

        n_valid = sum(1 for f in seq_fds if f is not None)
        print(f"  Valid frames: {n_valid} / {len(img_paths)}")
        per_seq_frames[seq_name] = seq_fds

    if not all_frame_data:
        print("ERROR: no valid frames found. Check --sequence_dir or --data_root.")
        sys.exit(1)

    # ── Build flat numpy arrays for PCA and NN ────────────────────────
    all_embeds_np = np.vstack([f["active_embeds"].cpu().numpy()
                                for f in all_frame_data])
    all_ids_np    = np.concatenate([f["gt_ids"].numpy()
                                     for f in all_frame_data])
    print(f"\nTotal object-matched embeddings: {all_embeds_np.shape}")

    # ── Run diagnostics ───────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("RUNNING DIAGNOSTICS")
    print("=" * 55)

    # Diag 1 — aggregate across sequences (per-pair similarity)
    print("Diag 1: Similarity Gap...")
    agg_d1     = {k: defaultdict(list)
                  for k in ["positive", "unmatched_cross", "within_frame"]}
    agg_d1_gap = defaultdict(list)

    for seq_name, seq_fds in per_seq_frames.items():
        stats, gap = diag1_similarity_gap(seq_fds, box_iou, box_cxcywh_to_xyxy)
        for key in stats:
            for b, s in stats[key].items():
                agg_d1[key][b].extend(s.get("values", []))
        for b, g in gap.items():
            agg_d1_gap[b].append(g)

    d1_gap_final = {b: float(np.mean(v)) for b, v in agg_d1_gap.items()}
    d1_stats_final = {}
    for key in agg_d1:
        d1_stats_final[key] = {}
        for b in ["easy", "medium", "hard"]:
            v = list(agg_d1[key][b])
            if v:
                d1_stats_final[key][b] = _stats(v)

    print(f"  gap easy={d1_gap_final.get('easy')}"
          f"  medium={d1_gap_final.get('medium')}"
          f"  hard={d1_gap_final.get('hard')}")

    # Diag 2 — PCA
    print("Diag 2: Intrinsic Dimensionality...")
    d2 = diag2_dimensionality(all_embeds_np)
    print(f"  n_90={d2['n_components_90']}  n_99={d2['n_components_99']}")

    # Diag 4 — active vs inactive
    print("Diag 4: Active vs Inactive Norms...")
    agg_an, agg_in = [], []
    for seq_fds in per_seq_frames.values():
        r = diag4_active_inactive(seq_fds)
        if r:
            agg_an.extend(r.get("_active_norms", []))
            agg_in.extend(r.get("_inactive_norms", []))
    d4 = {}
    if agg_an:
        d4 = {
            "active_norm_mean":   float(np.mean(agg_an)),
            "active_norm_std":    float(np.std(agg_an)),
            "inactive_norm_mean": float(np.mean(agg_in)),
            "inactive_norm_std":  float(np.std(agg_in)),
            "n_active":           len(agg_an),
            "n_inactive":         len(agg_in),
            "_active_norms":      agg_an,
            "_inactive_norms":    agg_in,
        }
        print(f"  active μ={d4['active_norm_mean']:.3f}  "
              f"inactive μ={d4['inactive_norm_mean']:.3f}")

    # Diag 5 — per-sequence temporal decay (never merge sequences)
    print("Diag 5: Object-Matched Temporal Decay...")
    agg_d5 = defaultdict(list)
    for seq_fds in per_seq_frames.values():
        r = diag5_temporal_decay(seq_fds)
        for gap, s in r.items():
            agg_d5[gap].extend(s.get("values", []))
    d5 = {g: _stats(v) for g, v in agg_d5.items() if v}
    if d5:
        g1 = d5.get(1, {})
        print(f"  gap=1: mean={g1.get('mean', 'N/A'):.4f}"
              f"  std={g1.get('std', 'N/A'):.4f}"
              f"  n={g1.get('n', 0)}")

    # Diag 6 — NN accuracy
    print("Diag 6: Identity Separability (NN accuracy)...")
    d6 = diag6_nn_accuracy(all_embeds_np, all_ids_np)
    print(f"  nn_accuracy={d6['nn_accuracy']:.4f}  "
          f"chance={d6['chance_level']:.4f}  "
          f"n_ids={d6['n_unique_ids']}")

    # Diag 7 — hard negatives
    print("Diag 7: Hard Negatives...")
    agg_hard, agg_easy = [], []
    for seq_fds in per_seq_frames.values():
        r = diag7_hard_negatives(seq_fds, box_iou, box_cxcywh_to_xyxy)
        agg_hard.extend(r["hard_negatives"].get("values", []))
        agg_easy.extend(r["easy_negatives"].get("values", []))
    d7 = {
        "hard_negatives": _stats(agg_hard),
        "easy_negatives": _stats(agg_easy),
    }
    hn = d7["hard_negatives"].get("mean")
    en = d7["easy_negatives"].get("mean")
    if hn:
        print(f"  hard_neg μ={hn:.4f}  easy_neg μ={en:.4f}  "
              f"n_hard={len(agg_hard)}  n_easy={len(agg_easy)}")

    # ── Final decision ────────────────────────────────────────────────
    decision = make_decision(d1_gap_final, d2, d4, d5, d6, d7)

    # ── Plots ─────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_all(od, d1_stats_final, d1_gap_final, d2, d4, d5, d7,
             all_embeds_np=all_embeds_np, all_ids_np=all_ids_np)

    # ── Save JSON ──────────────────────────────────────────────────────
    def _strip(obj):
        """Remove raw value lists and internal _prefixed keys for JSON output."""
        if isinstance(obj, dict):
            return {k: _strip(v) for k, v in obj.items()
                    if k not in ("values", "_active_norms", "_inactive_norms")}
        return obj

    results = {
        "checkpoint":              args.checkpoint,
        "sequences":               [s[0] for s in sequences],
        "n_frames":                len(all_frame_data),
        "n_active_embeddings":     int(all_embeds_np.shape[0]),
        "embedding_dim":           int(all_embeds_np.shape[1]),
        "model_info": {
            "hidden_dim":  int(getattr(ckpt_args, "hidden_dim", 0)),
            "num_queries": int(getattr(ckpt_args, "num_queries", 0)),
            "dec_layers":  int(getattr(ckpt_args, "dec_layers", 0)),
            "aux_loss":    bool(getattr(ckpt_args, "aux_loss", False)),
            "two_stage":   bool(getattr(ckpt_args, "two_stage", False)),
        },
        "diag1_gap":               d1_gap_final,
        "diag1_stats":             _strip(d1_stats_final),
        "diag2_dimensionality":    _strip(d2),
        "diag4_active_inactive":   _strip(d4),
        "diag5_temporal_decay":    _strip(d5),
        "diag6_nn_accuracy":       d6,
        "diag7_hard_negatives":    _strip(d7),
        "decision":                decision,
    }

    json_path = od / "object_matched_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # ── Print summary ──────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("DECISION SUMMARY")
    print("=" * 55)
    print(json.dumps(decision, indent=2))
    print()
    print(f"Diag 1 gap:    easy={d1_gap_final.get('easy')}  "
          f"medium={d1_gap_final.get('medium')}  "
          f"hard={d1_gap_final.get('hard')}")
    print(f"Diag 2 PCA:    n_90={d2['n_components_90']}  "
          f"n_99={d2['n_components_99']}")
    if d5.get(1):
        print(f"Diag 5 gap=1:  mean={d5[1]['mean']:.4f}  std={d5[1]['std']:.4f}")
    print(f"Diag 6 NN:     accuracy={d6['nn_accuracy']:.4f}  "
          f"chance={d6['chance_level']:.4f}")
    if hn:
        print(f"Diag 7:        hard_neg={hn:.4f}  easy_neg={en:.4f}")
    print()
    print(f"JSON   → {json_path}")
    print(f"Plots  → {od}/")
    print("=" * 55)


if __name__ == "__main__":
    main()