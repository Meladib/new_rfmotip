"""
diag_rfdetr_temporal_stability.py
==================================
Comprehensive diagnostic of fine-tuned RF-DETR Small embedding quality
on DanceTrack val set.

Scientific purpose
------------------
Proves (or disproves) that the frozen fine-tuned RF-DETR provides a viable
feature space for identity association. Four independent diagnostics:

  A — Slot-level temporal stability
      Cosine similarity of ALL 300 query slots between consecutive frames.
      Expected to be low (~0.53) due to background slot dilution.
      This is the MISLEADING metric — included to explain why it is misleading.

  B — Object-matched temporal stability
      Cosine similarity of the SAME OBJECT's embedding across frames.
      Computed at gap = 1, 5, 10, 20 frames.
      Expected to be high (~0.96 at gap=1) — proves embeddings are stable.

  C — Background slot ratio
      Fraction of 300 query slots that match a GT box (IoU > 0.5) per frame.
      Explains the dilution: if only X/300 slots are object-matched,
      slot-level stability is dominated by background slot variance.

  D — Inter-identity discriminability
      Intra-object similarity (same ID, different frames) vs
      inter-object similarity (different IDs, same frame).
      Proves embeddings are discriminative, not just stable.

  E — Embedding norm stability
      L2 norm of object-matched embeddings across frames for same object.
      Checks that embedding magnitude is consistent (not drifting).

Paper claim being tested
------------------------
"The frozen fine-tuned RF-DETR provides temporally stable and
identity-discriminative embeddings. The apparent low slot-level stability
is a dilution artifact from background-assigned slots, not a re-ID failure."

Outputs
-------
  Console : per-diagnostic summary table
  File    : diagnostics/rfdetr_temporal_stability.json

Run from repo root:
  python diagnostics/diag_rfdetr_temporal_stability.py
"""

import os
import sys
import json
import torch
import numpy as np
import torchvision.transforms.functional as TF
from pathlib import Path
from PIL import Image
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CKPT_PATH   = "rfdetr_dancetrack_motip/checkpoint_best_total.pth"
DATA_ROOT   = "/data/pos+mot/Datadir/DanceTrack"
SPLIT       = "val"
OUTPUT_PATH = "diagnostics/rfdetr_temporal_stability.json"

IOU_MATCH_THRESH  = 0.5    # minimum IoU to accept a GT-to-slot match
GAP_VALUES        = [1, 5, 10, 20]   # frame gaps for stability curve
MAX_FRAMES        = None   # set to int to limit frames per sequence (None = all)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Model loading — RF-DETR only, no MOTIP wrapper
# ---------------------------------------------------------------------------
def build_rfdetr_only(ckpt_path: str, device: torch.device):
    """
    Load RF-DETR standalone. Returns (model, resolution).
    The model forward() returns a dict containing:
      out["pred_logits"]  — (1, 300, num_classes)
      out["pred_boxes"]   — (1, 300, 4)  cx,cy,w,h normalised
      out["outputs"]      — (1, 300, 256) hs[-1] embeddings  ← our target
    """
    from models.rfdetr.models.lwdetr import build_model

    ckpt      = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args_ckpt = ckpt["args"]

    print(f"  Checkpoint: {ckpt_path}")
    print(f"  num_classes={args_ckpt.num_classes}, "
          f"resolution={args_ckpt.resolution}, "
          f"num_queries={args_ckpt.num_queries}, "
          f"group_detr={getattr(args_ckpt, 'group_detr', 1)}, "
          f"dec_layers={args_ckpt.dec_layers}")

    args_ckpt.num_classes -= 1
    model = build_model(args=args_ckpt)
    args_ckpt.num_classes += 1

    # Load weights — strip DDP prefix, skip shape-mismatched keys
    raw_state  = ckpt["model"]
    own_state  = model.state_dict()
    state      = {}
    for k, v in raw_state.items():
        k_clean = k.replace("module.", "").replace("detr.", "")
        if k_clean in own_state and v.shape == own_state[k_clean].shape:
            state[k_clean] = v
    missing, _ = model.load_state_dict(state, strict=False)
    non_head   = [k for k in missing if "class_embed" not in k and "bbox_embed" not in k]
    if non_head:
        print(f"  WARNING — unexpected missing keys: {non_head[:5]}")

    model.eval().to(device)
    return model, int(args_ckpt.resolution)


# ---------------------------------------------------------------------------
# GT loader
# ---------------------------------------------------------------------------
def load_gt(gt_path: str) -> dict:
    """
    Returns {frame_id: {obj_id: [x,y,w,h]}}
    Skips conf=0 (ignore regions) and zero-area boxes.
    """
    gt = defaultdict(dict)
    with open(gt_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            frame_id = int(parts[0])
            obj_id   = int(parts[1])
            conf     = float(parts[6]) if len(parts) > 6 else 1.0
            if conf == 0:
                continue
            x, y, w, h = float(parts[2]), float(parts[3]), \
                          float(parts[4]), float(parts[5])
            if w <= 0 or h <= 0:
                continue
            gt[frame_id][obj_id] = [x, y, w, h]
    return gt


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess(img_path: str, resolution: int, device: torch.device):
    img    = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size
    t = TF.to_tensor(img)
    t = TF.normalize(t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    t = TF.resize(t, [resolution, resolution])
    return t.unsqueeze(0).to(device), orig_w, orig_h


# ---------------------------------------------------------------------------
# IoU helpers
# ---------------------------------------------------------------------------
def box_cxcywh_to_xyxy_norm(boxes_norm: np.ndarray,
                              orig_w: int, orig_h: int) -> np.ndarray:
    """Convert (cx,cy,w,h) normalised → (x1,y1,x2,y2) absolute."""
    cx, cy, w, h = boxes_norm[:, 0], boxes_norm[:, 1], \
                    boxes_norm[:, 2], boxes_norm[:, 3]
    x1 = (cx - w / 2) * orig_w
    y1 = (cy - h / 2) * orig_h
    x2 = (cx + w / 2) * orig_w
    y2 = (cy + h / 2) * orig_h
    return np.stack([x1, y1, x2, y2], axis=1)


def iou_matrix(gt_boxes_xyxy: np.ndarray,
               pred_boxes_xyxy: np.ndarray) -> np.ndarray:
    """
    Compute IoU between GT boxes (N,4) and predicted boxes (M,4).
    Returns (N, M) IoU matrix.
    """
    N, M = len(gt_boxes_xyxy), len(pred_boxes_xyxy)
    if N == 0 or M == 0:
        return np.zeros((N, M))

    gt   = gt_boxes_xyxy[:, None, :]   # (N, 1, 4)
    pred = pred_boxes_xyxy[None, :, :] # (1, M, 4)

    inter_x1 = np.maximum(gt[..., 0], pred[..., 0])
    inter_y1 = np.maximum(gt[..., 1], pred[..., 1])
    inter_x2 = np.minimum(gt[..., 2], pred[..., 2])
    inter_y2 = np.minimum(gt[..., 3], pred[..., 3])

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter   = inter_w * inter_h

    area_gt   = (gt_boxes_xyxy[:, 2] - gt_boxes_xyxy[:, 0]) * \
                (gt_boxes_xyxy[:, 3] - gt_boxes_xyxy[:, 1])
    area_pred = (pred_boxes_xyxy[:, 2] - pred_boxes_xyxy[:, 0]) * \
                (pred_boxes_xyxy[:, 3] - pred_boxes_xyxy[:, 1])

    union = area_gt[:, None] + area_pred[None, :] - inter
    union = np.maximum(union, 1e-6)
    return inter / union


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Per-sequence extraction
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_sequence_embeddings(seq_dir: Path, model, resolution: int,
                                  device: torch.device):
    """
    For one sequence, extract object-matched embeddings for every GT object
    at every frame where a valid IoU match exists.

    Returns
    -------
    frame_embeddings : dict  {frame_id: {obj_id: np.array(256,)}}
    frame_raw_slots  : dict  {frame_id: np.array(300, 256)}  — ALL slots
    frame_obj_counts : dict  {frame_id: int}  matched objects / 300
    """
    gt_path = seq_dir / "gt" / "gt.txt"
    img_dir = seq_dir / "img1"

    if not gt_path.exists():
        return None, None, None

    gt_by_frame = load_gt(str(gt_path))
    img_files   = sorted(list(img_dir.glob("*.jpg")) +
                          list(img_dir.glob("*.png")))
    if MAX_FRAMES is not None:
        img_files = img_files[:MAX_FRAMES]

    frame_embeddings = {}
    frame_raw_slots  = {}
    frame_obj_counts = {}

    for img_path in img_files:
        frame_id = int(img_path.stem)   # DanceTrack: filename = zero-padded frame id
        gt_objs  = gt_by_frame.get(frame_id, {})

        tensor, orig_w, orig_h = preprocess(str(img_path), resolution, device)
        out = model(tensor)

        # hs[-1] embeddings — shape (1, 300, 256)
        if "outputs" not in out:
            raise KeyError(
                "'outputs' key not found in model output. "
                "Ensure lwdetr.py has: out['outputs'] = hs[-1]"
            )
        embeddings = out["outputs"][0].cpu().numpy()   # (300, 256)
        pred_boxes_norm = out["pred_boxes"][0].cpu().numpy()  # (300, 4) cx,cy,w,h

        frame_raw_slots[frame_id] = embeddings

        if not gt_objs:
            frame_obj_counts[frame_id] = 0
            continue

        # Convert predicted boxes to absolute xyxy
        pred_xyxy = box_cxcywh_to_xyxy_norm(pred_boxes_norm, orig_w, orig_h)

        # Build GT boxes array and obj_id list
        obj_ids   = list(gt_objs.keys())
        gt_boxes  = np.array([[b[0], b[1], b[0]+b[2], b[1]+b[3]]
                               for b in gt_objs.values()])  # xyxy absolute

        iou_mat   = iou_matrix(gt_boxes, pred_xyxy)  # (N_gt, 300)
        best_slot = np.argmax(iou_mat, axis=1)       # (N_gt,)
        best_iou  = iou_mat[np.arange(len(obj_ids)), best_slot]

        matched = {}
        for i, obj_id in enumerate(obj_ids):
            if best_iou[i] >= IOU_MATCH_THRESH:
                matched[obj_id] = embeddings[best_slot[i]]  # (256,)

        frame_embeddings[frame_id] = matched
        frame_obj_counts[frame_id] = len(matched)

    return frame_embeddings, frame_raw_slots, frame_obj_counts


# ---------------------------------------------------------------------------
# Diagnostic computations
# ---------------------------------------------------------------------------
def diag_A_slot_level(frame_raw_slots: dict, max_pairs: int = 500) -> dict:
    """
    DIAG A: Slot-level temporal stability (gap=1).
    Cosine similarity of ALL 300 slots between consecutive frames.
    """
    frame_ids = sorted(frame_raw_slots.keys())
    sims = []
    for i in range(min(len(frame_ids) - 1, max_pairs)):
        f1, f2 = frame_ids[i], frame_ids[i + 1]
        if f2 != f1 + 1:
            continue
        slots1 = frame_raw_slots[f1]   # (300, 256)
        slots2 = frame_raw_slots[f2]
        # Per-slot cosine similarity, then mean over all 300 slots
        norms1 = np.linalg.norm(slots1, axis=1, keepdims=True) + 1e-8
        norms2 = np.linalg.norm(slots2, axis=1, keepdims=True) + 1e-8
        cos    = np.sum((slots1 / norms1) * (slots2 / norms2), axis=1)  # (300,)
        sims.append(float(np.mean(cos)))

    return {
        "mean":  float(np.mean(sims)) if sims else 0.0,
        "std":   float(np.std(sims))  if sims else 0.0,
        "n_pairs": len(sims),
    }


def diag_B_object_matched(frame_embeddings: dict,
                            gap_values: list) -> dict:
    """
    DIAG B: Object-matched temporal stability at multiple gaps.
    For each (object_id, frame_t), find frame_(t+gap) and compute cosine sim.
    """
    # Build per-object frame→embedding lookup
    obj_to_frames = defaultdict(dict)
    for frame_id, objs in frame_embeddings.items():
        for obj_id, emb in objs.items():
            obj_to_frames[obj_id][frame_id] = emb

    results = {}
    for gap in gap_values:
        sims = []
        for obj_id, frames in obj_to_frames.items():
            frame_ids = sorted(frames.keys())
            for f in frame_ids:
                f2 = f + gap
                if f2 in frames:
                    sim = cosine_sim(frames[f], frames[f2])
                    sims.append(sim)
        results[f"gap_{gap}"] = {
            "mean": float(np.mean(sims)) if sims else 0.0,
            "std":  float(np.std(sims))  if sims else 0.0,
            "n_pairs": len(sims),
        }
    return results


def diag_C_background_ratio(frame_obj_counts: dict) -> dict:
    """
    DIAG C: What fraction of 300 slots are object-matched per frame?
    Explains why slot-level stability (DIAG A) is low.
    """
    counts = list(frame_obj_counts.values())
    matched_fracs = [c / 300.0 for c in counts]
    return {
        "mean_matched_slots": float(np.mean(counts)),
        "std_matched_slots":  float(np.std(counts)),
        "mean_background_fraction": float(1 - np.mean(matched_fracs)),
        "mean_object_fraction":     float(np.mean(matched_fracs)),
        "max_matched_in_frame":     int(max(counts)) if counts else 0,
    }


def diag_D_discriminability(frame_embeddings: dict,
                              max_frames: int = 200) -> dict:
    """
    DIAG D: Intra-object vs inter-object cosine similarity.
    In each frame with ≥2 objects:
      - intra: same object, consecutive frame  (gap=1)
      - inter: different objects, same frame
    """
    frame_ids = sorted(frame_embeddings.keys())

    intra_sims = []   # same object, consecutive frames
    inter_sims = []   # different objects, same frame

    for i, fid in enumerate(frame_ids[:max_frames]):
        objs = frame_embeddings.get(fid, {})
        if not objs:
            continue

        obj_ids = list(objs.keys())
        embs    = list(objs.values())

        # Inter: all pairs in this frame
        for j in range(len(embs)):
            for k in range(j + 1, len(embs)):
                inter_sims.append(cosine_sim(embs[j], embs[k]))

        # Intra: same object in next frame
        next_fid = fid + 1
        next_objs = frame_embeddings.get(next_fid, {})
        for obj_id in obj_ids:
            if obj_id in next_objs:
                intra_sims.append(cosine_sim(objs[obj_id], next_objs[obj_id]))

    return {
        "intra_mean": float(np.mean(intra_sims)) if intra_sims else 0.0,
        "intra_std":  float(np.std(intra_sims))  if intra_sims else 0.0,
        "inter_mean": float(np.mean(inter_sims)) if inter_sims else 0.0,
        "inter_std":  float(np.std(inter_sims))  if inter_sims else 0.0,
        "separability_gap": float(np.mean(intra_sims) - np.mean(inter_sims))
                             if intra_sims and inter_sims else 0.0,
        "n_intra": len(intra_sims),
        "n_inter": len(inter_sims),
    }


def diag_E_norm_stability(frame_embeddings: dict) -> dict:
    """
    DIAG E: L2 norm of object-matched embeddings.
    Checks that embedding magnitude is consistent across frames
    for the same object (not drifting or collapsing).
    """
    obj_to_frames = defaultdict(dict)
    for frame_id, objs in frame_embeddings.items():
        for obj_id, emb in objs.items():
            obj_to_frames[obj_id][frame_id] = emb

    all_norms       = []
    per_obj_cv      = []   # coefficient of variation per object

    for obj_id, frames in obj_to_frames.items():
        norms = [float(np.linalg.norm(e)) for e in frames.values()]
        all_norms.extend(norms)
        if len(norms) > 1 and np.mean(norms) > 1e-8:
            cv = float(np.std(norms) / np.mean(norms))
            per_obj_cv.append(cv)

    return {
        "global_norm_mean": float(np.mean(all_norms)) if all_norms else 0.0,
        "global_norm_std":  float(np.std(all_norms))  if all_norms else 0.0,
        "per_object_cv_mean": float(np.mean(per_obj_cv)) if per_obj_cv else 0.0,
        "per_object_cv_std":  float(np.std(per_obj_cv))  if per_obj_cv else 0.0,
        "interpretation": (
            "CV close to 0 means embedding magnitude is stable per object. "
            "High CV (>0.2) would indicate magnitude drift."
        ),
    }


# ---------------------------------------------------------------------------
# Aggregation across sequences
# ---------------------------------------------------------------------------
def aggregate(seq_results: list, key: str) -> dict:
    vals = [r[key] for r in seq_results if key in r and r[key] is not None]
    if not vals:
        return {}
    if isinstance(vals[0], dict):
        # Aggregate nested dicts (e.g. gap results)
        all_keys = vals[0].keys()
        out = {}
        for k in all_keys:
            nums = [v[k] for v in vals if isinstance(v[k], float)]
            out[k] = {"mean": float(np.mean(nums)), "std": float(np.std(nums))} \
                      if nums else {}
        return out
    nums = [v for v in vals if isinstance(v, float)]
    return {"mean": float(np.mean(nums)), "std": float(np.std(nums))}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    print("Loading RF-DETR...")
    model, resolution = build_rfdetr_only(CKPT_PATH, device)
    print(f"  Resolution: {resolution}×{resolution}\n")

    val_dir   = Path(DATA_ROOT) / SPLIT
    sequences = sorted([d for d in val_dir.iterdir() if d.is_dir()])
    print(f"Found {len(sequences)} sequences\n")

    seq_results = []

    for seq_idx, seq_dir in enumerate(sequences):
        seq_name = seq_dir.name
        print(f"[{seq_idx+1}/{len(sequences)}] {seq_name} ...", end=" ", flush=True)

        frame_embeddings, frame_raw_slots, frame_obj_counts = \
            extract_sequence_embeddings(seq_dir, model, resolution, device)

        if frame_embeddings is None:
            print("SKIPPED (no GT)")
            continue

        result = {"sequence": seq_name}

        result["diag_A"] = diag_A_slot_level(frame_raw_slots)
        result["diag_B"] = diag_B_object_matched(frame_embeddings, GAP_VALUES)
        result["diag_C"] = diag_C_background_ratio(frame_obj_counts)
        result["diag_D"] = diag_D_discriminability(frame_embeddings)
        result["diag_E"] = diag_E_norm_stability(frame_embeddings)

        seq_results.append(result)
        print(f"A={result['diag_A']['mean']:.3f}  "
              f"B_gap1={result['diag_B']['gap_1']['mean']:.3f}  "
              f"obj_slots={result['diag_C']['mean_matched_slots']:.1f}/300  "
              f"sep={result['diag_D']['separability_gap']:.3f}")

    # Global summary
    print("\n" + "=" * 70)
    print("GLOBAL SUMMARY — all val sequences")
    print("=" * 70)

    def flat_mean(key, subkey=None):
        vals = []
        for r in seq_results:
            v = r.get(key)
            if subkey and v:
                v = v.get(subkey)
            if isinstance(v, dict):
                v = v.get("mean")
            if v is not None:
                vals.append(v)
        return float(np.mean(vals)) if vals else 0.0

    a_mean   = flat_mean("diag_A", "mean")
    b_gap1   = flat_mean("diag_B", "gap_1")    # nested — handled below
    b_gap5   = flat_mean("diag_B", "gap_5")
    b_gap10  = flat_mean("diag_B", "gap_10")
    b_gap20  = flat_mean("diag_B", "gap_20")
    obj_frac = flat_mean("diag_C", "mean_object_fraction")
    intra    = flat_mean("diag_D", "intra_mean")
    inter    = flat_mean("diag_D", "inter_mean")
    sep      = flat_mean("diag_D", "separability_gap")
    norm_cv  = flat_mean("diag_E", "per_object_cv_mean")

    # Fix nested diag_B aggregation
    b_vals = {"gap_1": [], "gap_5": [], "gap_10": [], "gap_20": []}
    for r in seq_results:
        for gk in b_vals:
            v = r.get("diag_B", {}).get(gk, {}).get("mean")
            if v is not None:
                b_vals[gk].append(v)
    b_gap1  = float(np.mean(b_vals["gap_1"]))  if b_vals["gap_1"]  else 0.0
    b_gap5  = float(np.mean(b_vals["gap_5"]))  if b_vals["gap_5"]  else 0.0
    b_gap10 = float(np.mean(b_vals["gap_10"])) if b_vals["gap_10"] else 0.0
    b_gap20 = float(np.mean(b_vals["gap_20"])) if b_vals["gap_20"] else 0.0

    intra_vals = [r["diag_D"]["intra_mean"] for r in seq_results]
    inter_vals = [r["diag_D"]["inter_mean"] for r in seq_results]
    sep_vals   = [r["diag_D"]["separability_gap"] for r in seq_results]
    intra = float(np.mean(intra_vals))
    inter = float(np.mean(inter_vals))
    sep   = float(np.mean(sep_vals))

    print(f"\n  DIAG A — Slot-level stability (gap=1, all 300 slots)")
    print(f"    Mean cosine sim : {a_mean:.4f}")
    print(f"    Interpretation  : Low — dominated by {(1-obj_frac)*100:.1f}% background slots")

    print(f"\n  DIAG B — Object-matched stability (same object, N frames apart)")
    print(f"    Gap  1 : {b_gap1:.4f}")
    print(f"    Gap  5 : {b_gap5:.4f}")
    print(f"    Gap 10 : {b_gap10:.4f}")
    print(f"    Gap 20 : {b_gap20:.4f}")

    print(f"\n  DIAG C — Background slot ratio")
    bg_frac_vals = [r["diag_C"]["mean_background_fraction"] for r in seq_results]
    obj_frac_vals = [r["diag_C"]["mean_object_fraction"] for r in seq_results]
    print(f"    Object  slots per frame : {flat_mean('diag_C','mean_matched_slots'):.1f} / 300")
    print(f"    Object  fraction        : {float(np.mean(obj_frac_vals))*100:.1f}%")
    print(f"    Background fraction     : {float(np.mean(bg_frac_vals))*100:.1f}%")
    print(f"    → This explains the gap between DIAG A and DIAG B")

    print(f"\n  DIAG D — Inter-identity discriminability")
    print(f"    Intra-object sim (same ID, gap=1) : {intra:.4f}")
    print(f"    Inter-object sim (diff ID, same f): {inter:.4f}")
    print(f"    Separability gap                  : {sep:.4f}")
    print(f"    Interpretation : gap > 0 → embeddings discriminate identities")

    print(f"\n  DIAG E — Embedding norm stability")
    cv_vals = [r["diag_E"]["per_object_cv_mean"] for r in seq_results]
    print(f"    Per-object norm CV (mean) : {float(np.mean(cv_vals)):.4f}")
    print(f"    Interpretation : CV < 0.1 → stable norms (not drifting)")

    # Save
    global_summary = {
        "diag_A_slot_level_sim":         round(a_mean, 4),
        "diag_B_object_matched": {
            "gap_1":  round(b_gap1,  4),
            "gap_5":  round(b_gap5,  4),
            "gap_10": round(b_gap10, 4),
            "gap_20": round(b_gap20, 4),
        },
        "diag_C_object_slots_per_frame": round(flat_mean("diag_C","mean_matched_slots"), 2),
        "diag_C_background_fraction":    round(float(np.mean(bg_frac_vals)), 4),
        "diag_D_intra_sim":              round(intra, 4),
        "diag_D_inter_sim":              round(inter, 4),
        "diag_D_separability_gap":       round(sep, 4),
        "diag_E_norm_cv_mean":           round(float(np.mean(cv_vals)), 4),
    }

    output = {
        "checkpoint":    CKPT_PATH,
        "split":         SPLIT,
        "iou_threshold": IOU_MATCH_THRESH,
        "global_summary": global_summary,
        "per_sequence":   seq_results,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {OUTPUT_PATH}")
    print("\n" + "=" * 70)
    print("PAPER CLAIM CHECK")
    print("=" * 70)
    print(f"  Slot-level sim {a_mean:.3f}  ← expected ~0.53  "
          f"{'✓' if a_mean < 0.7 else '?'}")
    print(f"  Object-matched gap=1 {b_gap1:.3f}  ← expected ~0.97  "
          f"{'✓' if b_gap1 > 0.90 else '✗ LOWER THAN EXPECTED'}")
    print(f"  Separability gap {sep:.3f}  ← expected > 0  "
          f"{'✓' if sep > 0 else '✗ NOT DISCRIMINATIVE'}")
    print(f"  Norm CV {float(np.mean(cv_vals)):.3f}  ← expected < 0.10  "
          f"{'✓' if float(np.mean(cv_vals)) < 0.10 else '✗ NORM DRIFTING'}")
    print()
    if b_gap1 > 0.90 and sep > 0:
        print("  CONCLUSION: Frozen detector embeddings are viable for association.")
        print("  The detector is NOT the bottleneck. Proceed to association diagnostics.")
    else:
        print("  CONCLUSION: Detector embedding quality is INSUFFICIENT.")
        print("  Tracking gap may originate at the detector level. Investigate further.")


if __name__ == "__main__":
    main()