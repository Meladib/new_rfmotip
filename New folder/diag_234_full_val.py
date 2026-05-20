#!/usr/bin/env python3
"""
diag_234_full_val.py
=====================
Combined diagnostic: DIAG 2, 3, 4 across all DanceTrack val sequences.

DIAG 2 — IDDecoder temporal attention weight distribution (CV per layer)
DIAG 3 — Newborn inflation rate and crowd correlation
DIAG 4 — ID score distribution: correct vs Case A / B / C

Optimisations applied
---------------------
- Attention accumulation fully vectorised via numpy broadcasting (no Python loops)
- id_decoder hook registered once per sequence, not per frame
- DIAG 4 analysis done inline per frame — no storing full id_scores tensors
- DIAG 4 IoU matching vectorised per frame via numpy
- All heavy imports hoisted to module level

Run from repo root:
  python diagnostics/diag_234_full_val.py \
    --config  configs/rf_detr_motip_dancetrack.yaml \
    --checkpoint outputsV2/rfmotip_dancetrack/train/checkpoint_2.pth \
    --val_dir /data/pos+mot/Datadir/DanceTrack/val \
    --output_dir diagnostics/diag234_results/
"""

import sys
import os
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────
# ARGS
# ─────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",      required=True)
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--val_dir",     required=True)
    p.add_argument("--output_dir",  default="diagnostics/diag234_results/")
    p.add_argument("--device",      default=None)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# GT / SEQUENCE LOADING
# ─────────────────────────────────────────────────────────────
def load_sequence(seq_dir):
    from configparser import ConfigParser
    seq = Path(seq_dir)
    ini = ConfigParser()
    ini.read(seq / "seqinfo.ini")
    img_w   = int(ini["Sequence"]["imWidth"])
    img_h   = int(ini["Sequence"]["imHeight"])
    seq_len = int(ini["Sequence"]["seqLength"])

    image_paths = [str(seq / "img1" / f"{i+1:08d}.jpg") for i in range(seq_len)]

    fd = defaultdict(lambda: {"ids": [], "bboxes": []})
    with open(seq / "gt" / "gt.txt") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            conf = float(parts[6]) if len(parts) > 6 else 1.0
            if conf == 0:
                continue
            fd[int(parts[0])]["ids"].append(int(parts[1]))
            fd[int(parts[0])]["bboxes"].append(
                [float(parts[2]), float(parts[3]),
                 float(parts[4]), float(parts[5])])

    annotations, gt_dict = [], {}
    for i in range(seq_len):
        fid = i + 1
        ids = fd[fid]["ids"]
        bbs = fd[fid]["bboxes"]
        annotations.append({
            "id":      torch.tensor(ids,  dtype=torch.int64)   if ids
                       else torch.zeros(0, dtype=torch.int64),
            "bbox":    torch.tensor(bbs,  dtype=torch.float32) if bbs
                       else torch.zeros((0, 4), dtype=torch.float32),
            "is_legal": True,
        })
        gt_dict[fid] = {oid: bb for oid, bb in zip(ids, bbs)}

    return image_paths, annotations, gt_dict, img_w, img_h, seq_len


def compute_gt_newborns(annotations):
    seen, result = set(), []
    for ann in annotations:
        ids = ann["id"].tolist()
        result.append(set(ids) - seen)
        seen.update(ids)
    return result


# ─────────────────────────────────────────────────────────────
# CROWD DENSITY  (vectorised)
# ─────────────────────────────────────────────────────────────
def crowd_density(bboxes_xywh_tensor, img_w, img_h):
    n = len(bboxes_xywh_tensor)
    if n < 2:
        return 0.0
    x, y, w, h = bboxes_xywh_tensor.unbind(-1)
    x1 = (x / img_w).unsqueeze(1)
    y1 = (y / img_h).unsqueeze(1)
    x2 = ((x + w) / img_w).unsqueeze(1)
    y2 = ((y + h) / img_h).unsqueeze(1)
    ix1 = torch.max(x1, x1.T);  iy1 = torch.max(y1, y1.T)
    ix2 = torch.min(x2, x2.T);  iy2 = torch.min(y2, y2.T)
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    area  = (x2 - x1).squeeze(1) * (y2 - y1).squeeze(1)
    union = area.unsqueeze(1) + area.unsqueeze(0) - inter
    iou   = inter / (union + 1e-6)
    iou.fill_diagonal_(0.0)
    return float(iou.sum() / (n * (n - 1)))


# ─────────────────────────────────────────────────────────────
# IOU MATRIX  (vectorised numpy)
# ─────────────────────────────────────────────────────────────
def iou_matrix_np(pred_xywh, gt_xywh):
    """pred_xywh: (M,4)  gt_xywh: (N,4)  pixel x,y,w,h  -> (M,N) IoU"""
    if len(pred_xywh) == 0 or len(gt_xywh) == 0:
        return np.zeros((len(pred_xywh), len(gt_xywh)))
    px1 = pred_xywh[:, 0];              py1 = pred_xywh[:, 1]
    px2 = pred_xywh[:, 0] + pred_xywh[:, 2]
    py2 = pred_xywh[:, 1] + pred_xywh[:, 3]
    gx1 = gt_xywh[:, 0];               gy1 = gt_xywh[:, 1]
    gx2 = gt_xywh[:, 0] + gt_xywh[:, 2]
    gy2 = gt_xywh[:, 1] + gt_xywh[:, 3]
    ix1 = np.maximum(px1[:, None], gx1[None, :])
    iy1 = np.maximum(py1[:, None], gy1[None, :])
    ix2 = np.minimum(px2[:, None], gx2[None, :])
    iy2 = np.minimum(py2[:, None], gy2[None, :])
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    ap    = pred_xywh[:, 2] * pred_xywh[:, 3]
    ag    = gt_xywh[:, 2]   * gt_xywh[:, 3]
    union = ap[:, None] + ag[None, :] - inter
    return inter / np.maximum(union, 1e-6)


# ─────────────────────────────────────────────────────────────
# DIAG 2 — ATTENTION HOOK  (installed once on model)
# ─────────────────────────────────────────────────────────────
def install_attention_hooks(model):
    """
    DIAG 2: accumulate attention weight by KEY POSITION (= trajectory slot index).
    Position 0 = oldest trajectory slot, position T-1 = most recent.
    CV of mean-weight-by-position tells whether attention is flat or recency-biased.
    No dependency on trajectory_times values — avoids all dtype/format issues.
    """
    from models.misc import get_model
    inner      = get_model(model)
    id_decoder = inner.id_decoder
    num_layers = id_decoder.num_layers

    container = {
        "num_layers":     num_layers,
        "weights_by_pos": {li: defaultdict(list) for li in range(num_layers)},
        "traj_id_labels": None,
    }

    # Capture traj_id_labels for DIAG 4
    orig_fwd = id_decoder.forward
    def patched_fwd(seq_info, use_decoder_checkpoint=False):
        if "trajectory_id_labels" in seq_info:
            container["traj_id_labels"] = seq_info["trajectory_id_labels"].detach().clone()
        return orig_fwd(seq_info, use_decoder_checkpoint=False)
    id_decoder.forward = patched_fwd

    # Cross-attention hook: accumulate by key position
    for li in range(num_layers):
        orig_ca = id_decoder.cross_attn_layers[li].forward

        def make_ca_patch(layer_idx, ca_fwd):
            def patched_ca(query, key, value,
                           key_padding_mask=None, attn_mask=None,
                           need_weights=False, **kw):
                out, attn_w = ca_fwd(
                    query, key, value,
                    key_padding_mask=key_padding_mask,
                    attn_mask=attn_mask,
                    need_weights=True,
                    average_attn_weights=True,
                    **kw,
                )
                # attn_w: (BG, n_queries, n_keys) averaged over heads
                if attn_w is not None:
                    try:
                        aw = attn_w.detach().cpu().numpy()  # (BG, Q, K)
                        n_keys = aw.shape[-1]
                        # Mean over batch and query dims → weight per key position
                        mean_per_pos = aw.mean(axis=(0, 1))  # (K,)
                        for pos in range(n_keys):
                            container["weights_by_pos"][layer_idx][pos].append(
                                float(mean_per_pos[pos]))
                    except Exception:
                        pass
                return out, attn_w
            return patched_ca

        id_decoder.cross_attn_layers[li].forward = make_ca_patch(li, orig_ca)

    return container


def reset_container(container):
    nl = container["num_layers"]
    container["weights_by_pos"] = {li: defaultdict(list) for li in range(nl)}
    container["traj_id_labels"] = None


# ─────────────────────────────────────────────────────────────
# PER-SEQUENCE RUN
# ─────────────────────────────────────────────────────────────
def run_sequence(seq_dir, model, config, attn_container, device):
    import torchvision.transforms.functional as TF
    from PIL import Image
    from models.runtime_tracker import RuntimeTracker
    from models.misc import get_model
    from utils.nested_tensor import nested_tensor_from_tensor_list

    seq_name = Path(seq_dir).name
    image_paths, annotations, gt_dict, img_w, img_h, seq_len = load_sequence(seq_dir)
    gt_newborns_per_frame = compute_gt_newborns(annotations)

    id_thresh      = config.get("ID_THRESH",           0.2)
    det_thresh     = config.get("DET_THRESH",           0.3)
    newborn_thresh = config.get("NEWBORN_THRESH",       0.6)
    use_sigmoid    = config.get("USE_FOCAL_LOSS",       False)
    protocol       = config.get("ASSIGNMENT_PROTOCOL",  "object-max")
    miss_tol       = config.get("MISS_TOLERANCE",       30)
    area_thresh    = config.get("AREA_THRESH",          0)
    SIZE_DIV       = config.get("SIZE_DIVISIBILITY",    32)
    MAX_LONGER     = config.get("INFERENCE_MAX_LONGER", 1440)
    MEANS          = [0.485, 0.456, 0.406]
    STDS           = [0.229, 0.224, 0.225]

    rt = RuntimeTracker(
        model=model, sequence_hw=(img_h, img_w),
        use_sigmoid=use_sigmoid, assignment_protocol=protocol,
        miss_tolerance=miss_tol, det_thresh=det_thresh,
        newborn_thresh=newborn_thresh, id_thresh=id_thresh,
        area_thresh=area_thresh, only_detr=False, dtype=torch.float32,
    )
    num_id_vocab = rt.num_id_vocabulary

    reset_container(attn_container)

    # ── DIAG 4: register hook ONCE per sequence ───────────────────────
    _id_dec       = get_model(model).id_decoder
    _logits_store = {"logits": None}

    def _logit_hook(module, inp, output):
        val = output[0] if isinstance(output, (tuple, list)) else output
        _logits_store["logits"] = val.detach()

    _hook_handle = _id_dec.register_forward_hook(_logit_hook)

    # Shared dict written by patched_get_id, read in the main loop
    d4_frame = {}

    orig_get_id = rt._get_id_pred_labels

    def patched_get_id(boxes, output_embeds):
        _logits_store["logits"] = None
        result = orig_get_id(boxes, output_embeds)

        raw = _logits_store["logits"]
        if raw is not None:
            lgt = raw.reshape(-1, raw.shape[-1])
            scores = (lgt.softmax(dim=-1) if not use_sigmoid
                      else lgt.sigmoid()).cpu().numpy()
        else:
            scores = None

        tl = attn_container["traj_id_labels"]
        traj_labels = tl.flatten().cpu().numpy() if tl is not None \
                      else np.array([], dtype=np.int64)

        d4_frame["boxes"]       = boxes.cpu().numpy()
        d4_frame["assignment"]  = (result.cpu().numpy()
                                   if isinstance(result, torch.Tensor)
                                   else np.array(result))
        d4_frame["id_scores"]   = scores
        d4_frame["traj_labels"] = traj_labels
        return result

    rt._get_id_pred_labels = patched_get_id

    # ── Accumulators ─────────────────────────────────────────────────
    d3_stats       = []
    correct_scores = []
    case_a_scores  = []
    case_b_scores  = []
    case_c_count   = 0
    max_scores_all = []
    newborn_max    = []

    # ── Main inference loop ───────────────────────────────────────────
    with torch.no_grad():
        for t in range(seq_len):
            ann      = annotations[t]
            n_gt_new = len(gt_newborns_per_frame[t])
            crowd    = (crowd_density(ann["bbox"], img_w, img_h)
                        if len(ann["bbox"]) >= 2 else 0.0)
            gt_this  = gt_dict.get(t + 1, {})

            # Preprocess
            img   = Image.open(image_paths[t]).convert("RGB")
            img_t = TF.to_tensor(img)
            h0, w0 = img_t.shape[-2:]
            scale = 800.0 / min(h0, w0)
            if max(h0, w0) * scale > MAX_LONGER:
                scale = MAX_LONGER / max(h0, w0)
            img_t = TF.resize(img_t, [int(round(h0*scale)), int(round(w0*scale))])
            img_t = TF.normalize(img_t, MEANS, STDS)
            frame = nested_tensor_from_tensor_list([img_t], SIZE_DIV).to(device)

            # D3: snapshot before
            _tl     = rt.trajectory_id_labels
            pre_ids = set(_tl.flatten().tolist()) if _tl.numel() > 0 else set()

            d4_frame.clear()
            rt.update(image=frame)

            # D3: count new IDs
            _tl2     = rt.trajectory_id_labels
            post_ids = set(_tl2.flatten().tolist()) if _tl2.numel() > 0 else set()
            n_pred_new = len(post_ids - pre_ids)

            d3_stats.append({
                "gt_newborns":   n_gt_new,
                "pred_newborns": n_pred_new,
                "spurious":      max(0, n_pred_new - n_gt_new),
                "crowd_density": crowd,
            })

            # D4: inline analysis
            id_scores   = d4_frame.get("id_scores")
            assignment  = d4_frame.get("assignment")
            pred_boxes  = d4_frame.get("boxes")
            traj_labels = d4_frame.get("traj_labels")

            if (id_scores is not None and assignment is not None
                    and pred_boxes is not None and len(pred_boxes) > 0):

                # Vectorised pred→GT IoU matching
                pred_xywh = np.column_stack([
                    (pred_boxes[:, 0] - pred_boxes[:, 2] / 2) * img_w,
                    (pred_boxes[:, 1] - pred_boxes[:, 3] / 2) * img_h,
                    pred_boxes[:, 2] * img_w,
                    pred_boxes[:, 3] * img_h,
                ])

                if gt_this:
                    gt_arr   = np.array(list(gt_this.values()), dtype=np.float32)
                    iou_mat  = iou_matrix_np(pred_xywh, gt_arr)
                    best_iou = iou_mat.max(axis=1)
                    matched  = best_iou >= 0.5
                else:
                    matched = np.zeros(len(pred_boxes), dtype=bool)

                valid_traj = set(traj_labels.tolist()) - {num_id_vocab} \
                             if traj_labels is not None else set()

                for obj_idx in range(len(assignment)):
                    ms = float(id_scores[obj_idx].max())
                    max_scores_all.append(ms)

                    if not matched[obj_idx]:
                        continue

                    assigned = int(assignment[obj_idx])
                    if assigned != num_id_vocab:
                        correct_scores.append(float(id_scores[obj_idx, assigned]))
                    else:
                        newborn_max.append(ms)
                        if not valid_traj:
                            case_c_count += 1
                        else:
                            best_traj = max(
                                float(id_scores[obj_idx, lbl])
                                for lbl in valid_traj
                                if lbl < id_scores.shape[1]
                            )
                            if best_traj < id_thresh:
                                case_a_scores.append(best_traj)
                            else:
                                case_b_scores.append(best_traj)

    _hook_handle.remove()

    # ── D3 aggregate ─────────────────────────────────────────────────
    gt_arr   = np.array([s["gt_newborns"]   for s in d3_stats])
    pr_arr   = np.array([s["pred_newborns"] for s in d3_stats])
    sp_arr   = np.array([s["spurious"]      for s in d3_stats])
    cr_arr   = np.array([s["crowd_density"] for s in d3_stats])

    mean_gt   = float(gt_arr.mean())
    mean_pred = float(pr_arr.mean())
    inflation = mean_pred / (mean_gt + 1e-6)
    spur_rate = float(sp_arr.sum()) / float(max(pr_arr.sum(), 1))
    corr_d3   = float(np.corrcoef(cr_arr, sp_arr)[0, 1]) if cr_arr.std() > 1e-6 else 0.0

    d3 = {
        "sequence": seq_name, "n_frames": seq_len,
        "mean_gt_newborns": mean_gt, "mean_pred_newborns": mean_pred,
        "newborn_inflation_ratio": inflation,
        "spurious_newborn_rate":   spur_rate,
        "crowd_spurious_corr":     corr_d3,
    }

    # ── D4 aggregate ─────────────────────────────────────────────────
    def _stats(v):
        if not v:
            return {"mean": None, "std": None, "n": 0}
        a = np.array(v)
        return {"mean": float(a.mean()), "std": float(a.std()), "n": len(a)}

    total_nb = len(case_a_scores) + len(case_b_scores) + case_c_count
    d4 = {
        "sequence": seq_name,
        "correct":    _stats(correct_scores),
        "case_A":     _stats(case_a_scores),
        "case_B":     _stats(case_b_scores),
        "case_C_n":   case_c_count,
        "max_all":    _stats(max_scores_all),
        "max_newborn":_stats(newborn_max),
        "breakdown": {
            "total":    total_nb,
            "case_A":   len(case_a_scores),
            "case_B":   len(case_b_scores),
            "case_C":   case_c_count,
            "case_A_pct": round(len(case_a_scores) / max(total_nb, 1) * 100, 1),
            "case_B_pct": round(len(case_b_scores) / max(total_nb, 1) * 100, 1),
            "case_C_pct": round(case_c_count        / max(total_nb, 1) * 100, 1),
        },
    }

    # ── D2 per-sequence CV ────────────────────────────────────────────
    d2 = {"sequence": seq_name}
    cvs = []
    for li in range(attn_container["num_layers"]):
        pos_data = attn_container["weights_by_pos"][li]
        if not pos_data:
            d2[f"layer_{li}_cv"] = None
            continue
        # One mean per key position — list of scalars already averaged in hook
        positions = sorted(pos_data.keys())
        means = np.array([np.mean(pos_data[p]) for p in positions
                          if len(pos_data[p]) > 0])
        if len(means) < 2 or np.isnan(means).any() or means.mean() < 1e-12:
            d2[f"layer_{li}_cv"] = None
            continue
        cv = float(means.std() / means.mean())
        d2[f"layer_{li}_cv"] = cv
        cvs.append(cv)
    d2["mean_cv"] = float(np.mean(cvs)) if cvs else 0.0

    print(f"  D2 cv={d2['mean_cv']:.3f}  "
          f"D3 inf={inflation:.2f}x spur={spur_rate:.1%} corr={corr_d3:.3f}  "
          f"D4 B%={d4['breakdown']['case_B_pct']}%")

    return d2, d3, d4, attn_container


# ─────────────────────────────────────────────────────────────
# AGGREGATION + PLOTS
# ─────────────────────────────────────────────────────────────
def aggregate_and_plot(all_d2, all_d3, all_d4, attn_global, num_layers, od, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    id_thresh = config.get("ID_THRESH", 0.2)
    od = Path(od)

    # ── DIAG 2 ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DIAG 2 — Attention Weight Distribution (global)")
    print("=" * 60)

    fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 5))
    if num_layers == 1:
        axes = [axes]

    d2_global = {}
    global_cvs = []
    for li in range(num_layers):
        pos_data = attn_global["weights_by_pos"][li]
        if not pos_data:
            d2_global[f"layer_{li}"] = {}
            continue
        positions = sorted(pos_data.keys())
        means = np.array([np.mean(pos_data[p]) for p in positions
                          if len(pos_data[p]) > 0])
        positions = [p for p in positions if len(pos_data[p]) > 0]
        if len(means) < 2 or np.isnan(means).any() or means.mean() < 1e-12:
            d2_global[f"layer_{li}"] = {}
            continue
        cv = float(means.std() / means.mean())
        d2_global[f"layer_{li}"] = {"positions": positions, "means": means.tolist(), "cv": cv}
        global_cvs.append(cv)

        ax = axes[li]
        ax.bar(positions, means, alpha=0.7, color="steelblue")
        ax.axhline(1.0 / (len(positions) + 1e-8), color="red", ls="--",
                   alpha=0.6, label="Uniform")
        ax.set_xlabel("Key Position (0=oldest, T-1=newest)")
        ax.set_ylabel("Mean Attn Weight")
        tag = "FLAT" if cv < 0.1 else ("RECENCY" if cv > 0.3 else "MODERATE")
        ax.set_title(f"Layer {li}  CV={cv:.3f}  {tag}")
        ax.legend(fontsize=7)
        print(f"  Layer {li}: CV={cv:.4f}  {tag}")

    mean_cv = float(np.mean(global_cvs)) if global_cvs else 0.0
    d2_global["mean_cv"] = mean_cv
    print(f"  Mean CV: {mean_cv:.4f}")

    plt.suptitle(f"IDDecoder Attention vs Frame Age — {len(all_d2)} Sequences  "
                 f"Mean CV={mean_cv:.3f}")
    plt.tight_layout()
    plt.savefig(od / "diag2_attention_global.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── DIAG 3 ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DIAG 3 — Newborn Inflation (global)")
    print("=" * 60)

    inflations = [r["newborn_inflation_ratio"] for r in all_d3]
    spur_rates = [r["spurious_newborn_rate"]   for r in all_d3]
    corrs      = [r["crowd_spurious_corr"]     for r in all_d3]
    gt_m       = [r["mean_gt_newborns"]        for r in all_d3]
    pr_m       = [r["mean_pred_newborns"]      for r in all_d3]

    global_infl = float(sum(pr_m) / (sum(gt_m) + 1e-6))
    mean_spur   = float(np.mean(spur_rates))
    mean_corr   = float(np.mean(corrs))
    print(f"  Global inflation ratio : {global_infl:.2f}x")
    print(f"  Mean spurious rate     : {mean_spur:.1%}")
    print(f"  Mean crowd-spur corr   : {mean_corr:.3f}")

    seq_names = [r["sequence"] for r in all_d3]
    x = range(len(seq_names))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    ax1.bar(x, inflations,
            color=["red" if v > 1.5 else "steelblue" for v in inflations])
    ax1.axhline(1.0, color="black", ls="--", lw=1.5, label="No inflation")
    ax1.axhline(global_infl, color="red", ls="--", lw=1.5,
                label=f"Global={global_infl:.2f}x")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([s[-4:] for s in seq_names], rotation=45, fontsize=8)
    ax1.set_ylabel("Inflation ratio")
    ax1.set_title("Newborn Inflation per Sequence")
    ax1.legend()

    ax2.bar(x, spur_rates, color="orange")
    ax2.axhline(mean_spur, color="red", ls="--", lw=1.5,
                label=f"Mean={mean_spur:.1%}")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels([s[-4:] for s in seq_names], rotation=45, fontsize=8)
    ax2.set_ylabel("Spurious rate")
    ax2.set_title("Spurious Newborn Rate per Sequence")
    ax2.legend()

    plt.suptitle(f"DIAG 3 — {len(all_d3)} Val Sequences")
    plt.tight_layout()
    plt.savefig(od / "diag3_newborn_global.png", dpi=150, bbox_inches="tight")
    plt.close()

    d3_global = {
        "global_inflation_ratio":   global_infl,
        "mean_spurious_rate":       mean_spur,
        "mean_crowd_spurious_corr": mean_corr,
        "per_sequence":             all_d3,
    }

    # ── DIAG 4 ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DIAG 4 — ID Score Distribution (global)")
    print("=" * 60)

    total_A  = sum(r["breakdown"]["case_A"] for r in all_d4)
    total_B  = sum(r["breakdown"]["case_B"] for r in all_d4)
    total_C  = sum(r["breakdown"]["case_C"] for r in all_d4)
    total_nb = total_A + total_B + total_C

    correct_m = [r["correct"]["mean"] for r in all_d4
                 if r["correct"]["mean"] is not None]
    global_correct = float(np.mean(correct_m)) if correct_m else None

    print(f"  Total spurious newborns : {total_nb}")
    print(f"  Case A (score<thresh)   : {total_A}  "
          f"({total_A/max(total_nb,1)*100:.1f}%)")
    print(f"  Case B (score>=thresh)  : {total_B}  "
          f"({total_B/max(total_nb,1)*100:.1f}%)")
    print(f"  Case C (label absent)   : {total_C}  "
          f"({total_C/max(total_nb,1)*100:.1f}%)")
    if global_correct:
        print(f"  Correct score mean      : {global_correct:.4f}")

    dominant = "B" if total_B > total_A else "A"
    if dominant == "B":
        print("\n  FINDING: Case B dominates -> protocol failure, not score miscalibration.")
    else:
        print("\n  FINDING: Case A dominates -> score miscalibration is the failure mode.")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    caseA_m = [r["case_A"]["mean"] for r in all_d4 if r["case_A"]["mean"] is not None]
    caseB_m = [r["case_B"]["mean"] for r in all_d4 if r["case_B"]["mean"] is not None]
    ax.bar(["Correct", "Case A", "Case B"],
           [np.mean(correct_m) if correct_m else 0,
            np.mean(caseA_m)   if caseA_m   else 0,
            np.mean(caseB_m)   if caseB_m   else 0],
           color=["green", "red", "orange"], alpha=0.8)
    ax.axhline(id_thresh, color="black", ls="--", lw=1.5,
               label=f"id_thresh={id_thresh}")
    ax.set_ylabel("Mean ID score")
    ax.set_title("Mean Score by Outcome")
    ax.legend()

    ax = axes[1]
    bars = ax.bar(["Case A", "Case B", "Case C"],
                  [total_A, total_B, total_C],
                  color=["red", "orange", "grey"], alpha=0.8)
    for bar, cnt, pct in zip(bars,
                              [total_A, total_B, total_C],
                              [total_A/max(total_nb,1)*100,
                               total_B/max(total_nb,1)*100,
                               total_C/max(total_nb,1)*100]):
        ax.text(bar.get_x() + bar.get_width()/2, cnt + 0.5,
                f"{pct:.1f}%", ha="center", fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title(f"Root Cause (total={total_nb})")

    ax = axes[2]
    b_pcts = [r["breakdown"]["case_B_pct"] for r in all_d4]
    ax.bar(range(len(b_pcts)), b_pcts,
           color=["red" if v > 50 else "steelblue" for v in b_pcts])
    ax.set_xticks(range(len(b_pcts)))
    ax.set_xticklabels([r["sequence"][-4:] for r in all_d4],
                       rotation=45, fontsize=7)
    ax.axhline(50, color="black", ls="--", lw=1)
    ax.set_ylabel("Case B %")
    ax.set_title("Case B% per Sequence")

    plt.suptitle(
        f"DIAG 4 — {len(all_d4)} Sequences  "
        f"A={total_A}({total_A/max(total_nb,1)*100:.0f}%)  "
        f"B={total_B}({total_B/max(total_nb,1)*100:.0f}%)  "
        f"C={total_C}({total_C/max(total_nb,1)*100:.0f}%)"
    )
    plt.tight_layout()
    plt.savefig(od / "diag4_score_global.png", dpi=150, bbox_inches="tight")
    plt.close()

    d4_global = {
        "total_spurious_newborns": total_nb,
        "case_A_total": total_A,
        "case_B_total": total_B,
        "case_C_total": total_C,
        "case_A_pct":   round(total_A / max(total_nb, 1) * 100, 1),
        "case_B_pct":   round(total_B / max(total_nb, 1) * 100, 1),
        "case_C_pct":   round(total_C / max(total_nb, 1) * 100, 1),
        "global_correct_mean": global_correct,
        "dominant_case": dominant,
        "per_sequence":  all_d4,
    }

    return d2_global, d3_global, d4_global


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    args = get_args()
    from utils.misc import yaml_to_dict
    from configs.util import load_super_config
    from models.motip import build as build_motip
    from models.misc import load_checkpoint, get_model

    device = torch.device(
        args.device if args.device else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )

    od = Path(args.output_dir)
    od.mkdir(parents=True, exist_ok=True)

    config = yaml_to_dict(args.config)
    config = load_super_config(config, config.get("SUPER_CONFIG_PATH"))

    print("Loading model...")
    model, _ = build_motip(config=config)
    load_checkpoint(model, path=args.checkpoint)
    model.eval().to(device)

    print("Installing attention hooks...")
    attn_container = install_attention_hooks(model)
    num_layers     = get_model(model).id_decoder.num_layers

    attn_global = {
        "weights_by_pos": {li: defaultdict(list) for li in range(num_layers)},
        "num_layers": num_layers,
    }

    val_dir   = Path(args.val_dir)
    sequences = sorted([d for d in val_dir.iterdir() if d.is_dir()])
    print(f"Found {len(sequences)} sequences in {val_dir}\n")

    all_d2, all_d3, all_d4 = [], [], []

    for idx, seq_dir in enumerate(sequences):
        print(f"[{idx+1}/{len(sequences)}] {seq_dir.name}", end="  ")
        d2, d3, d4, seq_attn = run_sequence(
            seq_dir, model, config, attn_container, device)
        all_d2.append(d2)
        all_d3.append(d3)
        all_d4.append(d4)

        for li in range(num_layers):
            for pos, ws in seq_attn["weights_by_pos"][li].items():
                attn_global["weights_by_pos"][li][pos].extend(ws)

    d2_g, d3_g, d4_g = aggregate_and_plot(
        all_d2, all_d3, all_d4, attn_global, num_layers, od, config)

    out = {
        "checkpoint":  args.checkpoint,
        "val_dir":     args.val_dir,
        "n_sequences": len(sequences),
        "diag2": d2_g,
        "diag3": d3_g,
        "diag4": d4_g,
    }
    with open(od / "diag234_results.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved to {od}/")
    print("  diag2_attention_global.png")
    print("  diag3_newborn_global.png")
    print("  diag4_score_global.png")
    print("  diag234_results.json")


if __name__ == "__main__":
    main()