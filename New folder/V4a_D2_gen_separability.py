#!/usr/bin/env python3
"""
D2_gen_separability.py
======================
D2-GEN: Does reid_proj learn training-specific features that fail on val?

Method:
  1. Extract frozen RF-DETR embeddings on VAL set (GT-matched, IoU≥0.5).
     DETR frozen → embeddings identical across all checkpoints.
  2. Baseline LDA separability (no reid_proj).
  3. Apply checkpoint_3 reid_proj weights → separability (V4a peak).
  4. Apply checkpoint_6 reid_proj weights → separability (post-LR-drop).
  5. Compare: ep3>baseline → projection helped val; ep6<ep3 → overfit confirmed.

Run from RF-MOTIPV4 repo root:
    python "New folder/D2_gen_separability.py" \
        --ckpt_detr rfdetr_dancetrack_motip/checkpoint_best_total.pth \
        --ckpt3     outputs/rfmotip_dancetrack_V3_full/checkpoint_3.pth \
        --ckpt6     outputs/rfmotip_dancetrack_V3_full/checkpoint_6.pth \
        --val_dir   /data/pos+mot/Datadir/DanceTrack/val \
        --output_dir "New folder/D2_output/"
"""

import os, sys, json, argparse
import numpy as np
import torch
import torchvision.transforms.functional as TF
from pathlib import Path
from PIL import Image
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Args ─────────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_detr",  required=True)
    p.add_argument("--ckpt3",      required=True)
    p.add_argument("--ckpt6",      required=True)
    p.add_argument("--val_dir",    default="/data/pos+mot/Datadir/DanceTrack/val")
    p.add_argument("--output_dir", default="New folder/D2_output/")
    p.add_argument("--max_frames_per_seq", type=int, default=200)
    p.add_argument("--min_samples_per_id", type=int, default=3)
    p.add_argument("--lda_components",     type=int, default=128)
    p.add_argument("--iou_thresh",         type=float, default=0.5)
    return p.parse_args()


# ── RF-DETR builder (from diag_d_new_proj.py — confirmed working) ─────────────
def build_rfdetr(ckpt_path: str, device: torch.device):
    from models.rfdetr.models.lwdetr import build_model
    ckpt      = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args_ckpt = ckpt["args"]
    args_ckpt.num_classes -= 1
    model = build_model(args=args_ckpt)
    args_ckpt.num_classes += 1

    state    = ckpt["model"]
    own      = model.state_dict()
    filtered = {k.replace("module.", ""): v for k, v in state.items()
                if k.replace("module.", "") in own
                and v.shape == own[k.replace("module.", "")].shape}
    model.load_state_dict(filtered, strict=False)
    model.eval().to(device)
    return model, int(args_ckpt.resolution)


# ── Preprocessing (from diag_rfdetr_temporal_stability.py — confirmed) ────────
def preprocess(img_path: str, resolution: int, device: torch.device):
    img    = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size
    t = TF.to_tensor(img)
    t = TF.normalize(t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    t = TF.resize(t, [resolution, resolution])
    return t.unsqueeze(0).to(device), orig_w, orig_h


# ── GT loading (from diag_rfdetr_temporal_stability.py — confirmed) ───────────
def load_gt(gt_path: str) -> dict:
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
            x, y, w, h = (float(parts[2]), float(parts[3]),
                           float(parts[4]), float(parts[5]))
            if w <= 0 or h <= 0:
                continue
            gt[frame_id][obj_id] = [x, y, w, h]
    return gt


# ── Box conversion (from diag_rfdetr_temporal_stability.py — confirmed) ───────
def box_cxcywh_to_xyxy(boxes_norm: np.ndarray,
                        orig_w: int, orig_h: int) -> np.ndarray:
    cx, cy, w, h = (boxes_norm[:, 0], boxes_norm[:, 1],
                    boxes_norm[:, 2], boxes_norm[:, 3])
    x1 = (cx - w / 2) * orig_w
    y1 = (cy - h / 2) * orig_h
    x2 = (cx + w / 2) * orig_w
    y2 = (cy + h / 2) * orig_h
    return np.stack([x1, y1, x2, y2], axis=1)


# ── IoU matrix ────────────────────────────────────────────────────────────────
def iou_matrix(gt_xyxy: np.ndarray, pred_xyxy: np.ndarray) -> np.ndarray:
    if len(gt_xyxy) == 0 or len(pred_xyxy) == 0:
        return np.zeros((len(gt_xyxy), len(pred_xyxy)))
    ix1 = np.maximum(gt_xyxy[:, 0:1], pred_xyxy[None, :, 0])
    iy1 = np.maximum(gt_xyxy[:, 1:2], pred_xyxy[None, :, 1])
    ix2 = np.minimum(gt_xyxy[:, 2:3], pred_xyxy[None, :, 2])
    iy2 = np.minimum(gt_xyxy[:, 3:4], pred_xyxy[None, :, 3])
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    ag = ((gt_xyxy[:, 2] - gt_xyxy[:, 0]) *
          (gt_xyxy[:, 3] - gt_xyxy[:, 1]))[:, None]
    ap = ((pred_xyxy[:, 2] - pred_xyxy[:, 0]) *
          (pred_xyxy[:, 3] - pred_xyxy[:, 1]))[None, :]
    union = ag + ap - inter
    return inter / np.maximum(union, 1e-6)


# ── Extract frozen val embeddings ─────────────────────────────────────────────
@torch.no_grad()
def extract_val_embeddings(detr_model, resolution, val_dir,
                           max_frames, min_samples, iou_thresh, device):
    val_seqs = sorted([d for d in Path(val_dir).iterdir() if d.is_dir()])
    print(f"  Val sequences: {len(val_seqs)}")

    all_X, all_y = [], []

    for seq_idx, seq_dir in enumerate(val_seqs):
        seq_name = seq_dir.name
        gt_path  = seq_dir / "gt" / "gt.txt"
        img_dir  = seq_dir / "img1"
        if not gt_path.exists():
            print(f"  [{seq_idx+1}] {seq_name}: SKIP (no GT)")
            continue

        gt     = load_gt(str(gt_path))
        frames = sorted(img_dir.glob("*.jpg"))
        if max_frames > 0:
            frames = frames[:max_frames]

        seq_embs  = defaultdict(list)
        n_matched = 0

        for frame_path in frames:
            frame_id = int(frame_path.stem)
            if frame_id not in gt or not gt[frame_id]:
                continue

            inp, orig_w, orig_h = preprocess(str(frame_path), resolution, device)
            out = detr_model(inp)

            # Confirmed key: out["outputs"] = hs[-1], shape (1, num_queries, 256)
            if "outputs" not in out:
                raise KeyError("'outputs' key not in model output. "
                               "Check lwdetr.py: out['outputs'] = hs[-1]")
            embs      = out["outputs"][0].cpu().float().numpy()   # (Q, 256)
            pred_norm = out["pred_boxes"][0].cpu().numpy()        # (Q, 4) cx,cy,w,h

            pred_xyxy = box_cxcywh_to_xyxy(pred_norm, orig_w, orig_h)
            gt_frame  = gt[frame_id]
            obj_ids   = list(gt_frame.keys())
            gt_xyxy   = np.array([[b[0], b[1], b[0]+b[2], b[1]+b[3]]
                                   for b in gt_frame.values()])

            iou_mat   = iou_matrix(gt_xyxy, pred_xyxy)   # (N_gt, Q)
            best_slot = np.argmax(iou_mat, axis=1)
            best_iou  = iou_mat[np.arange(len(obj_ids)), best_slot]

            for i, obj_id in enumerate(obj_ids):
                if best_iou[i] >= iou_thresh:
                    seq_embs[obj_id].append(embs[best_slot[i]])
                    n_matched += 1

        kept = 0
        for obj_id, emb_list in seq_embs.items():
            if len(emb_list) >= min_samples:
                global_id = seq_idx * 10000 + obj_id
                all_X.extend(emb_list)
                all_y.extend([global_id] * len(emb_list))
                kept += 1

        print(f"  [{seq_idx+1}/{len(val_seqs)}] {seq_name}: "
              f"{n_matched} matched, {kept} IDs kept")

    return (np.array(all_X, dtype=np.float32),
            np.array(all_y,  dtype=np.int64))


# ── Load reid_proj weights ────────────────────────────────────────────────────
def load_reid_proj(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd   = ckpt["model"]
    return (sd["reid_proj.0.weight"].float().numpy(),   # (256, 256)
            sd["reid_proj.1.weight"].float().numpy(),   # (256,)
            sd["reid_proj.1.bias"].float().numpy())     # (256,)


def apply_reid_proj(X: np.ndarray, W, LN_w, LN_b, eps=1e-5) -> np.ndarray:
    """Linear(bias=False) + LayerNorm applied to rows of X."""
    out  = X @ W.T
    mean = out.mean(axis=1, keepdims=True)
    std  = out.std(axis=1,  keepdims=True)
    out  = (out - mean) / (std + eps)
    return out * LN_w + LN_b


# ── Separability ──────────────────────────────────────────────────────────────
def compute_separability(X: np.ndarray, y: np.ndarray,
                          max_pairs: int = 8000) -> dict:
    rng      = np.random.default_rng(42)
    uid_list = [uid for uid in np.unique(y) if (y == uid).sum() >= 2]
    intra, inter = [], []

    for i, uid_i in enumerate(uid_list):
        embs_i = X[y == uid_i]
        embs_i = embs_i / (np.linalg.norm(embs_i, axis=1, keepdims=True) + 1e-8)
        idx = rng.choice(len(embs_i), min(len(embs_i), 8), replace=False)
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                intra.append(float(embs_i[idx[a]] @ embs_i[idx[b]]))

        for uid_j in uid_list[i+1:i+5]:
            embs_j = X[y == uid_j]
            embs_j = embs_j / (np.linalg.norm(embs_j, axis=1, keepdims=True) + 1e-8)
            jdx = rng.choice(len(embs_j), min(3, len(embs_j)), replace=False)
            for a in idx[:3]:
                for b in jdx:
                    inter.append(float(embs_i[a] @ embs_j[b]))

        if len(intra) > max_pairs:
            break

    return {
        "intra_mean": float(np.mean(intra)) if intra else 0.0,
        "inter_mean": float(np.mean(inter)) if inter else 0.0,
        "gap":        float(np.mean(intra) - np.mean(inter))
                      if (intra and inter) else 0.0,
        "n_intra": len(intra), "n_inter": len(inter),
    }


def fit_lda(X: np.ndarray, y: np.ndarray, n_components: int) -> np.ndarray:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import LabelEncoder
    le    = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_comp = min(n_components, len(le.classes_) - 1, X.shape[1])
    lda   = LinearDiscriminantAnalysis(n_components=n_comp)
    lda.fit(X, y_enc)
    return lda.transform(X)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = get_args()
    od   = Path(args.output_dir)
    od.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Step 1: Frozen val embeddings
    print("=" * 65)
    print("STEP 1: Extracting frozen RF-DETR embeddings from VAL set")
    print("=" * 65)
    detr, resolution = build_rfdetr(args.ckpt_detr, device)
    X, y = extract_val_embeddings(
        detr, resolution, args.val_dir,
        args.max_frames_per_seq, args.min_samples_per_id,
        args.iou_thresh, device)
    del detr; torch.cuda.empty_cache()
    print(f"\n  Total embeddings: {len(X)}  |  Unique IDs: {len(np.unique(y))}\n")

    # Step 2: Baseline
    print("=" * 65)
    print("STEP 2: Baseline separability (no reid_proj)")
    print("=" * 65)
    base = compute_separability(X, y)
    print(f"  Intra: {base['intra_mean']:.4f}  Inter: {base['inter_mean']:.4f}"
          f"  Gap: {base['gap']:.4f}\n")

    # Step 3: LDA oracle
    print("=" * 65)
    print("STEP 3: LDA oracle (upper bound)")
    print("=" * 65)
    X_lda = fit_lda(X, y, args.lda_components)
    lda   = compute_separability(X_lda, y)
    print(f"  Gap (LDA oracle): {lda['gap']:.4f}\n")

    # Step 4: checkpoint_3
    print("=" * 65)
    print("STEP 4: checkpoint_3 reid_proj (V4a peak, epoch 3)")
    print("=" * 65)
    W3, LN_w3, LN_b3 = load_reid_proj(args.ckpt3)
    drift3 = float(np.linalg.norm(W3 - np.eye(256), ord='fro'))
    print(f"  Drift from identity: {drift3:.4f}")
    X3    = apply_reid_proj(X, W3, LN_w3, LN_b3)
    sep3  = compute_separability(X3, y)
    print(f"  Intra: {sep3['intra_mean']:.4f}  Inter: {sep3['inter_mean']:.4f}"
          f"  Gap: {sep3['gap']:.4f}\n")

    # Step 5: checkpoint_6
    print("=" * 65)
    print("STEP 5: checkpoint_6 reid_proj (post-LR-drop, epoch 6)")
    print("=" * 65)
    W6, LN_w6, LN_b6 = load_reid_proj(args.ckpt6)
    drift6 = float(np.linalg.norm(W6 - np.eye(256), ord='fro'))
    print(f"  Drift from identity: {drift6:.4f}")
    X6    = apply_reid_proj(X, W6, LN_w6, LN_b6)
    sep6  = compute_separability(X6, y)
    print(f"  Intra: {sep6['intra_mean']:.4f}  Inter: {sep6['inter_mean']:.4f}"
          f"  Gap: {sep6['gap']:.4f}\n")

    # Verdict
    ep3_helps = sep3['gap'] > base['gap']
    ep6_worse = sep6['gap'] < sep3['gap']
    overfit   = ep3_helps and ep6_worse

    print("=" * 65)
    print("D2-GEN VERDICT")
    print("=" * 65)
    print(f"\n  {'Condition':<42} {'Gap':>8}  {'vs Baseline':>12}")
    print(f"  {'-'*64}")
    for label, sep, drift in [
        ("Baseline (no reid_proj)",         base, None),
        ("LDA oracle (upper bound)",         lda,  None),
        (f"checkpoint_3 (drift={drift3:.2f})", sep3, drift3),
        (f"checkpoint_6 (drift={drift6:.2f})", sep6, drift6),
    ]:
        ratio = f"{sep['gap']/max(base['gap'],1e-8):+.2f}×"
        print(f"  {label:<42} {sep['gap']:>8.4f}  {ratio:>12}")

    print(f"\n  ep3 improves val separability: {ep3_helps}")
    print(f"  ep6 DEGRADES vs ep3:           {ep6_worse}")
    print(f"\n  OVERFITTING HYPOTHESIS: {'CONFIRMED' if overfit else 'NOT CONFIRMED'}")

    if overfit:
        print("""
  → reid_proj learned features that help val at epoch 3 but degraded
    by epoch 6. Transformation overfit to training-sequence geometry.
  → NEXT: D3-OVERFIT — compute separability at ALL checkpoints (0-9)
           to find the drift threshold where generalization breaks.""")
    elif ep3_helps and not ep6_worse:
        print("""
  → reid_proj improves val separability at both epochs.
    Feature quality is NOT the cause of AssA degradation.
  → NEXT: D3-ASSIGN — check Case B rate with vs without reid_proj.""")
    elif not ep3_helps:
        print("""
  → reid_proj NEVER improves val separability — not even at peak.
    Projection learned training-ID memorization from epoch 0.
  → NEXT: D3-OVERFIT with all checkpoints to confirm pattern.""")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["Baseline", "LDA oracle",
              f"ckpt_3\n(drift={drift3:.2f})",
              f"ckpt_6\n(drift={drift6:.2f})"]
    gaps   = [base['gap'], lda['gap'], sep3['gap'], sep6['gap']]
    colors = ["steelblue", "green", "orange", "red"]
    bars   = ax.bar(labels, gaps, color=colors, alpha=0.8)
    ax.axhline(base['gap'], ls="--", color="steelblue", alpha=0.5)
    for bar, gap in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width()/2, gap + 0.0003,
                f"{gap:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Separability gap (intra − inter cosine sim)")
    ax.set_title(f"D2-GEN: val separability — "
                 f"overfitting {'CONFIRMED' if overfit else 'NOT CONFIRMED'}")
    plt.tight_layout()
    plt.savefig(od / "D2_gen_separability.png", dpi=150, bbox_inches="tight")
    plt.close()

    result = {
        "n_val_embeddings": int(len(X)), "n_val_ids": int(len(np.unique(y))),
        "baseline": base, "lda_oracle": lda,
        "checkpoint_3": {**sep3, "drift": drift3},
        "checkpoint_6": {**sep6, "drift": drift6},
        "ep3_improves_val": bool(ep3_helps),
        "ep6_degrades_vs_ep3": bool(ep6_worse),
        "overfitting_confirmed": bool(overfit),
    }
    with open(od / "D2_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: {od}/D2_result.json")
    print(f"  Saved: {od}/D2_gen_separability.png")


if __name__ == "__main__":
    main()