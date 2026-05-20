#!/usr/bin/env python3
"""
diag_d_new_proj.py  —  D-NEW-PROJ
===================================
Answers: Can a LINEAR projection improve the separability of frozen
RF-DETR embeddings enough to justify adding a re-ID projection head?

Method
------
1. Extract frozen RF-DETR embeddings for GT-matched objects across
   ALL DanceTrack TRAINING sequences (no val knowledge used).
2. Measure baseline separability gap:
     gap_before = mean(intra_sim) - mean(inter_sim)
3. Fit a supervised Linear Discriminant Analysis (LDA) projection
   using GT track IDs as class labels — this is the best-case linear
   projection (oracle upper bound for what a Linear layer can achieve).
4. Measure separability gap after LDA:
     gap_after = mean(intra_sim_lda) - mean(inter_sim_lda)
5. Report ratio = gap_after / gap_before.

Pass condition (projection head confirmed):  ratio > 2.0
Fail condition (drop V4c):                  ratio < 1.5

Scientific justification
------------------------
LDA finds the optimal linear projection for class separability.
If LDA cannot substantially improve separability (ratio < 1.5),
no learnable linear layer will either — the frozen features are
fundamentally not linearly separable by identity.

If LDA achieves ratio > 2.0, a trainable 256→256 linear layer
has sufficient capacity to learn a useful re-ID mapping,
and the projection head is a justified architectural addition.

Run from repo root:
  python diagnostics/diag_d_new_proj.py \\
    --config   configs/rf_detrV3_motip_dancetrack.yaml \\
    --ckpt     rfdetr_dancetrack_motip/checkpoint_best_total.pth \\
    --train_dir /data/pos+mot/Datadir/DanceTrack/train \\
    --output_dir diagnostics/diag_proj_results/
"""

import os
import sys
import json
import argparse
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


# ─────────────────────────────────────────────────────────────
# ARGS
# ─────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     required=True)
    p.add_argument("--ckpt",       required=True,
                   help="Fine-tuned RF-DETR checkpoint (checkpoint_best_total.pth)")
    p.add_argument("--train_dir",  default="/data/pos+mot/Datadir/DanceTrack/train")
    p.add_argument("--output_dir", default="diagnostics/diag_proj_results/")
    p.add_argument("--max_frames_per_seq", type=int, default=200,
                   help="Limit frames per sequence for speed (0=all)")
    p.add_argument("--lda_components", type=int, default=128,
                   help="LDA output dimensionality")
    p.add_argument("--min_samples_per_id", type=int, default=3,
                   help="Minimum frames per track ID to include in analysis")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# GT LOADING
# ─────────────────────────────────────────────────────────────
def load_gt(seq_dir: str) -> dict:
    """Returns {frame_id: {obj_id: [x,y,w,h]}}. Skips conf=0."""
    gt = defaultdict(dict)
    with open(os.path.join(seq_dir, "gt", "gt.txt")) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 7 or float(p[6]) == 0:
                continue
            gt[int(p[0])][int(p[1])] = [float(p[2]), float(p[3]),
                                          float(p[4]), float(p[5])]
    return gt


# ─────────────────────────────────────────────────────────────
# RF-DETR LOADER
# ─────────────────────────────────────────────────────────────
def build_rfdetr(ckpt_path: str, device: torch.device):
    from models.rfdetr.models.lwdetr import build_model
    ckpt      = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args_ckpt = ckpt["args"]
    args_ckpt.num_classes -= 1
    model = build_model(args=args_ckpt)
    args_ckpt.num_classes += 1

    state     = ckpt["model"]
    own       = model.state_dict()
    filtered  = {k.replace("module.", ""): v for k, v in state.items()
                 if k.replace("module.", "") in own
                 and v.shape == own[k.replace("module.", "")].shape}
    model.load_state_dict(filtered, strict=False)
    model.eval().to(device)
    return model, int(args_ckpt.resolution)


# ─────────────────────────────────────────────────────────────
# IoU MATCHING (GT box → query slot)
# ─────────────────────────────────────────────────────────────
def iou_matrix_np(pred_xywh: np.ndarray, gt_xywh: np.ndarray) -> np.ndarray:
    if len(pred_xywh) == 0 or len(gt_xywh) == 0:
        return np.zeros((len(pred_xywh), len(gt_xywh)))
    px2 = pred_xywh[:, 0] + pred_xywh[:, 2]
    py2 = pred_xywh[:, 1] + pred_xywh[:, 3]
    gx2 = gt_xywh[:, 0]  + gt_xywh[:, 2]
    gy2 = gt_xywh[:, 1]  + gt_xywh[:, 3]
    ix1 = np.maximum(pred_xywh[:, 0:1], gt_xywh[None, :, 0])
    iy1 = np.maximum(pred_xywh[:, 1:2], gt_xywh[None, :, 1])
    ix2 = np.minimum(px2[:, None],       gx2[None, :])
    iy2 = np.minimum(py2[:, None],       gy2[None, :])
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    ap = pred_xywh[:, 2] * pred_xywh[:, 3]
    ag = gt_xywh[:, 2]   * gt_xywh[:, 3]
    union = ap[:, None] + ag[None, :] - inter
    return inter / np.maximum(union, 1e-6)


# ─────────────────────────────────────────────────────────────
# SEPARABILITY METRICS
# ─────────────────────────────────────────────────────────────
def compute_separability(embeddings: np.ndarray, labels: np.ndarray,
                          max_pairs: int = 5000) -> dict:
    """
    Compute intra-ID and inter-ID cosine similarity.
    Samples pairs randomly to keep runtime bounded.
    """
    unique_ids = np.unique(labels)
    intra_sims, inter_sims = [], []

    rng = np.random.default_rng(42)

    # Intra-ID: same identity, different samples
    for uid in unique_ids:
        idx = np.where(labels == uid)[0]
        if len(idx) < 2:
            continue
        pairs = min(len(idx) * (len(idx) - 1) // 2, 200)
        chosen = rng.choice(len(idx), size=(pairs, 2), replace=True)
        chosen = chosen[chosen[:, 0] != chosen[:, 1]]
        for i, j in chosen[:100]:
            a, b = embeddings[idx[i]], embeddings[idx[j]]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > 1e-8 and nb > 1e-8:
                intra_sims.append(float(np.dot(a, b) / (na * nb)))

    # Inter-ID: different identities
    n_inter = min(max_pairs, 3000)
    idx_all = np.arange(len(labels))
    for _ in range(n_inter):
        i, j = rng.choice(len(labels), size=2, replace=False)
        if labels[i] == labels[j]:
            continue
        a, b = embeddings[i], embeddings[j]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 1e-8 and nb > 1e-8:
            inter_sims.append(float(np.dot(a, b) / (na * nb)))

    return {
        "intra_mean": float(np.mean(intra_sims)) if intra_sims else 0.0,
        "intra_std":  float(np.std(intra_sims))  if intra_sims else 0.0,
        "inter_mean": float(np.mean(inter_sims)) if inter_sims else 0.0,
        "inter_std":  float(np.std(inter_sims))  if inter_sims else 0.0,
        "gap":        float(np.mean(intra_sims) - np.mean(inter_sims))
                      if intra_sims and inter_sims else 0.0,
        "n_intra":    len(intra_sims),
        "n_inter":    len(inter_sims),
    }


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    args = get_args()
    od = Path(args.output_dir)
    od.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading RF-DETR (frozen, fine-tuned on DanceTrack)...")
    model, resolution = build_rfdetr(args.ckpt, device)
    print(f"  Resolution: {resolution}×{resolution}\n")

    train_seqs = sorted(
        p for p in Path(args.train_dir).iterdir() if p.is_dir())
    print(f"Found {len(train_seqs)} training sequences\n")

    all_embeddings = []   # (M, 256)
    all_labels     = []   # (M,)  track_id
    all_seq_ids    = []   # (M,)  sequence index (for per-seq analysis)

    MEANS = [0.485, 0.456, 0.406]
    STDS  = [0.229, 0.224, 0.225]

    with torch.no_grad():
        for seq_idx, seq_dir in enumerate(train_seqs):
            seq_name = seq_dir.name
            gt       = load_gt(str(seq_dir))
            imgs     = sorted(
                list(seq_dir.glob("img1/*.jpg")) +
                list(seq_dir.glob("img1/*.png")))

            if args.max_frames_per_seq > 0:
                # Sample evenly spaced frames
                step  = max(1, len(imgs) // args.max_frames_per_seq)
                imgs  = imgs[::step]

            seq_embs   = defaultdict(list)   # {track_id: [emb, ...]}
            n_matched  = 0

            for img_path in imgs:
                frame_id = int(img_path.stem)
                gt_frame = gt.get(frame_id, {})
                if not gt_frame:
                    continue

                # Preprocess
                img   = Image.open(str(img_path)).convert("RGB")
                orig_w, orig_h = img.size
                t = TF.to_tensor(img)
                t = TF.normalize(t, MEANS, STDS)
                t = TF.resize(t, [resolution, resolution])
                t = t.unsqueeze(0).to(device)

                out = model(t)
                if "outputs" not in out:
                    raise KeyError("'outputs' key missing. Ensure lwdetr.py has: out['outputs'] = hs[-1]")

                embs      = out["outputs"][0].cpu().numpy()   # (300, 256)
                boxes_norm = out["pred_boxes"][0].cpu().numpy()  # (300,4) cx,cy,w,h

                # Convert to pixel xywh
                pred_xywh = np.column_stack([
                    (boxes_norm[:, 0] - boxes_norm[:, 2] / 2) * orig_w,
                    (boxes_norm[:, 1] - boxes_norm[:, 3] / 2) * orig_h,
                    boxes_norm[:, 2] * orig_w,
                    boxes_norm[:, 3] * orig_h,
                ])

                gt_ids   = list(gt_frame.keys())
                gt_xywh  = np.array([gt_frame[i] for i in gt_ids], dtype=np.float32)
                iou_mat  = iou_matrix_np(pred_xywh, gt_xywh)   # (300, N_gt)
                best_slot = iou_mat.argmax(axis=0)               # (N_gt,)
                best_iou  = iou_mat.max(axis=0)                  # (N_gt,)

                for i, track_id in enumerate(gt_ids):
                    if best_iou[i] >= 0.5:
                        seq_embs[track_id].append(embs[best_slot[i]])
                        n_matched += 1

            # Filter IDs with enough samples
            for track_id, emb_list in seq_embs.items():
                if len(emb_list) >= args.min_samples_per_id:
                    # Use a globally unique label: seq_idx * 10000 + track_id
                    global_label = seq_idx * 10000 + track_id
                    all_embeddings.extend(emb_list)
                    all_labels.extend([global_label] * len(emb_list))
                    all_seq_ids.extend([seq_idx] * len(emb_list))

            print(f"  [{seq_idx+1}/{len(train_seqs)}] {seq_name}: "
                  f"{n_matched} matched, "
                  f"{len(seq_embs)} tracks, "
                  f"{sum(len(v) for v in seq_embs.values() if len(v) >= args.min_samples_per_id)} kept")

    X = np.array(all_embeddings, dtype=np.float32)   # (M, 256)
    y = np.array(all_labels, dtype=np.int64)          # (M,)

    print(f"\nTotal embeddings: {len(X)}")
    print(f"Unique identities: {len(np.unique(y))}")

    # ── STEP 1: Baseline separability (raw frozen embeddings) ─────────
    print("\nComputing baseline separability (frozen embeddings)...")
    baseline = compute_separability(X, y)
    print(f"  Intra-ID sim: {baseline['intra_mean']:.4f} ± {baseline['intra_std']:.4f}")
    print(f"  Inter-ID sim: {baseline['inter_mean']:.4f} ± {baseline['inter_std']:.4f}")
    print(f"  Gap (before): {baseline['gap']:.4f}")

    # ── STEP 2: L2-normalize first (baseline for fair comparison) ─────
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    baseline_norm = compute_separability(X_norm, y)
    print(f"\nBaseline (L2-normalized):")
    print(f"  Intra-ID sim: {baseline_norm['intra_mean']:.4f}")
    print(f"  Inter-ID sim: {baseline_norm['inter_mean']:.4f}")
    print(f"  Gap (L2-norm): {baseline_norm['gap']:.4f}")

    # ── STEP 3: LDA projection ────────────────────────────────────────
    print(f"\nFitting LDA (oracle linear projection, {args.lda_components} components)...")
    print("  (LDA = best-case supervised linear projection for class separability)")

    try:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        n_classes = len(le.classes_)
        n_components = min(args.lda_components, n_classes - 1, X.shape[1])
        print(f"  n_classes={n_classes}, n_components={n_components}")

        # Subsample if too large for LDA (LDA scales poorly with many classes)
        if len(X) > 50000:
            idx = np.random.default_rng(42).choice(len(X), 50000, replace=False)
            X_fit, y_fit = X[idx], y_encoded[idx]
        else:
            X_fit, y_fit = X, y_encoded

        lda = LinearDiscriminantAnalysis(n_components=n_components)
        lda.fit(X_fit, y_fit)
        X_lda = lda.transform(X)   # (M, n_components)
        print(f"  LDA fit complete. Output shape: {X_lda.shape}")

        lda_sep = compute_separability(X_lda, y)
        print(f"\nSeparability after LDA projection:")
        print(f"  Intra-ID sim: {lda_sep['intra_mean']:.4f} ± {lda_sep['intra_std']:.4f}")
        print(f"  Inter-ID sim: {lda_sep['inter_mean']:.4f} ± {lda_sep['inter_std']:.4f}")
        print(f"  Gap (after):  {lda_sep['gap']:.4f}")

        if baseline_norm['gap'] > 1e-6:
            ratio = lda_sep['gap'] / baseline_norm['gap']
        else:
            ratio = float('inf') if lda_sep['gap'] > 0 else 1.0
        print(f"\n  Improvement ratio: {ratio:.2f}×")

    except ImportError:
        print("  sklearn not available. Falling back to PCA.")
        from numpy.linalg import svd

        # Center the data
        X_centered = X - X.mean(axis=0)
        _, _, Vt = svd(X_centered, full_matrices=False)
        X_pca = X_centered @ Vt[:args.lda_components].T
        lda_sep = compute_separability(X_pca, y)
        ratio = lda_sep['gap'] / max(baseline_norm['gap'], 1e-6)
        print(f"  PCA gap: {lda_sep['gap']:.4f}  ratio: {ratio:.2f}×")

    # ── STEP 4: Simple linear layer simulation (random init) ──────────
    print("\nSimulating random-init linear projection (256→256):")
    rng = np.random.default_rng(42)
    W   = rng.normal(0, 1.0 / np.sqrt(256), (256, 256)).astype(np.float32)
    # Xavier: std = 1/sqrt(fan_in)
    X_random = X_norm @ W
    X_random = X_random / (np.linalg.norm(X_random, axis=1, keepdims=True) + 1e-8)
    random_sep = compute_separability(X_random, y)
    print(f"  Gap (random linear): {random_sep['gap']:.4f}")
    print(f"  (Expected near baseline — confirms linear projection starts neutral)")

    # ── VERDICT ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("D-NEW-PROJ VERDICT")
    print("="*60)
    print(f"  Gap before (L2-norm):  {baseline_norm['gap']:.4f}")
    print(f"  Gap after  (LDA):      {lda_sep['gap']:.4f}")
    print(f"  Improvement ratio:     {ratio:.2f}×")
    print()

    if ratio > 2.0:
        verdict = "PASS"
        conclusion = (
            f"Linear projection improves separability by {ratio:.1f}×. "
            "The frozen feature space IS linearly separable by identity. "
            "Re-ID projection head (V4c) is CONFIRMED — a trained linear "
            "layer can learn a meaningful re-ID mapping from these features."
        )
    elif ratio > 1.5:
        verdict = "MARGINAL"
        conclusion = (
            f"Linear projection improves separability by {ratio:.1f}× (moderate). "
            "The feature space has limited linear separability. Projection head "
            "may help but effect will be small. Consider deeper projection (256→256→128)."
        )
    else:
        verdict = "FAIL"
        conclusion = (
            f"Linear projection improves separability by only {ratio:.1f}×. "
            "The frozen features are NOT linearly separable by identity. "
            "A linear projection head will not meaningfully help. "
            "DROP V4c — focus on density sampler and training schedule instead."
        )

    print(f"  RESULT: {verdict}")
    print(f"  {conclusion}")

    # ── PLOTS ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (sep, label, color) in zip(axes, [
        (baseline_norm, "Raw frozen (L2-norm)", "steelblue"),
        (random_sep,    "Random linear (256→256)", "orange"),
        (lda_sep,       f"LDA ({args.lda_components}d, oracle)", "green"),
    ]):
        ax.bar(["Intra-ID", "Inter-ID"],
               [sep["intra_mean"], sep["inter_mean"]],
               color=[color, "lightgrey"],
               yerr=[sep["intra_std"], sep["inter_std"]],
               capsize=5, alpha=0.85)
        ax.set_ylabel("Cosine similarity")
        ax.set_title(f"{label}\nGap = {sep['gap']:.4f}")
        ax.set_ylim(0, 1)
        ax.axhline(sep["intra_mean"], color=color, ls="--", alpha=0.4)

    plt.suptitle(f"D-NEW-PROJ: Linear Separability Analysis\n"
                 f"Ratio = {ratio:.2f}×  →  {verdict}", fontsize=13)
    plt.tight_layout()
    plt.savefig(od / "d_new_proj_separability.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── SAVE JSON ─────────────────────────────────────────────────────
    result = {
        "n_embeddings":      len(X),
        "n_unique_ids":      int(len(np.unique(y))),
        "n_train_sequences": len(train_seqs),
        "baseline_raw":      baseline,
        "baseline_l2norm":   baseline_norm,
        "lda_projection":    lda_sep,
        "random_linear":     random_sep,
        "improvement_ratio": round(ratio, 4),
        "verdict":           verdict,
        "conclusion":        conclusion,
        "pass_threshold":    2.0,
        "marginal_threshold": 1.5,
    }
    with open(od / "d_new_proj_results.json", "w") as f:
        import json
        json.dump(result, f, indent=2)

    print(f"\nSaved to {od}/")
    print("  d_new_proj_separability.png")
    print("  d_new_proj_results.json")


if __name__ == "__main__":
    main()