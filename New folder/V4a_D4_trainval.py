#!/usr/bin/env python3
"""
D4_trainval.py
==============
D4-TRAINVAL: Test H1 — IDDecoder co-adaptation failure.

Hypothesis: IDDecoder co-adapted to training-sequence ID distributions
during V4a training (moving coordinate system from reid_proj drift).
Prediction: V4a shows larger train-val discriminability gap than V3.

Measurement:
  For each sequence (train + val), capture per-frame:
    - non_newborn_scores: max id_score for objects assigned existing IDs
    - newborn_max_scores: max id_score (over non-newborn vocab slots) for
                          objects assigned as newborn
  Discriminability gap = mean(non_newborn_scores) - mean(newborn_max_scores)
  A larger gap = IDDecoder more confidently separates tracked vs new objects.

H1 confirmed if:
  gap_V4a_train >> gap_V4a_val  AND
  gap_V3_train ≈ gap_V3_val
  (V4a's IDDecoder is overfit to training sequences, V3's is not)

Run from RF-MOTIPV4 repo root:
    python "New folder/D4_trainval.py" \
        --config     configs/rf_detrV4_motip_dancetrack.yaml \
        --ckpt_v3    /data/adib/new/github/RF-MOTIP/outputsV3/rfmotip_dancetrack/train/checkpoint_7.pth \
        --ckpt_v4_6  outputs/rfmotip_dancetrack_V3_full/checkpoint_6.pth \
        --data_root  /data/pos+mot/Datadir/ \
        --num_train_seqs  10 \
        --num_val_seqs    10 \
        --output_dir "New folder/D4_output/"
"""

import os, sys, json, argparse
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Args ──────────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",          required=True)
    p.add_argument("--ckpt_v3",         required=True)
    p.add_argument("--ckpt_v4_6",       required=True)
    p.add_argument("--data_root",       default="/data/pos+mot/Datadir/")
    p.add_argument("--num_train_seqs",  type=int, default=10)
    p.add_argument("--num_val_seqs",    type=int, default=10)
    p.add_argument("--output_dir",      default="New folder/D4_output/")
    return p.parse_args()


# ── Model loader ──────────────────────────────────────────────────────────────
def load_model(config_path, ckpt_path, device, use_reid_proj=None):
    from utils.misc import yaml_to_dict
    from configs.util import load_super_config
    from models.motip import build as build_motip
    from models.misc import load_checkpoint

    config = yaml_to_dict(config_path)
    config = load_super_config(config, config.get("SUPER_CONFIG_PATH"))
    config["RESUME_MODEL"] = None
    if use_reid_proj is not None:
        config["USE_REID_PROJ"] = use_reid_proj
    model, _ = build_motip(config=config)
    load_checkpoint(model, path=ckpt_path)
    model.eval().to(device)
    return model, config


# ── Run sequences, capture score distributions ────────────────────────────────
@torch.no_grad()
def run_sequences(model, config, data_root, split, num_seqs, device):
    """
    Returns:
        non_newborn_scores : list of float — max id_score for non-newborn assignments
        newborn_vocab_max  : list of float — max score over non-newborn vocab slots
                             for objects assigned as newborn
    Both lists contain one entry per (frame, object) pair where trajectories exist.
    Frames with empty trajectory buffer (first frame of each sequence) are excluded.
    """
    from models.runtime_tracker import RuntimeTracker
    from data.dancetrack import DanceTrack
    from data.seq_dataset import SeqDataset

    dt = DanceTrack(data_root=data_root, split=split, load_annotation=False)
    seq_names = sorted(dt.sequence_infos.keys())[:num_seqs]
    print(f"  Split={split}, sequences={len(seq_names)}")

    non_newborn_scores = []  # score for the assigned existing ID
    newborn_vocab_max  = []  # max score over vocab slots (excl. newborn col)
                             # for objects assigned as newborn
    num_vocab = None

    for seq_idx, seq_name in enumerate(seq_names):
        seq_ds = SeqDataset(
            seq_info=dt.sequence_infos[seq_name],
            image_paths=dt.image_paths[seq_name],
            max_shorter=800,
            max_longer=config.get("INFERENCE_MAX_LONGER", 1440),
            size_divisibility=config.get("SIZE_DIVISIBILITY", 32),
            dtype=torch.float32,
        )
        loader = DataLoader(
            seq_ds, batch_size=1, shuffle=False,
            num_workers=0, collate_fn=lambda x: x[0]
        )

        rt = RuntimeTracker(
            model=model,
            sequence_hw=seq_ds.seq_hw(),
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
        num_vocab = rt.num_id_vocabulary

        # ── Capture container ──────────────────────────────────────────────
        # Written by the patch, read in the main loop.
        # Cleared each frame before rt.update().
        capture = {"id_scores": None, "id_labels": None,
                   "has_trajectories": False}

        orig_get_id = rt._get_id_pred_labels

        def patched_get_id(boxes, output_embeds):
            # Check trajectory buffer BEFORE the forward pass changes it.
            # trajectory_features.shape[0] == 0 means no active tracks.
            capture["has_trajectories"] = (
                rt.trajectory_features.shape[0] > 0
            )
            result = orig_get_id(boxes=boxes, output_embeds=output_embeds)
            return result

        # Patch _object_max_assignment to capture id_scores and labels.
        # This is called INSIDE _get_id_pred_labels, so capture is set first.
        orig_oma = rt._object_max_assignment

        def patched_oma(id_scores):
            # id_scores: (N_obj, num_vocab + 1)
            # Clone before passing to avoid mutation.
            capture["id_scores"] = id_scores.detach().cpu().clone()
            result = orig_oma(id_scores=id_scores)
            # result is a plain Python list of ints (vocabulary labels or num_vocab)
            capture["id_labels"] = result
            return result

        rt._get_id_pred_labels   = patched_get_id
        rt._object_max_assignment = patched_oma

        n_non_nb = 0
        n_nb     = 0

        for image, _ in loader:
            image.tensors = image.tensors.to(device)
            image.mask    = image.mask.to(device)

            # Clear capture before each frame
            capture["id_scores"]       = None
            capture["id_labels"]       = None
            capture["has_trajectories"] = False

            rt.update(image)

            # Skip frames where there were no active trajectories
            # (first frame of sequence, or after a long gap).
            # In these frames _get_id_pred_labels returns all-newborn
            # without calling _object_max_assignment.
            if not capture["has_trajectories"]:
                continue

            id_scores = capture["id_scores"]   # (N_obj, num_vocab+1) or None
            id_labels = capture["id_labels"]   # list of ints or None

            # _object_max_assignment is only called when has_trajectories=True
            # AND the trajectory buffer is non-empty. Confirm both are set.
            if id_scores is None or id_labels is None:
                continue

            N_obj = id_scores.shape[0]
            assert len(id_labels) == N_obj, (
                f"id_labels length {len(id_labels)} != id_scores rows {N_obj}"
            )

            for i in range(N_obj):
                label = id_labels[i]
                if label != num_vocab:
                    # Non-newborn: the assigned label IS the argmax.
                    # Confirm: id_scores[i, label] should equal
                    # the maximum score for this object over all vocab slots.
                    assigned_score = float(id_scores[i, label])
                    non_newborn_scores.append(assigned_score)
                    n_non_nb += 1
                else:
                    # Newborn: capture max score over non-newborn vocabulary
                    # slots (columns 0..num_vocab-1).
                    # This tells us: how close was this object to being matched?
                    vocab_max = float(id_scores[i, :num_vocab].max())
                    newborn_vocab_max.append(vocab_max)
                    n_nb += 1

        print(f"    [{seq_idx+1}/{len(seq_names)}] {seq_name}: "
              f"non_nb={n_non_nb}  nb={n_nb}")

    return non_newborn_scores, newborn_vocab_max


# ── Compute discriminability statistics ───────────────────────────────────────
def compute_stats(non_nb, nb_max):
    """
    Returns dict with discriminability metrics.
    non_nb : scores for non-newborn (correctly assigned) objects
    nb_max : max vocab scores for newborn objects
    """
    if not non_nb or not nb_max:
        return {k: float("nan") for k in
                ["mean_non_nb", "std_non_nb", "p10_non_nb",
                 "mean_nb_max", "std_nb_max", "p10_nb_max",
                 "discriminability_gap", "n_non_nb", "n_nb"]}
    non_nb = np.array(non_nb, dtype=np.float32)
    nb_max = np.array(nb_max, dtype=np.float32)
    return {
        "mean_non_nb":          float(np.mean(non_nb)),
        "std_non_nb":           float(np.std(non_nb)),
        "p10_non_nb":           float(np.percentile(non_nb, 10)),
        "mean_nb_max":          float(np.mean(nb_max)),
        "std_nb_max":           float(np.std(nb_max)),
        "p10_nb_max":           float(np.percentile(nb_max, 10)),
        # Key metric: how well does IDDecoder separate tracked vs new objects?
        "discriminability_gap": float(np.mean(non_nb) - np.mean(nb_max)),
        "n_non_nb":             int(len(non_nb)),
        "n_nb":                 int(len(nb_max)),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args   = get_args()
    od     = Path(args.output_dir)
    od.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    results = {}

    # ── V3 on train ───────────────────────────────────────────────────────────
    print("=" * 65)
    print("V3 — TRAIN sequences")
    print("=" * 65)
    model_v3, config = load_model(
        args.config, args.ckpt_v3, device, use_reid_proj=False)
    non_nb, nb_max = run_sequences(
        model_v3, config, args.data_root, "train",
        args.num_train_seqs, device)
    results["V3_train"] = compute_stats(non_nb, nb_max)

    # ── V3 on val ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("V3 — VAL sequences")
    print("=" * 65)
    non_nb, nb_max = run_sequences(
        model_v3, config, args.data_root, "val",
        args.num_val_seqs, device)
    results["V3_val"] = compute_stats(non_nb, nb_max)
    del model_v3
    torch.cuda.empty_cache()

    # ── V4a_6 on train ────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("V4a_6 — TRAIN sequences")
    print("=" * 65)
    model_v4, _ = load_model(
        args.config, args.ckpt_v4_6, device, use_reid_proj=None)
    non_nb, nb_max = run_sequences(
        model_v4, config, args.data_root, "train",
        args.num_train_seqs, device)
    results["V4a_6_train"] = compute_stats(non_nb, nb_max)

    # ── V4a_6 on val ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("V4a_6 — VAL sequences")
    print("=" * 65)
    non_nb, nb_max = run_sequences(
        model_v4, config, args.data_root, "val",
        args.num_val_seqs, device)
    results["V4a_6_val"] = compute_stats(non_nb, nb_max)
    del model_v4
    torch.cuda.empty_cache()

    # ── Verdict ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("D4-TRAINVAL VERDICT")
    print("=" * 65)

    def gap(key_train, key_val):
        return (results[key_train]["discriminability_gap"]
                - results[key_val]["discriminability_gap"])

    v3_trainval_gap  = gap("V3_train",    "V3_val")
    v4a_trainval_gap = gap("V4a_6_train", "V4a_6_val")

    print(f"\n  {'Condition':<20} {'mean_non_nb':>12} {'mean_nb_max':>12} "
          f"{'disc_gap':>10} {'n_non_nb':>10}")
    print(f"  {'-'*66}")
    for key in ["V3_train", "V3_val", "V4a_6_train", "V4a_6_val"]:
        r = results[key]
        print(f"  {key:<20} {r['mean_non_nb']:>12.4f} {r['mean_nb_max']:>12.4f} "
              f"{r['discriminability_gap']:>10.4f} {r['n_non_nb']:>10}")

    print(f"\n  V3  train-val discriminability gap:  {v3_trainval_gap:+.4f}")
    print(f"  V4a train-val discriminability gap:  {v4a_trainval_gap:+.4f}")
    print(f"  Difference (V4a - V3):               {v4a_trainval_gap - v3_trainval_gap:+.4f}")

    # H1 confirmed if V4a shows substantially larger train-val gap than V3.
    # Use a conservative threshold of 2× V3's gap.
    threshold = 2.0 * abs(v3_trainval_gap) if v3_trainval_gap != 0 else 0.01
    h1_confirmed = (v4a_trainval_gap > v3_trainval_gap + threshold)

    print(f"\n  H1 (IDDecoder co-adaptation): "
          f"{'CONFIRMED' if h1_confirmed else 'NOT CONFIRMED'}")

    if h1_confirmed:
        print("""
  V4a's IDDecoder shows significantly larger train-val discriminability
  gap than V3. The IDDecoder co-adapted to the training-sequence feature
  distribution during the moving coordinate system of reid_proj drift.
  This co-adaptation does not generalize to val sequences.
  → Next: design V4b to decouple reid_proj optimization from IDDecoder
    (e.g., separate LR schedule, gradient stopping, or staged training).""")
    else:
        print("""
  V4a's IDDecoder does NOT show a substantially larger train-val gap
  than V3. H1 is not supported.
  → The root cause of the generalization gap remains unidentified.
  → Possible direction: analyze IDDecoder attention patterns on val
    sequences at checkpoint_3 (peak) vs checkpoint_6 (collapse).""")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: discriminability gap train vs val
    ax = axes[0]
    models  = ["V3", "V4a_6"]
    x       = np.arange(len(models))
    train_g = [results["V3_train"]["discriminability_gap"],
               results["V4a_6_train"]["discriminability_gap"]]
    val_g   = [results["V3_val"]["discriminability_gap"],
               results["V4a_6_val"]["discriminability_gap"]]
    w = 0.35
    ax.bar(x - w/2, train_g, w, label="Train", color=["steelblue", "orange"],
           alpha=0.8)
    ax.bar(x + w/2, val_g,   w, label="Val",   color=["steelblue", "orange"],
           alpha=0.4)
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("Discriminability gap\n(mean_non_nb - mean_nb_max)")
    ax.set_title("IDDecoder discriminability: Train vs Val\n"
                 "(H1: V4a gap should be larger on Train than Val)")
    ax.legend()

    # Panel 2: train-val gap comparison
    ax = axes[1]
    gaps   = [v3_trainval_gap, v4a_trainval_gap]
    colors = ["steelblue", "orange"]
    bars   = ax.bar(["V3", "V4a_6"], gaps, color=colors, alpha=0.8)
    ax.axhline(0, color="black", lw=0.8)
    for bar, g in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width()/2,
                g + 0.0005 * np.sign(g),
                f"{g:+.4f}", ha="center",
                va="bottom" if g >= 0 else "top", fontsize=9)
    ax.set_ylabel("Train discriminability − Val discriminability")
    ax.set_title(f"H1 train-val gap\n"
                 f"({'CONFIRMED' if h1_confirmed else 'NOT CONFIRMED'})")

    plt.tight_layout()
    plt.savefig(od / "D4_trainval.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        "results":             results,
        "v3_trainval_gap":     float(v3_trainval_gap),
        "v4a_trainval_gap":    float(v4a_trainval_gap),
        "h1_confirmed":        bool(h1_confirmed),
    }
    with open(od / "D4_result.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {od}/D4_result.json")
    print(f"  Saved: {od}/D4_trainval.png")


if __name__ == "__main__":
    main()