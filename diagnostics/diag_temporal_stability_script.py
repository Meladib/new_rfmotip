#!/usr/bin/env python3
"""
Temporal stability diagnostic for RF-DETR query embeddings within MOTIP.

Measures per-query cosine similarity between consecutive frames to quantify
how consistently the decoder produces the same embedding for the same object
across time. Low similarity = high instability = broken re-ID.

Usage:
    python diagnostics/diag_temporal_stability_script.py \
        --checkpoint rfdetr_dancetrack_motip/checkpoint_best_total.pth \
        --sequence_dir /data/DanceTrack/val/dancetrack0001 \
        [--layer_index -1] \
        [--output_dir diagnostics/]
"""

import argparse
import json
import os
import sys
import glob

import numpy as np
import torch
import torch.nn.functional as F

# Add repo root to sys.path so model modules resolve.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------

def _load_frames(sequence_dir, start_idx, count):
    """Load `count` consecutive frames from DanceTrack sequence directory.

    DanceTrack layout: {sequence_dir}/img1/{frame_idx:08d}.jpg  (1-indexed).
    Returns a list of float32 torch tensors of shape [3, H, W], values in [0, 1].
    """
    from PIL import Image
    import torchvision.transforms.functional as TF

    frames = []
    for i in range(start_idx, start_idx + count):
        img_path = os.path.join(sequence_dir, "img1", f"{i:08d}.jpg")
        if not os.path.exists(img_path):
            raise FileNotFoundError(
                f"Frame not found: {img_path}. "
                f"Expected 1-indexed 8-digit filenames in {sequence_dir}/img1/"
            )
        img = Image.open(img_path).convert("RGB")
        frames.append(TF.to_tensor(img))
    return frames


def _build_nested_tensor(frame_tensor, device, size_divisibility=32):
    """Wrap a single [3, H, W] tensor into a NestedTensor-like object.

    RF-DETR's backbone expects a NestedTensor with `.tensors` and `.mask`.
    We create a minimal stand-in that satisfies the forward contract.
    """
    from structures.nested_tensor import NestedTensor  # noqa: E402

    t = frame_tensor.unsqueeze(0).to(device)  # [1, 3, H, W]
    _, _, H, W = t.shape

    # Pad H and W to multiples of size_divisibility.
    pad_h = (size_divisibility - H % size_divisibility) % size_divisibility
    pad_w = (size_divisibility - W % size_divisibility) % size_divisibility
    if pad_h > 0 or pad_w > 0:
        t = F.pad(t, (0, pad_w, 0, pad_h))

    _, _, H_pad, W_pad = t.shape
    mask = torch.zeros((1, H_pad, W_pad), dtype=torch.bool, device=device)
    return NestedTensor(t, mask)


# ---------------------------------------------------------------------------
# Model building and weight loading
# ---------------------------------------------------------------------------

def _build_and_load_rfdetr(checkpoint_path, device):
    """Build RF-DETR model from checkpoint args and apply the checkpoint weights.

    models/motip/__init__.py:build() intentionally omits load_state_dict() for
    the DETR model (it only reads ckpt["args"]).  This function explicitly
    applies ckpt["model"] so the diagnostic runs on trained weights.

    If checkpoint["model"] contains MOTIP wrapper keys (e.g. "detr."), this
    function strips the prefix so only the bare RF-DETR weights are loaded.
    """
    from models.rfdetr.models.lwdetr import build_model  # noqa: E402

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args_ckpt = ckpt["args"]
    model = build_model(args=args_ckpt)

    model_state = ckpt["model"]

    # Detect whether this is a MOTIP wrapper checkpoint (keys prefixed "detr.").
    if any(k.startswith("detr.") for k in model_state):
        model_state = {
            k[len("detr."):]: v
            for k, v in model_state.items()
            if k.startswith("detr.")
        }

    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing:
        print(f"[WARN] Missing keys in checkpoint: {missing[:5]}{'…' if len(missing)>5 else ''}")
    if unexpected:
        print(f"[WARN] Unexpected keys in checkpoint: {unexpected[:5]}{'…' if len(unexpected)>5 else ''}")

    model.eval()
    model.to(device)
    return model, args_ckpt


# ---------------------------------------------------------------------------
# Embedding extraction with layer-index support
# ---------------------------------------------------------------------------

def _extract_embeddings(model, nested_tensor, layer_index, captured):
    """Run one forward pass and return query embeddings at `layer_index`.

    The `TransformerDecoder` stores all layer outputs in `intermediate` and
    returns `torch.stack(intermediate)` with shape
    [dec_layers, B, num_queries, hidden_dim].  We capture this via a forward
    hook on `model.transformer`.

    layer_index follows Python list indexing: -1 = final layer, 0 = first.
    """
    captured.clear()

    def _hook(module, input, output):
        # output[0] = stacked hs: [dec_layers, B, num_queries, hidden_dim]
        if isinstance(output, (list, tuple)) and isinstance(output[0], torch.Tensor):
            captured["hs"] = output[0].detach().cpu()
        else:
            # Fallback: decoder returned output directly (non-intermediate mode)
            captured["hs"] = output.detach().cpu().unsqueeze(0)

    hook_handle = model.transformer.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            _ = model(nested_tensor)
    finally:
        hook_handle.remove()

    if "hs" not in captured:
        raise RuntimeError(
            "Forward hook did not capture decoder outputs. "
            "Verify that model.transformer.decoder has return_intermediate=True."
        )

    hs = captured["hs"]  # [dec_layers, B, num_queries, hidden_dim]
    return hs[layer_index, 0]  # [num_queries, hidden_dim]


# ---------------------------------------------------------------------------
# Cosine similarity statistics
# ---------------------------------------------------------------------------

def _cosine_stats(emb_a, emb_b):
    """Compute per-query cosine similarity between two [num_queries, D] tensors.

    Returns a dict with scalar statistics and a 10-bin histogram.
    """
    # Normalize rows to unit length.
    a_norm = F.normalize(emb_a.float(), dim=-1)
    b_norm = F.normalize(emb_b.float(), dim=-1)
    sims = (a_norm * b_norm).sum(dim=-1)  # [num_queries]

    sims_np = sims.numpy()
    hist_counts, hist_edges = np.histogram(sims_np, bins=10, range=(-1.0, 1.0))

    return {
        "mean": float(sims_np.mean()),
        "std": float(sims_np.std()),
        "min": float(sims_np.min()),
        "max": float(sims_np.max()),
        "n_unstable": int((sims_np < 0.5).sum()),
        "n_stable": int((sims_np > 0.9).sum()),
        "n_queries": int(len(sims_np)),
        "histogram": {
            "counts": hist_counts.tolist(),
            "edges": [round(float(e), 2) for e in hist_edges.tolist()],
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure temporal stability of RF-DETR query embeddings."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to RF-DETR checkpoint (.pth). May be the pre-trained RF-DETR "
             "checkpoint (rfdetr_dancetrack_motip/checkpoint_best_total.pth) or a "
             "MOTIP resume checkpoint (outputs/rfmotip_dancetrack/checkpoint_N.pth).",
    )
    parser.add_argument(
        "--sequence_dir",
        required=True,
        help="Path to a DanceTrack sequence directory, e.g. "
             "/data/DanceTrack/val/dancetrack0001. "
             "Must contain img1/XXXXXXXX.jpg (1-indexed, 8-digit).",
    )
    parser.add_argument(
        "--deformable_checkpoint",
        default=None,
        help="(Optional) Path to a second checkpoint to load over the first. "
             "Useful for loading a MOTIP resume checkpoint on top of the base "
             "RF-DETR architecture checkpoint.",
    )
    parser.add_argument(
        "--layer_index",
        type=int,
        default=-1,
        help="Which decoder layer to extract embeddings from. "
             "0 = first layer, -1 = final layer (default). "
             "For 3-layer RF-DETR-small: 0, 1, or 2 (or -1 = 2).",
    )
    parser.add_argument(
        "--output_dir",
        default="diagnostics",
        help="Directory to write temporal_stability_results.json (default: diagnostics/).",
    )
    parser.add_argument(
        "--start_frame",
        type=int,
        default=1,
        help="First frame index (1-indexed) to begin the 6-frame window (default: 1).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda if available, else cpu).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[INFO] Loading checkpoint: {args.checkpoint}")
    model, args_ckpt = _build_and_load_rfdetr(args.checkpoint, device)

    if args.deformable_checkpoint is not None:
        print(f"[INFO] Loading secondary checkpoint: {args.deformable_checkpoint}")
        sec_ckpt = torch.load(
            args.deformable_checkpoint, map_location="cpu", weights_only=False
        )
        sec_state = sec_ckpt["model"]
        if any(k.startswith("detr.") for k in sec_state):
            sec_state = {
                k[len("detr."):]: v
                for k, v in sec_state.items()
                if k.startswith("detr.")
            }
        model.load_state_dict(sec_state, strict=False)
        print("[INFO] Secondary checkpoint applied.")

    dec_layers = getattr(args_ckpt, "dec_layers", "unknown")
    hidden_dim = getattr(args_ckpt, "hidden_dim", "unknown")
    num_queries = getattr(args_ckpt, "num_queries", "unknown")
    print(
        f"[INFO] RF-DETR config: dec_layers={dec_layers}, "
        f"hidden_dim={hidden_dim}, num_queries={num_queries}"
    )
    print(f"[INFO] Probing decoder layer index: {args.layer_index}")

    # Load 6 consecutive frames (5 consecutive pairs).
    n_frames = 6
    print(
        f"[INFO] Loading {n_frames} frames starting at index {args.start_frame} "
        f"from {args.sequence_dir}"
    )
    frames = _load_frames(args.sequence_dir, args.start_frame, n_frames)

    # Extract embeddings for each frame.
    captured = {}
    embeddings = []
    for i, frame in enumerate(frames):
        nt = _build_nested_tensor(frame, device, size_divisibility=32)
        emb = _extract_embeddings(model, nt, args.layer_index, captured)
        embeddings.append(emb)
        print(
            f"  Frame {args.start_frame + i:08d}: "
            f"embedding shape={tuple(emb.shape)}, "
            f"norm_mean={emb.norm(dim=-1).mean().item():.4f}"
        )

    # Compute cosine similarity for each consecutive pair.
    pair_results = []
    for i in range(len(embeddings) - 1):
        stats = _cosine_stats(embeddings[i], embeddings[i + 1])
        pair_label = f"frame_{args.start_frame + i:08d}_to_{args.start_frame + i + 1:08d}"
        pair_results.append({"pair": pair_label, **stats})
        print(
            f"  {pair_label}: "
            f"mean_sim={stats['mean']:.4f}  std={stats['std']:.4f}  "
            f"min={stats['min']:.4f}  max={stats['max']:.4f}  "
            f"n_unstable(<0.5)={stats['n_unstable']}  n_stable(>0.9)={stats['n_stable']}"
        )

    # Aggregate across all pairs.
    all_means = [r["mean"] for r in pair_results]
    aggregate = {
        "mean_of_means": float(np.mean(all_means)),
        "std_of_means": float(np.std(all_means)),
        "min_mean": float(np.min(all_means)),
        "max_mean": float(np.max(all_means)),
    }

    print("\n[SUMMARY]")
    print(f"  Mean cosine similarity across 5 pairs: {aggregate['mean_of_means']:.4f}")
    print(f"  Std of pair means:                     {aggregate['std_of_means']:.4f}")
    print(f"  Per-pair range:                        [{aggregate['min_mean']:.4f}, {aggregate['max_mean']:.4f}]")
    if aggregate["mean_of_means"] < 0.5:
        print("  INTERPRETATION: SEVERE INSTABILITY (mean < 0.5)")
    elif aggregate["mean_of_means"] < 0.7:
        print("  INTERPRETATION: HIGH INSTABILITY (0.5 ≤ mean < 0.7)")
    elif aggregate["mean_of_means"] < 0.9:
        print("  INTERPRETATION: MODERATE INSTABILITY (0.7 ≤ mean < 0.9)")
    else:
        print("  INTERPRETATION: STABLE (mean ≥ 0.9)")

    # Histogram summary table.
    print("\n[HISTOGRAM — first pair]")
    h = pair_results[0]["histogram"]
    for count, lo, hi in zip(h["counts"], h["edges"][:-1], h["edges"][1:]):
        bar = "#" * min(count // max(1, num_queries // 40 if isinstance(num_queries, int) else 8), 40)
        print(f"  [{lo:+.2f}, {hi:+.2f}): {count:4d}  {bar}")

    # Write JSON results.
    results = {
        "checkpoint": args.checkpoint,
        "sequence_dir": args.sequence_dir,
        "layer_index": args.layer_index,
        "start_frame": args.start_frame,
        "dec_layers": dec_layers,
        "hidden_dim": hidden_dim,
        "num_queries": num_queries,
        "pair_results": pair_results,
        "aggregate": aggregate,
    }
    out_path = os.path.join(args.output_dir, "temporal_stability_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] Results written to: {out_path}")


if __name__ == "__main__":
    main()
