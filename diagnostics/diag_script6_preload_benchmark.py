#!/usr/bin/env python3
"""
diag_script6_preload_benchmark.py
===================================
Isolated benchmark comparing original SeqDataset (disk I/O per frame)
vs preloaded SeqDataset (all frames in RAM) without touching any repo file.

Both versions run on the same checkpoint and sequence.
Reports per-frame latency breakdown and FPS for each.

Run from inside MOTIP/:
  python diagnostics/diag_script6_preload_benchmark.py \\
    --config configs/rf_detr_motip_dancetrack.yaml \\
    --checkpoint outputsV2/rfmotip_dancetrack/train/checkpoint_7.pth \\
    --sequence_dir /data/pos+mot/Datadir/DanceTrack/val/dancetrack0041 \\
    --output_dir diagnostics/diag6_preload/ \\
    --num_frames 200 \\
    --warmup 10 \\
    --dtype FP16
"""

import sys
import os
import json
import argparse
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",       required=True)
    p.add_argument("--checkpoint",   required=True)
    p.add_argument("--sequence_dir", required=True)
    p.add_argument("--output_dir",   default="diagnostics/diag6_preload/")
    p.add_argument("--num_frames",   type=int, default=200)
    p.add_argument("--warmup",       type=int, default=10)
    p.add_argument("--dtype",        default="FP16", choices=["FP16", "FP32"])
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Preloaded SeqDataset — local definition, does NOT touch repo files
# ─────────────────────────────────────────────────────────────────────────────

class SeqDatasetPreloaded:
    """
    Drop-in replacement for SeqDataset that pre-loads all frames into RAM.
    Identical interface: __len__, __getitem__, seq_hw.
    Does not modify data/seq_dataset.py.
    """

    def __init__(self, seq_info, image_paths,
                 max_shorter=800, max_longer=1536,
                 size_divisibility=0, dtype=None):
        import torch
        from PIL import Image
        from torchvision.transforms import v2

        self.seq_info        = seq_info
        self.image_paths     = image_paths
        self.size_divisibility = size_divisibility
        self.dtype           = dtype or torch.float32

        transform = v2.Compose([
            v2.Resize(size=max_shorter, max_size=max_longer),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        print(f"  [PreloadedDataset] Loading {len(image_paths)} frames into RAM...")
        t0 = time.perf_counter()
        self._frames = []
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            t   = transform(img)
            if self.dtype != torch.float32:
                t = t.to(self.dtype)
            self._frames.append(t)
        elapsed = time.perf_counter() - t0
        mb = sum(f.element_size() * f.nelement() for f in self._frames) / 1e6
        print(f"  [PreloadedDataset] Loaded in {elapsed:.1f}s, "
              f"{mb:.0f} MB RAM used")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        from utils.nested_tensor import nested_tensor_from_tensor_list
        nt = nested_tensor_from_tensor_list(
            [self._frames[item]], self.size_divisibility)
        return nt, self.image_paths[item]

    def seq_hw(self):
        return self.seq_info["height"], self.seq_info["width"]


# ─────────────────────────────────────────────────────────────────────────────
# Timed inference loop
# ─────────────────────────────────────────────────────────────────────────────

def run_timed(dataset, model, config, dtype, device,
              num_frames, warmup, label):
    """
    Run inference on `dataset` for `num_frames` frames.
    Returns dict with per-frame latency breakdown.
    """
    import torch
    import numpy as np
    from torch.utils.data import DataLoader
    from models.runtime_tracker import RuntimeTracker
    sequence_wh = dataset.seq_hw()

    loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4 if label == "original" else 0,  # preloaded needs 0 workers
        pin_memory=True if label == "original" else False,
        collate_fn=lambda x: x[0],
    )

    rt = RuntimeTracker(
        model=model,
        sequence_hw=sequence_wh,
        use_sigmoid=config.get("USE_FOCAL_LOSS", False),
        assignment_protocol=config.get("ASSIGNMENT_PROTOCOL", "object-max"),
        miss_tolerance=config.get("MISS_TOLERANCE", 30),
        det_thresh=config.get("DET_THRESH", 0.3),
        newborn_thresh=config.get("NEWBORN_THRESH", 0.6),
        id_thresh=config.get("ID_THRESH", 0.2),
        area_thresh=config.get("AREA_THRESH", 0),
        only_detr=False,
        dtype=dtype,
    )

    times_load   = []   # time to get frame from DataLoader (includes disk I/O or RAM access)
    times_gpu    = []   # time for GPU transfer + tracker update
    times_total  = []

    n = min(num_frames + warmup, len(loader))

    loader_iter = iter(loader)
    for t in range(n):
        # ── Time the DataLoader step (disk I/O or RAM) ──────────────
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        frame, _ = next(loader_iter)
        t_load = (time.perf_counter() - t0) * 1000

        # ── GPU transfer + tracker update ───────────────────────────
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        frame.tensors = frame.tensors.to(device)
        frame.mask    = frame.mask.to(device)
        rt.update(image=frame)
        torch.cuda.synchronize()
        t_gpu = (time.perf_counter() - t1) * 1000

        if t >= warmup:
            times_load.append(t_load)
            times_gpu.append(t_gpu)
            times_total.append(t_load + t_gpu)

        if (t + 1) % 50 == 0:
            print(f"  [{label}] Frame {t+1}/{n}")

    mean_load  = float(np.mean(times_load))
    mean_gpu   = float(np.mean(times_gpu))
    mean_total = float(np.mean(times_total))
    fps        = 1000.0 / mean_total if mean_total > 0 else 0

    return {
        "label":          label,
        "mean_load_ms":   round(mean_load,  2),
        "mean_gpu_ms":    round(mean_gpu,   2),
        "mean_total_ms":  round(mean_total, 2),
        "fps":            round(fps,        2),
        "n_frames":       len(times_total),
        "load_pct":       round(mean_load  / mean_total * 100, 1),
        "gpu_pct":        round(mean_gpu   / mean_total * 100, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    import torch
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from configparser import ConfigParser
    from utils.misc import yaml_to_dict
    from configs.util import load_super_config
    from models.motip import build as build_motip
    from models.misc import load_checkpoint
    from data.seq_dataset import SeqDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float16 if args.dtype == "FP16" else torch.float32

    od = Path(args.output_dir)
    od.mkdir(parents=True, exist_ok=True)

    config = yaml_to_dict(args.config)
    config = load_super_config(config, config.get("SUPER_CONFIG_PATH"))

    # ── Load model once, share between both runs ──────────────────────
    print("Loading model...")
    model, _ = build_motip(config=config)
    load_checkpoint(model, path=args.checkpoint)
    model.eval().to(device)
    if dtype == torch.float16:
        model.half()

    # ── Sequence info ─────────────────────────────────────────────────
    seq = Path(args.sequence_dir)
    ini = ConfigParser()
    ini.read(seq / "seqinfo.ini")
    seq_info = {
        "height": int(ini["Sequence"]["imHeight"]),
        "width":  int(ini["Sequence"]["imWidth"]),
        "length": int(ini["Sequence"]["seqLength"]),
    }
    image_paths = [
        str(seq / "img1" / f"{i+1:08d}.jpg")
        for i in range(min(args.num_frames + args.warmup,
                           seq_info["length"]))
    ]

    MAX_SHORTER = 800
    MAX_LONGER  = config.get("INFERENCE_MAX_LONGER", 1440)
    SIZE_DIV    = config.get("SIZE_DIVISIBILITY", 32)

    # ── Run 1: Original (disk I/O per frame) ─────────────────────────
    print(f"\n{'='*55}")
    print("RUN 1 — Original SeqDataset (disk I/O per frame)")
    print(f"{'='*55}")
    ds_original = SeqDataset(
        seq_info=seq_info,
        image_paths=image_paths,
        max_shorter=MAX_SHORTER,
        max_longer=MAX_LONGER,
        size_divisibility=SIZE_DIV,
        dtype=dtype,
    )
    result_original = run_timed(
        ds_original, model, config, dtype, device,
        args.num_frames, args.warmup, "original"
    )

    # ── Run 2: Preloaded (RAM) ────────────────────────────────────────
    print(f"\n{'='*55}")
    print("RUN 2 — Preloaded SeqDataset (frames in RAM)")
    print(f"{'='*55}")
    ds_preloaded = SeqDatasetPreloaded(
        seq_info=seq_info,
        image_paths=image_paths,
        max_shorter=MAX_SHORTER,
        max_longer=MAX_LONGER,
        size_divisibility=SIZE_DIV,
        dtype=dtype,
    )
    result_preloaded = run_timed(
        ds_preloaded, model, config, dtype, device,
        args.num_frames, args.warmup, "preloaded"
    )

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("RESULTS")
    print(f"{'='*55}")
    for r in [result_original, result_preloaded]:
        print(f"\n  [{r['label']}]")
        print(f"    Load/preproc:  {r['mean_load_ms']:>7.2f} ms  ({r['load_pct']}%)")
        print(f"    GPU (tracker): {r['mean_gpu_ms']:>7.2f} ms  ({r['gpu_pct']}%)")
        print(f"    Total:         {r['mean_total_ms']:>7.2f} ms")
        print(f"    FPS:           {r['fps']:>7.2f}")

    speedup = result_original["mean_total_ms"] / result_preloaded["mean_total_ms"]
    fps_gain = result_preloaded["fps"] - result_original["fps"]
    print(f"\n  Speedup:  {speedup:.2f}x")
    print(f"  FPS gain: +{fps_gain:.1f} FPS")
    print(f"{'='*55}")

    # ── Plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    labels   = ["Original\n(disk I/O)", "Preloaded\n(RAM)"]
    loads    = [result_original["mean_load_ms"], result_preloaded["mean_load_ms"]]
    gpus     = [result_original["mean_gpu_ms"],  result_preloaded["mean_gpu_ms"]]
    totals   = [result_original["mean_total_ms"],result_preloaded["mean_total_ms"]]
    fpss     = [result_original["fps"],           result_preloaded["fps"]]

    x = [0, 1]
    ax = axes[0]
    bars_load = ax.bar(x, loads, color="tomato",    alpha=0.8, label="Load/preproc")
    bars_gpu  = ax.bar(x, gpus,  color="steelblue", alpha=0.8,
                       bottom=loads, label="GPU (tracker+detr)")
    for i, (tot, fps) in enumerate(zip(totals, fpss)):
        ax.text(i, tot + 0.5, f"{tot:.1f}ms\n{fps:.1f} FPS",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean latency per frame (ms)")
    ax.set_title(f"Latency Breakdown\ndtype={args.dtype}  "
                 f"sequence={seq.name}")
    ax.legend()

    ax = axes[1]
    colors = ["tomato", "steelblue"]
    bars = ax.bar(labels, fpss, color=colors, alpha=0.8)
    ax.bar_label(bars, fmt="%.1f FPS", fontsize=10)
    ax.axhline(25, color="green", ls="--", alpha=0.7, label="25 FPS (real-time)")
    ax.set_ylabel("FPS")
    ax.set_title(f"FPS Comparison\nSpeedup: {speedup:.2f}x  Gain: +{fps_gain:.1f} FPS")
    ax.legend()

    plt.tight_layout()
    plt.savefig(od / "diag6_preload_benchmark.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Save JSON ──────────────────────────────────────────────────────
    results = {
        "original":   result_original,
        "preloaded":  result_preloaded,
        "speedup":    round(speedup, 3),
        "fps_gain":   round(fps_gain, 2),
        "sequence":   seq.name,
        "dtype":      args.dtype,
        "num_frames": args.num_frames,
    }
    with open(od / "diag6_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nOutputs → {od}/")


if __name__ == "__main__":
    main()