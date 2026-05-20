#!/usr/bin/env python3
"""
diag_script1_id_loss_split.py  (v2 — reads existing logs only)
================================================================
Reads train id_loss from log.txt and AssA/HOTA from eval_during_train/
to plot the train loss vs evaluation metric gap across epochs.

No model forward pass required — everything is read from files you
already have.

Saturation confirmed if:
  train id_loss keeps falling after the epoch where AssA peaks.

Run from inside MOTIP/:
  python diagnostics/diag_script1_id_loss_split.py \\
    --train_log_file outputsV2/rfmotip_dancetrack/train/log.txt \\
    --eval_dirs      outputsV2/rfmotip_dancetrack/train/eval_during_train/ \\
    --output_dir     diagnostics/diag1/
"""

import sys
import os
import re
import json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_log_file", required=True)
    p.add_argument("--eval_dirs",      required=True)
    p.add_argument("--output_dir",     default="diagnostics/diag1/")
    return p.parse_args()


def parse_train_metrics(log_file):
    """
    Parse [Finish epoch: N] lines from MOTIP log.txt.
    Returns {epoch: {id_loss, detr_loss, total_loss}}
    """
    results = {}
    with open(log_file) as f:
        for line in f:
            if "Finish epoch" not in line:
                continue
            em = re.search(r'Finish epoch:\s*(\d+)', line)
            if not em:
                continue
            epoch = int(em.group(1))

            def extract(name):
                m = re.search(name + r'\s*=\s*([\d.]+)', line)
                return float(m.group(1)) if m else None

            results[epoch] = {
                "total_loss": extract("loss"),
                "detr_loss":  extract("detr_loss"),
                "id_loss":    extract("id_loss"),
            }
    return results


def parse_eval_from_log(log_file):
    """
    Parse [Eval epoch: N] lines from the same log.txt.
    Format: [Metrics] [Eval epoch: N] HOTA = X; AssA = X; DetA = X; ...
    """
    results = {}
    with open(log_file) as f:
        for line in f:
            if "Eval epoch" not in line:
                continue
            em = re.search(r'Eval epoch:\s*(\d+)', line)
            if not em:
                continue
            epoch = int(em.group(1))

            def extract(name):
                m = re.search(name + r'\s*=\s*([\d.]+)', line)
                return float(m.group(1)) if m else None

            results[epoch] = {
                "HOTA": extract("HOTA"),
                "AssA": extract("AssA"),
                "DetA": extract("DetA"),
            }
    return results


def parse_eval_from_dirs(eval_dirs):
    """
    Read HOTA/AssA/DetA from eval_during_train/epoch_N/ subdirs.
    Tries pedestrian_summary.txt (TrackEval format) and .json files.
    """
    eval_dir = Path(eval_dirs)
    results  = {}

    for epoch_dir in sorted(eval_dir.glob("epoch_*")):
        em = re.search(r'epoch_(\d+)', epoch_dir.name)
        if not em:
            continue
        epoch = int(em.group(1))

        # Try pedestrian_summary.txt
        for tf in epoch_dir.rglob("pedestrian_summary.txt"):
            try:
                with open(tf) as f:
                    lines = f.readlines()
                if len(lines) >= 2:
                    keys = lines[0].strip().split()
                    vals = lines[1].strip().split()
                    d    = {k: float(v) for k, v in zip(keys, vals)}
                    results[epoch] = {
                        "HOTA": d.get("HOTA"),
                        "AssA": d.get("AssA"),
                        "DetA": d.get("DetA"),
                    }
                    break
            except Exception:
                pass

        if epoch in results:
            continue

        # Try any .json with HOTA key
        for jf in epoch_dir.rglob("*.json"):
            try:
                with open(jf) as f:
                    d = json.load(f)
                if "HOTA" in d:
                    results[epoch] = {
                        "HOTA": d.get("HOTA"),
                        "AssA": d.get("AssA"),
                        "DetA": d.get("DetA"),
                    }
                    break
            except Exception:
                pass

    return results


def main():
    args = get_args()

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    od = Path(args.output_dir)
    od.mkdir(parents=True, exist_ok=True)

    # ── Parse train metrics ───────────────────────────────────────────
    print("Parsing training log...")
    train_metrics = parse_train_metrics(args.train_log_file)
    print(f"  Train epochs found: {sorted(train_metrics.keys())}")
    for e, m in sorted(train_metrics.items()):
        print(f"  Epoch {e}: id_loss={m['id_loss']}  "
              f"detr_loss={m['detr_loss']}  total={m['total_loss']}")

    # ── Parse eval metrics (dirs first, log fallback) ─────────────────
    print("\nParsing eval metrics...")
    eval_metrics = parse_eval_from_dirs(args.eval_dirs)
    if not eval_metrics:
        print("  No eval files in dirs — reading from log.txt...")
        eval_metrics = parse_eval_from_log(args.train_log_file)
    print(f"  Eval epochs found: {sorted(eval_metrics.keys())}")
    for e, m in sorted(eval_metrics.items()):
        print(f"  Epoch {e}: HOTA={m.get('HOTA')}  "
              f"AssA={m.get('AssA')}  DetA={m.get('DetA')}")

    if not train_metrics:
        print("ERROR: no training metrics found. Check --train_log_file.")
        sys.exit(1)

    # ── Plot ──────────────────────────────────────────────────────────
    epochs_train = sorted(train_metrics.keys())
    epochs_eval  = sorted(eval_metrics.keys())

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Panel 1: train id_loss
    ax = axes[0]
    id_losses = [train_metrics[e]["id_loss"] for e in epochs_train
                 if train_metrics[e].get("id_loss") is not None]
    valid_te  = [e for e in epochs_train
                 if train_metrics[e].get("id_loss") is not None]

    ax.plot(valid_te, id_losses,
            marker="o", lw=2, color="steelblue", label="Train id_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("id_loss")
    ax.set_title("Train ID Loss per Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: AssA / HOTA / DetA
    ax = axes[1]
    if epochs_eval:
        def plot_metric(name, color, marker):
            vals = [eval_metrics[e].get(name) for e in epochs_eval]
            if any(v is not None for v in vals):
                ax.plot(epochs_eval, vals, marker=marker, lw=2,
                        color=color, label=name)

        plot_metric("HOTA", "green",  "o")
        plot_metric("AssA", "orange", "s")
        plot_metric("DetA", "purple", "^")

        valid_assa = [(e, eval_metrics[e]["AssA"])
                      for e in epochs_eval
                      if eval_metrics[e].get("AssA") is not None]
        if valid_assa:
            peak_e, peak_a = max(valid_assa, key=lambda x: x[1])
            ax.axvline(x=peak_e, color="red", ls="--", alpha=0.7,
                       label=f"Peak AssA={peak_a:.2f} @ epoch {peak_e}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Evaluation Metrics per Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Saturation annotation
    saturation_confirmed = False
    if valid_te and epochs_eval:
        valid_assa = [(e, eval_metrics[e]["AssA"])
                      for e in epochs_eval
                      if eval_metrics[e].get("AssA") is not None]
        if valid_assa:
            peak_epoch = max(valid_assa, key=lambda x: x[1])[0]
            post_peak  = [e for e in valid_te if e > peak_epoch]
            id_at_peak = train_metrics.get(peak_epoch, {}).get("id_loss", 99)
            if post_peak:
                post_losses = [train_metrics[e]["id_loss"]
                               for e in post_peak
                               if train_metrics[e].get("id_loss") is not None]
                if post_losses and min(post_losses) < id_at_peak * 0.98:
                    saturation_confirmed = True
                    axes[0].axvline(x=peak_epoch, color="red", ls="--",
                                    alpha=0.7, label=f"AssA peak (epoch {peak_epoch})")
                    axes[0].legend()

    plt.suptitle(
        "Diagnostic 1 — ID Loss Saturation\n"
        + ("SATURATION CONFIRMED: train id_loss keeps falling after AssA peak"
           if saturation_confirmed else
           "Saturation not yet confirmed with current epochs"),
        fontsize=11,
        color="red" if saturation_confirmed else "black",
    )
    plt.tight_layout()
    plt.savefig(od / "diag1_saturation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot: {od}/diag1_saturation.png")

    # ── Save results ──────────────────────────────────────────────────
    interpretation = (
        "SATURATION CONFIRMED — train id_loss continues decreasing after AssA "
        "peak. Closed-set vocabulary memorization is the structural bottleneck."
        if saturation_confirmed else
        "Need more epochs (5+) to confirm saturation. "
        "Current data: train id_loss trend = " + str(id_losses)
    )

    results = {
        "train_id_loss_per_epoch":  {str(e): train_metrics[e]["id_loss"]
                                     for e in sorted(train_metrics.keys())},
        "eval_metrics_per_epoch":   {str(e): eval_metrics[e]
                                     for e in sorted(eval_metrics.keys())},
        "saturation_confirmed":     saturation_confirmed,
        "interpretation":           interpretation,
    }
    with open(od / "diag1_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nRESULTS:")
    print(f"  saturation_confirmed: {saturation_confirmed}")
    print(f"  interpretation: {interpretation}")
    print(f"\nOutputs → {od}/")


if __name__ == "__main__":
    main()