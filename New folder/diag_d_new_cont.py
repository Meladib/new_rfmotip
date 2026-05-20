#!/usr/bin/env python3
"""
diag_d_new_cont.py  —  D-NEW-CONT
====================================
Answers: Is CONTRASTIVE_WEIGHT: 0.1 justified over the current 0.05?

Method
------
1. Parse V3 training log for per-step id_loss and con_loss values.
2. Compute effective gradient ratio per step:
     ratio(t) = (con_loss(t) * weight) / id_loss(t)
   This approximates the relative gradient magnitude of the
   contrastive signal vs the primary ID signal.
3. Identify whether ratio stays below a meaningful threshold (0.05)
   throughout training — confirming contrastive is too weak.
4. Estimate what weight would produce ratio ≈ 0.1 (meaningful signal).
5. Check con_pos_sim convergence — if it never exceeds 0.75,
   the loss did not converge.

Scientific justification
------------------------
The contrastive loss is designed as a REGULARIZER (secondary signal).
id_loss weight = 1.0. At weight=0.05:
  ratio = con_loss * 0.05 / id_loss

If this ratio is consistently < 0.02, the contrastive gradient is
negligible — effectively not training the projection head at all.
In that case, weight=0.1 is justified to make the signal meaningful.

If ratio is already > 0.05 at weight=0.05, then the non-convergence
of con_pos_sim has a different cause (e.g., temperature too high,
projection head capacity, positive pair quality) and simply doubling
the weight may not help.

Pass condition (weight=0.1 confirmed):
  mean_ratio < 0.03 AND max_con_pos_sim < 0.75

Fail condition (investigate other causes first):
  mean_ratio > 0.05 OR max_con_pos_sim > 0.75

Run from repo root:
  python diagnostics/diag_d_new_cont.py \\
    --log_dir outputs/rfmotip_dancetrack_ctsv/train/ \\
    --output_dir diagnostics/diag_cont_results/

The script looks for log.txt or training_log.txt in log_dir.
"""

import os
import sys
import re
import json
import argparse
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--log_dir",    required=True,
                   help="Training log directory (contains log.txt)")
    p.add_argument("--output_dir", default="diagnostics/diag_cont_results/")
    p.add_argument("--weight",     type=float, default=0.05,
                   help="Contrastive weight used in training (default: 0.05)")
    p.add_argument("--target_ratio", type=float, default=0.1,
                   help="Target gradient ratio (what weight should produce)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# LOG PARSING
# ─────────────────────────────────────────────────────────────
def find_log_file(log_dir: str) -> str:
    """Find training log file in the given directory."""
    for name in ["log.txt", "training_log.txt", "train.log", "output.log"]:
        path = os.path.join(log_dir, name)
        if os.path.exists(path):
            return path
    # Try any .txt or .log file
    for f in Path(log_dir).glob("*.txt"):
        return str(f)
    for f in Path(log_dir).glob("*.log"):
        return str(f)
    raise FileNotFoundError(f"No log file found in {log_dir}")


def parse_log(log_path: str) -> dict:
    """
    Parse training log for per-step metrics.
    Handles the MOTIP logger format:
      [Epoch: X] [Step Y/Z] id_loss: A  con_loss: B  con_pos_sim: C ...
    Returns dict with lists keyed by metric name, plus 'epoch' and 'step'.
    """
    records = []

    # Patterns to try
    patterns = [
        # Format: key: value pairs on one line
        r"(?:Epoch[:\s]+(\d+))?.*?(?:step[:\s]+(\d+))?.*?id_loss[:\s]+([\d.]+).*?con_loss[:\s]+([\d.]+).*?con_pos_sim[:\s]+([\d.]+)",
        # Format: JSON-like
        r'"id_loss":\s*([\d.]+).*?"con_loss":\s*([\d.]+).*?"con_pos_sim":\s*([\d.]+)',
    ]

    # MOTIP log format:
    # [Metrics] [Epoch: X] [step/total] ... id_loss = V (V); con_loss = V (V); con_pos_sim = V (V); ...

    _pat_epoch  = re.compile(r"\[Epoch:\s*(\d+)\]")
    _pat_step   = re.compile(r"\[(\d+)/\d+\]")
    _pat_metric = re.compile(r"\b(\w+)\s*=\s*([\d.eE+\-]+)\s*\(")

    step_abs = 0
    with open(log_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "id_loss" not in line or "con_loss" not in line:
                continue

            em = _pat_epoch.search(line)
            sm = _pat_step.search(line)
            epoch    = int(em.group(1)) if em else None
            step_abs += 1

            # Parse all key=value pairs on the line
            vals = {m.group(1): float(m.group(2)) for m in _pat_metric.finditer(line)}

            if "id_loss" in vals and "con_loss" in vals:
                records.append({
                    "epoch":       epoch,
                    "step":        int(sm.group(1)) if sm else step_abs,
                    "id_loss":     vals["id_loss"],
                    "con_loss":    vals["con_loss"],
                    "con_pos_sim": vals.get("con_pos_sim"),
                    "con_warmup":  vals.get("con_warmup"),
                })

    if not records:
        raise ValueError(
            f"Could not parse id_loss/con_loss from {log_path}. "
            "Check log format — expected lines containing 'id_loss: X  con_loss: Y'."
        )

    print(f"  Parsed {len(records)} steps from {log_path}")
    return records


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    args = get_args()
    od = Path(args.output_dir)
    od.mkdir(parents=True, exist_ok=True)

    print(f"Looking for log in: {args.log_dir}")
    log_path = find_log_file(args.log_dir)
    print(f"Found: {log_path}\n")

    records = parse_log(log_path)

    # ── Extract arrays ────────────────────────────────────────────────
    steps       = np.array([r["step"]       for r in records])
    epochs      = np.array([r["epoch"] if r["epoch"] is not None else -1
                            for r in records])
    id_losses   = np.array([r["id_loss"]    for r in records], dtype=np.float32)
    con_losses  = np.array([r["con_loss"]   for r in records], dtype=np.float32)
    pos_sims    = np.array([r["con_pos_sim"] if r["con_pos_sim"] is not None else np.nan
                            for r in records], dtype=np.float32)
    warmups     = np.array([r["con_warmup"] if r["con_warmup"] is not None else np.nan
                            for r in records], dtype=np.float32)

    # ── Compute effective gradient ratio ──────────────────────────────
    # ratio(t) = con_loss(t) * weight / id_loss(t)
    # After warmup scaling:
    if not np.all(np.isnan(warmups)):
        # Use warmup-scaled ratio
        ws = np.where(np.isnan(warmups), 1.0, warmups)
        ratio = (con_losses * args.weight * ws) / np.maximum(id_losses, 1e-6)
        print("  Using warmup-scaled gradient ratio.")
    else:
        ratio = (con_losses * args.weight) / np.maximum(id_losses, 1e-6)
        print("  Using unscaled gradient ratio (no warmup data found).")

    # Post-warmup only (warmup period can distort)
    warmup_steps = 500  # CONTRASTIVE_WARMUP default
    post_warmup_mask = steps > warmup_steps

    # ── Summary statistics ────────────────────────────────────────────
    mean_ratio_all   = float(np.mean(ratio))
    mean_ratio_post  = float(np.mean(ratio[post_warmup_mask])) if post_warmup_mask.any() else mean_ratio_all
    max_ratio        = float(np.max(ratio))

    max_pos_sim      = float(np.nanmax(pos_sims)) if not np.all(np.isnan(pos_sims)) else np.nan
    final_pos_sim    = float(np.nanmean(pos_sims[-20:])) if len(pos_sims) > 20 else np.nan

    # Estimate what weight would achieve target ratio
    if mean_ratio_post > 1e-6:
        suggested_weight = args.target_ratio / (mean_ratio_post / args.weight)
        suggested_weight = round(suggested_weight, 3)
    else:
        suggested_weight = args.target_ratio

    # ── Per-epoch summary ─────────────────────────────────────────────
    unique_epochs = sorted(set(epochs[epochs >= 0]))
    epoch_stats   = {}
    for ep in unique_epochs:
        mask = epochs == ep
        epoch_stats[int(ep)] = {
            "mean_id_loss":  float(np.mean(id_losses[mask])),
            "mean_con_loss": float(np.mean(con_losses[mask])),
            "mean_ratio":    float(np.mean(ratio[mask])),
            "mean_pos_sim":  float(np.nanmean(pos_sims[mask])),
        }

    # ── Print report ──────────────────────────────────────────────────
    print("=" * 65)
    print("D-NEW-CONT — Contrastive Weight Analysis")
    print("=" * 65)
    print(f"\n  Current weight:          {args.weight}")
    print(f"  Total steps parsed:      {len(records)}")
    print(f"  Post-warmup steps:       {post_warmup_mask.sum()}")
    print()
    print(f"  Mean gradient ratio (all):        {mean_ratio_all:.4f}")
    print(f"  Mean gradient ratio (post-warmup): {mean_ratio_post:.4f}")
    print(f"  Max  gradient ratio:              {max_ratio:.4f}")
    print()
    print(f"  Max con_pos_sim reached:          {max_pos_sim:.4f}")
    print(f"  Final con_pos_sim (last 20 steps): {final_pos_sim:.4f}")
    print()
    print(f"  Target ratio:                     {args.target_ratio}")
    print(f"  Suggested weight to reach target: {suggested_weight}")

    print(f"\n  Per-epoch summary:")
    print(f"  {'Epoch':>6}  {'id_loss':>8}  {'con_loss':>9}  "
          f"{'ratio':>7}  {'pos_sim':>8}")
    print("  " + "-" * 50)
    for ep, s in sorted(epoch_stats.items()):
        print(f"  {ep:>6}  {s['mean_id_loss']:>8.4f}  "
              f"{s['mean_con_loss']:>9.4f}  "
              f"{s['mean_ratio']:>7.4f}  "
              f"{s['mean_pos_sim']:>8.4f}")

    # ── Verdict ───────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("D-NEW-CONT VERDICT")
    print(f"{'='*65}")

    converged = max_pos_sim > 0.75 if not np.isnan(max_pos_sim) else False

    if mean_ratio_post < 0.03 and not converged:
        verdict = "PASS"
        conclusion = (
            f"Mean gradient ratio = {mean_ratio_post:.4f} < 0.03 and "
            f"con_pos_sim max = {max_pos_sim:.3f} < 0.75 (did not converge). "
            f"Contrastive signal is too weak at weight={args.weight}. "
            f"Suggested weight: {suggested_weight} to achieve ratio ≈ {args.target_ratio}. "
            f"CONTRASTIVE_WEIGHT: {min(0.1, suggested_weight)} CONFIRMED."
        )
    elif converged:
        verdict = "FAIL — already converged"
        conclusion = (
            f"con_pos_sim reached {max_pos_sim:.3f} > 0.75. "
            "The contrastive loss DID converge. Non-convergence diagnosis "
            "was incorrect or the log showed partial training. "
            "Increasing weight is NOT the fix. "
            "Investigate temperature (CONTRASTIVE_TEMP) or positive pair quality instead."
        )
    elif mean_ratio_post > 0.05:
        verdict = "FAIL — ratio adequate"
        conclusion = (
            f"Mean gradient ratio = {mean_ratio_post:.4f} > 0.05. "
            "Contrastive signal is already meaningful. "
            "Non-convergence of con_pos_sim has a different cause. "
            "Investigate temperature or positive pair diversity, not weight."
        )
    else:
        verdict = "MARGINAL"
        conclusion = (
            f"Mean ratio = {mean_ratio_post:.4f} (borderline). "
            f"con_pos_sim max = {max_pos_sim:.3f}. "
            f"Weight increase to {suggested_weight} may help but effect is uncertain."
        )

    print(f"\n  RESULT: {verdict}")
    print(f"  {conclusion}")

    # ── Plots ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ax = axes[0, 0]
    ax.semilogy(steps, id_losses, alpha=0.4, color="steelblue", label="id_loss")
    ax.semilogy(steps, con_losses, alpha=0.4, color="orange", label="con_loss")
    ax.axvline(warmup_steps, color="grey", ls="--", lw=1, label=f"warmup={warmup_steps}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("id_loss vs con_loss (raw)")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(steps, ratio, alpha=0.4, color="red", lw=0.5)
    ax.axhline(args.target_ratio, color="green", ls="--",
               label=f"target ratio={args.target_ratio}")
    ax.axhline(mean_ratio_post, color="red", ls="--",
               label=f"mean post-warmup={mean_ratio_post:.4f}")
    ax.axvline(warmup_steps, color="grey", ls="--", lw=1)
    ax.set_xlabel("Step")
    ax.set_ylabel("Gradient ratio")
    ax.set_title("Effective Gradient Ratio = con_loss*weight / id_loss")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    valid_sim = ~np.isnan(pos_sims)
    if valid_sim.any():
        ax.plot(steps[valid_sim], pos_sims[valid_sim], alpha=0.5,
                color="green", lw=0.8)
        ax.axhline(0.75, color="red", ls="--", label="convergence threshold=0.75")
        ax.axhline(max_pos_sim, color="green", ls="--",
                   label=f"max achieved={max_pos_sim:.3f}")
    ax.set_xlabel("Step")
    ax.set_ylabel("con_pos_sim")
    ax.set_title("Positive Pair Cosine Similarity (projection space)")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    eps = sorted(epoch_stats.keys())
    ax.bar([e - 0.2 for e in eps],
           [epoch_stats[e]["mean_ratio"] for e in eps],
           0.35, label="gradient ratio", color="red", alpha=0.7)
    ax2 = ax.twinx()
    ax2.plot(eps, [epoch_stats[e]["mean_pos_sim"] for e in eps],
             "go-", ms=6, lw=2, label="pos_sim")
    ax.axhline(args.target_ratio, color="green", ls="--", lw=1.5,
               label=f"target={args.target_ratio}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient ratio", color="red")
    ax2.set_ylabel("con_pos_sim", color="green")
    ax.set_title("Per-Epoch: Gradient Ratio + pos_sim")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)

    plt.suptitle(
        f"D-NEW-CONT: Contrastive Weight Analysis\n"
        f"weight={args.weight}  mean_ratio={mean_ratio_post:.4f}  "
        f"max_pos_sim={max_pos_sim:.3f}  →  {verdict}",
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig(od / "d_new_cont_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Save JSON ─────────────────────────────────────────────────────
    result = {
        "current_weight":           args.weight,
        "target_ratio":             args.target_ratio,
        "n_steps_parsed":           len(records),
        "mean_ratio_all":           round(mean_ratio_all,  5),
        "mean_ratio_post_warmup":   round(mean_ratio_post, 5),
        "max_ratio":                round(max_ratio,       5),
        "max_con_pos_sim":          round(float(max_pos_sim), 4)
                                    if not np.isnan(max_pos_sim) else None,
        "final_con_pos_sim":        round(float(final_pos_sim), 4)
                                    if not np.isnan(final_pos_sim) else None,
        "suggested_weight":         suggested_weight,
        "verdict":                  verdict,
        "conclusion":               conclusion,
        "per_epoch":                epoch_stats,
    }
    with open(od / "d_new_cont_results.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved to {od}/")
    print("  d_new_cont_analysis.png")
    print("  d_new_cont_results.json")


if __name__ == "__main__":
    main()