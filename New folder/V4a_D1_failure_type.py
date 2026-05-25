#!/usr/bin/env python3
"""
V4a_D1_failure_type.py
==================
Diagnostic D1: classify V4a failure as one of:
  - GENERALIZATION GAP  : V4a id_loss < V3 AND V4a AssA < V3
  - TRAINING INTERFERENCE: V4a id_loss > V3 AND V4a AssA < V3
  - NEUTRAL TRAINING    : V4a id_loss ≈ V3 AND V4a AssA < V3

Reads [Finish epoch: N] lines from both log files.
Eval metrics hardcoded from confirmed results.

Run from RF-MOTIPV4 repo root:
    python "New folder/D1_failure_type.py" \
        --v3_log  /data/adib/new/github/RF-MOTIP/outputsV3/rfmotip_dancetrack/train/log.txt \
        --v4_log  outputs/rfmotip_dancetrack_V3_full/train/log.txt \
        --output_dir diagnostics/D1/
"""

import re
import sys
import argparse
import numpy as np
from pathlib import Path


# ── Hardcoded eval results (confirmed across all 25 val sequences) ────────────
V3_ASSA = {
    0: 25.37, 1: 30.36, 2: 32.20, 3: 33.18,
    4: 34.73, 5: 36.30, 6: 38.03, 7: 38.712,
}

V4_ASSA = {
    0: 26.01, 1: 31.46, 2: 34.98, 3: 35.27,
    4: 34.34, 5: 34.35, 6: 36.77, 7: 36.24,
    8: 36.04, 9: 36.25,
}

V3_HOTA = {
    0: 42.36, 1: 46.37, 2: 47.82, 3: 48.53,
    4: 49.64, 5: 50.76, 6: 52.01, 7: 52.477,
}

V4_HOTA = {
    0: 42.85, 1: 47.28, 2: 49.84, 3: 50.11,
    4: 49.32, 5: 49.32, 6: 51.02, 7: 50.81,
    8: 50.63, 9: 50.76,
}
# ──────────────────────────────────────────────────────────────────────────────


def parse_id_loss(log_path: str) -> dict:
    """Parse [Finish epoch: N] lines → {epoch: id_loss}."""
    results = {}
    with open(log_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "Finish epoch" not in line:
                continue
            em = re.search(r"Finish epoch:\s*(\d+)", line)
            im = re.search(r"\bid_loss\s*=\s*([\d.]+)", line)
            if em and im:
                results[int(em.group(1))] = float(im.group(1))
    return results


def classify(delta_id_loss_values, delta_assa_values, tol=0.01):
    """
    delta = V4 - V3 (negative = V4 is better for id_loss, worse for AssA)
    """
    mean_did  = np.mean(delta_id_loss_values)
    mean_dassa = np.mean(delta_assa_values)

    if mean_did < -tol and mean_dassa < 0:
        return "GENERALIZATION_GAP"
    elif mean_did > tol and mean_dassa < 0:
        return "TRAINING_INTERFERENCE"
    elif abs(mean_did) <= tol and mean_dassa < 0:
        return "NEUTRAL_TRAINING"
    else:
        return "UNCLEAR"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v3_log",     required=True)
    p.add_argument("--v4_log",     required=True)
    p.add_argument("--output_dir", default="diagnostics/D1/")
    args = p.parse_args()

    od = Path(args.output_dir)
    od.mkdir(parents=True, exist_ok=True)

    # ── Parse train id_loss ───────────────────────────────────────────────────
    print("Parsing V3 log...")
    v3_loss = parse_id_loss(args.v3_log)
    print(f"  Epochs found: {sorted(v3_loss.keys())}")

    print("Parsing V4a log...")
    v4_loss = parse_id_loss(args.v4_log)
    print(f"  Epochs found: {sorted(v4_loss.keys())}")

    # ── Common epochs only ────────────────────────────────────────────────────
    common_loss  = sorted(set(v3_loss) & set(v4_loss))
    common_eval  = sorted(set(V3_ASSA) & set(V4_ASSA))

    # ── Per-epoch table ───────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("D1 — FAILURE TYPE ANALYSIS")
    print("=" * 75)
    print(f"\n{'Epoch':>6} | {'V3 id_loss':>10} | {'V4 id_loss':>10} | "
          f"{'Δ id_loss':>10} | {'V3 AssA':>8} | {'V4 AssA':>8} | {'Δ AssA':>8}")
    print("-" * 75)

    delta_loss_vals  = []
    delta_assa_vals  = []

    for ep in sorted(set(common_loss) | set(common_eval)):
        v3l  = v3_loss.get(ep)
        v4l  = v4_loss.get(ep)
        v3a  = V3_ASSA.get(ep)
        v4a  = V4_ASSA.get(ep)

        dl   = (v4l - v3l) if (v3l and v4l) else None
        da   = (v4a - v3a) if (v3a is not None and v4a is not None) else None

        if dl is not None:
            delta_loss_vals.append(dl)
        if da is not None:
            delta_assa_vals.append(da)

        print(f"{ep:>6} | "
              f"{v3l if v3l else '—':>10} | "
              f"{v4l if v4l else '—':>10} | "
              f"{f'{dl:+.4f}' if dl is not None else '—':>10} | "
              f"{v3a if v3a else '—':>8} | "
              f"{v4a if v4a else '—':>8} | "
              f"{f'{da:+.2f}' if da is not None else '—':>8}")

    # ── Epoch-by-epoch flip analysis ──────────────────────────────────────────
    print("\n── Where did V4a fall behind? ──")
    for ep in common_eval:
        v3a = V3_ASSA.get(ep)
        v4a = V4_ASSA.get(ep)
        if v3a is None or v4a is None:
            continue
        marker = " ← GAP OPENS" if (v4a < v3a and V4_ASSA.get(ep-1, 999) >= V3_ASSA.get(ep-1, 0)) else ""
        print(f"  Epoch {ep}: V4a AssA={v4a:.2f}  V3 AssA={v3a:.2f}  "
              f"Δ={v4a-v3a:+.2f}{marker}")

    # ── Classification ────────────────────────────────────────────────────────
    # Use epochs 1+ (exclude epoch 0 where warmup dominates)
    post0_loss = [v for i, v in enumerate(delta_loss_vals) if i > 0]
    post0_assa = [v for i, v in enumerate(delta_assa_vals) if i > 0]

    failure_type = classify(post0_loss, post0_assa)
    mean_dl = np.mean(post0_loss)
    mean_da = np.mean(post0_assa)

    print("\n" + "=" * 75)
    print("D1 VERDICT")
    print("=" * 75)
    print(f"\n  Mean Δ id_loss (epochs 1+): {mean_dl:+.4f}  "
          f"({'V4a trains BETTER' if mean_dl < 0 else 'V4a trains WORSE'})")
    print(f"  Mean Δ AssA   (epochs 1+): {mean_da:+.2f}   "
          f"({'V4a evals WORSE' if mean_da < 0 else 'V4a evals BETTER'})")

    print(f"\n  FAILURE TYPE: {failure_type}")

    branch = {
        "GENERALIZATION_GAP": (
            "V4a learns the training task BETTER than V3 "
            "(lower id_loss) but GENERALIZES WORSE (lower AssA).\n"
            "  → reid_proj is overfitting to training sequences.\n"
            "  → NEXT: run D2-GEN — LDA separability on VAL features\n"
            "           at checkpoint_3 (peak) vs checkpoint_6 (collapse)."
        ),
        "TRAINING_INTERFERENCE": (
            "V4a trains WORSE than V3 despite having reid_proj.\n"
            "  → reid_proj is interfering with the ID optimization itself.\n"
            "  → NEXT: run D2-TRAIN — gradient alignment between\n"
            "           id_loss and reid_proj gradients."
        ),
        "NEUTRAL_TRAINING": (
            "V4a trains at same level as V3 — reid_proj adds no signal.\n"
            "  → The projection learned a near-zero or degenerate transform.\n"
            "  → NEXT: inspect reid_proj weight spectrum at each checkpoint."
        ),
        "UNCLEAR": (
            "Pattern does not match expected failure modes.\n"
            "  → Inspect per-epoch table manually."
        ),
    }[failure_type]

    print(f"\n  {branch}")
    print("\n" + "=" * 75)

    # ── Save result ───────────────────────────────────────────────────────────
    result = {
        "failure_type":     failure_type,
        "mean_delta_id_loss_post0": float(mean_dl),
        "mean_delta_assa_post0":    float(mean_da),
        "per_epoch": {
            ep: {
                "v3_id_loss": v3_loss.get(ep),
                "v4_id_loss": v4_loss.get(ep),
                "delta_id_loss": (v4_loss[ep] - v3_loss[ep])
                                  if ep in v3_loss and ep in v4_loss else None,
                "v3_assa": V3_ASSA.get(ep),
                "v4_assa": V4_ASSA.get(ep),
                "delta_assa": (V4_ASSA[ep] - V3_ASSA[ep])
                               if ep in V3_ASSA and ep in V4_ASSA else None,
            }
            for ep in sorted(set(common_loss) | set(common_eval))
        }
    }

    import json
    out_path = od / "D1_result.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Result saved to {out_path}")


if __name__ == "__main__":
    main()