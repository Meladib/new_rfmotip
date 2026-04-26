#!/usr/bin/env python3
"""
Companion plotting script for temporal stability diagnostic results.

Reads temporal_stability_results.json and produces two PNGs:
  1. temporal_stability_histograms.png  — per-pair cosine-similarity histograms overlaid
  2. temporal_stability_linechart.png   — mean cosine similarity per frame-pair over time

Usage:
    python diagnostics/plot_temporal_stability.py \
        [--results diagnostics/temporal_stability_results.json] \
        [--output_dir diagnostics/]
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot temporal stability results from diag_temporal_stability_script.py."
    )
    parser.add_argument(
        "--results",
        default="diagnostics/temporal_stability_results.json",
        help="Path to temporal_stability_results.json (default: diagnostics/temporal_stability_results.json).",
    )
    parser.add_argument(
        "--output_dir",
        default="diagnostics",
        help="Directory to write PNG files (default: diagnostics/).",
    )
    return parser.parse_args()


def plot_histograms(pair_results, output_path, metadata):
    """Produce histogram overlay: one curve per consecutive frame pair."""
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(pair_results)))
    for i, result in enumerate(pair_results):
        h = result["histogram"]
        edges = np.array(h["edges"])
        counts = np.array(h["counts"], dtype=float)
        # Normalize to density so different query counts are comparable.
        total = counts.sum()
        if total > 0:
            counts = counts / total
        centers = (edges[:-1] + edges[1:]) / 2
        ax.plot(centers, counts, marker="o", markersize=3, linewidth=1.5,
                color=colors[i], label=result["pair"].replace("frame_", "").replace("_to_", "→"))

    ax.axvline(x=0.5, color="red", linestyle="--", linewidth=1.0, alpha=0.7, label="instability threshold (0.5)")
    ax.axvline(x=0.9, color="green", linestyle="--", linewidth=1.0, alpha=0.7, label="stable threshold (0.9)")

    ax.set_xlabel("Cosine Similarity (T → T+1)", fontsize=11)
    ax.set_ylabel("Fraction of Queries", fontsize=11)
    ax.set_title(
        f"Query Embedding Temporal Stability — Histograms\n"
        f"layer_index={metadata.get('layer_index', '?')}, "
        f"dec_layers={metadata.get('dec_layers', '?')}, "
        f"hidden_dim={metadata.get('hidden_dim', '?')}",
        fontsize=10,
    )
    ax.set_xlim(-1.0, 1.0)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Histogram plot saved to: {output_path}")


def plot_linechart(pair_results, output_path, metadata):
    """Produce line chart of mean cosine similarity per pair, with ±1 std band."""
    fig, ax = plt.subplots(figsize=(8, 4))

    means = [r["mean"] for r in pair_results]
    stds = [r["std"] for r in pair_results]
    n_unstable = [r["n_unstable"] for r in pair_results]
    n_stable = [r["n_stable"] for r in pair_results]
    x = np.arange(len(pair_results))
    labels = [r["pair"].replace("frame_", "").replace("_to_", "→") for r in pair_results]

    means_arr = np.array(means)
    stds_arr = np.array(stds)

    ax.fill_between(x, means_arr - stds_arr, means_arr + stds_arr,
                    alpha=0.2, color="steelblue", label="±1 std")
    ax.plot(x, means_arr, marker="o", linewidth=2, color="steelblue", label="mean cosine sim")

    ax.axhline(y=0.5, color="red", linestyle="--", linewidth=1.0, alpha=0.8, label="instability threshold (0.5)")
    ax.axhline(y=0.7, color="orange", linestyle="--", linewidth=1.0, alpha=0.8, label="high-instability threshold (0.7)")
    ax.axhline(y=0.9, color="green", linestyle="--", linewidth=1.0, alpha=0.8, label="stable threshold (0.9)")

    # Annotate n_unstable on each point.
    for xi, (m, nu) in enumerate(zip(means, n_unstable)):
        ax.annotate(f"n_unstab={nu}", xy=(xi, m), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=7, color="darkred")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean Cosine Similarity", fontsize=11)
    ax.set_xlabel("Consecutive Frame Pair", fontsize=11)
    ax.set_title(
        f"Query Embedding Temporal Stability — Per-Pair Means\n"
        f"layer_index={metadata.get('layer_index', '?')}, "
        f"dec_layers={metadata.get('dec_layers', '?')}, "
        f"hidden_dim={metadata.get('hidden_dim', '?')}",
        fontsize=10,
    )
    ax.set_ylim(-0.1, 1.1)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Line chart saved to: {output_path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.results):
        raise FileNotFoundError(
            f"Results file not found: {args.results}. "
            f"Run diag_temporal_stability_script.py first."
        )

    with open(args.results) as f:
        data = json.load(f)

    pair_results = data["pair_results"]
    metadata = {
        "layer_index": data.get("layer_index"),
        "dec_layers": data.get("dec_layers"),
        "hidden_dim": data.get("hidden_dim"),
    }

    print(f"[INFO] Loaded {len(pair_results)} pair(s) from {args.results}")

    hist_path = os.path.join(args.output_dir, "temporal_stability_histograms.png")
    line_path = os.path.join(args.output_dir, "temporal_stability_linechart.png")

    plot_histograms(pair_results, hist_path, metadata)
    plot_linechart(pair_results, line_path, metadata)

    agg = data.get("aggregate", {})
    print(f"\n[SUMMARY]")
    print(f"  Mean of pair means: {agg.get('mean_of_means', 'N/A'):.4f}")
    print(f"  Pair mean range:    [{agg.get('min_mean', 'N/A'):.4f}, {agg.get('max_mean', 'N/A'):.4f}]")


if __name__ == "__main__":
    main()
