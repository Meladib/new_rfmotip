# DIAG_TEMPORAL_STABILITY

## 1. FINDINGS

- **What the script measures:** Per-query cosine similarity of RF-DETR decoder output embeddings between consecutive frames `T` and `T+1`. For each query slot `k`, it computes `cos_sim(hs[layer][k, T], hs[layer][k, T+1])`. A similarity near 1.0 means the decoder produces nearly the same embedding for the same query slot in adjacent frames. A similarity near 0 or negative means the query's content is essentially random between frames.

- **Why this matters for MOTIP:** MOTIP's ID decoder relies on `trajectory_features` — embeddings gathered via Hungarian matching from `hs[-1]`. If the embedding for the same physical object shifts drastically between frames (low cosine similarity), the ID decoder's cross-attention cannot build a consistent trajectory model. AssA=20.5 (frozen experiment) directly reflects this: the detector can localize objects but the embeddings are not identity-consistent.

- **What it does not measure:** The script measures query-slot-level stability (same slot index across frames). Because RF-DETR re-ranks query slots per frame via top-K encoder selection, the matched query slot for a given object may differ between frames. The script therefore captures an upper bound on content stability — if even same-slot embeddings are unstable, object-matched embeddings will be at least as unstable.

- **Axis-5 relates to Blocker B5** from the investigation plan: embedding discriminability requires a runtime trace. This script provides that trace without modifying any source files. It uses a `register_forward_hook` on `model.transformer` to capture the full stacked `hs` tensor (`[dec_layers, B, num_queries, hidden_dim]`) before the lwdetr.py code discards all but `hs[-1]`.

## 2. CLI COMMANDS

### Basic run (final decoder layer, RF-DETR pre-trained checkpoint):

```bash
python diagnostics/diag_temporal_stability_script.py \
    --checkpoint rfdetr_dancetrack_motip/checkpoint_best_total.pth \
    --sequence_dir /data/DanceTrack/val/dancetrack0001 \
    --layer_index -1 \
    --output_dir diagnostics/
```

### Probe all 3 decoder layers (run three times):

```bash
# Layer 0 (first decoder layer)
python diagnostics/diag_temporal_stability_script.py \
    --checkpoint rfdetr_dancetrack_motip/checkpoint_best_total.pth \
    --sequence_dir /data/DanceTrack/val/dancetrack0001 \
    --layer_index 0 \
    --output_dir diagnostics/layer0/

# Layer 1 (second decoder layer — aux_outputs[-1])
python diagnostics/diag_temporal_stability_script.py \
    --checkpoint rfdetr_dancetrack_motip/checkpoint_best_total.pth \
    --sequence_dir /data/DanceTrack/val/dancetrack0001 \
    --layer_index 1 \
    --output_dir diagnostics/layer1/

# Layer 2 / final layer
python diagnostics/diag_temporal_stability_script.py \
    --checkpoint rfdetr_dancetrack_motip/checkpoint_best_total.pth \
    --sequence_dir /data/DanceTrack/val/dancetrack0001 \
    --layer_index 2 \
    --output_dir diagnostics/layer2/
```

### Run on a MOTIP resume checkpoint (post-training weights):

```bash
python diagnostics/diag_temporal_stability_script.py \
    --checkpoint rfdetr_dancetrack_motip/checkpoint_best_total.pth \
    --deformable_checkpoint outputs/rfmotip_dancetrack/checkpoint_4.pth \
    --sequence_dir /data/DanceTrack/val/dancetrack0001 \
    --layer_index -1 \
    --output_dir diagnostics/epoch4/
```

### Generate plots from results:

```bash
python diagnostics/plot_temporal_stability.py \
    --results diagnostics/temporal_stability_results.json \
    --output_dir diagnostics/
```

## 3. INTERPRETATION THRESHOLDS

| Mean cosine similarity (across 5 pairs) | Interpretation |
|---|---|
| < 0.5 | **Severe instability** — embeddings are effectively random between frames; ID decoder has no signal |
| 0.5 – 0.7 | **High instability** — significant frame-to-frame drift; ID decoder will train on noisy targets |
| 0.7 – 0.9 | **Moderate instability** — some consistency but insufficient for robust re-ID |
| > 0.9 | **Stable** — embeddings are consistent; instability is not the primary source of AssA degradation |

**Expected result based on static analysis:** Mean cosine similarity likely falls in the 0.3–0.6 range for `layer_index=-1` on the pre-trained RF-DETR checkpoint, based on:
1. NAS weight sharing: decoder layer 2 was not specialized for final-layer semantic consistency.
2. Per-frame encoder top-K re-ranking: different encoder memory tokens initialize queries each frame.
3. AssA=20.5 in the frozen experiment implies the features are not identity-consistent even with a trained detector.

**Layer comparison interpretation:**
- If layer 0 > layer 2: instability increases with depth → NAS specialization artifact; the first layer is more consistent because it's used in every NAS sub-network.
- If layer 2 > layer 0: depth refinement adds consistency → the 3-layer config is appropriate; instability comes from elsewhere.
- If all layers similar: the encoder re-ranking is the dominant source, not depth.

## 4. HOW TO PROBE ALL 3 DECODER LAYERS

RF-DETR-small uses `dec_layers=3` at inference (confirmed from paper Table 7). The three decoder layers are indexed 0, 1, 2 (equivalently: -3, -2, -1).

Layer 1 is the layer whose Hungarian matching indices are returned by `SetCriterion.forward()` when `aux_loss=True` and `two_stage=False` — this is the layer whose indices MOTIP actually uses to gather embeddings from `hs[-1]`. Comparing the stability of layer 1 vs layer 2 directly quantifies the severity of the stale-indices bug (Axis 4 finding): if layer 1 embeddings are very different from layer 2 embeddings for the same objects, the wrong features are being fed to the ID decoder.

## 5. FINDINGS (OPEN QUESTIONS REQUIRING RUNTIME)

- What is the actual mean cosine similarity at `layer_index=-1`? This determines whether temporal instability is the primary cause of AssA=20.5 or a secondary factor.
- Is there a per-layer stability gradient (layer 0 vs 1 vs 2)? This would confirm or refute the NAS weight sharing hypothesis.
- Does post-training (epoch 4 checkpoint) show lower or higher stability than the pre-trained checkpoint? Lower stability would confirm that joint MOTIP training actively corrupts feature consistency.
- Are embeddings for high-confidence detections more stable than low-confidence ones? This would point to encoder confidence rank as a stability predictor.

## 6. SEVERITY ASSESSMENT

**High** (pending runtime confirmation)

The static analysis from Axes 1–4 predicts severe instability. If runtime confirms mean cosine similarity < 0.7, temporal instability is a co-primary cause of HOTA drop from ~70 to 23.4, independent of the matcher indices bug.

## 7. HYPOTHESIS RANKING

\#1 [Confidence: High] — Mean cosine similarity at `layer_index=-1` will be < 0.7, consistent with the AssA=20.5 frozen baseline; NAS weight sharing and per-frame top-K reranking both contribute.

\#2 [Confidence: Med] — Layer 0 will show higher stability than layer 2 because layer 0 weights are included in every NAS sub-network and receive the most regularization pressure toward consistent representations.

\#3 [Confidence: Med] — The `deformable_checkpoint` run (post-epoch-4) will show lower stability than the pre-trained checkpoint, confirming that joint MOTIP training degrades the backbone feature consistency.
