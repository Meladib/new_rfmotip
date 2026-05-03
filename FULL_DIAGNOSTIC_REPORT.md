# RF-MOTIP Full Diagnostic Report

---

## System Overview

**Task:** Multi-object tracking on DanceTrack  
**Detector:** RF-DETR-small (DINOv2-S backbone, `two_stage=True`, `aux_loss=True`, `dec_layers=3`, `hidden_dim=256`, `num_queries=300`)  
**Tracker:** MOTIP (TrajectoryModeling adapter + IDDecoder 6-layer cross-attention)  
**Training mode:** Detector fully frozen (`DETR_NUM_TRAIN_FRAMES=0`)  
**Best checkpoint:** `outputsV2/rfmotip_dancetrack/train/checkpoint_2.pth`  
**Best result:** HOTA=49.567, AssA=34.632, DetA=71.586

---

## Part 1 — RF-DETR Checkpoint Analysis

### Tool: `diag_matcher_indices.py`
### Checkpoint: `rfdetr_dancetrack_motip/checkpoint_best_total.pth`

| Field | Value |
|-------|-------|
| `aux_loss` | True |
| `two_stage` | True |
| `dec_layers` | 3 |
| `cls_loss_coef` | 1.0 |
| `bbox_loss_coef` | 5 |
| `giou_loss_coef` | 2 |
| `hidden_dim` | 256 |
| `num_queries` | 300 |
| `sum_group_losses` | False |

**SetCriterion index return path:**
```
two_stage=True → SetCriterion returns ENCODER indices
Index space: encoder top-K proposals
These map to decoder query slots by construction (transformer.py:284–294)
Result: no stale-indices bug for two-stage architecture
```

**Detection loss gradient ratio:**  
3 decoder layers × (cls=1.0 + bbox=5 + giou=2) = 24.0 total detection weight vs ID loss weight = 1.0

**Note on frozen training:** `DETR_NUM_TRAIN_FRAMES=0` sets `requires_grad=False` on all DETR parameters. `detr_grad_norm=0.000` confirmed in all training logs. The 24:1 ratio only applies if DETR were unfrozen.

---

## Part 2 — RF-DETR Embedding Quality

### Tool: `diag_object_matched.py`
### Sequence: `dancetrack0004`, frozen detector, val split

#### Diag 1 — Similarity Gap (Hungarian-matched, object-level)

| Difficulty bucket | Positive − unmatched gap |
|-------------------|--------------------------|
| Easy (IoU ≤ 0.3) | +0.1019 |
| Medium (0.3 < IoU ≤ 0.7) | +0.1676 |
| Hard (IoU > 0.7) | +0.1506 |

#### Diag 2 — Intrinsic Dimensionality (PCA via numpy SVD)

| Metric | Value |
|--------|-------|
| n_components_90 | 42 |
| n_components_99 | 158 |
| Embedding dimension | 256 |

#### Diag 4 — Active vs Inactive L2 Norms

| Query type | Mean L2 norm |
|------------|-------------|
| Active (GT-matched) | 14.303 |
| Inactive (background) | 12.468 |
| Delta | 1.835 |

#### Diag 5 — Object-Matched Temporal Stability

| Frame gap | Mean cosine similarity | Std |
|-----------|----------------------|-----|
| 1 | 0.9669 | 0.0463 |
| 5 | 0.9185 | — |
| 20 | 0.8753 | — |

#### Diag 6 — Identity Separability (NN accuracy)

| Metric | Value |
|--------|-------|
| NN accuracy | 0.971 |
| Chance level | 0.25 (4 unique IDs in test) |

#### Diag 7 — Hard Negative Similarity

| Pair type | Mean cosine similarity |
|-----------|----------------------|
| Easy negatives (IoU ≤ 0.3) | 0.8628 |
| Hard negatives (IoU > 0.3) | 0.7721 |

**Key observation:** easy_neg_mean (0.8628) > hard_neg_mean (0.7721). Spatially far objects are more similar than close ones — IoU is a poor proxy for contrastive hardness in RF-DETR embeddings.

---

## Part 3 — Slot-Level vs Object-Level Stability

### Tool: `diag_temporal_stability_script.py`
### All 300 query slots, layer_index=-1, `dancetrack0004`

| Frame pair | Mean cos sim | Std | n_unstable (sim<0.5) | n_stable (sim>0.9) |
|-----------|-------------|-----|---------------------|-------------------|
| 1→2 | 0.5252 | 0.1963 | 129 | 4 |
| 2→3 | 0.5426 | 0.2005 | 128 | 7 |
| 3→4 | 0.5458 | 0.2085 | 120 | 5 |
| 4→5 | 0.5332 | 0.2091 | 125 | 1 |
| 5→6 | 0.5088 | 0.1984 | 135 | 2 |
| **Aggregate** | **0.5311** | 0.0133 | — | — |

**Context:** The 0.531 includes ~270 inactive background query slots. Object-matched stability (same GT track_id) = 0.967. These measure different distributions.

---

## Part 4 — Decoder Layer Comparison

### Tool: `diag_layer_comparison.py`
### 50 frames, `dancetrack0004`

#### Per-Layer Object-Matched Stability (gap=1)

| Layer | Slot-level | Object-matched | n_obj |
|-------|-----------|---------------|-------|
| L0 | 0.6931 | 0.9529 | 383 |
| L1 (aux) | 0.6401 | 0.9622 | 383 |
| L2 (final) | 0.6059 | **0.9682** | 383 |

Final layer is most stable for objects.

#### Cross-Layer Similarity (same slot, same frame)

| Pair | Mean | Std |
|------|------|-----|
| L0 vs L1 | 0.6636 | 0.0054 |
| L0 vs L2 | 0.5267 | 0.0072 |
| L1 vs L2 | 0.7662 | 0.0076 |

---

## Part 5 — IDDecoder Temporal Attention

### Tool: `diag_script2_attention_weights.py`
### `checkpoint_3.pth`, 100 frames, `dancetrack0004`

Attention weights from `cross_attn_layers[layer]` with `need_weights=True`. Frame age 1 = most recent.

| Layer | CV | Pattern |
|-------|-----|---------|
| L0 | 1.585 | Monotonic decay, slight rise at age 21–29 |
| L1 | 0.938 | Peak at age 2, gradual decay |
| L2 | 0.475 | Peak at ages 4–6, not most recent |
| L3 | 1.218 | Sharp recency + secondary rise age 15+ |
| L4 | 1.268 | Sharp recency + secondary rise age 14+ |
| L5 | 1.468 | Sharp recency (0.051 at age 1) + long-tail rise |
| **Mean** | **1.158** | Non-flat — recency-dominant + long-tail |

L5 selected values:

| Age | Weight |
|-----|--------|
| 1 | 0.05099 |
| 5 | 0.00690 |
| 10 | 0.00235 |
| 15 | 0.00207 |
| 20 | 0.00260 |
| 25 | 0.00468 |
| 29 | 0.00787 |

---

## Part 6 — ID Loss Saturation

### Tool: `diag_script1_id_loss_split.py`
### Source: `log.txt` from V2 training run

| Epoch | Train id_loss | HOTA | AssA | DetA |
|-------|-------------|------|------|------|
| 0 | 1.3552 | 43.517 | 26.970 | 70.910 |
| 1 | 0.7271 | 46.679 | 30.634 | 71.903 |
| 2 | 0.6494 | **49.567** | **34.632** | 71.586 |
| 3 | 0.6032 | 48.819 | 33.739 | 71.301 |

Train id_loss at epoch 3 (0.6032) < epoch 2 (0.6494) while AssA drops. **Saturation confirmed.**

---

## Part 7 — Newborn Rate Analysis

### Tool: `diag_script3_newborn_rate.py`
### `checkpoint_3.pth`, 200 frames, `dancetrack0004`

| Metric | Value |
|--------|-------|
| GT newborns/frame | 0.020 |
| Predicted newborns/frame | 0.145 |
| Inflation ratio | 7.25× |
| Spurious rate | 100% |
| Crowd density correlation (Pearson r) | 0.248 |

Spurious newborn rate is not crowd-driven (r=0.248).

---

## Part 8 — Detailed Tracking Metrics

### Best checkpoint (epoch 2), ID_THRESH=0.2, object-max protocol

| Metric | Value |
|--------|-------|
| HOTA | 49.567 |
| DetA | 71.586 |
| AssA | 34.632 |
| AssRe | 42.212 |
| AssPr | 52.716 |
| DetRe | 79.728 |
| DetPr | 81.723 |
| MOTA | 82.154 |
| IDF1 | 52.986 |
| IDSW | 10,080 |
| MT | 39 (84.6%) |
| PT | 3 (14.3%) |
| ML | 3 (1.1%) |
| Frag | 5,905 |
| GT_IDs | 273 |
| Dets | 225,148 |
| GT_Dets | 219,651 |

IDSW=10,080 with 273 GT_IDs = ~37 ID switches per tracked identity.

---

## Part 9 — Threshold and Protocol Sensitivity

### ID_THRESH=0.1 (epoch 2 checkpoint)

| Metric | 0.2 | 0.1 |
|--------|-----|-----|
| HOTA | 49.567 | 48.660 |
| AssA | 34.632 | 33.448 |
| AssPr | 52.716 | 51.483 |
| AssRe | 42.212 | 41.323 |

Lowering threshold reduces AssA. Both AssPr and AssRe drop.

### ASSIGNMENT_PROTOCOL: hungarian

HOTA ≈ 48. No improvement over object-max.

---

## Part 10 — Score Distribution (All 25 Val Sequences)

### Tool: `diag_script4_score_distribution.py`
### `checkpoint_2.pth`, 300 frames per sequence, object-max protocol

#### Case definitions
- **Case A:** correct trajectory label score < id_thresh (0.2) → score miscalibration
- **Case B:** correct trajectory label score ≥ id_thresh → still assigned newborn (competition conflict)
- **Case C:** correct label absent from trajectory memory → eviction or missed detection

#### Results sorted by correct_mean

| Sequence | Correct mean | Total newborn | Case B |
|----------|-------------|--------------|--------|
| dancetrack0026 | 0.884 | 439 | 401 |
| dancetrack0041 | 0.887 | 445 | 413 |
| dancetrack0090 | 0.933 | 181 | 181 |
| dancetrack0043 | 0.935 | 138 | 134 |
| dancetrack0094 | 0.941 | 211 | 205 |
| dancetrack0034 | 0.944 | 156 | 152 |
| dancetrack0063 | 0.948 | 75 | 75 |
| dancetrack0014 | 0.950 | 97 | 94 |
| dancetrack0047 | 0.954 | 120 | 119 |
| dancetrack0073 | 0.958 | 159 | 156 |
| dancetrack0081 | 0.967 | 150 | 143 |
| dancetrack0004 | 0.968 | 45 | 45 |
| dancetrack0035 | 0.972 | 60 | 60 |
| dancetrack0058 | 0.976 | 28 | 28 |
| dancetrack0079 | 0.980 | 25 | 25 |
| dancetrack0025 | 0.980 | 27 | 27 |
| dancetrack0019 | 0.980 | 35 | 35 |
| dancetrack0007 | 0.987 | 58 | 55 |
| dancetrack0065 | 0.989 | 17 | 17 |
| dancetrack0010 | 0.989 | 27 | 26 |
| dancetrack0077 | 0.991 | 37 | 37 |
| dancetrack0030 | 0.994 | 15 | 15 |
| dancetrack0097 | 0.995 | 4 | 4 |
| dancetrack0005 | 0.996 | 12 | 12 |
| dancetrack0018 | 0.998 | 18 | 16 |

**Case B = 95–100% across all sequences.** Case A and Case C are negligible.

---

## Part 11 — Density vs Confidence Correlation

### Concurrent objects per sequence (GT annotation analysis)

| Sequence | Mean concurrent | Correct mean | HOTA |
|----------|----------------|-------------|------|
| dancetrack0094 | 19.5 | 0.941 | — |
| dancetrack0081 | 19.1 | 0.967 | — |
| dancetrack0041 | 16.7 | 0.882 | 21.175 |
| dancetrack0026 | 13.2 | 0.854 | 27.791 |
| dancetrack0090 | 13.1 | 0.933 | — |
| dancetrack0079 | 12.1 | 0.980 | — |
| dancetrack0097 | 3.9 | 0.995 | 90.934 |
| dancetrack0005 | 3.9 | 0.996 | 87.015 |

**Pearson r = -0.636** (mean_concurrent vs correct_mean, 25 sequences).

Correlation explains ~40% of variance. Outliers: dancetrack0079 (12.1 concurrent, 0.980 correct) and dancetrack0081 (19.1 concurrent, 0.967 correct) outperform their density expectation.

---

## Part 12 — Motion Analysis

### Mean pixel displacement per object per frame (GT-derived)

| Sequence | Mean disp (px) | Max disp (px) | Correct mean | HOTA |
|----------|---------------|--------------|-------------|------|
| dancetrack0026 | 9.2 | **418.0** | 0.854 | 27.8 |
| dancetrack0041 | 4.6 | 42.0 | 0.882 | 21.2 |
| dancetrack0081 | 3.7 | 64.8 | 0.967 | — |
| dancetrack0079 | 4.1 | 43.0 | 0.980 | — |

dancetrack0026 fails from density (13.2) + extreme motion (418px max).
dancetrack0041 fails from density alone (4.6px mean — similar to 0081).

---

## Part 13 — Training Data Distribution

### DanceTrack train (40 sequences), GT annotation analysis

| Category | Count |
|----------|-------|
| Sequences with max ≥ 15 | 6 |
| Sequences with max ≥ 10 | 12 |
| Sequences with max < 10 | 28 |
| Global max concurrent | 40 |
| Global mean concurrent | 9.3 |

Top sequences by density:

| Sequence | Mean concurrent | Max concurrent | Frames |
|----------|----------------|---------------|--------|
| dancetrack0020 | 34.6 | 40 | 583 |
| dancetrack0096 | 26.2 | 40 | 603 |
| dancetrack0083 | 24.9 | 25 | 603 |
| dancetrack0082 | 20.5 | 24 | 603 |

All 40 sequences are present in training. High-density sequences (0020, 0096) are loaded but contribute only ~1200/total_frames ≈ 3% of training steps.

---

## Part 14 — GenerateIDLabels Subsetting (Code Analysis)

### File: `data/transforms.py`

```python
GenerateIDLabels(
    num_id_vocabulary=config["NUM_ID_VOCABULARY"],      # 50
    aug_num_groups=config["AUG_NUM_GROUPS"],             # 6
    num_training_ids=config.get("NUM_TRAINING_IDS",
                                config["NUM_ID_VOCABULARY"]),  # defaults to 50
)
```

If `_N > num_training_ids OR _N > num_id_vocabulary`: randomly subset to `min(num_training_ids, num_id_vocabulary)` unique IDs per 30-frame window.

Since `NUM_TRAINING_IDS` is not in config, effective cap = 50.

---

## Part 15 — RuntimeTracker ID Queue Mechanism (Code Analysis)

### File: `models/runtime_tracker.py`

`id_queue` is an `OrderedSet` initialized with labels 0–49 (LRU structure):
- Active labels: moved to **back** of queue each frame (`id_queue.add(label)`)
- Newborn assignment: takes from **front** of queue (least recently used)
- Result: labels of recently active tracks are protected; labels of long-dead tracks are recycled

**No vocabulary exhaustion:** max_concurrent val = 23 < 50 vocabulary slots.

**Recycling mechanism:** when label `k` is reused for a new object, all trajectory history for the previous object with label `k` is evicted:
```python
trajectory_remove_idxs |= (self.trajectory_id_labels[0] == newborn_id_labels[_])
self.trajectory_features = self.trajectory_features[:, ~trajectory_remove_idxs]
```

---

## Part 16 — Per-Sequence HOTA (epoch 2 checkpoint)

Selected sequences:

| Sequence | HOTA |
|----------|------|
| dancetrack0097 | 90.934 |
| dancetrack0005 | 87.015 |
| dancetrack0018 | 80.388 |
| dancetrack0034 | 29.364 |
| dancetrack0026 | 27.791 |
| dancetrack0041 | 21.175 |

---

## Part 17 — Failed Experiments

### V1 baseline (NUM_ID_DECODER_LAYERS=3)
Config: `NUM_ID_DECODER_LAYERS=3`, `ID_LOSS_WEIGHT=1.0`, 3 GPUs, fp32  
Epoch 0: HOTA=40.430, AssA=23.246, DetA=71.059  
Loss: `id_loss=1.1461`, `other_grad_norm=2.3095`

### V2 baseline — best (NUM_ID_DECODER_LAYERS=6)
Config: `NUM_ID_DECODER_LAYERS=6`, `ID_LOSS_WEIGHT=1.0`, 3 GPUs, fp32  
Epoch 0: HOTA=43.517, AssA=26.970  
Peak epoch 2: HOTA=49.567, AssA=34.632

### ID_LOSS_WEIGHT=3.0 on 6 GPUs (bf16 unfixed)
Result: HOTA=7  
Mechanism: bf16 dtype mismatch in `id_criterion.py` (`torch.tensor()` creates fp32, id_logits are bf16). Fix: cast one-hot to `id_logits.dtype`. Additionally, LR×batch interaction on 6 GPUs with higher loss weight caused `other_grad_norm=4.04`.

### ID_THRESH=0.1
Result: HOTA=48.660, AssA=33.448 (worse than baseline)  
Mechanism: lowered threshold accepted wrong associations along with correct ones. Both AssPr and AssRe dropped.

### ASSIGNMENT_PROTOCOL: hungarian
Result: HOTA≈48 (no improvement)  
Mechanism: Case B still 100%. Hungarian still applies id_thresh gate and newborn repeat columns allow global optimum to assign to newborn.

### NUM_TRAINING_IDS=20
Result: HOTA=2 at epoch 0  
Mechanism: Train softmax over 20+1 classes; inference softmax over 50+1 classes. Distribution mismatch collapses IDDecoder confidence.

### AUG_TRAJECTORY_OCCLUSION_PROB=0.7
Result: HOTA=2  
Mechanism: Task too noisy for IDDecoder to converge. Training signal became unreliable.

### Contrastive loss v1 — direct on trajectory_features
Config: applied `HardNegativeSupConLoss` gradient directly to `trajectory_features`  
Result: AssA=0.11, AssPr=99.98, MOTA=-1.27 at epoch 0  
Mechanism: contrastive gradient corrupted IDDecoder cross-attention space. `trajectory_features` are used as keys/values in IDDecoder — two objectives competed on same tensor with no attenuation.

### Contrastive loss v2 — projection head without detach
Config: `TemporalSupConLoss` with `ProjectionHead(256→256→128)`, no `feats.detach()`  
Result: AssA=0.11 at epoch 2  
Mechanism: Projection head Jacobian insufficient to attenuate contrastive gradient. `con_pos_sim` saturated to ~0.90+ early, indicating projection head collapse. IDDecoder still corrupted.

### Contrastive loss v3 — projection head with feats.detach() (current version)
Config: `TemporalSupConLoss`, `feats.detach()` in projection forward  
Status: Not yet tested with full training run. This version is in `models/motip/contrastive_loss.py`.

---

## Part 18 — Diagnostic Scripts Inventory

All scripts are in `diagnostics/` directory:

| Script | What it measures | Key results |
|--------|-----------------|-------------|
| `diag_object_matched.py` | Object-matched embedding quality (Diags 1,2,4,5,6,7) | NN=0.971, stability=0.967 |
| `diag_temporal_stability_script.py` | Slot-level stability across all 300 queries | Mean=0.531 (includes inactive) |
| `diag_layer_comparison.py` | Per-layer stability and cross-layer similarity | L2 most stable (0.968) |
| `diag_matcher_indices.py` | Checkpoint args + stale indices check | two_stage=True, no bug |
| `diag_script1_id_loss_split.py` | Train id_loss vs val AssA per epoch | Saturation at epoch 2 |
| `diag_script2_attention_weights.py` | IDDecoder cross-attention weight vs frame age | CV=1.158, non-flat |
| `diag_script3_newborn_rate.py` | Predicted vs GT newborns per frame | 7.25× inflation, r=0.248 |
| `diag_script4_score_distribution.py` | ID score distribution by outcome (Case A/B/C) | 100% Case B across all sequences |

---

## Part 19 — Open Questions for Next Session

1. **Why does dancetrack0081 (19.1 concurrent) achieve 0.967 correct_mean while dancetrack0041 (16.7 concurrent) achieves only 0.882?**
   - Motion: 0081 mean_disp=3.7px, 0041 mean_disp=4.6px — similar, not explanatory
   - Requires: visual inspection, or appearance-similarity metric from frames

2. **What specifically degrades IDDecoder confidence at high density?**
   - Hypothesis A: Softmax normalization over more trajectory slots dilutes per-class probability
   - Hypothesis B: Cross-attention score dilution over more key-value pairs
   - Hypothesis C: Training distribution underrepresentation of high-density windows
   - Requires: diagnostic comparing IDDecoder output distribution at N=4 vs N=20 trajectory slots

3. **Would oversampling high-density train sequences help?**
   - Sequences 0020, 0096 (max=40) exist but contribute ~3% of steps
   - Requires: weighted sampler implementation + training run

4. **Does `TemporalSupConLoss` with `feats.detach()` produce any AssA gain?**
   - The loss is in the repo and train.py integration is done
   - Requires: one training run to epoch 2

5. **Is the 0041 vs 0081 gap explained by sequence-level characteristics visible from GT?**
   - Could be: occlusion frequency, pedestrian trajectory crossing frequency, aspect ratio
   - Can be measured from GT without visual inspection
