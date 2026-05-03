# PROMPT.md — Session Context for RF-MOTIP Research

## What We Are Building

We are integrating **RF-DETR** (a DINOv2-based object detector) as the frozen backbone into **MOTIP** (a transformer-based multi-object tracker that uses trajectory embeddings and an ID decoder for association). The goal is to achieve state-of-the-art tracking performance on DanceTrack.

The detector is completely frozen (`DETR_NUM_TRAIN_FRAMES=0`). Only the MOTIP tracking head is trained: `TrajectoryModeling` (adapter FFN) and `IDDecoder` (6-layer cross-attention).

---

## Current Best Result

**Checkpoint:** `outputsV2/rfmotip_dancetrack/train/checkpoint_2.pth`

| Metric | Value |
|--------|-------|
| HOTA | 49.567 |
| DetA | 71.586 |
| AssA | 34.632 |
| AssRe | 42.212 |
| AssPr | 52.716 |
| MOTA | 82.154 |
| IDF1 | 52.986 |
| IDSW | 10,080 |
| Frag | 5,905 |
| MT | 84.6% |
| ML | 1.1% |

Peak at **epoch 2**. Train id_loss continues decreasing at epoch 3 while AssA drops.

---

## What We Know (Evidence-Based Summary)

### RF-DETR Embeddings Are Strong

From `diag_object_matched.py` on frozen detector:
- NN accuracy = **0.971** — embeddings already cluster by identity
- Object-matched temporal stability (gap=1) = **0.967**
- Similarity gap in hard cases = **+0.150** (positive > negative even in crowds)
- 42 PCA components explain 90% of identity variance

The frozen detector is not the bottleneck.

### The Failure Is in Association, Not Detection

- DetA = 71.6 is stable across all epochs
- IDSW = 10,080 with only 273 GT IDs = ~37 switches per identity
- Frag = 5,905

### Score Distribution Analysis (Diag 4 — confirmed)

From `diag_script4_score_distribution.py` across all 25 val sequences:

- **Case A** (correct label score < threshold): 0–5% of spurious newborns
- **Case B** (correct label score ≥ threshold, still newborn): 95–100% of spurious newborns
- **Case C** (label absent from memory): 0%

The IDDecoder knows the correct answer (scores it above threshold) but the `object-max` assignment protocol routes it to newborn due to competition conflicts. This is universal.

### Density-Confidence Correlation

Pearson r = **-0.636** between mean_concurrent and correct_assignment_score across 25 val sequences.

| Sequence | Mean concurrent | Correct score mean | HOTA |
|----------|----------------|-------------------|------|
| dancetrack0097 | 3.9 | 0.994 | 90.9 |
| dancetrack0041 | 16.7 | 0.882 | 21.2 |
| dancetrack0026 | 13.2 | 0.854 | 27.8 |

Training distribution: **28/40 sequences have max < 10 concurrent objects**. Global train mean = 9.3.

### Motion as Secondary Driver

- dancetrack0026: mean_disp=9.2px, max_disp=**418px** (person running across frame)
- dancetrack0041: mean_disp=4.6px (similar to well-performing sequences)
- dancetrack0026 fails from density + extreme motion
- dancetrack0041 fails from density alone

### ID Vocabulary Is Not Exhausted

- NUM_ID_VOCABULARY = 50, max_concurrent val = 23
- LRU recycling in `RuntimeTracker.id_queue` prevents exhaustion
- Recycling mechanism: evicts trajectory history of old tracks when label is reused

### Training Saturation Confirmed

- Train id_loss: 1.36 → 0.60 across 4 epochs (still decreasing at epoch 3)
- AssA peaks at epoch 2 and drops at epoch 3
- The IDDecoder overfits to training sequence identity patterns

### IDDecoder Attention Is Non-Flat

From `diag_script2_attention_weights.py`:
- Mean CV = 1.158 across 6 layers
- Layer 5 (final): sharp recency peak (age 1: 0.051) + secondary rise at ages 25–29
- Model has learned a recency-dominant + long-tail temporal strategy

---

## Open Questions (Not Yet Answered)

1. **Why does dancetrack0081 (mean=19.1 concurrent) achieve correct_mean=0.967 while dancetrack0041 (mean=16.7) achieves only 0.882?** Density alone does not explain the gap.

2. **Does appearance similarity (clothing, body type) explain the 0081 vs 0041 gap?** Not measurable from GT — requires visual inspection or optical flow.

3. **Does the IDDecoder softmax distribution change shape with N_concurrent?** The score degradation at high density could come from softmax normalization over more classes, from attention score dilution over more trajectory slots, or from training exposure. Not yet isolated.

4. **Would a sequence-weighted sampler (oversampling high-density sequences) help?** Not tested. The 6 high-density train sequences (max ≥ 15) exist and are loaded but contribute only ~5% of gradient steps.

5. **Does the contrastive loss with `feats.detach()` help?** Not yet tested with the detached version. Previous attempts without detach caused AssA=0.11. The detached version is in `models/motip/contrastive_loss.py`.

---

## What Has Been Tried and Failed

See CLAUDE.md for the full failure table. Most critically:
- Any change to the training objective (loss weights, augmentation) that destabilizes the IDDecoder causes AssA → ~0
- Inference-time threshold/protocol changes provide marginal improvement (<1 HOTA)
- Vocabulary size is not the constraint

---

## The Core Research Question

The IDDecoder achieves near-perfect correct-label scores in easy scenes (0.994) but degrades to 0.854–0.882 in dense scenes (13–17 concurrent objects). **What is the specific mechanism causing this degradation, and how can we train the IDDecoder to maintain high-confidence discrimination under high scene density?**

---

## Session Instructions

1. Read FULL_DIAGNOSTIC_REPORT.md and this file completely before any analysis
2. Read the relevant repo files before making any claim about code behavior
3. Every proposed change needs: root cause → mechanism → proposed fix → expected effect → risk assessment
4. Do not propose training changes unless the root cause of the degradation is isolated
5. Propose diagnostics for any open question before proposing a solution
