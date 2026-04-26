# CLAUDE.md — RF-MOTIP Diagnostic Project

## Project Purpose
This repository integrates RF-DETR-small as the detector backbone inside MOTIP
(a transformer-based multi-object tracker). The integration trains but produces
severely degraded tracking. This project's current goal is **pure diagnosis** —
understand the architectural mismatches before attempting any fix.

---

## Do Not Do
- Do not propose fixes or architectural changes
- Do not modify any source files
- Do not run training or evaluation loops
- Do not execute the Axis-5 diagnostic scripts — write them only
- Do not speculate beyond what the code and papers confirm

---

## Metric Baselines

| Configuration | HOTA | DetA | AssA | DetRe | AssRe |
|---|---|---|---|---|---|
| MOTIP baseline (Deformable DETR + ResNet-50) | ~70 | — | — | — | — |
| RF-DETR frozen detector (Epoch 3 plateau) | 38.5 | 72.9 | 20.5 | 81.5 | 27.3 |
| RF-DETR joint training (Epoch 4) | 23.4 | 40.1 | 13.9 | 47.6 | 18.4 |

**Key interpretations:**
- Frozen experiment: RF-DETR features are informative for detection (DetA=72.9)
  but association is broken (AssA=20.5) — features distinguish objects, not identities
- Joint training degrades DetA (72.9 → 40.1): ID loss is interfering with detection
  gradients — gradient conflict is active, not just a representation mismatch
- Joint training also degrades AssA (20.5 → 13.9): the detector is being corrupted
  by the combined loss, making features worse for both tasks simultaneously

---

## MOTIP Architecture (from paper — confirmed facts)

- Detector: Deformable DETR with ResNet-50 backbone
- **Global hidden dimension: C = 256** (used everywhere — non-negotiable constraint)
- DETR output embedding: C-dimensional (256) → this is `f^n_t`
- ID Dictionary: K+1 learnable C-dimensional embeddings, **K=50** default
- Special token `i_spec`: used for newborn objects (no trajectory yet)
- Tracklet token: `τ^m_t = concat(f^m_t, i^k_m)` → **2C = 512** dimensional
- ID Decoder: standard transformer decoder, **6 layers** (self-attn + cross-attn per layer)
- Detection loss: applied at **final decoder layer only** — no auxiliary losses
- Gradient strategy: backprop DETR on **4 frames** out of T+1; rest are `torch.no_grad()`
- Loss weights: `λ_cls=2.0, λ_L1=5.0, λ_giou=2.0, λ_id=1.0`
- Training: 8×RTX 4090, batch size 1 per GPU, T=29 (DanceTrack)
- Inference thresholds: `λ_det=0.3, λ_new=0.6, λ_id=0.2`
- Trajectory augmentation: `λ_occ=λ_sw=0.5`

### ID Decoder input contract (critical)
```
Keys/Values: historical tracklets τ^m,k (each 2C=512 dimensional)
Queries: current frame detections τ^n_t (each 2C=512 dimensional, using i_spec)
Output: per-detection ID logits over K+1 classes
Supervision: cross-entropy loss (ID criterion)
```

---

## RF-DETR Architecture (from paper — confirmed facts)

### Backbone
- **DINOv2-S (ViT-S/14)**: 12 transformer layers, hidden dim = **384**, patch size = 14
- Window attention at backbone layers: **{0, 1, 3, 4, 6, 7, 9, 10}**
- Global (non-windowed) attention at layers: **{2, 5, 8, 11}**
- Window count: **2** (optimal for small variant; class token duplicated per window)
- Per-layer multiplicative decay: **0.8** (to preserve DINOv2 pretraining)
- Gradient clip: 0.1

### Projector
- Multi-scale projector maps backbone features to decoder input
- Uses **layer norm** (not batch norm) for consumer GPU compatibility
- Output dimension: **unconfirmed — must be verified in code** (expected 256)

### Decoder (small variant — Pareto-optimal config)
- **3 decoder layers** at inference
- Each layer: Self-Attention → Deformable Cross-Attention → Feed Forward
- Detection loss applied at **every layer** (layers 0, 1, and 2)
- Query selection: top-K by `max sigmoid(class_logit)` at encoder output — **per-frame**
- **300 queries** at inference for small variant (fixed)
- Can drop all decoder layers at inference (→ single-stage YOLO-like behavior)

### NAS Training (critical for understanding weight behavior)
- At every training iteration: random config sampled from search space
- Search axes: patch size, decoder layer count (1–6), query count, resolution, window count
- Result: all sub-networks (1-layer, 2-layer, 3-layer decoder...) share weights and
  must all produce valid outputs → **no layer is specialized to its depth position**
- This "architecture augmentation" serves as regularization

### RF-DETR-small confirmed config (Table 7 in paper)
| Resolution | Patch Size | Windows | Decoder Layers | Queries | Backbone |
|---|---|---|---|---|---|
| 512 | 16 | 2 | 3 | 300 | DINOv2-S |

---

## Core Architectural Tensions (paper-derived)

### Tension 1: Auxiliary losses vs. single-layer supervision
- RF-DETR: detection loss at **all 3 decoder layers** (×3 detection gradient signal)
- MOTIP: detection loss at **final layer only** + ID loss at final layer
- Impact: RF-DETR's intermediate layer gradients conflict with MOTIP's single-point
  ID supervision. The ID loss (λ_id=1.0) competes against 3× detection losses.

### Tension 2: DINOv2 384-dim vs. MOTIP's required 256-dim
- DINOv2-S produces 384-dim features
- MOTIP requires exactly 256-dim features for `f^n_t`
- If the projector output ≠ 256, the 512-dim tracklet concat is silently broken

### Tension 3: Per-frame encoder-selected queries vs. stable trajectory queries
- RF-DETR: query selection changes per frame (ordered by confidence)
- MOTIP: trajectory history assumes query slot N = object identity across frames
- This is the most architecturally fundamental incompatibility

### Tension 4: NAS weight sharing vs. fixed-depth MOTIP expectation
- MOTIP expects a single stable decoder producing consistent feature semantics
- RF-DETR's decoder weights are shared across 1–6 depth configurations
- The 3-layer inference config was never the only config trained — features may lack depth-specific specialization

---

## Expected Output Files
```
diagnostics/
  diag_decoder_topology.md
  diag_feature_space.md
  diag_query_divergence.md
  diag_matcher_alignment.md
  diag_temporal_stability.md
  diag_temporal_stability_script.py   ← user runs this against their checkpoint
  plot_temporal_stability.py          ← companion plotting script
```

---

## PR Instructions
When all 7 files are written, open a PR titled **"RF-MOTIP Diagnostic Reports"**
targeting this branch's base. The PR description must contain:
- One line per output file: filename | severity rating | #1 hypothesis (one sentence)
- A brief summary paragraph of the most critical finding overall
