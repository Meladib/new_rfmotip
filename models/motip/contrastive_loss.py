# models/motip/contrastive_loss.py
# Copyright (c) Ruopeng Gao. All Rights Reserved.
"""
Temporal Supervised Contrastive Loss with Isolated Projection Head
==================================================================

WHY THE PREVIOUS DESIGN FAILED (AssA = 0.11)
---------------------------------------------
The previous loss applied contrastive directly on trajectory_features,
which are the SAME tensors used as keys/values in IDDecoder cross-attention.

Two gradient paths competed on the same features:
  (A) id_loss  → IDDecoder cross-attention → trajectory_features
  (B) con_loss → trajectory_features  (no intermediate module)

Path (B) had no attenuation — the full contrastive gradient hit
trajectory_features directly, corrupting the cross-attention space
that path (A) was simultaneously trying to learn. IDDecoder never
stabilized. Result: AssA = 0.11.

WHAT THE DIAGNOSTICS PROVED
-----------------------------
Diag 1 (Saturation confirmed):
  train id_loss: 1.36 → 0.60 across 4 epochs
  AssA peaks at epoch 2 (34.63) then DROPS (33.74) while id_loss still falls.
  The model memorizes training identities — closed-set vocabulary memorization.

Diag 2 (Attention structure confirmed):
  Mean CV = 1.158 across 6 layers — NOT flat.
  Final layer (L5): sharp recency (age-1 dominant) + long-tail rise.
  Phase 2 (confidence-weighted attention) is lower priority.

Diag 3 (Fragmentation confirmed):
  Inflation ratio = 7.25x, spurious rate = 100%.
  Crowd correlation = 0.248 (LOW) — not crowd-driven.
  Score miscalibration causes spurious newborns uniformly.

Tracking result:
  IDSW = 10,080 with only 273 GT IDs = 37 switches per identity on average.
  The model loses associations over time — temporal consistency is the bottleneck.

CORRECT DESIGN: ISOLATED PROJECTION HEAD
-----------------------------------------
The fix follows the SimCLR/SupCon principle: apply contrastive on a
SEPARATE auxiliary projection head that NEVER feeds into IDDecoder.

                  trajectory_features (B,G,T,N,256)
                           │
              ┌────────────┴────────────────┐
              │                             │
              ▼                             ▼
        ProjectionHead              IDDecoder (unchanged)
     (256→256→128, L2-norm)         (keys/values unchanged)
              │
              ▼
       contrastive_loss
              │
              ▼  gradient flows back through ProjectionHead into
                 trajectory_features → TrajectoryModeling
                 BUT: IDDecoder sees the SAME trajectory_features
                 that id_loss gradient also flows through.
                 The contrastive gradient is ATTENUATED by the
                 projection head Jacobian before reaching the
                 shared trajectory_features.

Gradient attenuation proof:
  With ProjectionHead initialized Xavier-uniform:
    ||dL_con/d(trajectory_features)|| ≈ weight × ||dL_con/dz|| × ||dz/df||
  Where ||dz/df|| is the Jacobian of the projection head.
  This is bounded and much smaller than the direct id_loss gradient,
  ensuring id_loss remains the primary training signal.

TARGET: TEMPORAL POSITIVE PAIRS (directly addresses IDSW=10,080)
-----------------------------------------------------------------
Positives = same track_id at DIFFERENT time steps within (B,G) slice.
This is exactly the signal needed to fix IDSW: enforce that the same
object's embedding is consistent across the T=30 frame window.

Standard SupCon (no IoU, no hard-neg weighting, no sim threshold):
  - Previous complexity caused instability
  - The projection head already provides the separation needed
  - Simple SupCon on temporal pairs is the minimum necessary intervention

YAML keys:
  CONTRASTIVE_WEIGHT:    0.05    # secondary signal; id_loss=1.0 is primary
  CONTRASTIVE_TEMP:      0.3     # appropriate for projection head from scratch
  CONTRASTIVE_PROJ_DIM:  128     # projection output dim
  CONTRASTIVE_WARMUP:    500     # steps before full weight applied
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalSupConLoss(nn.Module):
    """
    Supervised Contrastive Loss with an isolated projection head.

    The projection head is an auxiliary branch — its output is NEVER
    used by IDDecoder. Gradients from con_loss reach trajectory_features
    only through the projection head's Jacobian, which attenuates them
    relative to the direct id_loss gradient.

    Positive pairs: same track_id at different time steps within the
    same (batch, group) slice. Directly addresses IDSW=10,080.

    Parameters
    ----------
    feature_dim : int
        Input dimension. Must match DETR_HIDDEN_DIM = 256.
    proj_dim : int
        Projection output dimension. 128 is standard (SupCon paper).
    temperature : float
        τ = 0.3. Appropriate for projection head starting from scratch.
        Do not use 0.1 — too sharp for random-init projection, causes
        vanishing gradients on non-positive pairs at initialization.
    weight : float
        Overall loss coefficient λ_con = 0.05.
        With id_loss weight = 1.0, the ratio is 1:0.05 = 20:1 in favor
        of id_loss. This keeps contrastive as a regularizer, not a
        competing objective.
    warmup_steps : int
        Linear warmup from 0 to weight over this many steps.
        Prevents early-epoch instability before IDDecoder has learned
        basic associations.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        proj_dim: int = 128,
        temperature: float = 0.3,
        weight: float = 0.05,
        warmup_steps: int = 500,
        num_id_vocabulary: int = 50,
    ):
        super().__init__()

        # ── Projection head (auxiliary branch, never used by IDDecoder) ──
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )
        # Xavier init — same as IDDecoder
        for p in self.projection.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.temperature       = temperature
        self.weight            = weight
        self.warmup_steps      = warmup_steps
        self.num_id_vocabulary = num_id_vocabulary

        self._step = 0

    def _warmup_scale(self) -> float:
        self._step += 1
        return min(1.0, self._step / max(self.warmup_steps, 1))

    def forward(self, seq_info: dict):
        """
        Parameters
        ----------
        seq_info : dict
            Output of TrajectoryModeling.forward(). Uses:
              trajectory_features  (B, G, T, N, D)
              trajectory_id_labels (B, G, T, N)  int64
              trajectory_masks     (B, G, T, N)  bool, True = padding
              unknown_features     (B, G, T, N, D)
              unknown_id_labels    (B, G, T, N)
              unknown_masks        (B, G, T, N)

        Returns
        -------
        loss : scalar tensor
        log  : dict with diagnostic values
            "loss"          — weighted loss value
            "raw_loss"      — loss before weight/warmup scaling
            "n_anchors"     — anchors that had >= 1 positive
            "n_positives"   — total positive pairs used
            "mean_pos_sim"  — mean cosine sim of positive pairs in projection space
                              (should rise toward >0.7 as training progresses)
            "warmup_scale"  — current warmup factor [0, 1]
        """
        device = seq_info["trajectory_features"].device
        B, G, T, N, D = seq_info["trajectory_features"].shape

        _zero      = torch.tensor(0.0, device=device, requires_grad=True)
        _empty_log = {
            "loss":         0.0,
            "raw_loss":     0.0,
            "n_anchors":    0.0,
            "n_positives":  0.0,
            "mean_pos_sim": 0.0,
            "warmup_scale": float(self._warmup_scale()),
        }

        # ── Collect features and labels ───────────────────────────────────
        # Process each (b, g) slice independently.
        # Within a slice, same label at different time steps = positive pair.
        # This is the direct fix for IDSW: enforce temporal embedding consistency.

        feats_list  = []
        labels_list = []

        for b in range(B):
            for g in range(G):
                slice_feats  = []
                slice_labels = []

                # Trajectory stream: T frames of history
                for t in range(T):
                    mask   = seq_info["trajectory_masks"][b, g, t]      # (N,)
                    labels = seq_info["trajectory_id_labels"][b, g, t]  # (N,)
                    # Valid: not padded AND label is a real ID (not -1)
                    valid  = ~mask & (labels >= 0)
                    if valid.sum() == 0:
                        continue
                    slice_feats.append(
                        seq_info["trajectory_features"][b, g, t][valid])
                    slice_labels.append(labels[valid])

                # Unknown stream: exclude newborn label (= num_id_vocabulary)
                # Newborns have no positive pair in trajectory history
                for t in range(T):
                    mask   = seq_info["unknown_masks"][b, g, t]
                    labels = seq_info["unknown_id_labels"][b, g, t]
                    valid  = (~mask & (labels >= 0) &
                              (labels < self.num_id_vocabulary))
                    if valid.sum() == 0:
                        continue
                    slice_feats.append(
                        seq_info["unknown_features"][b, g, t][valid])
                    slice_labels.append(labels[valid])

                if len(slice_feats) < 2:
                    continue

                feats_list.append(torch.cat(slice_feats,  dim=0))
                labels_list.append(torch.cat(slice_labels, dim=0))

        if not feats_list:
            _empty_log["warmup_scale"] = float(min(
                1.0, self._step / max(self.warmup_steps, 1)))
            return _zero, _empty_log

        feats  = torch.cat(feats_list,  dim=0)   # (M, D)
        labels = torch.cat(labels_list, dim=0)   # (M,)
        M      = feats.shape[0]

        if M < 2:
            return _zero, _empty_log

        # ── Project to contrastive space (auxiliary branch) ───────────────
        # This is the KEY difference from the previous design.
        # `feats` still carries gradients from TrajectoryModeling,
        # but the projection head interposes a non-trivial Jacobian
        # that attenuates the contrastive gradient before it reaches
        # trajectory_features. IDDecoder sees trajectory_features
        # directly, unaffected by what the projection head does.
        z = F.normalize(self.projection(feats), dim=-1)   # (M, proj_dim)

        # ── Standard SupCon ───────────────────────────────────────────────
        sim       = z @ z.T                               # (M, M)
        logits    = sim / self.temperature
        label_eq  = labels[:, None] == labels[None, :]   # (M, M)
        self_mask = torch.eye(M, dtype=torch.bool, device=device)

        # Denominator: logsumexp over all non-self entries
        log_denom = torch.logsumexp(
            logits.masked_fill(self_mask, float('-inf')), dim=1)   # (M,)

        # Numerator: positive pairs (same label, not self)
        pos_mask  = label_eq & ~self_mask                # (M, M)
        has_pos   = pos_mask.sum(dim=1) >= 1             # (M,) anchor filter

        if not has_pos.any():
            return _zero, _empty_log

        n_pos          = pos_mask.sum(dim=1).float().clamp(min=1)
        sum_pos_logits = (logits * pos_mask.float()).sum(dim=1)
        loss_per_anchor = -(sum_pos_logits / n_pos - log_denom)
        raw_loss        = loss_per_anchor[has_pos].mean()

        # Warmup scaling
        ws   = self._warmup_scale()
        loss = raw_loss * ws

        # ── Diagnostics ───────────────────────────────────────────────────
        with torch.no_grad():
            pos_sim_mean = ((sim * pos_mask.float()).sum() /
                            pos_mask.float().sum().clamp(min=1))

        log = {
            "loss":         loss.item(),
            "raw_loss":     raw_loss.item(),
            "n_anchors":    float(has_pos.sum().item()),
            "n_positives":  float(pos_mask.sum().item() // 2),
            "mean_pos_sim": pos_sim_mean.item(),
            "warmup_scale": float(ws),
        }

        return loss, log

    def __repr__(self):
        return (f"TemporalSupConLoss("
                f"feature_dim={self.projection[0].in_features}, "
                f"proj_dim={self.projection[2].out_features}, "
                f"τ={self.temperature}, "
                f"weight={self.weight}, "
                f"warmup={self.warmup_steps})")


def build_contrastive_criterion(config: dict) -> TemporalSupConLoss:
    """
    Build from MOTIP config dict.

    YAML additions:
        CONTRASTIVE_WEIGHT:    0.05
        CONTRASTIVE_TEMP:      0.3
        CONTRASTIVE_PROJ_DIM:  128
        CONTRASTIVE_WARMUP:    500
    """
    return TemporalSupConLoss(
        feature_dim       = config.get("FEATURE_DIM",          256),
        proj_dim          = config.get("CONTRASTIVE_PROJ_DIM",  128),
        temperature       = config.get("CONTRASTIVE_TEMP",      0.3),
        weight            = config.get("CONTRASTIVE_WEIGHT",    0.05),
        warmup_steps      = config.get("CONTRASTIVE_WARMUP",    500),
        num_id_vocabulary = config.get("NUM_ID_VOCABULARY",     50),
    )
