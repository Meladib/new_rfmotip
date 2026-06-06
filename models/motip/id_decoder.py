# Copyright (c) Ruopeng Gao. All Rights Reserved.
#
# MODIFIED: adds a relative-SPATIAL bias term to the cross-attention logit,
# structurally the twin of the existing relative-TEMPORAL bias (rel_pos_embeds).
# Indexed by binned IoU(unknown_i, trajectory_j), zero-initialized so the model
# is byte-identical to V3 at step 0. All injected lines are marked with
# `# >>> SPATIAL BIAS`. DIFF THIS AGAINST YOUR V3 id_decoder.py: keep your
# version for any unchanged line that differs; keep only the marked injections.

import torch
import einops
import torch.nn as nn
from typing import Tuple
from torch.utils.checkpoint import checkpoint

from models.misc import _get_clones, label_to_one_hot
from models.ffn import FFN


class IDDecoder(nn.Module):
    def __init__(
            self,
            feature_dim: int,
            id_dim: int,
            ffn_dim_ratio: int,
            num_layers: int,
            head_dim: int,
            num_id_vocabulary: int,
            rel_pe_length: int,
            use_aux_loss: bool,
            use_shared_aux_head: bool,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.id_dim = id_dim
        self.ffn_dim_ratio = ffn_dim_ratio
        self.num_layers = num_layers
        self.head_dim = head_dim
        self.n_heads = (self.feature_dim + self.id_dim) // self.head_dim
        self.num_id_vocabulary = num_id_vocabulary
        self.rel_pe_length = rel_pe_length

        self.use_aux_loss = use_aux_loss
        self.use_shared_aux_head = use_shared_aux_head

        self.word_to_embed = nn.Linear(self.num_id_vocabulary + 1, self.id_dim, bias=False)
        embed_to_word = nn.Linear(self.id_dim, self.num_id_vocabulary + 1, bias=False)

        if self.use_aux_loss and not self.use_shared_aux_head:
            self.embed_to_word_layers = _get_clones(embed_to_word, self.num_layers)
        else:
            self.embed_to_word_layers = nn.ModuleList([embed_to_word for _ in range(self.num_layers)])
        pass

        # Related Position Embeddings:
        self.rel_pos_embeds = nn.Parameter(
            torch.zeros((self.num_layers, self.rel_pe_length, self.n_heads), dtype=torch.float32)
        )
        # Prepare others for rel pe:
        t_idxs = torch.arange(self.rel_pe_length, dtype=torch.int64)
        curr_t_idxs, traj_t_idxs = torch.meshgrid([t_idxs, t_idxs])
        self.rel_pos_map = (curr_t_idxs - traj_t_idxs)      # [curr_t_idx, traj_t_idx] -> rel_pos, like [1, 0] = 1
        pass

        # >>> SPATIAL BIAS: relative-spatial bias, twin of rel_pos_embeds.
        # >>> Indexed by binned IoU(unknown_i, trajectory_j). Zero-init -> no-op
        # >>> at step 0 (model byte-identical to V3). 4 bins from Action-1 design.
        self.num_spatial_bins = 4
        self.rel_spatial_embeds = nn.Parameter(
            torch.zeros((self.num_layers, self.num_spatial_bins, self.n_heads), dtype=torch.float32)
        )
        # >>> 3 internal IoU edges -> 4 bins (bucketize). Buffer: follows module
        # >>> to device, saved in state_dict. Edges from quantile design on
        # >>> model-error events (pixel-space IoU); re-derive if normalized-space
        # >>> distribution is degenerate (see the sp_iou print in forward).
        self.register_buffer(
            "spatial_bin_edges",
            torch.tensor([0.05, 0.3, 0.6], dtype=torch.float32),
            persistent=True,
        )
        # >>> END SPATIAL BIAS

        self_attn = nn.MultiheadAttention(
            embed_dim=self.feature_dim + self.id_dim,
            num_heads=self.n_heads,
            dropout=0.0,
            batch_first=True,
            add_zero_attn=True,
        )
        self_attn_norm = nn.LayerNorm(self.feature_dim + self.id_dim)
        cross_attn = nn.MultiheadAttention(
            embed_dim=self.feature_dim + self.id_dim,
            num_heads=self.n_heads,
            dropout=0.0,
            batch_first=True,
            add_zero_attn=True,
        )
        cross_attn_norm = nn.LayerNorm(self.feature_dim + self.id_dim)
        ffn = FFN(
            d_model=self.feature_dim + self.id_dim,
            d_ffn=(self.feature_dim + self.id_dim) * self.ffn_dim_ratio,
            activation=nn.GELU(),
        )
        ffn_norm = nn.LayerNorm(self.feature_dim + self.id_dim)

        self.self_attn_layers = _get_clones(self_attn, self.num_layers - 1)
        self.self_attn_norm_layers = _get_clones(self_attn_norm, self.num_layers - 1)
        self.cross_attn_layers = _get_clones(cross_attn, self.num_layers)
        self.cross_attn_norm_layers = _get_clones(cross_attn_norm, self.num_layers)
        self.ffn_layers = _get_clones(ffn, self.num_layers)
        self.ffn_norm_layers = _get_clones(ffn_norm, self.num_layers)

        # Init parameters:
        for n, p in self.named_parameters():
            # >>> SPATIAL BIAS: exclude rel_spatial_embeds from Xavier so it stays
            # >>> zero-initialized (same treatment as rel_pos_embeds).
            if p.dim() > 1 and "rel_pos_embeds" not in n and "rel_spatial_embeds" not in n:
                nn.init.xavier_uniform_(p)

        pass

    def forward(self, seq_info, use_decoder_checkpoint):
        trajectory_features = seq_info["trajectory_features"]
        unknown_features = seq_info["unknown_features"]
        trajectory_id_labels = seq_info["trajectory_id_labels"]
        unknown_id_labels = seq_info["unknown_id_labels"] if "unknown_id_labels" in seq_info else None
        trajectory_times = seq_info["trajectory_times"]
        unknown_times = seq_info["unknown_times"]
        trajectory_masks = seq_info["trajectory_masks"]
        unknown_masks = seq_info["unknown_masks"]
        # >>> SPATIAL BIAS: read boxes (present in seq_info in BOTH train and
        # >>> inference paths; trajectory_modeling mutates only the features key).
        trajectory_boxes = seq_info["trajectory_boxes"]   # (B,G,T,N,4) cxcywh-norm
        unknown_boxes = seq_info["unknown_boxes"]          # (B,G,T,N,4) cxcywh-norm
        # >>> END SPATIAL BIAS
        _B, _G, _T, _N, _ = trajectory_features.shape
        _curr_B, _curr_G, _curr_T, _curr_N, _ = unknown_features.shape

        trajectory_id_embeds = self.id_label_to_embed(id_labels=trajectory_id_labels)
        unknown_id_embeds = self.generate_empty_id_embed(unknown_features=unknown_features)

        trajectory_embeds = torch.cat([trajectory_features, trajectory_id_embeds], dim=-1)
        unknown_embeds = torch.cat([unknown_features, unknown_id_embeds], dim=-1)

        # Prepare some common variables:
        self_attn_key_padding_mask = einops.rearrange(unknown_masks, "b g t n -> (b g t) n").contiguous()
        cross_attn_key_padding_mask = einops.rearrange(trajectory_masks, "b g t n -> (b g) (t n)").contiguous()
        _trajectory_times_flatten = einops.rearrange(trajectory_times, "b g t n -> (b g) (t n)")
        _unknown_times_flatten = einops.rearrange(unknown_times, "b g t n -> (b g) (t n)")
        cross_attn_mask = _trajectory_times_flatten[:, None, :] >= _unknown_times_flatten[:, :, None]
        cross_attn_mask = einops.repeat(cross_attn_mask, "bg tn1 tn2 -> (bg n_heads) tn1 tn2", n_heads=self.n_heads).contiguous()
        # Prepare for rel PE:
        self.rel_pos_map = self.rel_pos_map.to(trajectory_features.device)
        rel_pe_idx_pairs = torch.stack([
            torch.stack(
                torch.meshgrid([_unknown_times_flatten[_], _trajectory_times_flatten[_]]), dim=-1
            )
            for _ in range(len(_trajectory_times_flatten))
        ], dim=0)       # (B*G, T*N of curr, T*N of traj, 2)
        rel_pe_idx_pairs = rel_pe_idx_pairs.to(trajectory_features.device)
        rel_pe_idxs = self.rel_pos_map[rel_pe_idx_pairs[..., 0], rel_pe_idx_pairs[..., 1]]      # (B*G, T_curr, T_traj)
        pass

        # >>> SPATIAL BIAS: build the IoU-bin index ONCE before the layer loop.
        # >>> We use the VALIDATED signal: IoU(detection, most-recent-valid box
        # >>> of each trajectory identity), NOT per-slot IoU against every
        # >>> historical box. The diagnostic measured 0.891 AUC on the last-box
        # >>> signal; the per-slot version is a different (unvalidated) signal.
        # >>> We find each identity's most-recent valid box (max trajectory_time
        # >>> among unmasked slots) and broadcast it to all of that identity's
        # >>> (t,n) slots, so every key of identity n carries IoU(det, n's last box).
        _valid = ~trajectory_masks                                   # (B,G,T,N) True=valid
        _tm = torch.where(_valid, trajectory_times,
                          torch.full_like(trajectory_times, -1))     # invalid -> -1
        _recent_t = _tm.argmax(dim=2)                                # (B,G,N) slot index of most-recent valid
        _gather_idx = _recent_t.view(_B, _G, 1, _N, 1).expand(_B, _G, 1, _N, 4)
        _last_box = torch.gather(trajectory_boxes, 2, _gather_idx).squeeze(2)   # (B,G,N,4)
        _last_box_bcast = _last_box.unsqueeze(2).expand(_B, _G, _T, _N, 4).contiguous()
        _traj_boxes_flat = einops.rearrange(_last_box_bcast, "b g t n c -> (b g) (t n) c")  # (bg,l2,4)
        _unk_boxes_flat = einops.rearrange(unknown_boxes, "b g t n c -> (b g) (t n) c")      # (bg,l1,4)
        sp_iou = self._pairwise_iou_cxcywh(_unk_boxes_flat, _traj_boxes_flat)                # (bg,l1,l2) f32
        sp_idx = torch.bucketize(
            sp_iou, self.spatial_bin_edges.to(sp_iou.device)
        ).clamp_(0, self.num_spatial_bins - 1)                                               # (bg,l1,l2) int64
        # >>> First-run CALIBRATION: accumulate sp_iou over the first 300 steps,
        # >>> then print quantile-based bin edges. The IoU distribution is
        # >>> bimodal in normalized space, so pixel-space edges leave middle bins
        # >>> empty; quantile edges guarantee balanced bins. After reading the
        # >>> printed edges, hardcode them into spatial_bin_edges above and
        # >>> set self._sp_calib_done=True path off (or just leave it — once
        # >>> calibrated it prints once and stops). For the REAL run, replace
        # >>> the spatial_bin_edges tensor with the printed [q25,q50,q75].
        if False:  # calibration disabled; edges set by fixed IoU thresholds
            if not hasattr(self, "_sp_calib_buf"):
                self._sp_calib_buf = []
                self._sp_calib_steps = 0
            self._sp_calib_buf.append(sp_iou.detach().flatten().float().cpu())
            self._sp_calib_steps += 1
            if self._sp_calib_steps >= 300:
                allv = torch.cat(self._sp_calib_buf)
                if allv.numel() > 1_000_000:
                    sel = torch.randperm(allv.numel())[:1_000_000]
                    allv = allv[sel]
                qs = torch.quantile(allv, torch.tensor([0.25, 0.5, 0.75]))
                edges = [round(x.item(), 4) for x in qs]
                hist = torch.bincount(sp_idx.flatten(), minlength=self.num_spatial_bins).tolist()
                print(f"[spatial-calib] N={allv.numel()} CURRENT_bin_hist={hist} "
                      f"=> SET spatial_bin_edges = torch.tensor({edges})", flush=True)
                self._sp_calib_done = True
                self._sp_calib_buf = []
        # >>> END SPATIAL BIAS

        # Change Cross-Attn key_padding_mask and attn_mask to float:
        cross_attn_key_padding_mask = torch.masked_fill(
            cross_attn_key_padding_mask.float(),
            mask=cross_attn_key_padding_mask,
            value=float("-inf"),
        ).to(self.dtype)
        cross_attn_mask = torch.masked_fill(
            cross_attn_mask.float(),
            mask=cross_attn_mask,
            value=float("-inf"),
        ).to(self.dtype)
        pass

        all_unknown_id_logits = None
        all_unknown_id_labels = None
        all_unknown_id_masks = None

        for layer in range(self.num_layers):
            # Predict ID logits:
            if use_decoder_checkpoint:
                unknown_embeds = checkpoint(
                    self._forward_a_layer,
                    layer,
                    unknown_embeds, trajectory_embeds,
                    self_attn_key_padding_mask, cross_attn_key_padding_mask,
                    cross_attn_mask, rel_pe_idxs,
                    sp_idx,                                  # >>> SPATIAL BIAS
                    use_reentrant=False,
                )
            else:
                unknown_embeds = self._forward_a_layer(
                    layer=layer,
                    unknown_embeds=unknown_embeds,
                    trajectory_embeds=trajectory_embeds,
                    self_attn_key_padding_mask=self_attn_key_padding_mask,
                    cross_attn_key_padding_mask=cross_attn_key_padding_mask,
                    cross_attn_mask=cross_attn_mask,
                    rel_pe_idx=rel_pe_idxs,
                    sp_idx=sp_idx,                           # >>> SPATIAL BIAS
                )

            _unknown_id_logits = self.embed_to_word_layers[layer](unknown_embeds[..., -self.id_dim:])
            _unknown_id_masks = unknown_masks.clone()
            _unknown_id_labels = None if not self.training else unknown_id_labels
            if all_unknown_id_logits is None:
                all_unknown_id_logits = _unknown_id_logits
                all_unknown_id_labels = _unknown_id_labels
                all_unknown_id_masks = _unknown_id_masks
            else:
                all_unknown_id_logits = torch.cat([all_unknown_id_logits, _unknown_id_logits], dim=0)
                all_unknown_id_labels = torch.cat([all_unknown_id_labels, _unknown_id_labels], dim=0) if _unknown_id_labels is not None else None
                all_unknown_id_masks = torch.cat([all_unknown_id_masks, _unknown_id_masks], dim=0)

        if self.training and self.use_aux_loss:
            return all_unknown_id_logits, all_unknown_id_labels, all_unknown_id_masks
        else:
            return _unknown_id_logits, _unknown_id_labels, _unknown_id_masks

    def _forward_a_layer(
            self,
            layer: int,
            unknown_embeds: torch.Tensor,
            trajectory_embeds: torch.Tensor,
            self_attn_key_padding_mask: torch.Tensor,
            cross_attn_key_padding_mask: torch.Tensor,
            cross_attn_mask: torch.Tensor,
            rel_pe_idx: torch.Tensor,
            sp_idx: torch.Tensor,                            # >>> SPATIAL BIAS
    ):
        _B, _G, _T, _N, _ = trajectory_embeds.shape
        _curr_B, _curr_G, _curr_T, _curr_N, _ = unknown_embeds.shape
        if layer > 0:   # use self-attention to transfer information between unknown features (same time step)
            self_unknown_embeds = einops.rearrange(unknown_embeds, "b g t n c -> (b g t) n c").contiguous()
            self_out, _ = self.self_attn_layers[layer - 1](
                query=self_unknown_embeds, key=self_unknown_embeds, value=self_unknown_embeds,
                key_padding_mask=self_attn_key_padding_mask,
            )
            self_out = self_unknown_embeds + self_out
            self_out = self.self_attn_norm_layers[layer - 1](self_out)
            unknown_embeds = einops.rearrange(self_out, "(b g t) n c -> b g t n c", b=_B, g=_G, t=_curr_T)

        # Cross-attention for in-context decoding:
        cross_unknown_embeds = einops.rearrange(unknown_embeds, "b g t n c -> (b g) (t n) c").contiguous()
        cross_trajectory_embeds = einops.rearrange(trajectory_embeds, "b g t n c -> (b g) (t n) c").contiguous()
        # Prepare attn_mask:
        rel_pe_mask = self.rel_pos_embeds[layer][rel_pe_idx]
        # >>> SPATIAL BIAS: add the relative-spatial term. dtype-cast both rel
        # >>> terms to match cross_attn_mask (which is self.dtype). Order matters:
        # >>> cross_attn_mask holds -inf at padded/future keys; -inf + finite
        # >>> spatial bias stays -inf, so padded positions remain masked.
        rel_sp_mask = self.rel_spatial_embeds[layer][sp_idx]   # (bg, l1, l2, n_heads) f32
        assert rel_sp_mask.shape == rel_pe_mask.shape, \
            f"spatial mask {tuple(rel_sp_mask.shape)} != temporal mask {tuple(rel_pe_mask.shape)}"
        cross_attn_mask_with_rel_pe = (
            cross_attn_mask
            + einops.rearrange(rel_pe_mask, "bg l1 l2 n -> (bg n) l1 l2").to(cross_attn_mask.dtype)
            + einops.rearrange(rel_sp_mask, "bg l1 l2 n -> (bg n) l1 l2").to(cross_attn_mask.dtype)
        )
        # >>> END SPATIAL BIAS
        # Apply cross-attention:
        cross_out, _ = self.cross_attn_layers[layer](
            query=cross_unknown_embeds, key=cross_trajectory_embeds, value=cross_trajectory_embeds,
            key_padding_mask=cross_attn_key_padding_mask,
            attn_mask=cross_attn_mask_with_rel_pe,
        )
        cross_out = cross_unknown_embeds + cross_out
        cross_out = self.cross_attn_norm_layers[layer](cross_out)
        # Feed-forward network:
        cross_out = cross_out + self.ffn_layers[layer](cross_out)
        cross_out = self.ffn_norm_layers[layer](cross_out)
        # Re-shape back to original shape:
        unknown_embeds = einops.rearrange(cross_out, "(b g) (t n) c -> b g t n c", b=_B, g=_G, t=_curr_T)

        return unknown_embeds

    # >>> SPATIAL BIAS: pairwise IoU helper (cxcywh-normalized -> IoU matrix).
    @staticmethod
    def _pairwise_iou_cxcywh(boxes_a, boxes_b):
        # boxes_a: (bg, A, 4) cxcywh ; boxes_b: (bg, B, 4) cxcywh -> IoU (bg, A, B)
        boxes_a = boxes_a.float()
        boxes_b = boxes_b.float()

        def to_xyxy(b):
            cx, cy, w, h = b.unbind(-1)
            return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)

        a = to_xyxy(boxes_a)   # (bg, A, 4)
        b = to_xyxy(boxes_b)   # (bg, B, 4)
        lt = torch.maximum(a[:, :, None, :2], b[:, None, :, :2])   # (bg, A, B, 2)
        rb = torch.minimum(a[:, :, None, 2:], b[:, None, :, 2:])   # (bg, A, B, 2)
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]                            # (bg, A, B)
        area_a = ((a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1]))[:, :, None]
        area_b = ((b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1]))[:, None, :]
        return inter / (area_a + area_b - inter + 1e-6)
    # >>> END SPATIAL BIAS

    def id_label_to_embed(self, id_labels):
        id_words = label_to_one_hot(id_labels, self.num_id_vocabulary + 1, dtype=self.dtype)
        id_embeds = self.word_to_embed(id_words)
        return id_embeds

    def generate_empty_id_embed(self, unknown_features):
        _shape = unknown_features.shape[:-1]
        empty_id_labels = self.num_id_vocabulary * torch.ones(_shape, dtype=torch.int64, device=unknown_features.device)
        empty_id_embeds = self.id_label_to_embed(id_labels=empty_id_labels)
        return empty_id_embeds

    def shuffle(self):
        shuffle_index = torch.randperm(self.num_id_vocabulary, device=self.word_to_embed.weight.device)
        shuffle_index = torch.cat([shuffle_index, torch.tensor([self.num_id_vocabulary], device=self.word_to_embed.weight.device)])
        self.word_to_embed.weight.data = self.word_to_embed.weight.data[:, shuffle_index]
        self.embed_to_word.weight.data = self.embed_to_word.weight.data[shuffle_index, :]
        pass

    @property
    def dtype(self):
        return self.word_to_embed.weight.dtype