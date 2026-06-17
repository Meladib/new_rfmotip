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

        # >>> MOTION BIAS (V5): time-conditioned motion-IoU bias — third twin of
        # >>> rel_pos_embeds / rel_spatial_embeds. Indexed by the FUSED cell
        # >>> (motion_iou_bin * num_dt_bins + dt_bin). Zero-init -> V5 forward is
        # >>> byte-identical to V4 epoch 11 at step 0 (Theorem 1).
        self.num_motion_iou_bins = 4
        self.num_dt_bins = 4
        self.num_motion_cells = self.num_motion_iou_bins * self.num_dt_bins   # 16
        self.rel_motion_embeds = nn.Parameter(
            torch.zeros((self.num_layers, self.num_motion_cells, self.n_heads), dtype=torch.float32)
        )
        # >>> PLACEHOLDER edges — MUST be recalibrated before the real fine-tune
        # >>> (Lemma 7: empty cell => zero gradient => untrained parameter — the
        # >>> exact W2 pathology V4 shipped with). Run one batch, read the
        # >>> [mot-hist] q25/q50/q75 of ALLOWED motion-IoU, hardcode them here.
        self.register_buffer(
            "motion_iou_edges",
            torch.tensor([0.05, 0.3, 0.6], dtype=torch.float32),
            persistent=True,
        )
        # >>> dt bins: {1}, {2..4}, {5..12}, {>=13} frames — spans the
        # >>> MISS_TOLERANCE=30 window; edges are half-integers so integer dt
        # >>> values bucketize unambiguously.
        self.register_buffer(
            "dt_bin_edges",
            torch.tensor([1.5, 4.5, 12.5], dtype=torch.float32),
            persistent=True,
        )
        # >>> END MOTION BIAS

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
            # >>> MOTION BIAS (V5): rel_motion_embeds excluded for the same reason.
            if p.dim() > 1 and "rel_pos_embeds" not in n and "rel_spatial_embeds" not in n \
                    and "rel_motion_embeds" not in n:
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
        if not hasattr(self, "_sp_hist_done"):   # one-shot edge-space diagnostic
            self._sp_hist_done = True
            _bg = _B * _G
            _L1, _L2 = sp_iou.shape[1], sp_iou.shape[2]
            _time_mask = cross_attn_mask.view(_bg, self.n_heads, _L1, _L2)[:, 0]   # True = causally masked
            _key_pad   = cross_attn_key_padding_mask.view(_bg, 1, _L2)             # True = padded key
            _allowed   = (~_time_mask) & (~_key_pad)                              # positions the bias affects
            _v   = sp_iou[_allowed].detach().float()
            _raw = sp_iou.detach().flatten().float()
            if _v.numel() > 0:
                _q = torch.quantile(_v, torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90], device=_v.device))
                _h = torch.bincount(sp_idx[_allowed].flatten(), minlength=self.num_spatial_bins)
                _hi   = (sp_idx == self.num_spatial_bins - 1)  
                _frac = (_hi & _allowed).sum().float() / _hi.sum().clamp(min=1).float()
                print(f"[sp-hist] ALLOWED n={_v.numel()} min={_v.min():.4f} mean={_v.mean():.4f} "
                      f"max={_v.max():.4f} q10/25/50/75/90={[round(x.item(),4) for x in _q]} "
                      f"bin_hist(edges={self.spatial_bin_edges.tolist()})={_h.tolist()}", flush=True)
                print(f"[sp-hist] RAW n={_raw.numel()} max={_raw.max():.4f} (box-format check)", flush=True)
                print(f"[sp-align] frac_hi_on_allowed={_frac.item():.4f} n_hi={int(_hi.sum())}", flush=True)
        # >>> END SPATIAL BIAS
        # >>> MOTION BIAS (V5): build the motion cell index ONCE before the layer
        # >>> loop. Signal: IoU(detection_i, velocity-extrapolated box of identity
        # >>> j), jointly binned with the observation gap dt — the G2-validated
        # >>> motion signal (GT-anchored AUC 0.807).
        # >>>
        # >>> CAUSAL PER-QUERY-TIME SELECTION. For a query with time u, only
        # >>> trajectory observations with time < u are used — the exact
        # >>> complement of the attention-mask predicate (traj_time >=
        # >>> unknown_time). NOTE this is deliberately stricter than the V4
        # >>> spatial index above (global argmax): global selection would (a)
        # >>> leak future boxes into training-time indices and (b) yield
        # >>> negative dt for early queries. Causal selection makes the train
        # >>> and inference index distributions identical by construction; at
        # >>> inference (_curr_T == 1, all trajectory slots in the past) it
        # >>> reduces to "most recent + second most recent".
        _valid_m = ~trajectory_masks                                          # (B,G,T,N) True = valid
        _bhat_list, _dt_list = [], []
        for _tq in range(_curr_T):
            _u = unknown_times[:, :, _tq, 0]                                  # (B,G) query time (uniform over n; asserted below)
            _lt = _valid_m & (trajectory_times < _u[:, :, None, None])        # obs strictly before this query
            _tm_q = torch.where(_lt, trajectory_times,
                                torch.full_like(trajectory_times, -1))        # invalid/future -> -1
            _k = min(2, _T)
            _top_t, _top_i = _tm_q.topk(k=_k, dim=2)                          # (B,G,k,N)
            _g_idx = _top_i.unsqueeze(-1).expand(_B, _G, _k, _N, 4)
            _top_b = torch.gather(trajectory_boxes, 2, _g_idx)                # (B,G,k,N,4)
            _t1, _b1 = _top_t[:, :, 0], _top_b[:, :, 0]                       # most recent (< u)
            if _k == 2:
                _t2, _b2 = _top_t[:, :, 1], _top_b[:, :, 1]
            else:
                _t2, _b2 = torch.full_like(_t1, -1), _b1
            _has2 = (_t1 >= 0) & (_t2 >= 0)                                   # >=2 prior observations
            _dt_now = (_u[:, :, None] - _t1).to(torch.float32)                # (B,G,N); >=1 on all allowed keys
            _dt_obs = (_t1 - _t2).to(torch.float32).clamp(min=1.0)
            _vel = torch.where(
                _has2.unsqueeze(-1),
                (_b1[..., :2] - _b2[..., :2]).to(torch.float32) / _dt_obs.unsqueeze(-1),
                torch.zeros((_B, _G, _N, 2), dtype=torch.float32, device=_b1.device),
            )                                                                  # center velocity; w,h NOT extrapolated
            _pred_c = (_b1[..., :2].to(torch.float32)
                       + _vel * _dt_now.unsqueeze(-1)).clamp(0.0, 1.0)
            _bhat_list.append(torch.cat([_pred_c, _b1[..., 2:].to(torch.float32)], dim=-1))
            _dt_list.append(_dt_now)
        _bhat = torch.stack(_bhat_list, dim=2)                                # (B,G,Tq,N,4) f32
        _dt_all = torch.stack(_dt_list, dim=2)                                # (B,G,Tq,N)  f32
        # IoU(query box at its own time, predicted box of identity j AS OF that time):
        _unk_bt = einops.rearrange(unknown_boxes, "b g t n c -> (b g t) n c").to(torch.float32)
        _hat_bt = einops.rearrange(_bhat, "b g t n c -> (b g t) n c")
        mot_iou = self._pairwise_iou_cxcywh(_unk_bt, _hat_bt)                 # ((b g t), Nu, Nt) f32
        mot_iou = einops.rearrange(mot_iou, "(b g t) nu nt -> (b g) (t nu) nt",
                                   b=_B, g=_G, t=_curr_T)                     # (bg, L1, Nt)
        _mu = torch.bucketize(
            mot_iou, self.motion_iou_edges.to(mot_iou.device)
        ).clamp_(0, self.num_motion_iou_bins - 1)                             # (bg, L1, Nt)
        _dt_mat = einops.repeat(
            einops.rearrange(_dt_all, "b g t n -> (b g) t n"),
            "bg t nt -> bg (t nu) nt", nu=_curr_N,
        ).contiguous()                                                         # (bg, L1, Nt)
        _delta = torch.bucketize(
            _dt_mat, self.dt_bin_edges.to(_dt_mat.device)
        ).clamp_(0, self.num_dt_bins - 1)
        mot_idx = (_mu * self.num_dt_bins + _delta)                            # fused cell in [0, 16)
        mot_idx = einops.repeat(mot_idx, "bg l1 nt -> bg l1 (tk nt)", tk=_T).contiguous()
        # ^ broadcast across KEY time: key axis was flattened "(t n)" with t
        #   outer, so "(tk nt)" matches the existing key layout exactly.

        # >>> One-shot pre-flight diagnostic (mirrors [sp-hist]; runs once):
        if not hasattr(self, "_mot_hist_done"):
            self._mot_hist_done = True
            _bg = _B * _G
            _L1, _L2 = mot_idx.shape[1], mot_idx.shape[2]
            _time_mask = cross_attn_mask.view(_bg, self.n_heads, _L1, _L2)[:, 0]
            _key_pad = cross_attn_key_padding_mask.view(_bg, 1, _L2)
            _allowed = (~_time_mask) & (~_key_pad)
            # (a) time-uniformity assumption used for _u:
            _ut_chk = unknown_times.float()
            _ut_spread = (_ut_chk.max(dim=3).values - _ut_chk.min(dim=3).values).abs().max()
            # (b) causal guarantee: dt >= 1 on every allowed position:
            _dt_keyspace = einops.repeat(_dt_mat, "bg l1 nt -> bg l1 (tk nt)", tk=_T)
            _dt_min_allowed = _dt_keyspace[_allowed].min() if _allowed.any() else torch.tensor(float("nan"))
            # (c) occupancy of the 16 cells on allowed positions (Lemma 7 gate):
            _h2 = torch.bincount(mot_idx[_allowed].flatten(),
                                 minlength=self.num_motion_cells).view(
                                 self.num_motion_iou_bins, self.num_dt_bins)
            # (d) edge-calibration quantiles of ALLOWED motion IoU:
            _miou_keyspace = einops.repeat(mot_iou, "bg l1 nt -> bg l1 (tk nt)", tk=_T)
            _v = _miou_keyspace[_allowed].detach().float()
            if _v.numel() > 0:
                _q = torch.quantile(_v, torch.tensor([0.25, 0.50, 0.75], device=_v.device))
                print(f"[mot-hist] ALLOWED n={_v.numel()} mean={_v.mean():.4f} "
                      f"max={_v.max():.4f} q25/50/75={[round(x.item(), 4) for x in _q]} "
                      f"edges={self.motion_iou_edges.tolist()}", flush=True)
            print(f"[mot-hist] cell_occupancy(iou_bin x dt_bin)=\n{_h2.tolist()}", flush=True)
            print(f"[mot-align] dt_min_on_allowed={float(_dt_min_allowed):.1f} "
                  f"(must be >= 1.0) | unknown_time_spread={float(_ut_spread):.1f} "
                  f"(must be 0.0)", flush=True)
        # >>> END MOTION BIAS


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
                    mot_idx,                                 # >>> MOTION BIAS (V5)
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
                    mot_idx=mot_idx,                         # >>> MOTION BIAS (V5)
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
            mot_idx: torch.Tensor,                           # >>> MOTION BIAS (V5)
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
        
        rel_sp_mask = self.rel_spatial_embeds[layer][sp_idx]    # (bg, l1, l2, n_heads) f32
        rel_mot_mask = self.rel_motion_embeds[layer][mot_idx]   # >>> MOTION BIAS (V5): same gather pattern, f32
        rel_bias = rel_pe_mask + rel_sp_mask + rel_mot_mask     # combine in compact layout (still ONE rearrange)
        cross_attn_mask_with_rel_pe = cross_attn_mask + einops.rearrange(rel_bias, "bg l1 l2 n -> (bg n) l1 l2")
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
