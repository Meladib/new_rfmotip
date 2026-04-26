# DIAG_FEATURE_SPACE

## 1. FINDINGS

- **Projector output dimension is `args.hidden_dim` from the checkpoint.** `models/rfdetr/models/lwdetr.py:795` passes `out_channels=args.hidden_dim` to `build_backbone()`, which passes it to `MultiScaleProjector.__init__()` as `out_channels`. Each projector stage ends with `C2f(in_dim, out_channels, ...)` followed by `get_norm('LN', out_channels)` (projector.py, final two lines of each stage's layer list). The projector output is therefore exactly `args.hidden_dim` channels.

- **DINOv2-S backbone produces 384-dim features natively.** The backbone is ViT-S/14 with hidden dim = 384. The projector must bridge 384 → `args.hidden_dim`. If `args.hidden_dim = 256`, the projector successfully bridges the gap; if `args.hidden_dim = 384` (DINOv2 native), the projector is an identity-scale operation and no compression occurs.

- **`TrajectoryModeling` is built with `detr_dim = config["DETR_HIDDEN_DIM"] = 256` (MOTIP YAML), not from the checkpoint.** `models/motip/__init__.py:71` calls `TrajectoryModeling(detr_dim=config["DETR_HIDDEN_DIM"], ...)`. Its `adapter = FFN(d_model=detr_dim=256)`. If RF-DETR emits dim ≠ 256, `adapter.forward(trajectory_features)` crashes with a shape mismatch because `FFN` applies a linear layer of shape `[256, 256*ffn_dim_ratio]` to input of wrong channel count.

- **MOTIP infers feature dim dynamically at runtime.** `train.py:698`: `_feature_dim = detr_outputs["outputs"].shape[-1]`. `trajectory_features` and `unknown_features` are allocated with shape `(..., _feature_dim)` (train.py:707–714). This dynamic inference means the `trajectory_features` tensor adapts to whatever RF-DETR emits — but `TrajectoryModeling.adapter` is statically built for 256.

- **IDDecoder `embed_dim = feature_dim + id_dim = 256 + 256 = 512`.** `models/motip/id_decoder.py:62–70`: `self_attn` and `cross_attn` are built with `embed_dim = self.feature_dim + self.id_dim`. `models/motip/__init__.py:82` builds IDDecoder with `feature_dim=config["FEATURE_DIM"]=256` and `id_dim=config["ID_DIM"]=256`. The tracklet concat at `id_decoder.py:111–112` must produce exactly 512-dim inputs: `trajectory_embeds = cat([trajectory_features, trajectory_id_embeds], dim=-1)`.

- **Layer norm in the projector suppresses magnitude information.** Every projector stage applies `get_norm('LN', out_channels)` (resolves to `LayerNorm` via `projector.py:58–65`) immediately after the C2f block. LayerNorm normalizes per-token across channels to zero mean and unit variance. This means the raw activation magnitude — a proxy for detection confidence or feature strength — is discarded before features reach the decoder. Re-ID across frames depends on relative feature angles, not magnitudes; layer norm is compatible with this but eliminates any magnitude-based signal.

- **No projection layer exists in the integration between RF-DETR and MOTIP.** `hs[-1]` from RF-DETR is used directly as `detr_outputs["outputs"]` (lwdetr.py:221) and then `detr_output_embeds` (train.py:732). No learned linear projection adapts RF-DETR's decoder output into MOTIP's feature space; the same weights must serve both the detection head and the re-ID purpose.

## 2. MISMATCH EVIDENCE

| Module | Input Shape | Output Shape | Consumed By |
|---|---|---|---|
| DINOv2-S backbone | `[B, 3, H, W]` | `[B, 384, H/14, W/14]` (per-layer) | MultiScaleProjector |
| MultiScaleProjector | List of `[B, 384, *, *]` feature maps | List of `[B, args.hidden_dim, *, *]` | Transformer encoder |
| Transformer encoder | `[B, Σ(h_i×w_i), args.hidden_dim]` | same shape | Decoder (as memory) |
| TransformerDecoder (layer 2) | `[B, num_queries, args.hidden_dim]` | `[B, num_queries, args.hidden_dim]` | `hs[-1]` → MOTIP |
| `detr_output_embeds` (matched subset) | `[N_t, args.hidden_dim]` | same | TrajectoryModeling.adapter |
| TrajectoryModeling.adapter (FFN) | **expects** `[*, 256]` | `[*, 256]` | trajectory_features |
| Tracklet concat | `[*, 256]` features + `[*, 256]` id_embeds | `[*, 512]` | IDDecoder |
| IDDecoder self-attn / cross-attn | `[*, 512]` | `[*, 512]` | ID logits head |

**Critical cell:** `detr_output_embeds` has shape `[N_t, args.hidden_dim]`. `TrajectoryModeling.adapter` expects `[*, 256]`. If `args.hidden_dim = 256`, these match. If `args.hidden_dim ≠ 256`, training crashes at the first `adapter.forward()` call.

| File | Observed | Expected | Impact |
|---|---|---|---|
| `lwdetr.py:795` | `out_channels=args.hidden_dim` | `out_channels=256` | If checkpoint has `hidden_dim≠256`, Tension 2 is unresolved |
| `models/motip/__init__.py:71` | `detr_dim=config["DETR_HIDDEN_DIM"]=256` | `detr_dim=args_ckpt.hidden_dim` | Guaranteed crash if `args_ckpt.hidden_dim≠256` |
| `projector.py` (each stage) | LayerNorm after C2f | No norm (Deformable DETR uses no projector norm) | Suppresses magnitude; may reduce discriminability |

## 3. SEVERITY ASSESSMENT

**Critical (conditional on checkpoint `hidden_dim`)**

If `args.hidden_dim = 256` (confirmed by runtime inspection), Tension 2 is resolved at the projector level and the TrajectoryModeling adapter is compatible. If `args.hidden_dim ≠ 256`, the integration crashes before producing any training signal. The layer norm suppression of magnitude is medium severity regardless: it eliminates a potential discriminability signal and was not present in the original MOTIP Deformable DETR pipeline, contributing to AssA=20.5 even in the frozen-detector experiment.

## 4. OPEN QUESTIONS

- **Critical blocker:** What is `args_ckpt.hidden_dim` in `rfdetr_dancetrack_motip/checkpoint_best_total.pth`? Run: `python -c "import torch; c=torch.load('rfdetr_dancetrack_motip/checkpoint_best_total.pth', weights_only=False); print(c['args'].hidden_dim)"`
- If `hidden_dim = 256`: does DINOv2-S feature quality survive the 384→256 compression without damaging re-ID discriminability? Requires runtime feature visualization.
- Does the absence of a dedicated re-ID projection head (task-specific linear layer) contribute to the AssA collapse? MOTIP's original design uses a single-purpose detection feature space; RF-DETR's features are trained for detection with auxiliary NAS configurations.

## 5. HYPOTHESIS RANKING

\#1 [Confidence: High] — Layer norm in the projector normalizes per-frame feature magnitudes to unit scale, eliminating any activation-magnitude-based discriminability that Deformable DETR's non-normalized features would preserve; combined with NAS weight sharing, this makes `hs[-1]` features poorly suited for cross-frame re-ID.

\#2 [Confidence: High] — If `args.hidden_dim = 256`, no dimension crash occurs but the projector compresses DINOv2-S's 384-dim instance-discriminative features into a detection-optimized 256-dim space without any re-ID supervision, explaining why frozen DetA=72.9 but AssA=20.5.

\#3 [Confidence: Med] — If `args.hidden_dim ≠ 256`, the `TrajectoryModeling.adapter` (built for 256-dim input) crashes on first use, meaning reported training metrics come from a configuration that never reached the ID decoder — the HOTA numbers reflect pure detection without any tracking association.
