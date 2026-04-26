# DIAG_MATCHER_ALIGNMENT

## 1. FINDINGS

- **CRITICAL BUG: `SetCriterion.forward()` overwrites `indices` at every matcher call and returns the last value.** The variable `indices` is set three times in sequence, with each call overwriting the previous result:
  1. `models/rfdetr/models/lwdetr.py:546`: `indices = self.matcher(outputs_without_aux, targets, ...)` — final decoder layer.
  2. Lines 563–573: `for i, aux_outputs in enumerate(outputs['aux_outputs']): indices = self.matcher(aux_outputs, ...)` — overwrites for each aux layer (layers 0..`dec_layers−2`).
  3. Lines 577–585: `if 'enc_outputs' in outputs: indices = self.matcher(enc_outputs, ...)` — overwrites again if `two_stage=True`.
  4. Line 598: `return losses, indices` — returns the **last** matcher's output.

- **For 3-layer RF-DETR + `aux_loss=True`: returned `indices` is from decoder layer 1 (the second-to-last layer), not the final layer.** `_set_aux_loss()` at `lwdetr.py:260–268` builds `aux_outputs` from `outputs_class[:-1]` — layers 0..`dec_layers−2`. For `dec_layers=3`, this is layers 0 and 1. The loop runs `i=0` (layer 0) then `i=1` (layer 1); after the loop, `indices` = layer 1 matching.

- **If `two_stage=True`: returned `indices` is from the encoder matcher, not any decoder layer.** The encoder output at `out['enc_outputs']` contains `pred_logits = cls_enc` and `pred_boxes = ref_enc` from the two-stage proposal head (lwdetr.py:212–217). Encoder indices are into the top-K selected memory tokens, which share index space with decoder queries only by construction (each decoder query slot `k` = initialized from encoder top-K proposal `k`).

- **`prepare_for_motip` uses these stale `detr_indices` to index `hs[-1]` (final decoder output).** `train.py:732`: `detr_output_embeds = detr_outputs["outputs"][flatten_idx][selected_pred_idxs]` where `selected_pred_idxs` derives from `detr_indices[flatten_idx]`. With iterative box refinement active (RF-DETR uses `bbox_embed` updates at `transformer.py:399–406`), the Hungarian matching at decoder layer 1 may assign GT `j` to query slot `k₁`, while the matching at layer 2 (final) assigns GT `j` to a different slot `k₂`. The embedding at `hs[-1][k₁]` is not the final-layer representation of the object matched at the final layer — it is an adjacent query's representation.

- **Matcher is called once per decoder layer per training frame; no_grad frames skip loss computation.** The group-detr training path (`if hasattr(detr_criterion, "group_detr")`) at `train.py:427–445` runs `detr_criterion(outputs=_out_bl, targets=...)` only for the first `_bl` training frames (where `_bl = min(detr_criterion_batch_len, ...)`) and `torch.no_grad()` for the rest. No_grad frames' indices are computed but their gradients are not tracked. Loss is accumulated only for training frames.

- **Detection loss weights come from checkpoint args, not MOTIP YAML.** `build_criterion_and_postprocessors(args=args_ckpt)` at `models/motip/__init__.py:64`. `weight_dict` is built at `lwdetr.py:840–851` using `args_ckpt.cls_loss_coef`, `args_ckpt.bbox_loss_coef`, `args_ckpt.giou_loss_coef`. MOTIP YAML values `DETR_CLS_LOSS_COEF=2.0`, `DETR_BBOX_LOSS_COEF=5.0`, `DETR_GIOU_LOSS_COEF=2.0` (base config lines 79–81) are passed to `detr_args` but used only for the Deformable DETR code path in `build()`. For RF-DETR, these YAML values are silently dead code.

- **Auxiliary weight dict is constructed for `dec_layers − 1` intermediate layers.** `lwdetr.py:845–851`: `for i in range(args.dec_layers - 1): aux_weight_dict.update({k + f'_{i}': v ...})`. For `dec_layers=3`, keys `loss_ce_0`, `loss_bbox_0`, `loss_giou_0`, `loss_ce_1`, `loss_bbox_1`, `loss_giou_1` are added to `weight_dict`. The total detection loss magnitude is approximately `3 × (cls + bbox + giou)` base values.

- **ID loss weight = 1.0 from YAML.** `id_criterion.weight = config["ID_LOSS_WEIGHT"] = 1.0` (base config line 55). Combined loss: `detr_loss + id_loss * 1.0` (train.py:477). With 3× detection terms, the effective detection-to-ID ratio is approximately `3 × (2.0 + 5.0 + 2.0) = 27` versus `1.0` for ID.

- **Trajectory augmentation is active and matches paper spec.** `configs/r50_deformable_detr_motip_dancetrack.yaml:41–42`: `AUG_TRAJECTORY_OCCLUSION_PROB: 0.5`, `AUG_TRAJECTORY_SWITCH_PROB: 0.5`. Applied via `data/transforms.py:456–504`. These match the paper's λ_occ = λ_sw = 0.5.

## 2. MISMATCH EVIDENCE

| File | Observed | Expected | Impact |
|---|---|---|---|
| `lwdetr.py:546,563–573,577–585` | `indices` overwritten at each of 3–4 matcher calls; returned = last | Final layer indices returned (MOTIP original Deformable DETR returns final-layer indices) | Wrong query slots indexed into `hs[-1]` for MOTIP trajectory building |
| `train.py:732` | `selected_pred_idxs` from stale `detr_indices` → indexes `hs[-1]` | Indices from final decoder layer matcher | Embedding for GT `j` may come from a neighboring query slot, not GT `j`'s actual final-layer match |
| `lwdetr.py:840–851` | Checkpoint's own loss coefs (unknown values) | MOTIP YAML: cls=2.0, bbox=5.0, giou=2.0 | Actual detection loss magnitude unknown; may differ from intended ratio |
| Total detection/ID ratio | ~27:1 (3× aux losses × base coefs) vs ID=1.0 | ~9:1 (1× final layer only) | 3× stronger detection gradient → ID loss cannot overcome the detection-optimum pull |

## 3. SEVERITY ASSESSMENT

**Critical**

The stale-indices bug means `prepare_for_motip` extracts the wrong embedding from `hs[-1]` whenever `aux_loss=True` and box positions differ between decoder layers. This is a systematic mismatch at every training step for every ground-truth object, providing consistently wrong features to the ID decoder. This single bug alone — independent of all other architectural tensions — would prevent the ID decoder from ever learning a coherent re-ID mapping, directly explaining AssA=13.9 in joint training. The 27:1 detection-to-ID gradient ratio further ensures that even if the correct embeddings were fed, detection pressure would dominate and prevent ID-compatible feature learning.

## 4. OPEN QUESTIONS

- What are `args_ckpt.aux_loss` and `args_ckpt.two_stage`? If both are True, `detr_indices` = encoder matcher indices (into a different index space entirely). If only `aux_loss=True`, `detr_indices` = last aux layer (layer `dec_layers−2`). One-liner: `python -c "import torch; c=torch.load('rfdetr_dancetrack_motip/checkpoint_best_total.pth', weights_only=False); a=c['args']; print(a.aux_loss, a.two_stage, a.cls_loss_coef, a.bbox_loss_coef, a.giou_loss_coef)"`
- With iterative box refinement, do Hungarian assignments actually differ between decoder layers 1 and 2? The degree of reassignment determines the severity of the stale-indices bug. Runtime trace with `diag_temporal_stability_script.py` can reveal this.
- Is `sum_group_losses` True or False in the checkpoint? Affects normalization of loss by `group_detr` (lwdetr.py:551).

## 5. HYPOTHESIS RANKING

\#1 [Confidence: High] — The stale-indices bug (`SetCriterion` overwrites `indices` and returns aux-layer or encoder indices) causes `prepare_for_motip` to extract wrong decoder-final-layer embeddings for every object at every training step, making the ID decoder's cross-entropy loss train against randomly misaligned features.

\#2 [Confidence: High] — The 27:1 detection-to-ID gradient ratio (3× aux + 1× final detection vs 1× ID) drives the RF-DETR decoder weights away from any re-ID-compatible feature regime, explaining the joint-training AssA collapse from 20.5 to 13.9.

\#3 [Confidence: Med] — The silently-different checkpoint loss coefficients may amplify or attenuate the detection gradient beyond what either the MOTIP YAML or the RF-DETR paper intended, creating a training regime that was never validated by either upstream system.
