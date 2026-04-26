# DIAG_DECODER_TOPOLOGY

## 1. FINDINGS

- **MOTIP receives only the final decoder layer output.** `models/rfdetr/models/lwdetr.py:221` adds `out["outputs"] = hs[-1]` with the comment "change 1: to follow motip's input format". `hs` is a stacked tensor `[dec_layers, B, num_queries, hidden_dim]`; `hs[-1]` is decoder layer index `dec_layers − 1`.

- **`return_intermediate_dec=True` is unconditionally set** in `build_transformer()` at `models/rfdetr/models/transformer.py:572`. The `TransformerDecoder` accumulates every layer output into `intermediate`, then returns `torch.stack(intermediate)` (transformer.py:433). All intermediate activations are computed, stored, and available for loss computation — they are not pruned or detached before MOTIP consumes `hs[-1]`.

- **Detection loss runs on every decoder layer + the encoder.** In `SetCriterion.forward()`:
  - Line 546: matcher called on final layer outputs (`outputs_without_aux`).
  - Lines 563–573: loop over `outputs['aux_outputs']` (= decoder layers 0..`dec_layers−2`, built at lwdetr.py:260–268 via `_set_aux_loss`). Matcher called once per entry; losses accumulated with suffix `_0`, `_1`, …
  - Lines 577–585: if `enc_outputs` present (`two_stage=True`), matcher called on encoder predictions; losses accumulated with suffix `_enc`.
  - **For 3-layer decoder + `aux_loss=True`:** 3 detection loss signals (2 aux + 1 final). If `two_stage=True`: 4 signals. Compare to MOTIP's design of 1 (final layer only).

- **ID loss backprops through `hs[-1]` only.** `prepare_for_motip` at `train.py:694–758` gathers `detr_output_embeds` from `detr_outputs["outputs"]` = `hs[-1]`. The gradient path from `id_loss` flows back through those embeddings into decoder layer `dec_layers−1` (and then through the prior layers via residual connections), not into the aux layer outputs directly. However, the detection losses at aux layers 0..`dec_layers−2` simultaneously push those same weights toward a detection optimum, creating a three-way gradient conflict on shared decoder parameters.

- **Zero-decoder config cannot silently reach MOTIP.** `transformer.py:267` guards with `if self.dec_layers > 0`; when `dec_layers=0`, `hs=None`. The line `out["outputs"] = hs[-1]` at `lwdetr.py:221` is executed unconditionally and would raise `TypeError: 'NoneType' object is not subscriptable`. The zero-decoder config terminates with an error before any MOTIP code is reached.

- **YAML loss coefficients are silently ignored for RF-DETR.** `models/motip/__init__.py:58–64` calls `build_criterion_and_postprocessors(args=args_ckpt)` with the checkpoint's own `args`, not the MOTIP YAML values. The `weight_dict` is constructed from `args_ckpt.cls_loss_coef`, `args_ckpt.bbox_loss_coef`, `args_ckpt.giou_loss_coef` at `lwdetr.py:840–842`.

- **RF-DETR checkpoint model weights are never applied to `detr` in `build()`.** `models/motip/__init__.py:59–64` loads `ckpt["args"]` for architecture but omits `detr.load_state_dict(ckpt["model"])`. The comment in `train.py:104–118` claims "weights already loaded from CKPT_PATH in build()" — this is incorrect for first-epoch training. Weights only exist if `RESUME_MODEL` is set (loaded at train.py:142–155 via `load_checkpoint()`).

## 2. MISMATCH EVIDENCE

| Location | Observed | Expected (MOTIP design) | Impact |
|---|---|---|---|
| `lwdetr.py:221` | `out["outputs"] = hs[-1]` (final layer only) | Final layer only | No mismatch on consumed output |
| `lwdetr.py:563–573` + `transformer.py:572` | 2 aux-layer detection losses (+ 1 final) = 3× gradient | 1× detection gradient (final only) | 3× detection signal vs 1× ID signal |
| `lwdetr.py:840–842` | Loss coefs from `args_ckpt` | MOTIP YAML `DETR_CLS=2.0, BBOX=5.0, GIOU=2.0` | Actual magnitudes unknown; silently diverge |
| `models/motip/__init__.py:59–64` | `ckpt["model"]` never applied | RF-DETR weights loaded from CKPT_PATH | Epoch-0 detector starts with random decoder/projector weights |

## 3. SEVERITY ASSESSMENT

**Critical**

Three simultaneous detection gradients per forward pass drive the RF-DETR decoder weights toward a multi-level detection optimum while ID loss (λ_id=1.0) provides only a single final-layer signal. This gradient imbalance directly explains the joint-training DetA collapse (72.9→40.1): the detection gradient at aux layers corrupts the final-layer features that ID loss depends on. The HOTA drop from ~70 to 23.4 follows directly from this combined pressure.

## 4. OPEN QUESTIONS

- What are `args_ckpt.aux_loss`, `args_ckpt.two_stage`, `args_ckpt.dec_layers` in the checkpoint? Resolves exact matcher call count. Run: `python -c "import torch; c=torch.load('rfdetr_dancetrack_motip/checkpoint_best_total.pth', weights_only=False); a=c['args']; print(a.aux_loss, a.two_stage, a.dec_layers)"`
- What are `args_ckpt.cls_loss_coef`, `bbox_loss_coef`, `giou_loss_coef`? Determines actual detection loss magnitude ratio to ID loss.
- Does epoch-0 training use the frozen-detector configuration (no gradient to DETR), or does the absent `ckpt["model"]` application mean the first `RESUME_MODEL`-free run trains random detector weights?

## 5. HYPOTHESIS RANKING

\#1 [Confidence: High] — RF-DETR's 3× auxiliary detection losses (vs MOTIP's 1×) generate gradient pressure on shared decoder parameters that directly corrupts the final-layer embeddings used by the ID decoder, explaining the DetA collapse in joint training.

\#2 [Confidence: High] — RF-DETR checkpoint model weights are never applied to `detr` in `build()`, so first-epoch training optimizes a randomly-initialized RF-DETR decoder against a detection+ID objective simultaneously, amplifying all instabilities.

\#3 [Confidence: Med] — The per-layer detection loss weight schedule in `weight_dict` (set from checkpoint args, not MOTIP YAML) may amplify intermediate-layer gradients beyond what the 3-layer inference config was designed to sustain, further destabilizing final-layer feature quality.
