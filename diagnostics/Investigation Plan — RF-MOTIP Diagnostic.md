# Investigation Plan — RF-MOTIP Diagnostic

## Context
RF-DETR-small is integrated as the detector backbone inside MOTIP. Training converges
but tracking is severely degraded (HOTA ~38 frozen, ~23 joint). The goal is to locate
every architectural mismatch between the two codebases before proposing any fix.

---

## Repo Map

```
/home/user/new_rfmotip/
├── train.py                          ← main training loop, prepare_for_motip, DETR dispatch
├── configs/
│   ├── r50_deformable_detr_motip_dancetrack.yaml   ← base/super config (DETR_HIDDEN_DIM=256)
│   └── rf_detr_motip_dancetrack.yaml               ← RF-DETR overlay config
├── models/
│   ├── motip/
│   │   ├── __init__.py               ← build(): RF-DETR checkpoint loading (CKPT_PATH)
│   │   ├── motip.py                  ← MOTIP wrapper (detr/trajectory_modeling/id_decoder)
│   │   ├── id_decoder.py             ← IDDecoder: 6-layer transformer, expects 512-dim input
│   │   ├── id_criterion.py           ← ID cross-entropy loss
│   │   └── trajectory_modeling.py   ← FFN adapter (256→256, residual)
│   └── rfdetr/
│       └── models/
│           ├── lwdetr.py             ← LWDETR class, forward(), SetCriterion, build_model()
│           ├── transformer.py        ← TransformerDecoder, query selection (top-K)
│           └── backbone/
│               ├── dinov2_with_windowed_attn.py  ← DINOv2-S backbone
│               └── projector.py      ← MultiScaleProjector (C2f blocks → out_channels)
└── data/
    └── dancetrack.py                 ← seq dir layout: {root}/{split}/{seq}/img1/XXXXXXXX.jpg
```

---

## Axis 1 — Decoder Topology

### Target file:line locations
| What | File:Line |
|---|---|
| RF-DETR forward() return dict | `models/rfdetr/models/lwdetr.py:190–223` |
| `out["outputs"] = hs[-1]` (MOTIP compatibility patch) | `models/rfdetr/models/lwdetr.py:221` |
| SetCriterion.forward() — final layer matcher | `models/rfdetr/models/lwdetr.py:546` |
| SetCriterion.forward() — aux_outputs loop (per intermediate layer) | `models/rfdetr/models/lwdetr.py:563–573` |
| SetCriterion.forward() — encoder matcher | `models/rfdetr/models/lwdetr.py:575–585` |
| train.py criterion dispatch (RF-DETR takes `group_detr` branch) | `train.py:427–445` |
| DETR loss summed, ID loss added | `train.py:474–477` |

### Key findings
- RF-DETR `SetCriterion` has `group_detr` attribute → train.py takes the group-detr path (lines 427–445), not the simpler Deformable DETR path (lines 447–449).
- Matcher call count per forward pass with RF-DETR:
  - 1 call for final decoder layer
  - N-1 calls for aux layers (if `args.aux_loss=True` in checkpoint)
  - 1 call for encoder outputs (if `args.two_stage=True` in checkpoint)
  - Total: up to **4 matcher calls** with 3 decoder layers + encoder
  - Compare to Deformable DETR: 6+1=7 calls (6 dec layers + encoder)
- Loss weights come from `build_criterion_and_postprocessors(args=args_ckpt)` — i.e., from the **RF-DETR checkpoint args**, NOT from the MOTIP YAML config values (`DETR_CLS_LOSS_COEF`, etc.). Those YAML keys are silently ignored for RF-DETR.
- `out["outputs"] = hs[-1]` (line 221) was added explicitly for MOTIP compat — confirmed code annotation says "change 1: to follow motip's input format".

### Key question for Phase 2
Are `args.aux_loss` and `args.two_stage` both `True` in the saved checkpoint, making the total matcher/loss call count 4? And does the aux weight dict (suffixed `_0`, `_1`, `_enc`) get properly included in `detr_weight_dict` at line 473?

### Blocker
`args.aux_loss` and `args.two_stage` values inside `CKPT_PATH` checkpoint are unknown without opening the file at runtime. Cannot confirm total matcher call count from code alone.

---

## Axis 2 — Feature Space

### Target file:line locations
| What | File:Line |
|---|---|
| Projector output channel = `out_channels` param | `models/rfdetr/models/backbone/projector.py:225` |
| Projector instantiation: `out_channels=args.hidden_dim` | `models/rfdetr/models/lwdetr.py:795` |
| MOTIP `TrajectoryModeling` initialized with `detr_dim=config["DETR_HIDDEN_DIM"]` | `models/motip/__init__.py:71` |
| `DETR_HIDDEN_DIM: 256` in base config | `configs/r50_deformable_detr_motip_dancetrack.yaml:68` |
| Feature dim inferred at runtime: `_feature_dim = detr_outputs["outputs"].shape[-1]` | `train.py:698` |
| Tracklet concat: `cat([trajectory_features, trajectory_id_embeds], dim=-1)` | `models/motip/id_decoder.py:111` |
| Unknown concat: `cat([unknown_features, unknown_id_embeds], dim=-1)` | `models/motip/id_decoder.py:112` |

### Key findings
- The projector's output dimension is `args.hidden_dim` from the RF-DETR checkpoint. DINOv2-S outputs 384-dim; the projector must project 384→`args.hidden_dim`.
- MOTIP's `trajectory_modeling` adapter expects `detr_dim=256` (from MOTIP config). If the checkpoint has `args.hidden_dim ≠ 256`, the adapter input size mismatches the checkpoint weight shape and training would crash with a shape error.
- `_feature_dim` is inferred at runtime from `detr_outputs["outputs"].shape[-1]`, so MOTIP silently adapts to whatever dimension RF-DETR emits — but IDDecoder's cross-attention is built with `feature_dim=config["FEATURE_DIM"]=256`. If RF-DETR emits 256-dim, cross-attention is compatible; if not, there is a silent broadcast error or a crash.
- `FEATURE_DIM: 256` and `ID_DIM: 256` both from YAML → tracklet concat expects 256+256=512.

### Key question for Phase 2
What is `args.hidden_dim` in the checkpoint? If it is 256, Tension 2 is resolved at the projector level. If it is 384 or another value, the integration is broken at the concat boundary.

### Blocker
`args.hidden_dim` in the RF-DETR checkpoint cannot be confirmed without reading the checkpoint at runtime (or printing `ckpt["args"]`). This is the single most important blocker.

---

## Axis 3 — Query Mechanism

### Target file:line locations
| What | File:Line |
|---|---|
| RF-DETR top-K query selection at encoder output | `models/rfdetr/models/transformer.py:246–247` |
| `prepare_for_motip` entry point | `train.py:694` |
| Matcher indices extracted: `pred_idxs`, `gt_idxs` | `train.py:714–715` |
| Sort-by-GT ordering (block layout) | `train.py:725–728` |
| Select column-0 representative per GT | `train.py:730` |
| Embed/box lookup from `detr_outputs["outputs"]` | `train.py:732–733` |
| Features assigned to MOTIP trajectory/unknown slots | `train.py:755–758` |

### Key findings
- `prepare_for_motip` uses **Hungarian matcher output** (GT-indexed), not raw query slot indices, to fill `trajectory_features` and `unknown_features`. This means the system already abstracts away query ordering — what matters is which embedding the matcher assigned to each GT box, not which query slot it came from.
- **BUT**: RF-DETR's query selection changes per frame (top-K by encoder confidence). This means the embedding at query slot N in frame t is physically different from slot N in frame t-1. For MOTIP's trajectory modeling, what matters is the *content* of the embedding (how discriminative it is for re-ID), not the slot number.
- The real tension is not slot ordering (abstracted by matcher) but **embedding discriminability**: are RF-DETR's final-layer embeddings (`hs[-1]`) discriminative enough for re-ID across frames? This is what AssA=20.5 suggests they are not.
- The gradient flow check at `train.py:735–740` is a runtime assertion, not a code-level guarantee.

### Key question for Phase 2
Does the encoder-based top-K selection in RF-DETR (confidence-ordered) introduce query-to-query inconsistency in the *embedding content* (not just ordering)? Specifically, does a high-confidence detection in one frame produce a different embedding than the same object detected at lower confidence in the next frame?

### Blocker
None from code reading; the feature discriminability question requires a runtime trace or the `diag_temporal_stability_script.py`.

---

## Axis 4 — Matcher Alignment

### Target file:line locations
| What | File:Line |
|---|---|
| Final layer matcher call in SetCriterion | `models/rfdetr/models/lwdetr.py:546` |
| Aux layer matcher loop | `models/rfdetr/models/lwdetr.py:563–573` |
| Encoder matcher call | `models/rfdetr/models/lwdetr.py:577` |
| weight_dict built from checkpoint args (not YAML) | `models/rfdetr/models/lwdetr.py:840–842` |
| Aux weight dict construction | `models/rfdetr/models/lwdetr.py:845–851` (probable range — confirmed at build_criterion_and_postprocessors) |
| Loss computation: `sum(detr_loss_dict[k] * detr_weight_dict[k] ...)` | `train.py:474–476` |
| Combined loss: `detr_loss + id_loss * id_criterion.weight` | `train.py:477` |
| ID loss weight = 1.0 | `configs/r50_deformable_detr_motip_dancetrack.yaml:55` |

### Key findings
- MOTIP YAML config values `DETR_CLS_LOSS_COEF`, `DETR_BBOX_LOSS_COEF`, `DETR_GIOU_LOSS_COEF` are passed to `detr_args` and then used only for Deformable DETR. For RF-DETR, `build_criterion_and_postprocessors(args=args_ckpt)` uses the checkpoint's own loss weights — the YAML values are silently ignored.
- The Hungarian matcher is called once per decoder layer + encoder per forward pass (for training frames only — no_grad frames skip the loss computation).
- RF-DETR's auxiliary losses produce gradient signals at every intermediate decoder layer (layers 0 and 1 in the 3-layer config), in addition to the final layer and encoder. MOTIP's original design has loss only at the final layer, making gradient pressure 3–4× more concentrated on the DETR weights than originally designed.

### Key question for Phase 2
What loss weights does the RF-DETR checkpoint store for `cls_loss_coef`, `bbox_loss_coef`, `giou_loss_coef`? Are they the same as the MOTIP YAML (2.0, 5.0, 2.0) or different?

### Blocker
Same as Axis 1: need `ckpt["args"]` values for `aux_loss`, `two_stage`, `cls_loss_coef`, `bbox_loss_coef`, `giou_loss_coef`.

---

## Axis 5 — Script Inputs (Temporal Stability Script)

### Embedding extraction call
- **Function**: `model(frames=batch_frames, part="detr")` → `self.detr(samples=frames)` → returns `out` dict
- **Embedding key**: `out["outputs"]` = `hs[-1]` (last decoder layer hidden states)
- **File:Line**: `models/rfdetr/models/lwdetr.py:221` (`out["outputs"] = hs[-1]`)
- **Shape**: `[B*T, num_queries, hidden_dim]` where `hidden_dim = args.hidden_dim` from checkpoint

### Checkpoint loading pattern
- **File:Line**: `models/misc.py:157–165`
- **Pattern**:
  ```python
  load_state = torch.load(path, map_location=..., weights_only=False)
  model_state = load_state["model"]
  model.load_state_dict(model_state)
  ```
- **Keys in checkpoint**: `model`, `optimizer`, `scheduler`, `states`
- **Special case**: If `"bbox_embed.0.layers.0.weight" in model_state`, redirects to `load_detr_pretrain()`

### Sequence directory format
- **Root structure**: `{DATA_ROOT}/DanceTrack/{split}/{sequence_name}/`
- **Images**: `img1/XXXXXXXX.jpg` (8-digit, 1-indexed, zero-padded)
- **Ground truth**: `gt/gt.txt` (CSV: frame_id,obj_id,x,y,w,h,…)
- **Sequence metadata**: `seqinfo.ini` (keys: `imWidth`, `imHeight`, `seqLength`)
- **Source**: `data/dancetrack.py:63–67`

### Blocker
**Critical**: The build() function in `models/motip/__init__.py:58–64` loads `ckpt["args"]` from `CKPT_PATH` but **never applies `ckpt["model"]` state dict to the `detr` model**. The comment in `train.py:104–118` claims "weights already loaded from CKPT_PATH in build()" but this is incorrect — the state dict is never applied. RF-DETR starts with random decoder/projector weights (backbone gets DINOv2 pretrained weights via `load_dinov2_weights=args.pretrain_weights is None`). Trained weights only appear if `RESUME_MODEL` checkpoint exists and is loaded.

---

## Open Blockers Summary

| # | Blocker | Axis | How to resolve |
|---|---|---|---|
| B1 | `args.hidden_dim` in CKPT_PATH checkpoint is unconfirmed (expected 256, DINOv2-S native is 384) | 2 | `python -c "import torch; c=torch.load('rfdetr_dancetrack_motip/checkpoint_best_total.pth', weights_only=False); print(c['args'].hidden_dim)"` |
| B2 | `args.aux_loss` and `args.two_stage` in checkpoint unknown → can't confirm total matcher call count | 1, 4 | Same one-liner: print `c['args'].aux_loss`, `c['args'].two_stage` |
| B3 | RF-DETR checkpoint loss weights may differ from MOTIP YAML (silently ignored for RF-DETR) | 4 | Print `c['args'].cls_loss_coef`, `bbox_loss_coef`, `giou_loss_coef` |
| B4 | `ckpt["model"]` is never applied to `detr` in build() — RF-DETR detector weights may be randomly initialized at epoch 0 | 5 | Code audit: verify whether any downstream call loads the state dict, or confirm training always resumes from a MOTIP checkpoint |
| B5 | Feature discriminability of RF-DETR `hs[-1]` for re-ID requires runtime trace | 3 | `diag_temporal_stability_script.py` (Axis-5 deliverable) |
