# DIAG_QUERY_DIVERGENCE

## 1. FINDINGS

- **RF-DETR re-ranks query slot assignments every frame via encoder top-K selection.** `models/rfdetr/models/transformer.py:246–247`: `topk_proposals_gidx = torch.topk(enc_outputs_class_unselected_gidx.max(-1)[0], topk, dim=1)[1]`. The top-K encoder memory tokens are selected by their maximum class logit. This set changes every frame based on what the encoder believes is worth attending to. Decoder query slot `k` in frame `t` is initialized from a different encoder memory token than slot `k` in frame `t−1`.

- **`prepare_for_motip` abstracts raw slot ordering via the Hungarian matcher output.** `train.py:714–715`: `pred_idxs = detr_indices[flatten_idx][0]` and `gt_idxs = detr_indices[flatten_idx][1]`. Lines 725–728 sort by `gt_idxs` to recover GT-indexed order. Line 730: `selected_pred_idxs = sorted_pred_idxs_2d[:, 0]` (first group's match per GT). Line 732: `detr_output_embeds = detr_outputs["outputs"][flatten_idx][selected_pred_idxs]`. The raw slot number is used only as an index into `hs[-1]`; the assignment of GT `j` to any slot is determined by the matcher, not by slot position.

- **Slot ordering is abstracted by the matcher; embedding content is not.** MOTIP does not assume query slot `N = object identity`. The real question is: does the embedding gathered at `selected_pred_idxs[j]` have consistent content across frames for the same object `j`? Each frame's selected slot is whatever query the Hungarian matcher assigned to GT `j` — but that query was initialized from a different encoder memory region (different spatial location, different confidence rank) in each frame.

- **NAS weight sharing means decoder layer 2 features are not specialized to depth=2.** During RF-DETR training, a random `dec_layers ∈ {1, 2, 3, …, 6}` is sampled every iteration (CLAUDE.md — confirmed NAS design). All decoder layers share weights and must produce valid outputs at any depth. Decoder layer 2 was trained while simultaneously being layer 0, 1, and later layers in other configurations. Features at inference-time layer 2 (`hs[-1]`) carry no depth-specific semantic specialization: they are not "refined final-stage features" but features that must be adequate at any intermediate depth.

- **Query count: 300 at inference, `300 × group_detr` during training.** `lwdetr.py:77`: `self.query_feat = nn.Embedding(num_queries * group_detr, hidden_dim)`. At inference, `group_detr=1` (lwdetr.py:167). Training uses `group_detr` groups, but `prepare_for_motip` selects only column 0 (`sorted_pred_idxs_2d[:, 0]`, train.py:730), discarding all group-detr duplicates. Group-detr embeddings are not averaged or ensembled — only one redundant prediction is used.

- **Gradient flow guard is an assertion, not a mechanistic guarantee.** `train.py:735–740`: if `detr_outputs["outputs"].requires_grad`, then asserts `detr_output_embeds.requires_grad`. This verifies gradient connectivity when the assertion is reached but does not prevent gradient disconnection in no_grad frames (which are gathered then concatenated with training frames at train.py:425–450).

- **Newborn objects always use `i_spec` (special embedding).** `id_decoder.py:113–119`: `unknown_id_embeds = self.generate_empty_id_embed(unknown_features=unknown_features)` fills all unknown slots with the special token regardless of frame history. This design is unchanged from original MOTIP and is not disrupted by the RF-DETR query mechanism. The newborn case does not depend on query slot consistency.

## 2. MISMATCH EVIDENCE

| File | Observed | Expected (MOTIP original) | Impact |
|---|---|---|---|
| `transformer.py:246–247` | Top-K re-ranked per frame by encoder class score | Fixed anchor points (Deformable DETR uses fixed object queries from learned `refpoint_embed`) | Different encoder memory regions initialize each frame's queries → embedding content shift per frame |
| `train.py:730` | Column-0 selection only from `group_detr` predictions | Only 1 prediction per GT in original MOTIP | No embedding ensemble; one potentially noisy prediction per object |
| `transformer.py:572` | `return_intermediate_dec=True` | N/A (Deformable DETR also returns intermediate) | Same pattern; not a new mismatch |
| NAS training (CLAUDE.md) | Layer 2 weights shared across depth configs 1–6 | Depth-specialized final-layer features | Features lack depth-semantic meaning needed for stable re-ID |

## 3. SEVERITY ASSESSMENT

**High**

The slot-ordering issue is abstracted by the Hungarian matcher and does not directly corrupt MOTIP's trajectory logic. However, the per-frame encoder-driven query initialization means that the same object's embedding in consecutive frames was computed from encoder tokens with different spatial origins and confidence ranks. NAS weight sharing compounds this by preventing any depth-specialization. AssA=20.5 (frozen) directly reflects this: features can detect objects (DetA=72.9) but cannot maintain consistent identity representations across frames.

## 4. OPEN QUESTIONS

- Does the per-frame top-K encoder re-ranking produce measurably different embedding content for the same object across frames? This is the core empirical question — requires `diag_temporal_stability_script.py` to answer quantitatively.
- Is there a systematic relationship between encoder confidence rank and embedding discriminability? (High-confidence detections may produce more consistent embeddings than low-confidence ones.)
- Does group_detr > 1 at training produce systematically different embeddings across groups for the same GT? If group 0's assignment is the worst match among the groups, discarding groups 1..G−1 would lose better embeddings.

## 5. HYPOTHESIS RANKING

\#1 [Confidence: High] — Per-frame encoder top-K reranking produces query embeddings whose content shifts with encoder confidence ordering, so the same physical object produces embeddings in different feature space regions across frames, directly explaining AssA=20.5 (features are object-class-discriminative but not identity-discriminative).

\#2 [Confidence: High] — NAS weight sharing prevents depth-specialization at decoder layer 2: `hs[-1]` features were never trained to be "final-stage refinements" and carry the statistical uncertainty of all NAS depth configurations, reducing their consistency as re-ID representations.

\#3 [Confidence: Med] — The exclusive use of group 0's Hungarian-matched prediction (column-0 selection at train.py:730) discards potentially more discriminative matches from groups 1..G−1, increasing embedding variance without a clear quality criterion for selection.
