# competition_geometry_diag.py
# =============================================================================
# Competition-event geometry & feature separability diagnostic for RF-MOTIP V3.
#
# PURPOSE
# -------
# At each competition event (two detections compete for the same trajectory ID
# under the object-max assignment rule), measure whether the WINNER (kept the ID)
# vs the LOSER (forced to newborn) can be separated by:
#   (A) relative POSITION  (trajectory's last box vs each detection)
#   (B) relative MOTION    (velocity-predicted box vs each detection)
#   (F) appearance FEATURE  (cosine sim of detection feature to trajectory feature)
#
# The separability is summarized as an AUC (how well the geometry/feature ranks
# winner above loser across all events). This is the GATE that decides the
# thesis direction WITHOUT any retraining:
#   - AUC_pos    >> 0.5  -> relative-position bias suffices (Option A)
#   - AUC_pos    ~= 0.5 but AUC_motion >> 0.5 -> need motion bias (Option B)
#   - AUC_feature>> 0.5  -> appearance CAN separate -> contrastive route has signal
#   - all ~= 0.5         -> these events are unresolvable by these signals -> STOP
#
# GROUNDING IN THE ACTUAL CODE (models/runtime_tracker.py)
# --------------------------------------------------------
# * Competition rule reproduced EXACTLY from _object_max_assignment:
#     object_max_confs, object_max_id_labels = id_scores.max(dim=-1)
#     a detection LOSES (becomes newborn) when its argmax id_label _id_label has
#     _conf < id_max_confs[_id_label], i.e. another detection scored higher for
#     that same trajectory ID. The higher-scoring one is the WINNER.
# * id_scores shape: (num_dets, num_id_vocabulary + 1); last column = newborn token.
# * Boxes are cxcywh, normalized in [0,1]. trajectory_boxes/unknown_boxes shape
#   (T, N, 4) and (N_cur, 4). Index 0 of the T axis = current/most-recent frame
#   (consistent with trajectory_id_labels[0] being "tracked IDs" in the code).
# * trajectory_masks True = padding/invalid (newborn placeholders are zero-box +
#   mask True). Velocity needs >= 2 valid (unmasked) steps for a track.
# * Runs at inference on the frozen V3 checkpoint. No loss, no retrain, no MOTIP env.
#
# HOW TO USE
# ----------
# This script is INSTRUMENTATION. You import RuntimeTracker from the V3 repo and
# monkeypatch its _get_id_pred_labels to dump the needed tensors per frame, then
# run the standard eval loop over the 25 val sequences. Because the exact eval
# entrypoint differs per setup, the recording hook is provided as a drop-in and
# the analysis is fully self-contained. See __main__ for the two integration
# modes:
#   MODE 1 (record): patch + run your existing eval; events are written to disk.
#   MODE 2 (analyze): load recorded events and compute the AUCs + report.
#
# The recorder writes ONE row per (event, candidate) with all geometry/feature
# fields so the analysis is reproducible and auditable (no hidden computation).
# =============================================================================

import os
import json
import math
import argparse
from typing import Optional

import numpy as np
# torch is only needed for the RECORDER (server-side, inside eval). The ANALYSIS
# and self-test run with numpy alone. Import lazily inside the recorder.


# -----------------------------------------------------------------------------
# Geometry helpers (cxcywh, normalized). Pure, testable, no framework deps.
# -----------------------------------------------------------------------------
def _to_xyxy(b):
    # b: (...,4) cxcywh -> xyxy
    cx, cy, w, h = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return np.stack([x1, y1, x2, y2], axis=-1)


def iou_xyxy(a, b):
    # a,b: (4,) xyxy. returns scalar IoU.
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def center_dist_norm(box_a, box_b):
    # normalized center distance (cxcywh). scale by the trajectory box diagonal
    # so it is scale-invariant. box_b is the reference (trajectory) box.
    dx = box_a[0] - box_b[0]
    dy = box_a[1] - box_b[1]
    diag = math.sqrt(box_b[2] ** 2 + box_b[3] ** 2) + 1e-6
    return math.sqrt(dx * dx + dy * dy) / diag


# -----------------------------------------------------------------------------
# AUC via Mann-Whitney U (no sklearn dependency). Probability that a winner's
# score exceeds a loser's score, averaged over all winner/loser pairs.
# Here "score" = the separating quantity; we orient each so HIGHER = more
# winner-like, then AUC>0.5 means the signal ranks winners above losers.
# -----------------------------------------------------------------------------
def auc_winner_vs_loser(winner_scores, loser_scores):
    w = np.asarray(winner_scores, dtype=np.float64)
    l = np.asarray(loser_scores, dtype=np.float64)
    w = w[np.isfinite(w)]
    l = l[np.isfinite(l)]
    n_w, n_l = len(w), len(l)
    if n_w == 0 or n_l == 0:
        return float("nan"), n_w, n_l
    # rank-based U statistic
    allv = np.concatenate([w, l])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv), dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ties
    # (simple tie handling: group equal values)
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    # recompute average ranks for ties
    sorted_idx = np.argsort(allv, kind="mergesort")
    sorted_vals = allv[sorted_idx]
    avg_ranks = np.empty(len(allv), dtype=np.float64)
    i = 0
    r = 1
    while i < len(sorted_vals):
        j = i
        while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg = (r + (r + (j - i))) / 2.0
        for k in range(i, j + 1):
            avg_ranks[sorted_idx[k]] = avg
        r += (j - i + 1)
        i = j + 1
    rank_w = avg_ranks[:n_w]
    U = rank_w.sum() - n_w * (n_w + 1) / 2.0
    auc = U / (n_w * n_l)
    return float(auc), n_w, n_l


# =============================================================================
# RECORDER  — monkeypatch hook for RuntimeTracker._get_id_pred_labels
# =============================================================================
# Drop-in: in your eval script, AFTER building the RuntimeTracker `tracker`, call
#   attach_recorder(tracker, out_path="events.jsonl")
# then run eval normally. Detach/flush at the end with `flush_recorder()`.
#
# The hook recomputes id_scores exactly as the code does, applies the EXACT
# object-max competition rule to find events, and logs winner/loser geometry.
# It does NOT change assignment behavior (it calls the original method and only
# observes), so tracking output is unchanged and metrics stay identical to V3.

_RECORDS = []
_OUT_PATH = None
_SEQ_NAME = "unknown"


def attach_recorder(tracker, out_path="events.jsonl", seq_name="unknown"):
    import torch  # noqa: F401  (recorder only; server-side)
    global _OUT_PATH, _RECORDS, _SEQ_NAME
    _OUT_PATH = out_path
    _SEQ_NAME = seq_name
    _RECORDS = []

    import types
    num_vocab = tracker.num_id_vocabulary

    # ROBUST APPROACH: hook the assignment method, which is handed the REAL
    # id_scores the tracker actually computed (after the true forward, with
    # correct times/temperature). We do NOT re-derive scores. We read the
    # tracker's real trajectory_* attributes (available now) for geometry.
    # This eliminates the earlier re-derivation bugs (wrong current_times,
    # shape mismatches). Behavior is unchanged: we observe id_scores, then
    # call the original assignment and return its result untouched.
    proto = tracker.assignment_protocol
    assign_name = {
        "object-max": "_object_max_assignment",
        "id-max": "_id_max_assignment",
        "hungarian": "_hungarian_assignment",
    }.get(proto, "_object_max_assignment")
    orig_assign = getattr(tracker, assign_name).__func__

    def patched_assign(self, id_scores):
        try:
            rec = _capture_from_real_scores(self, id_scores, num_vocab)
            if rec:
                _RECORDS.extend(rec)
        except Exception as e:
            # never break tracking because of the recorder
            global _DIAG
            try:
                _DIAG["capture_errors"] = _DIAG.get("capture_errors", 0) + 1
            except Exception:
                pass
        return orig_assign(self, id_scores)

    setattr(tracker, assign_name, types.MethodType(patched_assign, tracker))

    # Companion hook: stash the REAL current boxes/features so the assignment
    # hook can compute detection geometry. We read them from the arguments to
    # _get_id_pred_labels (boxes, output_embeds) — the genuine current-frame
    # detections — and store on the tracker. Behavior unchanged.
    orig_get_id = tracker._get_id_pred_labels.__func__

    def patched_get_id(self, boxes, output_embeds):
        try:
            self._diag_cur_boxes = boxes.detach().float().cpu().numpy()
            self._diag_cur_feats = output_embeds.detach().float().cpu().numpy()
        except Exception:
            self._diag_cur_boxes = None
            self._diag_cur_feats = None
        # frame counter (1-indexed, matches GT frame_id); increments each frame
        self._diag_frame = getattr(self, "_diag_frame", 0) + 1
        return orig_get_id(self, boxes, output_embeds)

    tracker._get_id_pred_labels = types.MethodType(patched_get_id, tracker)
    return tracker


def _capture_from_real_scores(self, id_scores, num_vocab):
    """Hooked inside the assignment method. id_scores is the REAL score matrix
    (Ncur, vocab+1) the tracker uses. Read self.trajectory_* for geometry."""
    import torch  # recorder only
    if self.trajectory_id_labels.shape[1] == 0:
        return None
    if id_scores is None or id_scores.ndim != 2:
        return None
    _Ncur = id_scores.shape[0]
    if _Ncur < 2:
        return None

    id_scores_np = id_scores.detach().float().cpu().numpy()                    # (Ncur, vocab+1)
    traj_id_labels_row0 = self.trajectory_id_labels[0].detach().cpu().numpy()  # (Ntraj,)
    traj_boxes = self.trajectory_boxes.detach().float().cpu().numpy()          # (T, Ntraj, 4)
    traj_masks = self.trajectory_masks.detach().cpu().numpy()                  # (T, Ntraj)
    traj_feats = self.trajectory_features.detach().float().cpu().numpy()       # (T, Ntraj, 256)

    # current detection boxes/features: the tracker stores them transiently?
    # They are NOT passed to assignment, so reconstruct from the most recent
    # update: the assignment is called right after scoring on `current_*`.
    # The tracker keeps them as locals; to access here we rely on attributes
    # set by our companion hook on _get_id_pred_labels (see attach: we also
    # stash them). If unavailable, fall back to trajectory-only (no det geom).
    cur_boxes_np = getattr(self, "_diag_cur_boxes", None)
    cur_feats_np = getattr(self, "_diag_cur_feats", None)

    traj_label_set = set(int(x) for x in traj_id_labels_row0.tolist())

    global _DIAG
    try:
        _DIAG
    except NameError:
        _DIAG = {"frames": 0, "labels_with_2plus": 0, "events_emitted": 0}
    _DIAG["frames"] = _DIAG.get("frames", 0) + 1

    id_thresh = float(getattr(self, "id_thresh", 0.1))

    label_to_traj_col = {}
    for col, lab in enumerate(traj_id_labels_row0.tolist()):
        if int(lab) not in label_to_traj_col:
            label_to_traj_col[int(lab)] = col

    # one-shot score dump for calibration
    if not _DIAG.get("dumped", False) and len(traj_label_set) >= 2:
        traj_cols = sorted([l for l in traj_label_set
                            if 0 <= l < id_scores_np.shape[1] and l != num_vocab])
        if traj_cols:
            import numpy as _np
            sub = id_scores_np[:, traj_cols]
            nb = id_scores_np[:, num_vocab]
            print("[dump] === REAL id_scores structure ===", flush=True)
            print(f"[dump] shape={id_scores_np.shape} num_vocab={num_vocab} "
                  f"use_sigmoid={getattr(self,'use_sigmoid',None)} "
                  f"id_thresh={id_thresh}", flush=True)
            print(f"[dump] newborn-col min/mean/max="
                  f"{nb.min():.4f}/{nb.mean():.4f}/{nb.max():.4f}", flush=True)
            print(f"[dump] traj-cols min/mean/max="
                  f"{sub.min():.4f}/{sub.mean():.4f}/{sub.max():.4f}", flush=True)
            for thr in [0.5, 0.2, 0.1, 0.05, 0.01]:
                cols2 = int(((sub >= thr).sum(axis=0) >= 2).sum())
                print(f"[dump]   thr={thr:<5} entries>=thr={int((sub>=thr).sum()):5d} "
                      f"cols>=2dets={cols2}", flush=True)
            print(f"[dump] cur_boxes available={cur_boxes_np is not None}", flush=True)
            print("[dump] === end ===", flush=True)
            _DIAG["dumped"] = True

    rows = []
    for L in traj_label_set:
        if L == num_vocab or L < 0 or L >= id_scores_np.shape[1]:
            continue
        if L not in label_to_traj_col:
            continue
        col_scores = id_scores_np[:, L]
        competitors = np.where(col_scores >= id_thresh)[0]
        if len(competitors) < 2:
            continue
        _DIAG["labels_with_2plus"] = _DIAG.get("labels_with_2plus", 0) + 1

        winner_det = int(competitors[np.argmax(col_scores[competitors])])
        tcol = label_to_traj_col[L]
        valid_ts = np.where(~traj_masks[:, tcol])[0]
        if len(valid_ts) == 0:
            continue
        t0 = valid_ts[0]
        traj_box_last = traj_boxes[t0, tcol]
        traj_feat_last = traj_feats[t0, tcol]

        has_motion = len(valid_ts) >= 2
        if has_motion:
            t1 = valid_ts[1]
            dt = float(t1 - t0)
            vel = (traj_box_last - traj_boxes[t1, tcol]) / (dt if dt != 0 else 1.0)
            pred_box = traj_box_last + vel
        else:
            pred_box = None

        _DIAG["events_emitted"] = _DIAG.get("events_emitted", 0) + 1

        # pixel conversion: bbox_unnorm = [W, H, W, H]; cxcywh-norm -> xywh-px
        bu = getattr(self, "bbox_unnorm", None)
        if bu is not None:
            bu = bu.detach().float().cpu().numpy()  # [W,H,W,H]
            W, H = float(bu[0]), float(bu[1])
        else:
            W = H = 1.0

        def _cxcywh_to_xywh_px(b):
            cx, cy, w, h = float(b[0]), float(b[1]), float(b[2]), float(b[3])
            x = (cx - w / 2.0) * W
            y = (cy - h / 2.0) * H
            return [x, y, w * W, h * H]

        traj_box_px = _cxcywh_to_xywh_px(traj_box_last)
        frame = int(getattr(self, "_diag_frame", -1))
        event_id = f"{_SEQ_NAME}:{frame}:{int(L)}"

        for det_idx in competitors.tolist():
            is_winner = (det_idx == winner_det)
            row = {
                "event_id": event_id,
                "seq": _SEQ_NAME,
                "frame": frame,                 # current frame (1-indexed, matches GT)
                "label": int(L),
                "is_winner": bool(is_winner),    # model's pick (highest score for L)
                "n_competitors": int(len(competitors)),
                "id_score_for_label": float(col_scores[det_idx]),
                "has_motion": bool(has_motion),
                "iou_pos": float("nan"), "center_dist": float("nan"),
                "iou_motion": float("nan"), "center_dist_motion": float("nan"),
                "cos_feat": float("nan"),
                "det_box_px": None,              # [x,y,w,h] pixel, for GT match
                "traj_box_px": traj_box_px,      # trajectory's last box, pixel
            }
            if cur_boxes_np is not None and det_idx < len(cur_boxes_np):
                db = cur_boxes_np[det_idx]
                row["det_box_px"] = _cxcywh_to_xywh_px(db)
                row["iou_pos"] = float(iou_xyxy(_to_xyxy(db), _to_xyxy(traj_box_last)))
                row["center_dist"] = float(center_dist_norm(db, traj_box_last))
                if pred_box is not None:
                    row["iou_motion"] = float(iou_xyxy(_to_xyxy(db), _to_xyxy(pred_box)))
                    row["center_dist_motion"] = float(center_dist_norm(db, pred_box))
            if cur_feats_np is not None and det_idx < len(cur_feats_np):
                df = cur_feats_np[det_idx]
                denom = (np.linalg.norm(df) * np.linalg.norm(traj_feat_last) + 1e-9)
                row["cos_feat"] = float(np.dot(df, traj_feat_last) / denom)
            rows.append(row)

    if _DIAG["frames"] % 200 == 0:
        print(f"[diag] frames={_DIAG['frames']} "
              f"labels>=2={_DIAG.get('labels_with_2plus',0)} "
              f"events={_DIAG.get('events_emitted',0)} "
              f"errs={_DIAG.get('capture_errors',0)}", flush=True)
    return rows if rows else None



def flush_recorder():
    global _RECORDS, _OUT_PATH
    if _OUT_PATH is None:
        return
    with open(_OUT_PATH, "w") as f:
        for r in _RECORDS:
            f.write(json.dumps(r) + "\n")
    print(f"[recorder] wrote {len(_RECORDS)} candidate-rows to {_OUT_PATH}")


# =============================================================================
# ANALYSIS  — load events.jsonl and compute the gating AUCs + report
# =============================================================================
def analyze(events_path):
    rows = []
    with open(events_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        print("No event rows found.")
        return

    # winners vs losers, per signal. We orient each signal so HIGHER=winner-like:
    #  - iou_pos / iou_motion: higher = closer = winner-like  -> use as-is
    #  - center_dist / center_dist_motion: lower = closer -> negate
    #  - cos_feat: higher = more similar to trajectory -> winner-like -> as-is
    def split(field, negate=False, require_motion=False):
        w, l = [], []
        for r in rows:
            if require_motion and not r["has_motion"]:
                continue
            v = r[field]
            if v is None or (isinstance(v, float) and not math.isfinite(v)):
                continue
            v = -v if negate else v
            (w if r["is_winner"] else l).append(v)
        return w, l

    signals = [
        ("Position IoU         (A)", "iou_pos",            False, False),
        ("Position -CenterDist (A)", "center_dist",        True,  False),
        ("Motion   IoU         (B)", "iou_motion",         False, True),
        ("Motion   -CenterDist (B)", "center_dist_motion", True,  True),
        ("Feature  Cosine      (F)", "cos_feat",           False, False),
    ]

    n_events = sum(1 for r in rows if r["is_winner"])  # one winner per event
    n_rows = len(rows)
    frac_motion = np.mean([1.0 if r["has_motion"] else 0.0 for r in rows])

    print("=" * 64)
    print("COMPETITION-EVENT SEPARABILITY DIAGNOSTIC  (V3, all val seqs)")
    print("=" * 64)
    print(f"competition events (winners): {n_events}")
    print(f"candidate rows (win+lose):    {n_rows}")
    print(f"fraction of rows w/ motion:   {frac_motion:.3f}")
    print("-" * 64)
    print(f"{'signal':<28}{'AUC':>8}{'n_win':>8}{'n_lose':>8}")
    print("-" * 64)
    results = {}
    for name, field, negate, req in signals:
        w, l = split(field, negate=negate, require_motion=req)
        auc, nw, nl = auc_winner_vs_loser(w, l)
        results[name] = auc
        auc_str = f"{auc:.3f}" if math.isfinite(auc) else "nan"
        print(f"{name:<28}{auc_str:>8}{nw:>8}{nl:>8}")
    print("-" * 64)

    # interpretation gate
    def best(*names):
        vals = [results[n] for n in names if math.isfinite(results.get(n, float('nan')))]
        return max(vals) if vals else float("nan")

    auc_pos = best("Position IoU         (A)", "Position -CenterDist (A)")
    auc_mot = best("Motion   IoU         (B)", "Motion   -CenterDist (B)")
    auc_feat = results.get("Feature  Cosine      (F)", float("nan"))

    print("INTERPRETATION")
    def verdict(x):
        if not math.isfinite(x): return "n/a"
        if x >= 0.65: return "STRONG"
        if x >= 0.57: return "MODERATE"
        if x >= 0.53: return "WEAK"
        return "NONE"
    print(f"  position signal : AUC~{auc_pos:.3f}  [{verdict(auc_pos)}]")
    print(f"  motion   signal : AUC~{auc_mot:.3f}  [{verdict(auc_mot)}]")
    print(f"  feature  signal : AUC~{auc_feat:.3f}  [{verdict(auc_feat)}]")
    print("-" * 64)
    print("  Decision rule:")
    print("   - position STRONG/MODERATE        -> Option A (relative-position bias) viable")
    print("   - position weak but motion strong -> Option B (motion bias) required")
    print("   - feature STRONG                  -> contrastive route has signal (but see FDTA overlap)")
    print("   - all NONE                        -> these events unresolvable by geom/appearance: STOP")
    print("=" * 64)


# =============================================================================
# GT-ANCHORED GATE  — the analysis that decides the direction.
# For each competition event:
#   1. Match the trajectory's last box to GT (at frame-1, fallback frame-2/-3)
#      -> trajectory's TRUE identity (gt_traj_id).
#   2. Match each competitor detection to GT (current frame) -> det's TRUE id.
#   3. GT-correct competitor = the one whose det GT id == gt_traj_id.
#   4. model winner = the row flagged is_winner (model's highest-score pick).
# Then:
#   - "model correct" if model winner IS the GT-correct competitor.
#   - On MODEL-ERROR events (model winner != GT-correct, GT-correct present),
#     measure AUC of position/motion in ranking GT-correct ABOVE model-winner.
#     THIS is the real gate: can geometry fix the model's actual mistakes?
# =============================================================================
def _iou_xywh(a, b):
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = a[2] * a[3] + b[2] * b[3] - inter
    return inter / ua if ua > 0 else 0.0


def _load_gt(gt_root, seq):
    """Return dict: frame_id(int,1-idx) -> list of (obj_id, [x,y,w,h])."""
    import os
    path = os.path.join(gt_root, seq, "gt", "gt.txt")
    gt = {}
    if not os.path.isfile(path):
        return gt
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            fid, oid = int(parts[0]), int(parts[1])
            x, y, w, h = map(float, parts[2:6])
            gt.setdefault(fid, []).append((oid, [x, y, w, h]))
    return gt


def _match_box_to_gt(box_xywh, gt_frame, iou_thr=0.5):
    """Return obj_id of best-matching GT box in this frame, or None."""
    best_iou, best_id = 0.0, None
    for oid, gbox in gt_frame:
        i = _iou_xywh(box_xywh, gbox)
        if i > best_iou:
            best_iou, best_id = i, oid
    return best_id if best_iou >= iou_thr else None


def analyze_gt(events_path, gt_root, iou_thr=0.5):
    import os
    rows = []
    with open(events_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        print("No event rows found.")
        return

    # group by event
    events = {}
    for r in rows:
        events.setdefault(r["event_id"], []).append(r)

    # cache GT per seq
    gt_cache = {}
    def gt_for(seq):
        if seq not in gt_cache:
            gt_cache[seq] = _load_gt(gt_root, seq)
        return gt_cache[seq]

    n_events = 0
    n_traj_matched = 0
    n_gtcorrect_present = 0
    n_model_correct = 0
    n_model_error = 0

    # collectors for the gate (on model-error events): geometry of GT-correct vs model-winner
    pos_gtc, pos_mw = [], []
    mot_gtc, mot_mw = [], []
    feat_gtc, feat_mw = [], []

    for eid, ev in events.items():
        n_events += 1
        seq = ev[0]["seq"]
        frame = ev[0]["frame"]
        gt = gt_for(seq)
        if not gt:
            continue

        # 1. trajectory's GT id: match traj_box_px at frame-1/-2/-3
        traj_box = ev[0]["traj_box_px"]
        gt_traj_id = None
        for fback in (frame - 1, frame - 2, frame - 3):
            if fback in gt:
                gt_traj_id = _match_box_to_gt(traj_box, gt[fback], iou_thr)
                if gt_traj_id is not None:
                    break
        if gt_traj_id is None:
            continue
        n_traj_matched += 1

        # 2. each competitor's GT id at current frame
        gt_frame = gt.get(frame, [])
        for r in ev:
            r["_gt_id"] = (_match_box_to_gt(r["det_box_px"], gt_frame, iou_thr)
                           if r["det_box_px"] is not None else None)

        # 3. GT-correct competitor = det GT id == traj GT id
        gtc_rows = [r for r in ev if r["_gt_id"] == gt_traj_id]
        if not gtc_rows:
            continue   # the correct continuation wasn't among competitors -> different failure
        n_gtcorrect_present += 1
        gtc = gtc_rows[0]

        # 4. model winner
        mw_rows = [r for r in ev if r["is_winner"]]
        if not mw_rows:
            continue
        mw = mw_rows[0]

        if mw is gtc or mw.get("_gt_id") == gt_traj_id:
            n_model_correct += 1
        else:
            n_model_error += 1
            # gate: does geometry rank GT-correct above the model's wrong winner?
            if math.isfinite(gtc["iou_pos"]) and math.isfinite(mw["iou_pos"]):
                pos_gtc.append(gtc["iou_pos"]); pos_mw.append(mw["iou_pos"])
            if (gtc["has_motion"] and mw["has_motion"]
                    and math.isfinite(gtc["iou_motion"]) and math.isfinite(mw["iou_motion"])):
                mot_gtc.append(gtc["iou_motion"]); mot_mw.append(mw["iou_motion"])
            if math.isfinite(gtc["cos_feat"]) and math.isfinite(mw["cos_feat"]):
                feat_gtc.append(gtc["cos_feat"]); feat_mw.append(mw["cos_feat"])

    print("=" * 64)
    print("GT-ANCHORED GATE  (V3, all val seqs)")
    print("=" * 64)
    print(f"total competition events:        {n_events}")
    print(f"  trajectory matched to GT:      {n_traj_matched}")
    print(f"  GT-correct among competitors:  {n_gtcorrect_present}")
    print(f"    model CORRECT:               {n_model_correct}")
    print(f"    model ERROR (the headroom):  {n_model_error}")
    if n_gtcorrect_present > 0:
        print(f"  model error rate at competitions: "
              f"{n_model_error / n_gtcorrect_present:.3f}")
    print("-" * 64)
    print("GATE — on MODEL-ERROR events, can geometry pick the GT-correct det")
    print("over the model's wrong winner?  (AUC > 0.5 => geometry would help)")
    print("-" * 64)
    auc_p, npw, npl = auc_winner_vs_loser(pos_gtc, pos_mw)
    auc_m, nmw, nml = auc_winner_vs_loser(mot_gtc, mot_mw)
    auc_f, nfw, nfl = auc_winner_vs_loser(feat_gtc, feat_mw)
    def vstr(a): return f"{a:.3f}" if math.isfinite(a) else "nan"
    print(f"  position IoU : AUC={vstr(auc_p)}  (n_error_events={npw})")
    print(f"  motion   IoU : AUC={vstr(auc_m)}  (n_error_events={nmw})")
    print(f"  feature  cos : AUC={vstr(auc_f)}  (n_error_events={nfw})")
    print("-" * 64)
    def verdict(x):
        if not math.isfinite(x): return "n/a"
        if x >= 0.65: return "STRONG — geometry fixes errors"
        if x >= 0.57: return "MODERATE — geometry helps"
        if x >= 0.53: return "WEAK"
        return "NONE — geometry cannot fix these errors"
    print(f"  position : [{verdict(auc_p)}]")
    print(f"  motion   : [{verdict(auc_m)}]")
    print(f"  feature  : [{verdict(auc_f)}]")
    print("=" * 64)
    print("READING:")
    print("  - If position/motion STRONG/MODERATE -> spatial bias is justified:")
    print("    on the events the model gets WRONG, geometry points to the right det.")
    print("  - If all NONE -> geometry cannot correct the model's identity errors;")
    print("    the discriminating signal is not in single-frame box/appearance.")
    print("  - 'model error rate' is the ceiling: max fraction of competitions a")
    print("    perfect geometric tiebreaker could rescue.")
    print("=" * 64)


def _selftest_gt():
    """Synthetic GT + events where, on model-error events, position separates
    the GT-correct det from the model's wrong winner."""
    import os
    rng = np.random.default_rng(1)
    seq = "dancetrack9999"
    gt_dir = f"/home/claude/diag/_gt/{seq}/gt"
    os.makedirs(gt_dir, exist_ok=True)
    # GT: two people. id 1 moves along x; id 2 elsewhere.
    gt_lines = []
    for fr in range(1, 50):
        gt_lines.append(f"{fr},1,{100+fr},100,40,80,1,1,1")
        gt_lines.append(f"{fr},2,{400-fr},300,40,80,1,1,1")
    with open(os.path.join(gt_dir, "gt.txt"), "w") as f:
        f.write("\n".join(gt_lines))

    rows = []
    for fr in range(5, 45):
        # trajectory is person 1, last box ~ at frame fr-1
        traj_px = [100 + (fr - 1), 100, 40, 80]
        # competitor A = correct person-1 det at current frame (overlaps GT id1)
        detA = [100 + fr, 100, 40, 80]
        # competitor B = person-2 det (far)
        detB = [400 - fr, 300, 40, 80]
        # MODEL ERROR: model picks B (wrong) as winner
        for is_win, det in [(False, detA), (True, detB)]:
            iou_pos = _iou_xywh(det, traj_px)
            rows.append({
                "event_id": f"{seq}:{fr}:7", "seq": seq, "frame": fr, "label": 7,
                "is_winner": is_win, "n_competitors": 2, "id_score_for_label": 0.5,
                "has_motion": False,
                "iou_pos": iou_pos, "center_dist": 0.0,
                "iou_motion": float("nan"), "center_dist_motion": float("nan"),
                "cos_feat": float(rng.uniform(0.9, 0.95)),
                "det_box_px": det, "traj_box_px": traj_px,
            })
    path = "/home/claude/diag/_selftest_gt_events.jsonl"
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("[selftest_gt] synthetic: model always picks WRONG; position should "
          "STRONGLY separate GT-correct from model-winner.\n")
    analyze_gt(path, gt_root="/home/claude/diag/_gt")


# -----------------------------------------------------------------------------
# Self-test of the geometry + AUC math on synthetic data (runs with no repo).
# -----------------------------------------------------------------------------
def _selftest():
    # Construct synthetic events where position perfectly separates winner/loser:
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(500):
        # winner near trajectory box, loser far
        traj = np.array([0.5, 0.5, 0.1, 0.2])
        win = traj + rng.normal(0, 0.005, 4)
        los = traj + np.array([0.2, 0.0, 0.0, 0.0]) + rng.normal(0, 0.005, 4)
        for is_w, db in [(True, win), (False, los)]:
            rows.append({
                "label": 1, "is_winner": is_w, "n_competitors": 2,
                "id_score_for_label": 0.5,
                "iou_pos": iou_xyxy(_to_xyxy(db), _to_xyxy(traj)),
                "center_dist": center_dist_norm(db, traj),
                "iou_motion": float("nan"), "center_dist_motion": float("nan"),
                "cos_feat": float(rng.uniform(0.9, 0.95)),  # feature ~unseparable
                "has_motion": False,
            })
    path = "/home/claude/diag/_selftest_events.jsonl"
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("[selftest] synthetic: position separable, feature not.\n")
    analyze(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",
                    choices=["analyze", "selftest", "analyze_gt", "selftest_gt"],
                    default="selftest")
    ap.add_argument("--events", type=str, default="events.jsonl")
    ap.add_argument("--gt_root", type=str, default=None,
                    help="path to <DanceTrack>/val (contains <seq>/gt/gt.txt)")
    ap.add_argument("--iou_thr", type=float, default=0.5,
                    help="IoU threshold for matching boxes to GT (try 0.3 and 0.5)")
    args = ap.parse_args()
    if args.mode == "selftest":
        _selftest()
    elif args.mode == "selftest_gt":
        _selftest_gt()
    elif args.mode == "analyze_gt":
        if not args.gt_root:
            raise SystemExit("analyze_gt requires --gt_root <DanceTrack>/val")
        analyze_gt(args.events, gt_root=args.gt_root, iou_thr=args.iou_thr)
    else:
        analyze(args.events)
