"""
DIAG 6 — Targeted No-Newborn Override
=======================================
Inference only. GPU required. Run from repo root:
    python diagnostics/diag_script6_no_newborn_override.py

Confirmed context (FULL_DIAGNOSTIC_REPORT.md):
  - Case B = 95-100% of all spurious newborns: correct label score >= id_thresh
    but detection still assigned newborn due to competition conflict.
  - Case B does NOT tell us whether the model's ARGMAX is the correct label.

This diagnostic distinguishes:
  Scenario α: argmax IS the correct label — assignment rule blocked it.
               Fix is inference-only (protocol change), no training needed.
  Scenario β: argmax is a WRONG label — model itself is confused.
               Fix requires training to sharpen IDDecoder under high density.

Also simulates TARGETED OVERRIDE: for qualifying objects (argmax in tracked
set AND conf >= id_thresh), skip the competition check.
"""

import os
import sys
import math
import types
import torch
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CONFIG_PATH     = "configs/rf_detrV2_motip_dancetrack.yaml"
CHECKPOINT_PATH = "ch/checkpoint_2.pth"
VAL_DIR         = "/data/pos+mot/Datadir/DanceTrack/val"
OUTPUT_DIR      = "diagnostics/diag6_outputs"
MAX_FRAMES      = 300
IOU_THRESH      = 0.5


# ---------------------------------------------------------------------------
# Config / model builders
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    config_path = os.path.abspath(config_path)
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}
    super_path = config.get("SUPER_CONFIG_PATH", None)
    if super_path:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        super_abs = os.path.join(repo_root, super_path)
        if not os.path.exists(super_abs):
            super_abs = os.path.abspath(os.path.join(os.path.dirname(config_path), super_path))
        super_config = load_config(super_abs)
        merged = {**super_config, **config}
        merged.pop("SUPER_CONFIG_PATH", None)
        return merged
    return config


def build_model(config: dict, device: torch.device):
    from models.rfdetr.models.lwdetr import build_model as build_rfdetr
    from models.motip.trajectory_modeling import TrajectoryModeling
    from models.motip.id_decoder import IDDecoder
    from models.motip.motip import MOTIP

    det_ckpt   = torch.load(config["CKPT_PATH"], map_location="cpu", weights_only=False)
    args_ckpt  = det_ckpt["args"]
    args_ckpt.num_classes -= 1
    detr = build_rfdetr(args=args_ckpt)
    args_ckpt.num_classes += 1
    det_state  = det_ckpt.get("model", None)
    if det_state is not None:
        model_sd = detr.state_dict()
        filtered = {(k[5:] if k.startswith("detr.") else k): v
                    for k, v in det_state.items()
                    if (k[5:] if k.startswith("detr.") else k) in model_sd
                    and v.shape == model_sd[(k[5:] if k.startswith("detr.") else k)].shape}
        detr.load_state_dict(filtered, strict=False)

    traj_mod = TrajectoryModeling(
        detr_dim=config["DETR_HIDDEN_DIM"],
        ffn_dim_ratio=config["FFN_DIM_RATIO"],
        feature_dim=config["FEATURE_DIM"],
    )
    id_dec = IDDecoder(
        feature_dim=config["FEATURE_DIM"],
        id_dim=config["ID_DIM"],
        ffn_dim_ratio=config["FFN_DIM_RATIO"],
        num_layers=config["NUM_ID_DECODER_LAYERS"],
        head_dim=config["HEAD_DIM"],
        num_id_vocabulary=config["NUM_ID_VOCABULARY"],
        rel_pe_length=config["REL_PE_LENGTH"],
        use_aux_loss=config["USE_AUX_LOSS"],
        use_shared_aux_head=config["USE_SHARED_AUX_HEAD"],
    )
    model = MOTIP(detr=detr, detr_framework="rf_detr", only_detr=False,
                  trajectory_modeling=traj_mod, id_decoder=id_dec)

    ckpt  = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    state = {k.replace("module.", ""): v for k, v in ckpt.get("model", ckpt).items()}
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    print(f"  Model loaded on {device}")
    return model


def build_tracker(model, config: dict, sequence_hw: tuple):
    from models.runtime_tracker import RuntimeTracker
    dtype_str = str(config.get("INFERENCE_DTYPE", "FP32")).upper()
    dtype = torch.float16 if dtype_str == "FP16" else torch.float32
    return RuntimeTracker(
        model=model,
        sequence_hw=sequence_hw,
        use_sigmoid=config.get("USE_SIGMOID", False),
        assignment_protocol=config.get("ASSIGNMENT_PROTOCOL", "object-max"),
        miss_tolerance=config.get("MISS_TOLERANCE", 30),
        det_thresh=config.get("DET_THRESH", 0.3),
        newborn_thresh=config.get("NEWBORN_THRESH", 0.6),
        id_thresh=config.get("ID_THRESH", 0.2),
        area_thresh=config.get("AREA_THRESH", 0),
        only_detr=False,
        dtype=dtype,
    )


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image_as_nested_tensor(img_path: str, max_longer: int,
                                 size_div: int, device: torch.device):
    """
    Replicates SeqDataset preprocessing:
      resize longest side → max_longer, enforce size_divisibility, normalize.
    Returns (NestedTensor, (H, W)) where H,W are the resized dimensions.
    """
    from PIL import Image
    import torchvision.transforms.functional as TF

    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size

    scale = max_longer / max(orig_h, orig_w)
    new_h = math.ceil(orig_h * scale / size_div) * size_div
    new_w = math.ceil(orig_w * scale / size_div) * size_div

    img    = img.resize((new_w, new_h), Image.BILINEAR)
    tensor = TF.to_tensor(img)
    tensor = TF.normalize(tensor,
                          mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])

    tensors = tensor.unsqueeze(0).to(device)
    mask    = torch.zeros(1, new_h, new_w, dtype=torch.bool, device=device)

    from utils.nested_tensor import NestedTensor


    return NestedTensor(tensors=tensors, mask=mask), (new_h, new_w)


# ---------------------------------------------------------------------------
# GT loading and IoU
# ---------------------------------------------------------------------------

def load_gt(seq_dir: str) -> dict:
    gt_path = os.path.join(seq_dir, "gt", "gt.txt")
    gt = defaultdict(dict)
    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue
            fid, tid = int(parts[0]), int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            if int(float(parts[6])) == 0:
                continue
            gt[fid][tid] = [x, y, w, h]
    return gt


def iou_xywh(a, b):
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    iy = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2]*a[3] + b[2]*b[3] - inter
    return inter / union if union > 0 else 0.0


def match_dets_to_gt(pred_boxes_xywh, gt_boxes_xywh, thresh=IOU_THRESH):
    """Returns {pred_idx: gt_track_id}."""
    gt_ids = list(gt_boxes_xywh.keys())
    if not gt_ids or not pred_boxes_xywh:
        return {}
    mat = np.zeros((len(pred_boxes_xywh), len(gt_ids)))
    for i, pb in enumerate(pred_boxes_xywh):
        for j, gid in enumerate(gt_ids):
            mat[i, j] = iou_xywh(pb, gt_boxes_xywh[gid])
    matched, used_p, used_g = {}, set(), set()
    for fi in np.argsort(-mat, axis=None):
        pi, gj = divmod(int(fi), len(gt_ids))
        if mat[pi, gj] < thresh or pi in used_p or gj in used_g:
            continue
        matched[pi] = gt_ids[gj]
        used_p.add(pi); used_g.add(gj)
    return matched


# ---------------------------------------------------------------------------
# Targeted override assignment
# ---------------------------------------------------------------------------

def targeted_override_assignment(id_scores, traj_set, num_vocab, id_thresh):
    """
    Skip competition check for objects whose argmax is tracked AND conf >= id_thresh.
    All others use original competition logic.
    """
    n = len(id_scores)
    obj_max_confs, obj_max_labels = torch.max(id_scores, dim=-1)

    qualifying = [(i, obj_max_labels[i].item(), obj_max_confs[i].item())
                  for i in range(n)
                  if obj_max_labels[i].item() in traj_set
                  and obj_max_confs[i].item() >= id_thresh]
    non_qual = [i for i in range(n)
                if not (obj_max_labels[i].item() in traj_set
                        and obj_max_confs[i].item() >= id_thresh)]

    result  = [num_vocab] * n
    claimed = set()

    qualifying.sort(key=lambda x: -x[2])
    for i, lbl, _ in qualifying:
        if lbl not in claimed:
            result[i] = lbl
            claimed.add(lbl)

    id_max_confs = {}
    for i in non_qual:
        lbl  = obj_max_labels[i].item()
        conf = obj_max_confs[i].item()
        id_max_confs[lbl] = max(id_max_confs.get(lbl, 0.0), conf)
    id_max_confs[num_vocab] = 0.0

    for i in non_qual:
        lbl  = obj_max_labels[i].item()
        conf = obj_max_confs[i].item()
        if lbl not in traj_set:
            result[i] = num_vocab
        elif conf < id_thresh or conf < id_max_confs.get(lbl, 0.0):
            result[i] = num_vocab
        elif lbl in claimed:
            result[i] = num_vocab
        else:
            result[i] = lbl
            claimed.add(lbl)

    return result


# ---------------------------------------------------------------------------
# Instrumented tracker (monkey-patches _object_max_assignment per frame)
# ---------------------------------------------------------------------------

class InstrumentedTracker:
    """
    Wraps RuntimeTracker.  Patches two methods on the instance to capture
    per-frame state needed for the α/β analysis:
      _get_id_pred_labels  → captures current detection boxes
      _object_max_assignment → captures id_scores + baseline_labels + override_labels
    """

    def __init__(self, tracker, config):
        self.tracker  = tracker
        self.config   = config
        self.num_vocab = tracker.num_id_vocabulary
        self.id_thresh = tracker.id_thresh

        self.last_boxes          = None   # (N_curr, 4) cxcywh normalized
        self.last_id_scores      = None   # (N_curr, vocab+1)
        self.last_baseline_labels = None  # list[int]
        self.last_override_labels = None  # list[int]

        self._patch()

    def _patch(self):
        tracker   = self.tracker
        instrumented = self

        # --- patch _get_id_pred_labels to capture boxes ---
        orig_gipl = tracker._get_id_pred_labels.__func__

        def patched_gipl(self_t, boxes, output_embeds):
            instrumented.last_boxes = boxes.detach().cpu()
            return orig_gipl(self_t, boxes, output_embeds)

        tracker._get_id_pred_labels = types.MethodType(patched_gipl, tracker)

        # --- patch _object_max_assignment to capture scores + run override ---
        orig_oma = tracker._object_max_assignment.__func__

        def patched_oma(self_t, id_scores):
            instrumented.last_id_scores = id_scores.detach().cpu()

            # Baseline (original logic)
            baseline = orig_oma(self_t, id_scores)
            instrumented.last_baseline_labels = list(baseline)

            # Targeted override
            traj_set = (set(self_t.trajectory_id_labels[0].tolist())
                        if self_t.trajectory_id_labels.shape[1] > 0 else set())
            instrumented.last_override_labels = targeted_override_assignment(
                id_scores, traj_set, self_t.num_id_vocabulary, self_t.id_thresh)

            return baseline   # tracker uses baseline for its own state

        tracker._object_max_assignment = types.MethodType(patched_oma, tracker)


# ---------------------------------------------------------------------------
# Per-frame analysis
# ---------------------------------------------------------------------------

def analyze_frame(instrumented: InstrumentedTracker,
                  gt_frame: dict, img_h: int, img_w: int) -> dict:

    boxes_norm    = instrumented.last_boxes           # (N, 4) cxcywh normalized
    id_scores     = instrumented.last_id_scores       # (N, vocab+1)
    base_labels   = instrumented.last_baseline_labels
    override_labels = instrumented.last_override_labels

    if id_scores is None or boxes_norm is None or not base_labels:
        return None

    num_vocab  = instrumented.num_vocab
    id_thresh  = instrumented.id_thresh

    # Convert pred boxes to pixel xywh for IoU
    pred_xywh = []
    for b in boxes_norm.tolist():
        cx, cy, w, h = b
        pred_xywh.append([(cx - w/2)*img_w, (cy - h/2)*img_h, w*img_w, h*img_h])

    pred_to_gt = match_dets_to_gt(pred_xywh, gt_frame)

    # Build gt_track_id → vocab_label from non-newborn assignments
    gt_to_vocab = {}
    for pi, gt_tid in pred_to_gt.items():
        if pi < len(base_labels) and base_labels[pi] != num_vocab:
            gt_to_vocab[gt_tid] = base_labels[pi]

    traj_set = (set(instrumented.tracker.trajectory_id_labels[0].tolist())
                if instrumented.tracker.trajectory_id_labels.shape[1] > 0 else set())

    counts = defaultdict(int)
    counts["total_detections"] = len(base_labels)
    counts["baseline_newborn"] = sum(1 for l in base_labels if l == num_vocab)
    counts["override_newborn"] = sum(1 for l in override_labels if l == num_vocab)

    for pi in range(len(base_labels)):
        if base_labels[pi] != num_vocab:
            continue  # not a newborn in baseline — skip

        correct_vocab = None
        if pi in pred_to_gt:
            correct_vocab = gt_to_vocab.get(pred_to_gt[pi], None)

        if correct_vocab is None or correct_vocab not in traj_set:
            counts["case_b_no_gt_match"] += 1
            continue

        if id_scores[pi, correct_vocab].item() < id_thresh:
            continue   # Case A — score below threshold (rare, confirmed)

        counts["case_b_total"] += 1
        argmax_label = int(id_scores[pi].argmax().item())

        if argmax_label == correct_vocab:
            counts["scenario_alpha"] += 1
        else:
            counts["scenario_beta"] += 1

        ov = override_labels[pi] if pi < len(override_labels) else num_vocab
        if ov == correct_vocab:
            counts["override_recovery"] += 1
        elif ov != num_vocab:
            counts["override_false_recovery"] += 1

    return dict(counts)


# ---------------------------------------------------------------------------
# seqinfo loader
# ---------------------------------------------------------------------------

def load_seq_info(seq_dir: str) -> dict:
    info = {"width": 1920, "height": 1080}
    ini  = os.path.join(seq_dir, "seqinfo.ini")
    if os.path.exists(ini):
        with open(ini) as f:
            for line in f:
                if "imWidth"  in line: info["width"]  = int(line.split("=")[1])
                elif "imHeight" in line: info["height"] = int(line.split("=")[1])
    return info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    config = load_config(CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model(config, device)

    max_longer  = config.get("INFERENCE_MAX_LONGER", 1440)
    size_div    = config.get("SIZE_DIVISIBILITY", 32)

    seq_names = sorted([s for s in os.listdir(VAL_DIR)
                        if os.path.isdir(os.path.join(VAL_DIR, s))])
    print(f"Val sequences: {len(seq_names)}")

    global_counts  = defaultdict(int)
    per_seq_results = []

    for seq_name in seq_names:
        seq_dir   = os.path.join(VAL_DIR, seq_name)
        gt        = load_gt(seq_dir)
        seq_info  = load_seq_info(seq_dir)
        img_w, img_h = seq_info["width"], seq_info["height"]

        img_dir   = os.path.join(seq_dir, "img1")
        img_files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)
                            if f.lower().endswith((".jpg", ".png"))])
        if not img_files:
            continue

        frames = sorted(gt.keys())[:MAX_FRAMES]
        print(f"  {seq_name}: {len(frames)} frames...")

        # Compute resized sequence_hw from first image
        _, seq_hw = load_image_as_nested_tensor(
            img_files[0], max_longer, size_div, device)

        # Build fresh tracker per sequence (sequence_hw is required and sequence-specific)
        tracker      = build_tracker(model, config, seq_hw)
        instrumented = InstrumentedTracker(tracker, config)

        seq_counts = defaultdict(int)

        for frame_idx, frame_id in enumerate(frames):
            if frame_idx >= len(img_files):
                break
            nested_img, _ = load_image_as_nested_tensor(
                img_files[frame_idx], max_longer, size_div, device)

            try:
                with torch.no_grad():
                    tracker.update(image=nested_img)
            except Exception as e:
                print(f"    [warn] frame {frame_id}: {e}")
                continue

            gt_frame = gt.get(frame_id, {})
            result   = analyze_frame(instrumented, gt_frame, img_h, img_w)
            if result:
                for k, v in result.items():
                    seq_counts[k] += v

        per_seq_results.append({"seq": seq_name, **dict(seq_counts)})
        for k, v in seq_counts.items():
            global_counts[k] += v

    _print_report(global_counts, per_seq_results, config.get("ID_THRESH", 0.2))
    _make_plots(per_seq_results, OUTPUT_DIR)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _print_report(gc, per_seq, id_thresh):
    total_det   = gc["total_detections"]
    total_bn_bl = gc["baseline_newborn"]
    total_caseb = gc["case_b_total"]
    total_alpha = gc["scenario_alpha"]
    total_beta  = gc["scenario_beta"]
    total_recov = gc["override_recovery"]
    total_false = gc["override_false_recovery"]
    total_bn_ov = gc["override_newborn"]

    print()
    print("=" * 68)
    print("DIAG 6 — Scenario α/β Split (No-Newborn Override)")
    print("=" * 68)
    print(f"  Total detections:              {total_det:>8,}")
    print(f"  Baseline newborns:             {total_bn_bl:>8,}  "
          f"({100*total_bn_bl/max(1,total_det):.1f}%)")
    print(f"  Case B classified:             {total_caseb:>8,}  "
          f"({100*total_caseb/max(1,total_bn_bl):.1f}% of newborns)")

    if total_caseb > 0:
        pa = 100 * total_alpha / total_caseb
        pb = 100 * total_beta  / total_caseb
        pr = 100 * total_recov / total_caseb
        bn_red = total_bn_bl - total_bn_ov
        print()
        print(f"  Scenario α (argmax=correct): {total_alpha:>7,}  ({pa:.1f}%)")
        print(f"  Scenario β (argmax=wrong):   {total_beta:>7,}  ({pb:.1f}%)")
        print(f"  Override recovery:           {total_recov:>7,}  ({pr:.1f}%)")
        print(f"  Override false recovery:     {total_false:>7,}")
        print(f"  Newborn reduction (override):{bn_red:>7,}  "
              f"({100*bn_red/max(1,total_bn_bl):.1f}%)")
        print()
        print("INTERPRETATION:")
        if pa >= 70:
            print(f"  >> SCENARIO α DOMINATES ({pa:.0f}%). Protocol fix may recover most spurious newborns.")
            print(f"     Investigate assignment rule changes before any training run.")
        elif pa >= 40:
            print(f"  >> MIXED ({pa:.0f}% α). Both protocol fix and training improvement needed.")
        else:
            print(f"  >> SCENARIO β DOMINATES ({pb:.0f}%). IDDecoder argmax is wrong under density.")
            print(f"     Training required. Protocol changes will not help significantly.")
            print(f"     Check DIAG 5 to identify which mechanism degrades argmax.")

    print()
    print(f"{'Sequence':<22} {'CaseB':>7} {'Alpha':>7} {'Beta':>7} {'α%':>7} {'Recovery':>9}")
    print("-" * 62)
    for r in sorted(per_seq_results, key=lambda x: -x.get("case_b_total", 0)):
        cb  = r.get("case_b_total", 0)
        a   = r.get("scenario_alpha", 0)
        b   = r.get("scenario_beta", 0)
        pct = 100*a/cb if cb > 0 else 0.0
        rec = r.get("override_recovery", 0)
        print(f"{r['seq']:<22} {cb:>7} {a:>7} {b:>7} {pct:>6.1f}% {rec:>9}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _make_plots(per_seq_results, output_dir):
    seqs = [r["seq"] for r in per_seq_results]
    alpha_pct = [100*r.get("scenario_alpha",0)/r.get("case_b_total",1)
                 if r.get("case_b_total",0) > 0 else 0.0 for r in per_seq_results]
    recov_pct = [100*r.get("override_recovery",0)/r.get("case_b_total",1)
                 if r.get("case_b_total",0) > 0 else 0.0 for r in per_seq_results]

    order = np.argsort(alpha_pct)
    seqs_s      = [seqs[i].replace("dancetrack","dt") for i in order]
    alpha_pct_s = [alpha_pct[i] for i in order]
    recov_pct_s = [recov_pct[i] for i in order]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("DIAG 6 — Scenario α/β and Override Recovery", fontsize=12)
    x = np.arange(len(seqs_s))

    axes[0].barh(x, alpha_pct_s, color="steelblue", alpha=0.8, label="α (argmax correct)")
    axes[0].barh(x, [100-a for a in alpha_pct_s], left=alpha_pct_s,
                 color="tomato", alpha=0.8, label="β (argmax wrong)")
    axes[0].set_yticks(x); axes[0].set_yticklabels(seqs_s, fontsize=8)
    axes[0].set_xlabel("% of Case B"); axes[0].axvline(50, color="black", linestyle="--")
    axes[0].set_title("Scenario α vs β per sequence"); axes[0].legend(); axes[0].set_xlim(0,100)

    axes[1].barh(x, recov_pct_s, color="green", alpha=0.8)
    axes[1].set_yticks(x); axes[1].set_yticklabels(seqs_s, fontsize=8)
    axes[1].set_xlabel("% of Case B recovered"); axes[1].set_xlim(0,100)
    axes[1].set_title("Override recovery rate per sequence")

    plt.tight_layout()
    out = os.path.join(output_dir, "scenario_alpha_beta.png")
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved: {out}")


if __name__ == "__main__":
    main()