"""
DIAG 5 — Raw Logit Magnitude and Attention Entropy vs N_concurrent
===================================================================
Inference only. GPU required. Run from repo root:
    python diagnostics/diag_script5_logit_attn_analysis.py

Confirmed context:
  - Density-confidence correlation r = -0.636 (FULL_DIAGNOSTIC_REPORT Part 11)
  - Three competing hypotheses for WHY confidence degrades at high density:

  Hypothesis A — Softmax dilution:
    Raw logit for correct label is FLAT across N bins.
    Softmax score drops because denominator grows (more classes).
    Signature: raw_logit_correct flat, softmax_correct drops.
    Fix: inference-only temperature scaling. No training.

  Hypothesis B — Attention dilution:
    Cross-attention spreads over more KV slots at high N.
    Query representation degrades → lower raw logit.
    Signature: attn_entropy_l5 rises with N, raw_logit drops.
    Fix: architectural/training change.

  Hypothesis C — Training underexposure:
    IDDecoder rarely seen N>=15 during training.
    All metrics degrade; correlates with DIAG 8 window rarity.
    Fix: weighted sequence sampler (training).
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
OUTPUT_DIR      = "diagnostics/diag5_outputs"
MAX_FRAMES      = 300
IOU_THRESH      = 0.5
N_BINS          = [(1,5),(6,10),(11,15),(16,20),(21,50)]
BIN_LABELS      = ["N=1–5","N=6–10","N=11–15","N=16–20","N=21+"]


# ---------------------------------------------------------------------------
# Config / model builders  (identical to DIAG 6)
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

    det_ckpt  = torch.load(config["CKPT_PATH"], map_location="cpu", weights_only=False)
    args_ckpt = det_ckpt["args"]
    args_ckpt.num_classes -= 1
    detr = build_rfdetr(args=args_ckpt)
    args_ckpt.num_classes += 1
    det_state = det_ckpt.get("model", None)
    if det_state is not None:
        msd = detr.state_dict()
        filtered = {(k[5:] if k.startswith("detr.") else k): v
                    for k, v in det_state.items()
                    if (k[5:] if k.startswith("detr.") else k) in msd
                    and v.shape == msd[(k[5:] if k.startswith("detr.") else k)].shape}
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

def load_image_as_nested_tensor(img_path, max_longer, size_div, device):
    from PIL import Image
    import torchvision.transforms.functional as TF

    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size
    scale = max_longer / max(orig_h, orig_w)
    new_h = math.ceil(orig_h * scale / size_div) * size_div
    new_w = math.ceil(orig_w * scale / size_div) * size_div
    img    = img.resize((new_w, new_h), Image.BILINEAR)
    tensor = TF.to_tensor(img)
    tensor = TF.normalize(tensor, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    tensors = tensor.unsqueeze(0).to(device)
    mask    = torch.zeros(1, new_h, new_w, dtype=torch.bool, device=device)
    try:
        from structures.nested_tensor import NestedTensor
    except ImportError:
        from utils.misc import NestedTensor
    return NestedTensor(tensors=tensors, mask=mask), (new_h, new_w)


# ---------------------------------------------------------------------------
# GT loading and matching
# ---------------------------------------------------------------------------

def load_gt(seq_dir):
    gt = defaultdict(dict)
    with open(os.path.join(seq_dir, "gt", "gt.txt")) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 7: continue
            fid, tid = int(parts[0]), int(parts[1])
            x,y,w,h = float(parts[2]),float(parts[3]),float(parts[4]),float(parts[5])
            if int(float(parts[6])) == 0: continue
            gt[fid][tid] = [x,y,w,h]
    return gt


def iou_xywh(a, b):
    ax2,ay2 = a[0]+a[2], a[1]+a[3]
    bx2,by2 = b[0]+b[2], b[1]+b[3]
    ix = max(0., min(ax2,bx2)-max(a[0],b[0]))
    iy = max(0., min(ay2,by2)-max(a[1],b[1]))
    inter = ix*iy
    union = a[2]*a[3]+b[2]*b[3]-inter
    return inter/union if union > 0 else 0.


def match_dets_to_gt(pred_xywh, gt_xywh, thresh=IOU_THRESH):
    gt_ids = list(gt_xywh.keys())
    if not gt_ids or not pred_xywh: return {}
    mat = np.zeros((len(pred_xywh), len(gt_ids)))
    for i,pb in enumerate(pred_xywh):
        for j,gid in enumerate(gt_ids):
            mat[i,j] = iou_xywh(pb, gt_xywh[gid])
    matched, up, ug = {}, set(), set()
    for fi in np.argsort(-mat, axis=None):
        pi,gj = divmod(int(fi), len(gt_ids))
        if mat[pi,gj]<thresh or pi in up or gj in ug: continue
        matched[pi]=gt_ids[gj]; up.add(pi); ug.add(gj)
    return matched


# ---------------------------------------------------------------------------
# IDDecoder hooks (registered once on model — persist across sequences)
# ---------------------------------------------------------------------------

class IDDecoderHooks:
    """
    Forward hooks on IDDecoder final layer (index 5 for NUM_ID_DECODER_LAYERS=6):
      cross_attn_layers[5]    → attention weights (B*G, N_curr, T*N_traj)
      embed_to_word_layers[5] → raw logits (B, G, T, N_curr, vocab+1)
    """
    def __init__(self, id_decoder, final_layer_idx: int = 5):
        self.attn_weights  = None
        self.raw_logits    = None
        self._handles      = []

        def attn_hook(module, inputs, output):
            if isinstance(output, (tuple, list)) and len(output) >= 2 and output[1] is not None:
                self.attn_weights = output[1].detach().cpu()
            else:
                self.attn_weights = None

        def logit_hook(module, inputs, output):
            self.raw_logits = output.detach().cpu()

        self._handles.append(
            id_decoder.cross_attn_layers[final_layer_idx].register_forward_hook(attn_hook))
        self._handles.append(
            id_decoder.embed_to_word_layers[final_layer_idx].register_forward_hook(logit_hook))

    def clear(self):
        self.attn_weights = None
        self.raw_logits   = None

    def remove(self):
        for h in self._handles: h.remove()
        self._handles.clear()


# ---------------------------------------------------------------------------
# Entropy helpers
# ---------------------------------------------------------------------------

def entropy(probs: np.ndarray) -> float:
    probs = np.clip(probs, 1e-12, 1.)
    return float(-np.sum(probs * np.log(probs)))


def softmax_entropy(logits: np.ndarray) -> float:
    logits = logits - logits.max()
    p = np.exp(logits) / np.sum(np.exp(logits))
    return entropy(p)


# ---------------------------------------------------------------------------
# Per-frame data extraction
# ---------------------------------------------------------------------------

def extract_frame_records(hooks: IDDecoderHooks,
                           tracker,
                           id_scores_softmax: torch.Tensor,   # (N, vocab+1)
                           baseline_labels: list,
                           gt_frame: dict,
                           pred_xywh: list,
                           num_vocab: int) -> list:

    if hooks.raw_logits is None:
        return []

    # raw_logits: (1,1,1,N,vocab+1) at inference → (N, vocab+1)
    rl = hooks.raw_logits
    if rl.dim() == 5:   rl = rl[0,0,0]
    elif rl.dim() != 2: return []
    rl_np = rl.numpy()                            # (N, vocab+1)
    sc_np = id_scores_softmax.cpu().numpy()       # (N, vocab+1)

    # Attention weights: (BG, N_curr, kv_len)
    attn_np = None
    if hooks.attn_weights is not None:
        aw = hooks.attn_weights
        if aw.dim() == 3:
            attn_np = aw.mean(0).numpy() if aw.shape[0] > 1 else aw[0].numpy()

    # N_concurrent = active trajectory slots
    n_concurrent = (tracker.trajectory_id_labels.shape[1]
                    if tracker.trajectory_id_labels.shape[1] > 0 else 0)

    # kv_len = T * N
    try:
        kv_len = tracker.trajectory_features.shape[0] * tracker.trajectory_features.shape[1]
    except Exception:
        kv_len = 0

    # GT match → gt_track_id → vocab_label
    pred_to_gt  = match_dets_to_gt(pred_xywh, gt_frame)
    gt_to_vocab = {}
    for pi, gt_tid in pred_to_gt.items():
        if pi < len(baseline_labels) and baseline_labels[pi] != num_vocab:
            gt_to_vocab[gt_tid] = baseline_labels[pi]

    records = []
    for pi in range(min(len(baseline_labels), rl_np.shape[0])):
        logits_i = rl_np[pi]
        scores_i = sc_np[pi]
        bl       = baseline_labels[pi]

        argmax_lbl      = int(np.argmax(logits_i))
        raw_logit_argmax = float(logits_i[argmax_lbl])
        sfmx_ent         = softmax_entropy(logits_i)

        attn_ent = float("nan")
        if attn_np is not None and pi < attn_np.shape[0]:
            row = attn_np[pi]
            s   = float(np.sum(row))
            if s > 1e-9:
                attn_ent = entropy(row / s)

        # GT-matched values
        correct_vocab  = None
        raw_logit_corr = float("nan")
        softmax_corr   = float("nan")
        is_alpha       = False

        if pi in pred_to_gt:
            cv = gt_to_vocab.get(pred_to_gt[pi], None)
            if cv is not None and cv < logits_i.shape[0]:
                correct_vocab  = cv
                raw_logit_corr = float(logits_i[cv])
                softmax_corr   = float(scores_i[cv])
                is_alpha       = (argmax_lbl == cv)

        records.append({
            "n_concurrent":      n_concurrent,
            "kv_len":            kv_len,
            "raw_logit_argmax":  raw_logit_argmax,
            "raw_logit_correct": raw_logit_corr,
            "softmax_correct":   softmax_corr,
            "softmax_entropy":   sfmx_ent,
            "attn_entropy_l5":   attn_ent,
            "is_newborn":        int(bl == num_vocab),
            "is_alpha":          int(is_alpha),
            "has_gt_match":      int(correct_vocab is not None),
        })
    return records


# ---------------------------------------------------------------------------
# Tracker step with id_scores capture
# ---------------------------------------------------------------------------

def run_step_and_capture(tracker, hooks: IDDecoderHooks,
                          nested_img, frame_id) -> tuple:
    """Run tracker.update(), capture id_scores and baseline_labels."""
    captured = {"id_scores": None, "labels": None}
    orig_oma = tracker._object_max_assignment.__func__

    def patched_oma(self_t, id_scores_inner):
        captured["id_scores"] = id_scores_inner.detach().cpu()
        result = orig_oma(self_t, id_scores_inner)
        captured["labels"] = list(result)
        return result

    tracker._object_max_assignment = types.MethodType(patched_oma, tracker)
    hooks.clear()

    try:
        with torch.no_grad():
            tracker.update(image=nested_img)
    except Exception as e:
        print(f"    [warn] frame {frame_id}: {e}")
        tracker._object_max_assignment = types.MethodType(orig_oma, tracker)
        return None, None
    finally:
        tracker._object_max_assignment = types.MethodType(orig_oma, tracker)

    return captured["id_scores"], captured["labels"]


# ---------------------------------------------------------------------------
# Seqinfo / box helpers
# ---------------------------------------------------------------------------

def load_seq_info(seq_dir):
    info = {"width": 1920, "height": 1080}
    ini  = os.path.join(seq_dir, "seqinfo.ini")
    if os.path.exists(ini):
        with open(ini) as f:
            for line in f:
                if "imWidth"  in line: info["width"]  = int(line.split("=")[1])
                elif "imHeight" in line: info["height"] = int(line.split("=")[1])
    return info


def get_pred_xywh(tracker, img_w, img_h) -> list:
    """Convert tracker's current trajectory_boxes[-1] to pixel xywh."""
    try:
        if tracker.trajectory_features.shape[0] == 0:
            return []
        # Get only the LAST time step (current frame detections)
        boxes = tracker.trajectory_boxes[-1]       # (N_vocab, 4) cxcywh normalised
        masks = tracker.trajectory_masks[-1]       # (N_vocab,) True=masked/invalid
        valid = ~masks
        out   = []
        for i, (b, m) in enumerate(zip(boxes.tolist(), valid.tolist())):
            if not m:   # skip invalid slots
                continue
            cx,cy,w,h = b
            out.append([(cx-w/2)*img_w, (cy-h/2)*img_h, w*img_w, h*img_h])
        return out
    except Exception:
        return []


def get_bin(n: int) -> int:
    for i,(lo,hi) in enumerate(N_BINS):
        if lo <= n <= hi: return i
    return len(N_BINS)-1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    config = load_config(CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model   = build_model(config, device)
    num_vocab = config.get("NUM_ID_VOCABULARY", 50)

    # Register hooks once — they persist across sequences since model doesn't change
    final_layer = config.get("NUM_ID_DECODER_LAYERS", 6) - 1
    hooks = IDDecoderHooks(model.id_decoder, final_layer_idx=final_layer)

    max_longer = config.get("INFERENCE_MAX_LONGER", 1440)
    size_div   = config.get("SIZE_DIVISIBILITY", 32)

    seq_names = sorted([s for s in os.listdir(VAL_DIR)
                        if os.path.isdir(os.path.join(VAL_DIR, s))])
    print(f"Val sequences: {len(seq_names)}")

    all_records = []

    for seq_name in seq_names:
        seq_dir  = os.path.join(VAL_DIR, seq_name)
        gt       = load_gt(seq_dir)
        seq_info = load_seq_info(seq_dir)
        img_w, img_h = seq_info["width"], seq_info["height"]

        img_dir   = os.path.join(seq_dir, "img1")
        img_files = sorted([os.path.join(img_dir,f) for f in os.listdir(img_dir)
                            if f.lower().endswith((".jpg",".png"))])
        if not img_files: continue

        frames = sorted(gt.keys())[:MAX_FRAMES]
        print(f"  {seq_name}: {len(frames)} frames...")

        # Compute sequence_hw from first image, build fresh tracker
        _, seq_hw = load_image_as_nested_tensor(img_files[0], max_longer, size_div, device)
        tracker   = build_tracker(model, config, seq_hw)

        for frame_idx, frame_id in enumerate(frames):
            if frame_idx >= len(img_files): break
            nested_img, _ = load_image_as_nested_tensor(
                img_files[frame_idx], max_longer, size_div, device)

            id_scores, baseline_labels = run_step_and_capture(
                tracker, hooks, nested_img, frame_id)
            if id_scores is None or not baseline_labels:
                continue

            pred_xywh = get_pred_xywh(tracker, img_w, img_h)
            gt_frame  = gt.get(frame_id, {})

            recs = extract_frame_records(
                hooks, tracker, id_scores, baseline_labels,
                gt_frame, pred_xywh, num_vocab)

            for r in recs:
                r["seq"] = seq_name
                r["bin"] = get_bin(r["n_concurrent"])
            all_records.extend(recs)

    hooks.remove()
    print(f"\nTotal records: {len(all_records):,}")

    if not all_records:
        print("No records collected."); sys.exit(1)

    _print_report(all_records)
    _make_plots(all_records)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _print_report(records):
    gt_recs = [r for r in records if r["has_gt_match"]]
    print()
    print("="*76)
    print("DIAG 5 — Raw Logit & Attention Entropy vs N_concurrent")
    print("="*76)
    print(f"GT-matched records: {len(gt_recs):,} / {len(records):,}")

    metrics = [
        ("raw_logit_correct", "Raw logit (correct label)  "),
        ("softmax_correct",   "Softmax score (correct)    "),
        ("softmax_entropy",   "Softmax entropy            "),
        ("attn_entropy_l5",   "Attn entropy L5            "),
    ]

    print()
    hdr = f"  {'Metric':<32}" + "".join(f"  {l:>11}" for l in BIN_LABELS)
    print(hdr); print("-"*(34 + 13*len(N_BINS)))

    bin_data = {}
    for key, label in metrics:
        row = f"  {label}"
        bin_means = []
        for bi in range(len(N_BINS)):
            vals = [r[key] for r in gt_recs if r["bin"]==bi and not math.isnan(r[key])]
            m    = float(np.mean(vals)) if vals else float("nan")
            bin_means.append(m)
            row += f"  {m:>11.4f}" if not math.isnan(m) else f"  {'N/A':>11}"
        print(row)
        n_row = f"  {'n':>32}" + "".join(
            f"  {sum(1 for r in gt_recs if r['bin']==bi and not math.isnan(r[key])):>11,}"
            for bi in range(len(N_BINS)))
        print(n_row)
        bin_data[key] = bin_means

    # --- Hypothesis discrimination ---
    print()
    print("HYPOTHESIS DISCRIMINATION:")
    rl = bin_data["raw_logit_correct"]
    sc = bin_data["softmax_correct"]
    ae = bin_data["attn_entropy_l5"]

    valid = [i for i in range(len(N_BINS)) if not math.isnan(rl[i]) and not math.isnan(sc[i])]
    if len(valid) >= 2:
        lo, hi = valid[0], valid[-1]
        logit_drop   = rl[hi] - rl[lo]
        softmax_drop = sc[hi] - sc[lo]
        ae_valid     = [i for i in range(len(N_BINS)) if not math.isnan(ae[i])]
        attn_rise    = (ae[ae_valid[-1]] - ae[ae_valid[0]]) if len(ae_valid)>=2 else float("nan")

        print(f"  Raw logit change  ({BIN_LABELS[lo]}→{BIN_LABELS[hi]}): {logit_drop:+.4f}")
        print(f"  Softmax chg:  {softmax_drop:+.4f}   "
              f"Attn entropy rise: {attn_rise:+.4f}" if not math.isnan(attn_rise)
              else f"  Softmax chg:  {softmax_drop:+.4f}")
        print()

        logit_flat  = abs(logit_drop) < 0.5
        sfmx_drops  = softmax_drop < -0.05
        attn_rises  = (not math.isnan(attn_rise)) and attn_rise > 0.3
        logit_drops = logit_drop < -0.5

        print(f"  Hyp A (softmax dilution):   ", end="")
        if logit_flat and sfmx_drops:
            print("SUPPORTED  → logit flat, softmax drops.  Fix: temperature scaling (no training).")
        elif logit_flat:
            print("PARTIAL    → logit flat but softmax barely drops.")
        else:
            print("NOT supported → logit changes with N.")

        print(f"  Hyp B (attn dilution):      ", end="")
        if attn_rises and logit_drops:
            print("SUPPORTED  → attn entropy rises, logit drops.  Fix: training required.")
        elif attn_rises:
            print("PARTIAL    → attn entropy rises but logit drop is small.")
        else:
            print("NOT supported.")

        print(f"  Hyp C (underexposure):      ", end="")
        if logit_drops:
            print("POSSIBLE   → cross-check with DIAG 8 window distribution.")
        else:
            print("Insufficient evidence → check DIAG 8.")

    # α rate by N bin
    print()
    print("Scenario α rate (argmax=correct) by N bin:")
    row = "  "
    for bi in range(len(N_BINS)):
        recs = [r for r in gt_recs if r["bin"]==bi]
        rate = float(np.mean([r["is_alpha"] for r in recs])) if recs else float("nan")
        row += f"  {BIN_LABELS[bi]}:{rate:.3f}" if not math.isnan(rate) else f"  {BIN_LABELS[bi]}:N/A"
    print(row)
    print("  (drops with N → argmax degrades at density → Hyp B/C + training needed)")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _make_plots(records):
    gt_recs = [r for r in records if r["has_gt_match"]]
    by_n    = defaultdict(list)
    for r in gt_recs: by_n[r["n_concurrent"]].append(r)
    ns_all  = sorted(by_n.keys())
    ns      = [n for n in ns_all if len(by_n[n]) >= 5]

    def agg(key):
        ms, ss, ns_out = [], [], []
        for n in ns:
            vals = [r[key] for r in by_n[n] if not math.isnan(r[key])]
            if vals:
                ms.append(np.mean(vals)); ss.append(np.std(vals)); ns_out.append(n)
        return np.array(ns_out), np.array(ms), np.array(ss)

    fig, axes = plt.subplots(2,2, figsize=(16,12))
    fig.suptitle("DIAG 5 — IDDecoder Logit & Attention vs N_concurrent", fontsize=12)

    specs = [
        ("raw_logit_correct", "Raw logit (correct label)",
         "Hyp A: flat=softmax dilution; drop=attn/exposure issue", "steelblue", axes[0,0]),
        ("softmax_correct",   "Softmax score (correct label)",
         "Should drop with N; question is: does raw logit also drop?",    "green",      axes[0,1]),
        ("attn_entropy_l5",   "Cross-attn entropy L5",
         "Hyp B: rises with N = attention diluting",                      "darkorange", axes[1,0]),
        ("softmax_entropy",   "Softmax entropy (all vocab+1)",
         "Higher = more uncertain",                                        "purple",     axes[1,1]),
    ]
    for key, ylabel, title, color, ax in specs:
        ns_p, ms, ss = agg(key)
        if len(ns_p):
            ax.plot(ns_p, ms, "o-", color=color, linewidth=2, markersize=4)
            ax.fill_between(ns_p, ms-ss, ms+ss, alpha=0.2, color=color)
        ax.set_xlabel("N_concurrent (active trajectory slots)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=8)
        ax.grid(True, alpha=0.3)
        if key == "softmax_correct":
            ax.axhline(0.2, color="red", linestyle="--", linewidth=1, label="id_thresh=0.2")
            ax.legend(fontsize=8); ax.set_ylim(0,1)

    plt.tight_layout()
    out1 = os.path.join(OUTPUT_DIR, "logit_attn_vs_N.png")
    plt.savefig(out1, dpi=150); print(f"Plot saved: {out1}")

    # α rate by bin
    fig2, ax5 = plt.subplots(figsize=(9,5))
    alpha_rates = []
    for bi in range(len(N_BINS)):
        recs = [r for r in gt_recs if r["bin"]==bi]
        alpha_rates.append(float(np.mean([r["is_alpha"] for r in recs])) if recs else float("nan"))

    valid_bi = [i for i in range(len(N_BINS)) if not math.isnan(alpha_rates[i])]
    colors   = ["green" if alpha_rates[i]>=0.7 else "orange" if alpha_rates[i]>=0.4
                else "tomato" for i in valid_bi]
    ax5.bar([BIN_LABELS[i] for i in valid_bi], [alpha_rates[i] for i in valid_bi],
            color=colors, alpha=0.8)
    ax5.axhline(0.5, color="black", linestyle="--")
    ax5.set_ylim(0,1.1); ax5.grid(True, alpha=0.3, axis="y")
    ax5.set_title("Scenario α rate by N bin  (green≥0.7: argmax reliable)", fontsize=10)
    ax5.set_ylabel("α rate (argmax = correct label)")
    plt.tight_layout()
    out2 = os.path.join(OUTPUT_DIR, "alpha_rate_by_N_bin.png")
    plt.savefig(out2, dpi=150); print(f"Plot saved: {out2}")

    np.save(os.path.join(OUTPUT_DIR, "all_records.npy"),
            np.array([[r["n_concurrent"],r["raw_logit_correct"],r["softmax_correct"],
                       r["softmax_entropy"],r["attn_entropy_l5"],r["is_newborn"],
                       r["is_alpha"],r["has_gt_match"]] for r in records]))
    print(f"Raw records: {OUTPUT_DIR}/all_records.npy")


if __name__ == "__main__":
    main()