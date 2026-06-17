import json, sys, math, os
import numpy as np

# --- Configuration ---
# Usage: python analyze_cases.py <events.jsonl> <gt_root_folder> [id_thresh] [iou_thr]
events_path = sys.argv[1]
gt_root = sys.argv[2]
ID_THRESH = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
IOU_THR = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3   # GT-match thr

def iou_xywh(a, b):
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0]+a[2], a[1]+a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0]+b[2], b[1]+b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2-ix1), max(0.0, iy2-iy1)
    inter = iw * ih
    ua = a[2]*a[3] + b[2]*b[3] - inter
    return inter / ua if ua > 0 else 0.0

def load_gt(seq):
    path = os.path.join(gt_root, seq, "gt", "gt.txt")
    gt = {}
    if not os.path.isfile(path): 
        return gt
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): 
                continue
            p = line.split(",")
            # Handle potential float frame IDs safely
            fid, oid = int(float(p[0])), int(float(p[1])) 
            x, y, w, h = map(float, p[2:6])
            gt.setdefault(fid, []).append((oid, [x, y, w, h]))
    return gt

def match(box, gtf, thr):
    best, bid = 0.0, None
    for oid, g in gtf:
        i = iou_xywh(box, g)
        if i > best:
            best, bid = i, oid
    return bid if best >= thr else None

# --- Load Events ---
print(f"Loading events from: {events_path}")
rows = [json.loads(l) for l in open(events_path) if l.strip()]
events = {}
for r in rows: 
    events.setdefault(r["event_id"], []).append(r)

gtc_ = {}
def gt_for(s):
    if s not in gtc_: 
        gtc_[s] = load_gt(s)
    return gtc_[s]

n_err = 0
case_A = 0
case_B = 0
margins = []
correct_scores = []

print("Analyzing competition events...")

for eid, ev in events.items():
    seq = ev[0]["seq"]
    frame = ev[0]["frame"]
    tb = ev[0]["traj_box_px"]
    
    gt = gt_for(seq)
    if not gt: 
        continue
    
    # RESOLVED: Corrected off-by-one exclusive stop index for full 15-frame lookback
    gid = None
    for fb in range(frame - 1, max(0, frame - 16), -1):
        if fb in gt:
            gid = match(tb, gt[fb], IOU_THR)
            if gid is not None: 
                break
                
    if gid is None: 
        continue 
        
    gf = gt.get(frame, [])
    
    for r in ev:
        r["_g"] = match(r["det_box_px"], gf, IOU_THR) if r.get("det_box_px") else None
        
    gtc_list = [r for r in ev if r["_g"] == gid]
    mw_list = [r for r in ev if r.get("is_winner")]
    
    if not gtc_list or not mw_list: 
        continue
        
    # RESOLVED: Sort to prevent arbitrary index [0] selection noise
    gtc = sorted(gtc_list, key=lambda x: float(x.get("id_score_for_label", 0)), reverse=True)[0]
    mw = sorted(mw_list, key=lambda x: float(x.get("id_score_for_label", 0)), reverse=True)[0]
    
    if mw.get("_g") == gid: 
        continue
        
    s_correct = gtc.get("id_score_for_label")
    s_winner = mw.get("id_score_for_label")
    
    if s_correct is None or s_winner is None:
        continue 
        
    s_correct = float(s_correct)
    s_winner = float(s_winner)
    
    if s_winner <= s_correct:
        continue 
        
    n_err += 1
    correct_scores.append(s_correct)
    margins.append(s_winner - s_correct)
    
    if s_correct >= ID_THRESH:
        case_B += 1     
    else:
        case_A += 1
# --- Output Results ---
margins = np.array(margins)
correct_scores = np.array(correct_scores)
ver = os.path.basename(events_path).replace("events_", "").replace("_all.jsonl", "")

print("=" * 70)
print(f"G3a — CASE A/B DIAGNOSTIC ({ver})")
print(f"Threshold (tau_id): {ID_THRESH} | IOU Match Thr: {IOU_THR}")
print("=" * 70)
print(f"Total model-error competition events analyzed: {n_err}")

if n_err > 0:
    print(f"  Case B (Competition Failure, score >= {ID_THRESH}): {case_B} ({case_B/n_err:.4%})")
    print(f"  Case A (Confidence Failure,  score <  {ID_THRESH}): {case_A} ({case_A/n_err:.4%})")
    print("-" * 70)
    print(f"  Competition Margin (winner - correct):")
    print(f"    Mean: {margins.mean():.4f} | Median: {np.median(margins):.4f} | Std: {margins.std():.4f}")
    print(f"  Correct (lost) score:")
    print(f"    Mean: {correct_scores.mean():.4f} | Median: {np.median(correct_scores):.4f}")
else:
    print("  No valid error events found. Check your JSONL format and GT paths.")

print("=" * 70)
