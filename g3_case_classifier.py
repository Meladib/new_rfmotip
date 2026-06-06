# G3a: Case A/B classification on competition events (verified pipeline).
# Reads events_Vx_all.jsonl + val GT. No model run.
#
# For each model-ERROR competition event (the correct continuation lost):
#   Case B (competition failure): correct continuation's score >= id_thresh
#                                 but it lost the winner-takes-all assignment.
#   Case A (confidence failure):  correct continuation's score <  id_thresh.
# (Case C = label absent from vocab is not observable in competition events;
#  prior work found it ~0%. We report A/B among competition events, stated as such.)
#
# Also reports, for context, the score gap between the winning (wrong) detection
# and the correct (lost) detection -- the "competition margin".
import json, sys, math, os
import numpy as np

events_path = sys.argv[1]
gt_root = sys.argv[2]
ID_THRESH = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
IOU_THR = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3   # GT-match thr (hardened)

def iou_xywh(a,b):
    ax1,ay1,ax2,ay2=a[0],a[1],a[0]+a[2],a[1]+a[3]
    bx1,by1,bx2,by2=b[0],b[1],b[0]+b[2],b[1]+b[3]
    ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw,ih=max(0.0,ix2-ix1),max(0.0,iy2-iy1); inter=iw*ih
    ua=a[2]*a[3]+b[2]*b[3]-inter
    return inter/ua if ua>0 else 0.0

def load_gt(seq):
    path=os.path.join(gt_root,seq,"gt","gt.txt"); gt={}
    if not os.path.isfile(path): return gt
    for line in open(path):
        line=line.strip()
        if not line: continue
        p=line.split(","); fid,oid=int(p[0]),int(p[1]); x,y,w,h=map(float,p[2:6])
        gt.setdefault(fid,[]).append((oid,[x,y,w,h]))
    return gt

def match(box,gtf,thr):
    best,bid=0.0,None
    for oid,g in gtf:
        i=iou_xywh(box,g)
        if i>best: best,bid=i,oid
    return bid if best>=thr else None

rows=[json.loads(l) for l in open(events_path) if l.strip()]
events={}
for r in rows: events.setdefault(r["event_id"],[]).append(r)
gtc_={}
def gt_for(s):
    if s not in gtc_: gtc_[s]=load_gt(s)
    return gtc_[s]

n_err=0; case_A=0; case_B=0
margins=[]   # winner_score - correct_lost_score (the competition margin)
correct_scores=[]
for eid,ev in events.items():
    seq=ev[0]["seq"]; frame=ev[0]["frame"]; tb=ev[0]["traj_box_px"]
    gt=gt_for(seq)
    if not gt: continue
    gid=None
    for fb in (frame-1,frame-2,frame-3):
        if fb in gt:
            gid=match(tb,gt[fb],IOU_THR)
            if gid is not None: break
    if gid is None: continue
    gf=gt.get(frame,[])
    for r in ev:
        r["_g"]=match(r["det_box_px"],gf,IOU_THR) if r["det_box_px"] else None
    gtc=[r for r in ev if r["_g"]==gid]
    mw=[r for r in ev if r["is_winner"]]
    if not gtc or not mw: continue
    gtc=gtc[0]; mw=mw[0]
    if mw.get("_g")==gid: continue   # model correct, not an error
    # MODEL ERROR: the correct continuation (gtc) lost to mw.
    n_err+=1
    s_correct = gtc["id_score_for_label"]
    s_winner  = mw["id_score_for_label"]
    correct_scores.append(s_correct)
    margins.append(s_winner - s_correct)
    if s_correct >= ID_THRESH:
        case_B += 1     # correct was confident enough but lost competition
    else:
        case_A += 1     # correct was under-confident

margins=np.array(margins); correct_scores=np.array(correct_scores)
ver = os.path.basename(events_path).replace("events_","").replace("_all.jsonl","")
print("="*60)
print(f"G3a — CASE A/B  ({ver}, competition events, id_thresh={ID_THRESH})")
print("="*60)
print(f"model-error competition events: {n_err}")
if n_err>0:
    print(f"  Case B (competition failure, score>={ID_THRESH}): {case_B}  ({case_B/n_err:.1%})")
    print(f"  Case A (confidence failure,  score< {ID_THRESH}): {case_A}  ({case_A/n_err:.1%})")
    print(f"  competition margin (winner-correct): mean={margins.mean():.4f} median={np.median(margins):.4f}")
    print(f"  correct(lost) score: mean={correct_scores.mean():.4f} median={np.median(correct_scores):.4f}")
print("="*60)
print("Reading: Case B dominant => failure is the winner-takes-all ASSIGNMENT,")
print("not the score (the correct label scored high enough but lost). This is the")
print("protocol-level competition failure; the spatial bias adds the tiebreaker.")