"""Post-process fusion.npz: consistent L/R hand tracks + gap interpolation.

- builds two tracks (left/right) assigned by spatial continuity of the wrist,
  with predicted handedness only as a soft prior (labels flip; position doesn't)
- suppresses duplicate detections (two skeletons on one physical hand)
- linearly interpolates joints across short gaps (<= MAX_GAP frames)
- backs up the original to fusion_raw.npz, overwrites fusion.npz

Usage: track_hands.py <fusion_dir> [max_gap_frames]
"""
import sys
import numpy as np
from pathlib import Path

SAMPLES = Path(__file__).resolve().parent
fusion_dir = SAMPLES / (sys.argv[1] if len(sys.argv) > 1 else 'ego_fusion_full')
MAX_GAP = int(sys.argv[2]) if len(sys.argv) > 2 else 15

DUP_DIST = 0.09     # m: wrists closer than this = same physical hand
GATE0 = 0.12        # m: base assignment gate per frame
GATE_GROW = 0.03    # m per missed frame
HAND_PENALTY = 0.10 # m-equivalent cost for handedness mismatch

_raw = fusion_dir / 'fusion_raw.npz'
src = np.load(_raw if _raw.exists() else fusion_dir / 'fusion.npz')
names = sorted(n for n in src.files if '__' not in n and not n.startswith('meta'))
out = {k: src[k] for k in src.files}

def wrist(row):
    return row[1:64].reshape(21, 3)[0]

def wrist_raw(row):
    return row[64:127].reshape(21, 3)[0] if row.shape[0] >= 127 else wrist(row)

def is_dup(a, b):
    return (np.linalg.norm(wrist(a) - wrist(b)) < DUP_DIST or
            np.linalg.norm(wrist_raw(a) - wrist_raw(b)) < DUP_DIST)

tracks = {0: dict(last=None, last_i=-10**9, rows={}),   # left
          1: dict(last=None, last_i=-10**9, rows={})}   # right

for i, nm in enumerate(names):
    rows = list(src[nm])
    # --- duplicate suppression: cluster wrists, keep one row per cluster ---
    kept = []
    for r in rows:
        dup = next((k for k in kept if is_dup(k, r)), None)
        if dup is None:
            kept.append(r)
        # else: drop r (same physical hand detected twice)
    # --- assign kept rows to tracks by wrist continuity ---
    cands = []
    for ri, r in enumerate(kept):
        for t in (0, 1):
            tr = tracks[t]
            gap = i - tr['last_i']
            if tr['last'] is None:
                cost = 0.5 + HAND_PENALTY * (int(r[0]) != t)  # cold start
            else:
                gate = GATE0 + GATE_GROW * min(gap, 30)
                d = np.linalg.norm(wrist(r) - tr['last'])
                if d > gate:
                    continue
                cost = d + HAND_PENALTY * (int(r[0]) != t)
            cands.append((cost, ri, t))
    used_r, used_t = set(), set()
    for cost, ri, t in sorted(cands):
        if ri in used_r or t in used_t:
            continue
        used_r.add(ri); used_t.add(t)
        r = kept[ri].copy()
        r[0] = t                       # relabel handedness to track identity
        tracks[t]['rows'][i] = r
        tracks[t]['last'] = wrist(r)
        tracks[t]['last_i'] = i

# --- interpolate short gaps per track ---
n_interp = 0
for t in (0, 1):
    idxs = sorted(tracks[t]['rows'])
    for a, b in zip(idxs, idxs[1:]):
        if 1 < b - a <= MAX_GAP:
            ra, rb = tracks[t]['rows'][a], tracks[t]['rows'][b]
            for j in range(a + 1, b):
                w = (j - a) / (b - a)
                r = ra * (1 - w) + rb * w
                r[0] = t
                tracks[t]['rows'][j] = r
                n_interp += 1

# --- post-track smoothing (One-Euro on wrists; rigid shift per hand) ---
def one_euro(xs, min_cutoff=1.0, beta=0.5, fps=15.0):
    a = lambda cut: 1.0 / (1.0 + fps / (2 * np.pi * cut))
    xh, dxh = xs[0].copy(), np.zeros_like(xs[0])
    res = [xs[0].copy()]
    for x in xs[1:]:
        dx = (x - xh) * fps
        dxh = a(1.0) * dx + (1 - a(1.0)) * dxh
        cut = min_cutoff + beta * np.linalg.norm(dxh)
        xh = a(cut) * x + (1 - a(cut)) * xh
        res.append(xh.copy())
    return res

for t in (0, 1):
    idxs = sorted(tracks[t]['rows'])
    # smooth each contiguous run separately (don't smooth across long gaps)
    run = []
    for k, i in enumerate(idxs):
        run.append(i)
        end = (k == len(idxs) - 1) or (idxs[k + 1] - i > 1)
        if end:
            if len(run) > 2:
                for cols in (slice(1, 64), slice(64, 127)):
                    rows = [tracks[t]['rows'][j] for j in run]
                    if rows[0].shape[0] < cols.stop:
                        continue
                    Js = [r[cols].reshape(21, 3) for r in rows]
                    sm = one_euro([J[0] for J in Js])
                    for r, J, w_s in zip(rows, Js, sm):
                        r[cols] = (J + (w_s - J[0])).ravel()
            run = []

for i, nm in enumerate(names):
    frame_rows = [tracks[t]['rows'][i] for t in (0, 1) if i in tracks[t]['rows']]
    out[nm] = (np.stack(frame_rows).astype(np.float32) if frame_rows
               else np.zeros((0, src[nm].shape[1] if src[nm].ndim == 2 and src[nm].shape[1] else 127),
                             dtype=np.float32))

np.savez_compressed(fusion_dir / 'fusion_raw.npz', **{k: src[k] for k in src.files})
np.savez_compressed(fusion_dir / 'fusion.npz', **out)
n_frames_LR = sum(1 for i in range(len(names))
                  if i in tracks[0]['rows'] and i in tracks[1]['rows'])
print(f'TRACKED: {len(names)} frames | both hands {n_frames_LR} | interpolated {n_interp} '
      f'| L {len(tracks[0]["rows"])} R {len(tracks[1]["rows"])}')
