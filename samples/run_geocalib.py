"""Estimate camera intrinsics for a clip with GeoCalib.
Usage: run_geocalib.py <frames_dir> <out_json>
Samples a few frames, reports per-frame focal, saves the median.
"""
import sys, glob, json
import numpy as np
import torch
from geocalib import GeoCalib

frames_dir, out_json = sys.argv[1], sys.argv[2]
frames = sorted(glob.glob(f'{frames_dir}/frame_*.jpg'))
picks = frames[:: max(1, len(frames) // 5)][:5]

model = GeoCalib().to('cuda')
focals = []
for fp in picks:
    img = model.load_image(fp).to('cuda')
    res = model.calibrate(img)
    f = float(res['camera'].f.mean())
    focals.append(f)
    print(f'{fp}: f={f:.1f} px')

med = float(np.median(focals))
import cv2
H, W = cv2.imread(picks[0]).shape[:2]
json.dump({'focal_px': med, 'width': W, 'height': H,
           'hfov_deg': float(2*np.degrees(np.arctan(W/2/med)))}, open(out_json, 'w'), indent=1)
print(f'median focal {med:.1f}px  (nominal-guess was {max(H,W)});  hfov '
      f'{2*np.degrees(np.arctan(W/2/med)):.1f} deg -> {out_json}')
