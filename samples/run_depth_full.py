"""Depth Anything V2 metric on all frames. Usage: run_depth_full.py [frames_dir] [out_dir]"""
import torch, numpy as np, os, glob, sys
from PIL import Image
from transformers import pipeline

FRAMES_DIR = sys.argv[1] if len(sys.argv) > 1 else 'ego_sample_frames'
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else 'ego_depth_full'
pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
                device=0, torch_dtype=torch.float16)
os.makedirs(OUT_DIR, exist_ok=True)
frames = sorted(glob.glob(f'{FRAMES_DIR}/frame_*.jpg'))
for i, fp in enumerate(frames):
    name = os.path.basename(fp).replace('.jpg', '')
    out_npy = f'{OUT_DIR}/{name}_depth.npy'
    if os.path.exists(out_npy):
        continue
    img = Image.open(fp)
    depth = np.array(pipe(img)['predicted_depth'], dtype=np.float32)
    depth_full = np.array(Image.fromarray(depth).resize(img.size, Image.BILINEAR))
    np.save(out_npy, depth_full)
    if i % 20 == 0:
        print(f'{i+1}/{len(frames)}', flush=True)
print('DEPTH-FULL-DONE', len(frames))
