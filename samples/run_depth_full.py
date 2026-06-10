"""Depth Anything V2 metric on ALL extracted frames of the ego sample."""
import torch, numpy as np, os, glob
from PIL import Image
from transformers import pipeline

pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
                device=0, torch_dtype=torch.float16)
os.makedirs('ego_depth_full', exist_ok=True)
frames = sorted(glob.glob('ego_sample_frames/frame_*.jpg'))
for i, fp in enumerate(frames):
    name = os.path.basename(fp).replace('.jpg', '')
    out_npy = f'ego_depth_full/{name}_depth.npy'
    if os.path.exists(out_npy):
        continue
    img = Image.open(fp)
    depth = np.array(pipe(img)['predicted_depth'], dtype=np.float32)
    depth_full = np.array(Image.fromarray(depth).resize(img.size, Image.BILINEAR))
    np.save(out_npy, depth_full)
    if i % 20 == 0:
        print(f'{i+1}/{len(frames)}', flush=True)
print('DEPTH-FULL-DONE', len(frames))
