"""Run Depth Anything V2 (metric indoor) on the ego subset frames."""
import torch, numpy as np, os
from PIL import Image
from transformers import pipeline
import matplotlib

pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
                device=0, torch_dtype=torch.float16)
os.makedirs('ego_depth', exist_ok=True)
frames = ['0001','0015','0030','0045','0060','0075','0090','0105','0120']
cmap = matplotlib.colormaps['turbo']
for f in frames:
    img = Image.open(f'ego_subset/frame_{f}.jpg')
    out = pipe(img)
    depth = np.array(out['predicted_depth'], dtype=np.float32)
    # store at full image resolution for pixel lookups
    depth_full = np.array(Image.fromarray(depth).resize(img.size, Image.BILINEAR))
    np.save(f'ego_depth/frame_{f}_depth.npy', depth_full)
    d = np.clip(depth_full, 0, 3.0) / 3.0
    Image.fromarray((cmap(d)[..., :3] * 255).astype(np.uint8)).save(f'ego_depth/frame_{f}_depth.jpg')
    print(f, 'depth range %.2f - %.2f m' % (depth_full.min(), depth_full.max()))
print('DEPTH-DONE')
