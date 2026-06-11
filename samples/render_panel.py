"""Render the Panel-1 replica video from fusion.npz:
  top    (1a): RGB frame + projected 2D hand skeletons (red=R, blue=L)
  bottom (1b): metric 3D skeletons + fitted table plane, fixed camera
Stitched to ego_panel1.mp4 (30 fps).
"""
import numpy as np, cv2, glob, os, subprocess, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SAMPLES = Path(__file__).resolve().parent
EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),
         (11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
RED, BLUE = (60, 60, 230), (230, 120, 40)  # BGR

frames_dir = sys.argv[1] if len(sys.argv) > 1 else 'ego_sample_frames'
fusion_dir = sys.argv[2] if len(sys.argv) > 2 else 'ego_fusion_full'
panel_dir = sys.argv[3] if len(sys.argv) > 3 else 'ego_panel_frames'
out_name = sys.argv[4] if len(sys.argv) > 4 else 'ego_panel1.mp4'
data = np.load(SAMPLES / fusion_dir / 'fusion.npz')
frames = sorted(glob.glob(str(SAMPLES / frames_dir / 'frame_*.jpg')))
out_dir = SAMPLES / panel_dir
out_dir.mkdir(exist_ok=True)

W_out = 1280
for fp in frames:
    name = Path(fp).stem
    img = cv2.imread(fp)
    H, W = img.shape[:2]
    # must match the focal fuse_full used (calibrated if provided, else nominal)
    if len(sys.argv) > 5 and float(sys.argv[5]) > 0:
        f = float(sys.argv[5])
    elif 'meta__focal' in data.files and data['meta__focal'][0] > 0:
        f = float(data['meta__focal'][0])
    else:
        f = float(max(H, W))
    hands = data[name] if name in data.files else np.zeros((0, 64))
    plane = data.get(f'{name}__plane', np.zeros(4))

    # ---- 1a: 2D skeleton overlay ----
    top = img.copy()
    for row in hands:
        right = int(row[0])
        # cols 1:64 = metric (anchored) joints; cols 64:127 = HaMeR raw camera
        # joints (pixel-aligned). Draw 1a with raw when available.
        J = row[64:127].reshape(21, 3) if row.shape[0] >= 127 else row[1:64].reshape(21, 3)
        col = RED if right else BLUE
        pts2d = np.stack([J[:, 0]/J[:, 2]*f + W/2, J[:, 1]/J[:, 2]*f + H/2], 1).astype(int)
        for e in EDGES:
            cv2.line(top, tuple(pts2d[e[0]]), tuple(pts2d[e[1]]), col, 3, cv2.LINE_AA)
        for p in pts2d:
            cv2.circle(top, tuple(p), 5, col, -1, cv2.LINE_AA)
    top = cv2.resize(top, (W_out, int(H * W_out / W)))

    # ---- 1b: 3D view ----
    fig = plt.figure(figsize=(W_out/100, 4.2), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    if plane.any():
        n, d = plane[:3], plane[3]
        c0 = -d * n  # closest point on plane to origin
        b1 = np.cross(n, [1, 0, 0]); b1 /= np.linalg.norm(b1)
        b2 = np.cross(n, b1)
        # center patch under the hands if any, else under camera ray
        if len(hands):
            cm = np.mean([r[1:64].reshape(21, 3).mean(0) for r in hands], 0)
            c0 = cm - (cm @ n + d) * n
        g = np.linspace(-0.45, 0.45, 2)
        quad = c0 + np.outer(np.repeat(g, 2), b1) + np.outer(np.tile(g, 2), b2)
        ax.plot_trisurf(quad[:, 0], quad[:, 2], -quad[:, 1], alpha=0.35, color='tan')
        center = c0
    else:
        center = np.array([0, 0, 1.2])
    for row in hands:
        right = int(row[0]); J = row[1:64].reshape(21, 3)
        col = 'crimson' if right else 'royalblue'
        for e in EDGES:
            ax.plot(J[list(e), 0], J[list(e), 2], -J[list(e), 1], c=col, lw=2)
        ax.scatter(J[:, 0], J[:, 2], -J[:, 1], c=col, s=6)
    ax.set_xlim(center[0]-0.4, center[0]+0.4)
    ax.set_ylim(center[2]-0.4, center[2]+0.4)
    ax.set_zlim(-center[1]-0.4, -center[1]+0.4)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=28, azim=-60)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.canvas.draw()
    bottom = np.asarray(fig.canvas.buffer_rgba())[..., :3][..., ::-1].copy()
    plt.close(fig)

    panel = np.vstack([top, bottom])
    cv2.imwrite(str(out_dir / f'{name}.jpg'), panel, [cv2.IMWRITE_JPEG_QUALITY, 90])

print('frames rendered:', len(list(out_dir.glob('*.jpg'))))
fps = sys.argv[6] if len(sys.argv) > 6 else '30'
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', fps,
                '-pattern_type', 'glob', '-i', str(out_dir / 'frame_*.jpg'),
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                str(SAMPLES / out_name)], check=True)
print('PANEL-DONE ->', SAMPLES / out_name)
