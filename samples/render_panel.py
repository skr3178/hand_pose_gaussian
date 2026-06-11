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
GREEN = (80, 200, 80)
# COCO-17 body chains (no wrist edges: arms connect to hands by proximity)
BODY_EDGES = [(5,7),(6,8),(5,6),(5,11),(6,12),(11,12),(0,5),(0,6)]


def synth_skeleton(data, names, gravity):
    """Fixed synthetic torso: neck, shoulders, 3-segment spine, pelvis (clip-level).
    Built from median measured elbow position; returns dict or None."""
    g = np.asarray(gravity, dtype=float)
    g /= np.linalg.norm(g)
    if g[1] < 0: g = -g          # ensure g points image-down (world down)
    # only the pelvis is FIXED (egocentric anchor below the camera/head);
    # neck/shoulders are computed per frame and move freely
    s = np.cross(g, [0.0, 0.0, 1.0]); s /= np.linalg.norm(s)
    pelvis = np.array([0.0, 0.0, 0.12]) + 0.83 * g
    return dict(pelvis=pelvis, g=g, s=s)

FOREARM = 0.26   # m, drawn forearm length (direction measured, length clamped)
SPINE_L = 0.55   # m, pelvis->neck

UPPER_ARM = 0.28  # m, elbow->shoulder

def torso_frame(SY, elbow_pts):
    """Fully articulated per-frame torso. Only the pelvis is fixed.
    shoulders[i] corresponds to elbow_pts[i]."""
    pelvis, g, s = SY['pelvis'], SY['g'], SY['s']
    up = -g
    shoulders = []
    for e in elbow_pts:
        # pull each shoulder from its elbow toward the body axis (vertical
        # line through the pelvis) at upper-arm length
        t = max((e - pelvis) @ up, 0.25)
        p_axis = pelvis + t * up
        d = p_axis - e
        shoulders.append(e + UPPER_ARM * d / (np.linalg.norm(d) + 1e-9))
    if shoulders:
        neck = np.mean(np.stack(shoulders), axis=0)
        if len(shoulders) == 1:
            side = 1.0 if (shoulders[0] - pelvis) @ s > 0 else -1.0
            neck = shoulders[0] - side * 0.18 * s
    else:
        neck = pelvis + SPINE_L * up
    # clamp spine length, then bend it: vertical at the base, curving to neck
    v = neck - pelvis
    L = np.linalg.norm(v)
    Lc = min(max(L, 0.40), 0.62)
    neck = pelvis + v / (L + 1e-9) * Lc
    C = pelvis + 0.5 * Lc * up   # Bezier control: base rises along gravity
    spine = [(1-t)**2 * pelvis + 2*(1-t)*t * C + t**2 * neck
             for t in (0, 1/3, 2/3, 1.0)]
    return neck, spine, shoulders

def clamp_forearm(pairs):
    out = []
    for e, w in pairs:
        v = e - w; n = np.linalg.norm(v)
        out.append((w + FOREARM * v / n if n > 1e-9 else e, w))
    return out

def match_elbows(elbows, wrists):
    """Greedy unique nearest-neighbor: [(elbow_pos, wrist_pos), ...]"""
    pairs = sorted(((np.linalg.norm(e[1]-w[1]), ei, wi)
                    for ei, e in enumerate(elbows) for wi, w in enumerate(wrists)),
                   key=lambda t: t[0])
    used_e, used_w, out = set(), set(), []
    for _, ei, wi in pairs:
        if ei in used_e or wi in used_w: continue
        used_e.add(ei); used_w.add(wi)
        out.append((elbows[ei][1], wrists[wi][1]))
    return out

frames_dir = sys.argv[1] if len(sys.argv) > 1 else 'ego_sample_frames'
fusion_dir = sys.argv[2] if len(sys.argv) > 2 else 'ego_fusion_full'
panel_dir = sys.argv[3] if len(sys.argv) > 3 else 'ego_panel_frames'
out_name = sys.argv[4] if len(sys.argv) > 4 else 'ego_panel1.mp4'
data = np.load(SAMPLES / fusion_dir / 'fusion.npz')
frames = sorted(glob.glob(str(SAMPLES / frames_dir / 'frame_*.jpg')))
out_dir = SAMPLES / panel_dir
out_dir.mkdir(exist_ok=True)
GRAV = [0.0, 1.0, 0.0]
if len(sys.argv) > 7:
    import json as _json
    GRAV = _json.load(open(SAMPLES / sys.argv[7])).get('gravity_cam', GRAV)
_names_all = sorted(n for n in data.files if '__' not in n and not n.startswith('meta'))
SYNTH = synth_skeleton(data, _names_all, GRAV)

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
    body = data.get(f'{name}__body', np.zeros((17, 7)))

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
    # arm chains: match elbows to nearest hand wrist (labels are unreliable)
    elbows = [(j, body[j, :2].astype(float)) for j in (7, 8) if body[j, 2] > 0.3]
    wrists = []
    for row in hands:
        J = row[64:127].reshape(21, 3) if row.shape[0] >= 127 else row[1:64].reshape(21, 3)
        wrists.append((0, np.array([J[0,0]/J[0,2]*f + W/2, J[0,1]/J[0,2]*f + H/2])))
    for e_pos, w_pos in match_elbows(elbows, wrists):
        cv2.line(top, (int(e_pos[0]), int(e_pos[1])), (int(w_pos[0]), int(w_pos[1])),
                 GREEN, 3, cv2.LINE_AA)
    for a, b in BODY_EDGES:
        if body[a, 2] > 0.3 and body[b, 2] > 0.3:
            cv2.line(top, (int(body[a,0]), int(body[a,1])), (int(body[b,0]), int(body[b,1])),
                     GREEN, 3, cv2.LINE_AA)
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
    elbows3 = [(j, body[j, 3:6].astype(float)) for j in (7, 8) if body[j, 6] > 0]
    wrists3 = [(0, row[1:64].reshape(21, 3)[0].astype(float)) for row in hands]
    arm_pairs = clamp_forearm(match_elbows(elbows3, wrists3))
    for e_pos, w_pos in arm_pairs:
        ax.plot([e_pos[0], w_pos[0]], [e_pos[2], w_pos[2]], [-e_pos[1], -w_pos[1]],
                c='seagreen', lw=2.5)
    if SYNTH is not None:
        neck, spine, shoulders = torso_frame(SYNTH, [e for e, _ in arm_pairs])
        sp = np.stack(spine)
        ax.plot(sp[:, 0], sp[:, 2], -sp[:, 1], c='indigo', lw=3)
        ax.scatter(sp[:, 0], sp[:, 2], -sp[:, 1], c='k', s=20)
        ax.scatter([SYNTH['pelvis'][0]], [SYNTH['pelvis'][2]], [-SYNTH['pelvis'][1]],
                   c='k', s=60, marker='s')  # the fixed base
        for sh, (e_pos, _) in zip(shoulders, arm_pairs):
            ax.plot([neck[0], sh[0]], [neck[2], sh[2]], [-neck[1], -sh[1]], c='indigo', lw=3)
            ax.plot([sh[0], e_pos[0]], [sh[2], e_pos[2]], [-sh[1], -e_pos[1]],
                    c='gray', lw=1.8, ls='--')
    for a, b in BODY_EDGES:
        if body[a, 6] > 0 and body[b, 6] > 0:
            ax.plot([body[a,3], body[b,3]], [body[a,5], body[b,5]], [-body[a,4], -body[b,4]],
                    c='seagreen', lw=2.5)
    r_view = 0.4
    if SYNTH is not None:
        center = (center + SYNTH['pelvis']) / 2
        r_view = 0.7
    ax.set_xlim(center[0]-r_view, center[0]+r_view)
    ax.set_ylim(center[2]-r_view, center[2]+r_view)
    ax.set_zlim(-center[1]-r_view, -center[1]+r_view)
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
