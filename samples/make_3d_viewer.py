"""Interactive 3D viewer (Plotly HTML) of metric hands + table plane per frame.
Usage: make_3d_viewer.py <fusion_dir> <out_html> [stride]
Open the resulting .html in a browser: drag to rotate, wheel to zoom,
slider/play to move through the clip.
"""
import sys
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

SAMPLES = Path(__file__).resolve().parent
fusion_dir = sys.argv[1] if len(sys.argv) > 1 else 'ego_fusion_full'
out_html = sys.argv[2] if len(sys.argv) > 2 else 'ego_3d_viewer.html'
stride = int(sys.argv[3]) if len(sys.argv) > 3 else 2

EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),
         (11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
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
    pairs = sorted(((np.linalg.norm(e[1]-w[1]), ei, wi)
                    for ei, e in enumerate(elbows) for wi, w in enumerate(wrists)),
                   key=lambda t: t[0])
    used_e, used_w, out = set(), set(), []
    for _, ei, wi in pairs:
        if ei in used_e or wi in used_w: continue
        used_e.add(ei); used_w.add(wi)
        out.append((elbows[ei][1], wrists[wi][1]))
    return out

data = np.load(SAMPLES / fusion_dir / 'fusion.npz')
names = sorted(n for n in data.files
               if '__' not in n and not n.startswith('meta'))
names = [n for n in names if len(data[n])][::stride]

GRAV = [0.0, 1.0, 0.0]
if len(sys.argv) > 4:
    import json as _json
    GRAV = _json.load(open(SAMPLES / sys.argv[4])).get('gravity_cam', GRAV)
SYNTH = synth_skeleton(data, names, GRAV)

def world(p):  # cam coords -> display coords (x, depth, up)
    return p[:, 0], p[:, 2], -p[:, 1]

def hand_lines(J):
    xs, ys, zs = [], [], []
    for a, b in EDGES:
        xs += [J[a,0], J[b,0], None]; ys += [J[a,2], J[b,2], None]; zs += [-J[a,1], -J[b,1], None]
    return xs, ys, zs

def frame_traces(nm):
    traces = []
    plane = data[f'{nm}__plane']
    hands = data[nm]
    centers = [r[1:64].reshape(21,3).mean(0) for r in hands]
    cm = np.mean(centers, 0) if centers else np.array([0,0,1.0])
    if plane.any():
        n, d = plane[:3], plane[3]
        c0 = cm - (cm @ n + d) * n
        b1 = np.cross(n, [1,0,0]); b1 /= np.linalg.norm(b1)
        b2 = np.cross(n, b1)
        g = 0.45
        corners = np.array([c0 + sx*g*b1 + sy*g*b2 for sx, sy in
                            [(-1,-1), (1,-1), (1,1), (-1,1)]])
        x, y, z = world(corners)
        traces.append(go.Mesh3d(x=x, y=y, z=z, i=[0,0], j=[1,2], k=[2,3],
                                color='tan', opacity=0.45, name='table'))
    body = data.get(f'{nm}__body', np.zeros((17, 7)))
    elbows3 = [(j, body[j, 3:6].astype(float)) for j in (7, 8) if body[j, 6] > 0]
    wrists3 = [(0, row[1:64].reshape(21, 3)[0].astype(float)) for row in hands]
    arm_pairs = clamp_forearm(match_elbows(elbows3, wrists3))
    bxs, bys, bzs = [], [], []
    for e_pos, w_pos in arm_pairs:
        bxs += [e_pos[0], w_pos[0], None]
        bys += [e_pos[2], w_pos[2], None]
        bzs += [-e_pos[1], -w_pos[1], None]
    for a, b in BODY_EDGES:
        if body[a, 6] > 0 and body[b, 6] > 0:
            bxs += [body[a,3], body[b,3], None]
            bys += [body[a,5], body[b,5], None]
            bzs += [-body[a,4], -body[b,4], None]
    if SYNTH is not None:
        neck, spine, shoulders = torso_frame(SYNTH, [e for e, _ in arm_pairs])
        sp = np.stack(spine)
        sxs, sys_, szs = list(sp[:, 0]), list(sp[:, 2]), list(-sp[:, 1])
        traces.append(go.Scatter3d(x=sxs, y=sys_, z=szs, mode='lines+markers',
                                   line=dict(color='indigo', width=8),
                                   marker=dict(size=5, color='black'), name='torso (synthetic)'))
        traces.append(go.Scatter3d(x=[SYNTH['pelvis'][0]], y=[SYNTH['pelvis'][2]],
                                   z=[-SYNTH['pelvis'][1]], mode='markers',
                                   marker=dict(size=9, color='black', symbol='square'),
                                   name='fixed base'))
        for sh, (e_pos, _) in zip(shoulders, arm_pairs):
            traces.append(go.Scatter3d(
                x=[neck[0], sh[0], e_pos[0]], y=[neck[2], sh[2], e_pos[2]],
                z=[-neck[1], -sh[1], -e_pos[1]], mode='lines',
                line=dict(color='gray', width=4, dash='dash'), showlegend=False))
    if bxs:
        traces.append(go.Scatter3d(x=bxs, y=bys, z=bzs, mode='lines',
                                   line=dict(color='seagreen', width=7), name='body'))
    for row in hands:
        right = int(row[0]); J = row[1:64].reshape(21, 3)
        col = 'crimson' if right else 'royalblue'
        xs, ys, zs = hand_lines(J)
        traces.append(go.Scatter3d(x=xs, y=ys, z=zs, mode='lines',
                                   line=dict(color=col, width=6),
                                   name='right' if right else 'left'))
        x, y, z = world(J)
        traces.append(go.Scatter3d(x=x, y=y, z=z, mode='markers',
                                   marker=dict(size=3, color=col), showlegend=False))
    return traces

# global bounds for a stable view
allJ = np.concatenate([data[nm][:, 1:64].reshape(-1, 3) for nm in names])
cx, cy, cz = allJ[:,0], allJ[:,2], -allJ[:,1]
c = np.array([cx.mean(), cy.mean(), cz.mean()]); r = 0.6
if SYNTH is not None:
    c = (c + np.array([SYNTH['pelvis'][0], SYNTH['pelvis'][2], -SYNTH['pelvis'][1]])) / 2
    r = 0.9
rng = dict(xaxis=dict(range=[c[0]-r, c[0]+r], title='x (m)'),
           yaxis=dict(range=[c[1]-r, c[1]+r], title='depth (m)'),
           zaxis=dict(range=[c[2]-r, c[2]+r], title='up (m)'),
           aspectmode='cube')

frames = [go.Frame(data=frame_traces(nm), name=nm) for nm in names]
fig = go.Figure(data=frames[0].data, frames=frames)
steps = [dict(method='animate', label=nm.replace('frame_',''),
              args=[[nm], dict(mode='immediate', frame=dict(duration=0, redraw=True))])
         for nm in names]
fig.update_layout(
    scene=rng, title=f'{fusion_dir}: metric hands + table plane (drag to rotate)',
    sliders=[dict(steps=steps, currentvalue=dict(prefix='frame '))],
    updatemenus=[dict(type='buttons', buttons=[
        dict(label='▶ play', method='animate',
             args=[None, dict(frame=dict(duration=120, redraw=True), fromcurrent=True)]),
        dict(label='⏸ pause', method='animate',
             args=[[None], dict(mode='immediate')])])],
    margin=dict(l=0, r=0, t=40, b=0))
fig.write_html(SAMPLES / out_html, include_plotlyjs='cdn', auto_play=False)
print('VIEWER ->', SAMPLES / out_html, f'({len(names)} frames)')
