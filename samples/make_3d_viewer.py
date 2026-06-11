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

data = np.load(SAMPLES / fusion_dir / 'fusion.npz')
names = sorted(n for n in data.files if not n.endswith('__plane') and not n.startswith('meta__'))
names = [n for n in names if len(data[n])][::stride]

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
