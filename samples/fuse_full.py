"""Full-clip Stage 1-2-3: HaMeR on every frame, wrist depth anchor, plane fit.
Saves per-frame metric joints + plane to ego_fusion_full/fusion.npz.

Run: cd hamer && PYOPENGL_PLATFORM=egl ../.venv-hamer/bin/python ../samples/fuse_full.py
"""
import os, glob
import numpy as np
import torch
import cv2
from pathlib import Path

from hamer.models import load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.utils.renderer import cam_crop_to_full
from vitpose_model import ViTPoseModel
from detectron2.config import LazyConfig
import hamer

SAMPLES = Path(__file__).resolve().parent

def build_detector():
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    cfg_path = Path(hamer.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
    cfg = LazyConfig.load(str(cfg_path))
    cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    for i in range(3):
        cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    return DefaultPredictor_Lazy(cfg)

def ransac_plane(pts, iters=200, thresh=0.01):
    rng = np.random.default_rng(0)
    best_inl, best = 0, None
    for _ in range(iters):
        p0, p1, p2 = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(n)
        if nn < 1e-8: continue
        n /= nn
        d = -n.dot(p0)
        inl = (np.abs(pts @ n + d) < thresh).sum()
        if inl > best_inl: best_inl, best = inl, (n, d)
    n, d = best
    P = pts[np.abs(pts @ n + d) < thresh]
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c)
    n = vt[2]; d = -n.dot(c)
    if n[1] > 0: n, d = -n, -d
    return n, d

def main():
    model, model_cfg = load_hamer(DEFAULT_CHECKPOINT)
    model = model.to('cuda').eval()
    detector = build_detector()
    cpm = ViTPoseModel('cuda')

    frames = sorted(glob.glob(str(SAMPLES / 'ego_sample_frames' / 'frame_*.jpg')))
    out = {}   # frame name -> dict(hands=[(right, joints 21x3)], plane=(n,d))
    for fi, fp in enumerate(frames):
        name = Path(fp).stem
        dp = SAMPLES / 'ego_depth_full' / f'{name}_depth.npy'
        if not dp.exists(): continue
        img_cv2 = cv2.imread(fp)
        depth = np.load(dp)
        H, W = img_cv2.shape[:2]
        # realistic nominal intrinsics (~53 deg hfov); HaMeR cam_t adapts via
        # cam_crop_to_full so 2D projections stay aligned under this same K
        scaled_focal = float(max(H, W))

        det = detector(img_cv2)['instances']
        vmask = (det.pred_classes == 0) & (det.scores > 0.5)
        boxes_p = det.pred_boxes.tensor[vmask].cpu().numpy()
        scores_p = det.scores[vmask].cpu().numpy()
        if len(boxes_p) == 0:
            out[name] = dict(hands=[], plane=None); continue
        vitposes_out = cpm.predict_pose(img_cv2[:, :, ::-1],
                                        [np.concatenate([boxes_p, scores_p[:, None]], axis=1)])
        bboxes, is_right = [], []
        for vp in vitposes_out:
            for kps, r in [(vp['keypoints'][-42:-21], 0), (vp['keypoints'][-21:], 1)]:
                valid = kps[:, 2] > 0.5
                if valid.sum() > 3:
                    bboxes.append([kps[valid,0].min(), kps[valid,1].min(),
                                   kps[valid,0].max(), kps[valid,1].max()])
                    is_right.append(r)
        ys, xs = np.mgrid[0:H:8, 0:W:8]
        zs = depth[ys, xs]
        good = (zs > 0.1) & (zs < 4.0)
        pts = np.stack([(xs[good]-W/2)*zs[good]/scaled_focal,
                        (ys[good]-H/2)*zs[good]/scaled_focal, zs[good]], 1)
        n, d = ransac_plane(pts)

        hands = []
        if bboxes:
            ds = ViTDetDataset(model_cfg, img_cv2, np.stack(bboxes), np.stack(is_right), rescale_factor=2.0)
            for batch in torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False):
                batch = recursive_to(batch, 'cuda')
                with torch.no_grad():
                    o = model(batch)
                mult = (2 * batch['right'] - 1)
                pc = o['pred_cam']; pc[:, 1] = mult * pc[:, 1]
                cam_t = cam_crop_to_full(pc, batch['box_center'].float(), batch['box_size'].float(),
                                         batch['img_size'].float(), torch.tensor(scaled_focal)).cpu().numpy()
                k3 = o['pred_keypoints_3d'].cpu().numpy()
                rts = batch['right'].cpu().numpy()
                for i in range(len(k3)):
                    k = k3[i].copy(); k[:, 0] = (2*rts[i]-1)*k[:, 0]
                    jc = k + cam_t[i]
                    w = jc[0]
                    u = int(np.clip(w[0]/w[2]*scaled_focal + W/2, 0, W-1))
                    v = int(np.clip(w[1]/w[2]*scaled_focal + H/2, 0, H-1))
                    win = depth[max(0,v-4):v+5, max(0,u-4):u+5]
                    wv = win[(win > 0.1) & (win < 4.0)]
                    if wv.size:
                        t_new = w * (float(np.median(wv))/w[2]) - k[0]
                        jc = k + t_new
                    hands.append((int(rts[i]), jc))
        out[name] = dict(hands=hands, plane=(n, d))
        if fi % 20 == 0: print(f'{fi+1}/{len(frames)}', flush=True)

    # ---- Stage 5: One-Euro-ish smoothing of wrists (per handedness track) ----
    def one_euro(xs, min_cutoff=1.0, beta=0.3, fps=30.0):
        a = lambda cut: 1.0/(1.0 + fps/(2*np.pi*cut))
        xh, dxh = xs[0], np.zeros_like(xs[0])
        res = [xs[0]]
        for x in xs[1:]:
            dx = (x - xh) * fps
            dxh = a(1.0)*dx + (1-a(1.0))*dxh
            cut = min_cutoff + beta*np.linalg.norm(dxh)
            xh = a(cut)*x + (1-a(cut))*xh
            res.append(xh.copy())
        return res

    names = sorted(out.keys())
    for side in (0, 1):
        idxs = [nm for nm in names if any(h[0] == side for h in out[nm]['hands'])]
        if len(idxs) < 2: continue
        track = [next(h[1] for h in out[nm]['hands'] if h[0] == side) for nm in idxs]
        wrists = one_euro([t[0] for t in track])
        for nm, joints, w_s in zip(idxs, track, wrists):
            sm = joints + (w_s - joints[0])  # rigid shift to smoothed wrist
            out[nm]['hands'] = [(r, sm if r == side else j) for r, j in out[nm]['hands']]

    np.savez_compressed(SAMPLES / 'ego_fusion_full' / 'fusion.npz',
                        **{nm: np.array(
                            [np.concatenate([[h[0]], h[1].ravel()]) for h in v['hands']]
                            if v['hands'] else np.zeros((0, 64)), dtype=np.float32)
                           for nm, v in out.items()},
                        **{f'{nm}__plane': (np.concatenate([v['plane'][0], [v['plane'][1]]])
                                            if v['plane'] else np.zeros(4))
                           for nm, v in out.items()})
    n_det = sum(1 for v in out.values() if v['hands'])
    print(f'FUSION-FULL-DONE {n_det}/{len(out)} frames with hands')

if __name__ == '__main__':
    os.makedirs(SAMPLES / 'ego_fusion_full', exist_ok=True)
    main()
