"""Fuse HaMeR hand poses with metric depth: anchor wrists, fit table plane,
plot metric 3D skeletons (first static Panel-1b pass).

Run from the hamer/ directory so _DATA paths resolve:
    cd hamer && PYOPENGL_PLATFORM=egl ../.venv-hamer/bin/python ../samples/fuse_depth_hands.py
"""
import os, sys
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
FRAMES = ['0001','0015','0030','0045','0060','0075','0090','0105','0120']

# OpenPose hand skeleton edges
EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),
         (11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]

def build_detector():
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
    cfg_path = Path(hamer.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    from detectron2.engine import DefaultPredictor as _DP  # noqa
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    return DefaultPredictor_Lazy(detectron2_cfg)

def ransac_plane(pts, iters=300, thresh=0.01):
    """RANSAC plane fit. Returns (normal, d) with normal·p + d = 0."""
    rng = np.random.default_rng(0)
    best_inl, best = 0, None
    n_pts = pts.shape[0]
    for _ in range(iters):
        idx = rng.choice(n_pts, 3, replace=False)
        p0, p1, p2 = pts[idx]
        n = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(n)
        if nn < 1e-8:
            continue
        n = n / nn
        d = -n.dot(p0)
        dist = np.abs(pts @ n + d)
        inl = (dist < thresh).sum()
        if inl > best_inl:
            best_inl, best = inl, (n, d)
    n, d = best
    inliers = np.abs(pts @ n + d) < thresh
    # least-squares refine on inliers
    P = pts[inliers]
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c)
    n = vt[2]
    d = -n.dot(c)
    return n, d, inliers.mean()

def main():
    model, model_cfg = load_hamer(DEFAULT_CHECKPOINT)
    device = 'cuda'
    model = model.to(device).eval()
    detector = build_detector()
    cpm = ViTPoseModel(device)

    out_dir = SAMPLES / 'ego_fusion'
    out_dir.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    report = []
    for f in FRAMES:
        img_path = SAMPLES / 'ego_subset' / f'frame_{f}.jpg'
        depth_path = SAMPLES / 'ego_depth' / f'frame_{f}_depth.npy'
        if not depth_path.exists():
            print(f, 'no depth, skip'); continue
        img_cv2 = cv2.imread(str(img_path))
        depth = np.load(depth_path)  # HxW meters, full resolution
        H, W = img_cv2.shape[:2]

        # ---- detection chain (as in demo.py) ----
        det_out = detector(img_cv2)
        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()
        vitposes_out = cpm.predict_pose(
            img_cv2[:, :, ::-1],
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )
        bboxes, is_right = [], []
        for vitposes in vitposes_out:
            for kps, right in [(vitposes['keypoints'][-42:-21], 0), (vitposes['keypoints'][-21:], 1)]:
                valid = kps[:, 2] > 0.5
                if valid.sum() > 3:
                    bboxes.append([kps[valid, 0].min(), kps[valid, 1].min(),
                                   kps[valid, 0].max(), kps[valid, 1].max()])
                    is_right.append(right)
        if not bboxes:
            print(f, 'NO HANDS'); report.append((f, None)); continue
        boxes = np.stack(bboxes); right = np.stack(is_right)

        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=2.0)
        loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)

        # nominal intrinsics: HaMeR's scaled focal convention for this image size
        scaled_focal = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * max(H, W)
        K = np.array([[scaled_focal, 0, W / 2], [0, scaled_focal, H / 2], [0, 0, 1]])

        # ---- table plane from depth point cloud ----
        ys, xs = np.mgrid[0:H:8, 0:W:8]
        zs = depth[ys, xs]
        good = (zs > 0.1) & (zs < 4.0)
        pts = np.stack([(xs[good] - W/2) * zs[good] / scaled_focal,
                        (ys[good] - H/2) * zs[good] / scaled_focal,
                        zs[good]], axis=1)
        n, d, inl_frac = ransac_plane(pts)
        # orient normal to point up (towards camera, -y-ish in cam coords)
        if n[1] > 0: n, d = -n, -d

        frame_hands = []
        for batch in loader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)
            multiplier = (2 * batch['right'] - 1)
            pred_cam = out['pred_cam']
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            cam_t = cam_crop_to_full(pred_cam, box_center, box_size, img_size,
                                     torch.tensor(scaled_focal)).cpu().numpy()
            kps3d = out['pred_keypoints_3d'].cpu().numpy()  # (B,21,3) hand frame
            rights = batch['right'].cpu().numpy()
            for i in range(kps3d.shape[0]):
                k = kps3d[i].copy()
                k[:, 0] = (2 * rights[i] - 1) * k[:, 0]
                joints_cam = k + cam_t[i]  # camera-frame, HaMeR's depth guess
                # ---- anchor: wrist pixel -> metric depth ----
                wrist = joints_cam[0]
                u = int(np.clip(wrist[0] / wrist[2] * scaled_focal + W/2, 0, W-1))
                v = int(np.clip(wrist[1] / wrist[2] * scaled_focal + H/2, 0, H-1))
                # median depth in 9x9 window (robust to edges)
                win = depth[max(0,v-4):v+5, max(0,u-4):u+5]
                z_meas = float(np.median(win[(win > 0.1) & (win < 4.0)])) if win.size else None
                if z_meas is None or np.isnan(z_meas):
                    frame_hands.append(dict(joints=joints_cam, right=int(rights[i]), anchored=False))
                    continue
                ratio = z_meas / wrist[2]
                t_new = wrist * ratio - k[0]
                joints_metric = k + t_new
                # height above table for wrist + min over joints
                h_wrist = joints_metric[0] @ n + d
                h_min = float((joints_metric @ n + d).min())
                frame_hands.append(dict(joints=joints_metric, right=int(rights[i]), anchored=True,
                                        z_cam=z_meas, h_wrist=float(h_wrist), h_min=h_min))

        # ---- 3D plot: skeletons + plane patch ----
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        # plane patch around point cloud footprint
        c = pts[np.abs(pts @ n + d) < 0.01].mean(0)
        b1 = np.cross(n, [1, 0, 0]); b1 /= np.linalg.norm(b1)
        b2 = np.cross(n, b1)
        g = np.linspace(-0.5, 0.5, 2)
        quad = c + np.outer(np.repeat(g, 2), b1) + np.outer(np.tile(g, 2), b2)
        ax.plot_trisurf(quad[:, 0], quad[:, 2], -quad[:, 1], alpha=0.3, color='tan')
        for hd in frame_hands:
            J = hd['joints']
            col = 'crimson' if hd['right'] else 'royalblue'
            for e in EDGES:
                ax.plot(J[list(e), 0], J[list(e), 2], -J[list(e), 1], c=col, lw=2)
            ax.scatter(J[:, 0], J[:, 2], -J[:, 1], c=col, s=8)
        ax.set_xlabel('x (m)'); ax.set_ylabel('z depth (m)'); ax.set_zlabel('up (m)')
        ax.set_title(f'frame {f} — metric hands + table plane (plane inliers {inl_frac:.0%})')
        ax.view_init(elev=15, azim=-75)
        plt.tight_layout()
        plt.savefig(out_dir / f'frame_{f}_3d.png', dpi=110)
        plt.close()
        report.append((f, frame_hands))
        for hd in frame_hands:
            if hd.get('anchored'):
                print(f"frame {f} {'R' if hd['right'] else 'L'}: wrist {hd['z_cam']:.2f} m from cam, "
                      f"{hd['h_wrist']*100:+.1f} cm above plane, closest joint {hd['h_min']*100:+.1f} cm")
            else:
                print(f"frame {f}: hand without depth anchor")
    print('FUSION-DONE')

if __name__ == '__main__':
    main()
