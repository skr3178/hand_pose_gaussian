# Rebuilding the RLDX-1 §4-2 Human Data Capture Panel from Monocular RGB

**Goal:** Recreate Panel 1 (left) of "Video 7 — Human Data Capture Pipeline," driven by a single uncalibrated RGB camera. Turn bare-hand tabletop demonstrations into robot training data.

## Status (2026-06-10)

| Stage | State | Findings |
|---|---|---|
| 0 — frames | ✅ done | `samples/ego_sample.mp4` (HaWoR ego clip, 121 frames extracted) |
| 1 — hand pose | ✅ both models verified | HaMeR and WiLoR run in shared `.venv-hamer`; mesh quality near-identical on 9 test frames (`samples/ego_compare_3way.jpg`). WiLoR recovered 1 of the 2 motion-blur frames HaMeR missed; detector is 51 MB vs HaMeR's 2.4 GB chain. **Stage-1 pick: WiLoR** (also feeds HaWoR later); HaMeR kept as benchmark baseline. |
| 2 — metric lift | 🔄 in progress | Depth Anything V2 metric (indoor) run on test frames (`samples/ego_depth/`); wrist-anchor fusion script at `samples/fuse_depth_hands.py` |
| 3 — table plane | 🔄 in progress | RANSAC plane fit inside the same fusion script |
| 4–6 | ⏳ pending | HaWoR for world-frame on moving cameras; Rerun panel |

Setup details + gotchas: see `setup_log.md`. Per-frame mesh quality and blur
dropouts documented in `samples/ego_compare_3way.jpg` and `samples/ego_side_by_side/`.

## Panel 1 anatomy

Panel 1 is two synchronized sub-views fed by one pipeline:

- **1a (top):** Raw egocentric RGB with per-hand colored mask overlays (blue = left, red = right), 2D keypoint skeletons, and an object-pose wireframe on the manipulated object.
- **1b (bottom):** A world-frame 3D viewer showing lifted 3D hand skeletons floating over a fitted table plane.

## Stage 0 — Frame extraction

`ffmpeg` to extract frames (already in place). 30 fps is typically enough; keep original resolution for hand crops.

## Stage 1 — Hand detection, handedness, and 3D hand pose (per frame)

**Tool: WiLoR** — https://github.com/rolpotamias/WiLoR

- End-to-end multi-hand localization + reconstruction: a real-time fully convolutional hand detector plus a transformer-based 3D reconstruction model. Demonstrates smooth 3D hand tracking from monocular video without temporal components.
- Outputs per-hand MANO parameters, handedness, and 2D/3D keypoints.
- Sub-view 1a rendering:
  - 2D skeleton = projected MANO joints, drawn with OpenCV.
  - Blue/red overlay = MANO mesh rendered as a filled silhouette (pyrender or nvdiffrast), alpha-blended onto the frame.
  - For pixel-accurate masks instead of rendered-mesh masks, prompt **SAM 2** with WiLoR's hand boxes.
- Alternative: HaMeR (older, no built-in detector; WiLoR is faster).

## Stage 2 — Lifting to a consistent world frame

WiLoR output lives in a per-frame camera frame with weak-perspective scale — insufficient for 1b and for robot data.

- **Moving / head-mounted camera:** **HaWoR** — https://github.com/ThunderVVV/HaWoR
  - Reconstructs hand motion in world coordinates from egocentric monocular video by decoupling camera-frame hand motion from world-space camera trajectory, via an adaptive egocentric SLAM framework.
  - Includes a motion-infiller network that completes frames where hands leave the view frustum.
  - Same group as WiLoR; uses WiLoR internally.
- **Static tripod camera:** skip SLAM.
  - Estimate intrinsics once (GeoCalib, or assume nominal FOV).
  - Resolve metric scale with a monocular metric-depth model — **Depth Anything V2 (metric)**, **UniDepth**, or **MoGe** — by anchoring wrist/hand depth to the depth map.

## Stage 3 — Table plane

1. Run the metric depth model on a clean frame.
2. Back-project to a point cloud.
3. RANSAC-fit the dominant plane.
4. Define the world frame on it: table = z 0, gravity from the plane normal.

The fitted quad is the beige plane rendered in sub-view 1b, and it makes hand trajectories physically meaningful (heights above table, contact events).

## Stage 4 — Object 6-DoF pose

Hardest monocular piece. Two routes:

- **Model-based (recommended for repeatable robot data):**
  - Get a mesh once — phone photogrammetry, or an image-to-3D model like TRELLIS.
  - Track with **FoundationPose** — https://github.com/NVlabs/FoundationPose
  - FoundationPose nominally wants RGB-D; feeding it depth from Depth Anything V2 as pseudo-RGB-D is standard practice in monocular pipelines.
- **Template-free:**
  - **HOLD** — https://github.com/zc-alexfan/hold — jointly reconstructs articulated hands and objects from monocular video without a pre-scanned object template or 3D hand-object training data; supports custom videos and bimanual sequences. Slower (per-sequence optimization), zero per-object setup.
  - EasyHOI / MagicHOI are newer alternatives for short clips.

## Stage 5 — Temporal cleanup

- One-Euro filter or spline fit on MANO pose parameters and wrist SE(3) trajectories.
- HaWoR's infiller already handles occlusion dropouts if used.

## Stage 6 — Rendering the panel

- **Rerun** (https://rerun.io, open source): log the annotated RGB stream to one view and the 3D skeletons + table plane to a 3D view below it; screen-record or render headless.
- Alternative: draw 1a with OpenCV, render 1b with viser/Open3D, compose with `ffmpeg -filter_complex vstack` (and `hstack` for the full 3-panel layout).

## End-to-end reference pipelines

- **VideoManip** (arXiv:2602.09013) — device-free framework reconstructing explicit robot-object trajectories from monocular RGB human videos (hand poses + object meshes), then retargeting to robot hands.
- **RoboWheel** (arXiv:2512.02729) — end-to-end monocular video → HOI reconstruction → cross-embodiment retargeting → augmentation, with an RL optimizer enforcing contact and penetration constraints.

Their preprocessing code is the closest existing open-source analog to this rebuild.

## Practical warnings

- **Metric scale is the dominant error source.** Anchor everything to the table plane; if possible, place one object of known size in frame as a sanity check.
- **MANO licensing:** register at the official MANO site before WiLoR / HaWoR / HOLD will run.

## Output data format (handoff to retargeting, Panel 2)

Per frame, in the table/world frame:

- MANO pose θ and shape β per hand
- Wrist SE(3) trajectory per hand
- Fingertip keypoints
- Object SE(3) pose (if Stage 4 enabled)
- Contact events (hand-table / hand-object proximity thresholds)
