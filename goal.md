# Goal — Hand-Demonstration Pipeline from Monocular Video

Target: rebuild the RLDX-1 §4-2 human data capture pipeline ("Video 7 — Human Data
Capture Pipeline"), but driven by **monocular RGB video** as the input — turn
single-camera bare-hand demonstrations into robot training data.

## Input

- A single-camera (monocular) RGB video of a person at a table manipulating an
  object with bare hands. No multi-camera rig, no depth sensor, no calibration.
- Frames extracted with ffmpeg before processing.

This rules out multi-view triangulation pipelines (e.g. HO-Cap-Annotation, which
needs 8 calibrated RealSense cameras). Every stage below must work from one view.

## Pipeline to construct

- [ ] **Stage 0 — Preprocess.** ffmpeg video → frames; pick working clips.
- [ ] **Stage 1a — Hand extraction (HaMeR).** Per frame: ViTDet person detection →
      ViTPose hand boxes → HaMeR regresses MANO parameters (wrist 6-DoF, 15 finger
      joint rotations, shape) + 21 3D keypoints + camera.
      Downloads: `fetch_demo_data.sh` (~2.5 GB checkpoints) + `MANO_RIGHT.pkl`
      (registration at mano.is.tue.mpg.de).
      Add temporal smoothing — HaMeR is per-frame and jitters on video.
- [ ] **Stage 1b — Object 6-DoF track.** FoundationPose (needs object mesh) or
      BundleSDF (model-free); both run monocular RGB(-D) — without depth expect
      scale ambiguity, resolve against the MANO hand scale.
- [ ] **Stage 2 — Reconstruct workspace (3DGS).** Static scene scan (slow camera
      sweep of the empty table) → COLMAP poses → gaussian-splatting / gsplat.
      Mask the moving hand/object using stage-1 tracks; place tracked hand
      keypoints + object geometry in the reconstructed workspace.
- [ ] **Stage 3 — Retarget.** `dex-retargeting` maps MANO keypoints → robot hand
      joint angles (ALLEX URDF), with a feasibility/collision filter for grasps
      that have no robot equivalent.

(Stage 4 — sim rollout → LeRobot VLA data — is the downstream step once 1–3 work.)

## Known monocular-specific risks

- **Depth/scale ambiguity:** single view gives hand pose up to scale; aligning the
  hand, object, and 3DGS workspace into one metric frame is the new hard part the
  multi-view rig solved for free.
- **Occlusion under contact:** fingers + object occlude each other with no second
  view to recover from; bad tracking silently poisons everything downstream.

## Build order

1. Prototype Stage 1a alone on real footage (HaMeR demo on extracted frames).
   If hand tracking under contact-occlusion isn't reliable, the rest is moot.
2. Add object tracking (1b), then scale/frame alignment.
3. Only then build stages 2 → 3.

See `4.2.md` for the full pipeline notes, tool candidates per stage, caveats, and
checkpoint references.
