# Goal — Reproduce the RLDX-1 Human Data Capture Pipeline (§4-2)

Target: rebuild the stages shown in "Video 7 — Human Data Capture Pipeline"
so we can turn bare-hand demonstrations into robot training data.

## Reference (what the video depicts)

**Left (two stacked frames) — Stage 1 + 2.**
- Top: a person at a wooden table, bare hands overlaid with colored hand-tracking
  meshes (purple/blue on one hand, orange/red on the other), manipulating a small
  object, lab equipment in the background.
- Bottom: the reconstructed scene — a flat tan tabletop with sparse colored keypoint
  markers (hand joints + the tracked object) — i.e. the 3DGS workspace + tracked
  geometry.

**Middle (arrow in) — Stage 3.**
- A white humanoid robot (ALLEX) on a grid floor, arms reaching down to the object —
  the human motion retargeted onto the robot embodiment.

## Goals

- [ ] **Stage 1 — Capture + track.** Record bare-hand video at a table; recover
      per-frame 3D hand mesh/keypoints + object 6-DoF pose.
- [ ] **Stage 2 — Reconstruct.** Build the 3DGS workspace and place the tracked hand
      keypoints + object geometry in it.
- [ ] **Stage 3 — Retarget.** Map the human hand motion onto the ALLEX robot
      embodiment, with a feasibility/collision filter for infeasible grasps.

(Stage 4 — sim rollout → LeRobot VLA data — is the downstream step once 1–3 work.)

See `4.2.md` for the full pipeline notes, tool candidates per stage, caveats, and
the recommended build order.
