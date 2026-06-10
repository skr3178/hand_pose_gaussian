# hand_pose_gaussian

Paper-reproduction workspace for hand pose extraction and dexterous manipulation from human videos.

## Project notes (start here)

- [goal.md](goal.md) — the pipeline we are building: monocular RGB video → HaMeR hand
  extraction → object 6-DoF track → 3DGS workspace → robot retargeting.
- [4.2.md](4.2.md) — deep reference notes on the RLDX-1 §4-2 human data pipeline
  (tool candidates per stage, caveats, checkpoints).

## Papers (included as PDFs)

- HO-Cap: A Capture System for Hand-Object Interaction ([arXiv:2406.06843](https://arxiv.org/abs/2406.06843))
- Human2Sim2Robot: Crossing the Embodiment Gap ([arXiv:2504.12609](https://arxiv.org/abs/2504.12609))
- ManipTrans: Bimanual Manipulation Transfer ([arXiv:2503.21860](https://arxiv.org/abs/2503.21860))
- ViViDex: Dexterous Manipulation from Human Videos ([arXiv:2404.15709](https://arxiv.org/abs/2404.15709))
- RLDX-1 Technical Report (arXiv:2605.03269)

## External repos (git submodules — content not uploaded)

Pinned as submodules; fetch with `git submodule update --init` (add `--recursive` for RLDX-1's own submodules):

| Submodule | Upstream | Role |
|---|---|---|
| `RLDX-1/` | [RLWRLD/RLDX-1](https://github.com/RLWRLD/RLDX-1) | paper's released VLA trainer |
| `hamer/` | [geopavlakos/hamer](https://github.com/geopavlakos/hamer) | monocular 3D hand mesh recovery |
| `WiLoR/` | [rolpotamias/WiLoR](https://github.com/rolpotamias/WiLoR) | hand detection + 3D mesh, overlay demo |

## Other local assets (not uploaded)

| Item | Source |
|---|---|
| `mano_v1_2/models/MANO_{LEFT,RIGHT}.pkl` | Register + download at [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de) (license forbids redistribution) |
| `hand_body_estimation` | Local symlink to `/media/skr/storage/3DGS/RhodusAI/hand_body_estimation` (separate project) |
| Model checkpoints (`*.pt`, `*.pth`, `*.ckpt`) | See each tool's README |

## Related tools

- [HO-Cap toolkit](https://github.com/IRVLUTD/HO-Cap) / [HO-Cap-Annotation](https://github.com/IRVLUTD/HO-Cap-Annotation) — dataset and multi-view annotation pipeline
- [Human2Sim2Robot](https://github.com/tylerlum/human2sim2robot) — one RGB-D human demo → sim → real (closest open pipeline to RLDX-1 §4-2)
- [Real2Render2Real](https://github.com/uynitsuj/real2render2real) — smartphone scan + human demo + 3DGS → rendered robot data
- [dex-retargeting](https://github.com/dexsuite/dex-retargeting) — human hand → robot hand joint mapping (Stage 3)
