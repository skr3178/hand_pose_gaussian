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

## External dependencies (not uploaded — download separately)

| Folder | Source |
|---|---|
| `RLDX-1/` | `git clone https://github.com/RLWRLD/RLDX-1.git` |
| `hand_body_estimation` | Local symlink to `/media/skr/storage/3DGS/RhodusAI/hand_body_estimation` (separate project) |

## Related tools

- [HaMeR](https://github.com/geopavlakos/hamer) — monocular 3D hand mesh recovery (planned for hand/finger extraction)
- [HO-Cap toolkit](https://github.com/IRVLUTD/HO-Cap) / [HO-Cap-Annotation](https://github.com/IRVLUTD/HO-Cap-Annotation) — dataset and multi-view annotation pipeline
