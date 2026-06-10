# Setup Log — Stage 1 Hand Extraction (HaMeR / WiLoR)

What was done to get monocular hand-pose extraction running on this machine
(2026-06-10). Companion to `monocular_capture_pipeline_plan.md` (the plan) and
`goal.md` (the goal); this file records the actual setup and its gotchas.

## Machine

- GPU: NVIDIA RTX PRO 4000 Blackwell, 24 GB (sm_120 — needs CUDA 12.8 wheels)
- Driver 595.71.05, CUDA toolkit 12.8, Ubuntu, Python 3.10 via uv venv

## What's installed where

| Thing | Location | Notes |
|---|---|---|
| HaMeR repo | `hamer/` | cloned with ViTPose submodule |
| WiLoR repo | `WiLoR/` | cloned, not yet set up |
| Python env | `.venv-hamer/` | uv venv, Python 3.10 |
| HaMeR checkpoints + ViTPose ckpts | `hamer/_DATA/` | from 5.6 GB tarball |
| MANO hand models | `mano_v1_2/models/` and `hamer/_DATA/data/mano/` | `MANO_LEFT.pkl`, `MANO_RIGHT.pkl` |
| ViTDet detector ckpt (demo, auto-fetched) | `~/.torch/iopath_cache/detectron2/ViTDet/...` | ~2.4 GB, see gotcha 6 |

Env stack: torch 2.8.0+cu128, torchvision 0.23.0, detectron2 (compiled from git),
mmpose 0.24.0 (ViTPose), chumpy, smplx 0.1.28, numpy<2.

## Downloads and their sources

| File | Size | Source |
|---|---|---|
| `hamer_demo_data.tar.gz` | 5.6 GB | HF mirror [AlenZeng/hamer_demo_data.tar.gz](https://huggingface.co/AlenZeng/hamer_demo_data.tar.gz) (official UT-Austin server throttled to ~20 KB/s; Google Drive blocked gdown) |
| **`mano_v1_2.zip`** | 167 MB | **Register + download at <https://mano.is.tue.mpg.de>** → "Models & Code". License forbids redistribution — never commit; `.gitignore` blocks `*.pkl`/`*.zip`/`mano_v1_2/`. Only `MANO_LEFT/RIGHT.pkl` needed; tail of our download (SMPLH body files) is corrupt but unused. |
| ViTDet `model_final_f05665.pkl` | 2.4 GB | [dl.fbaipublicfiles.com](https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl), auto-fetched by `demo.py` on first run |

## Gotchas hit (and fixes) — read before redoing any of this

1. **Blackwell GPU**: README's cu117 torch won't work on sm_120 — must use
   `--index-url https://download.pytorch.org/whl/cu128`, torch ≥ 2.7 (pinned 2.8.0;
   uv's resolver chokes on torch 2.11's `cuda-toolkit` meta-dep).
2. **`uv venv` ships no pip**, and piping installs to `tail` masked the failures.
   Use `uv pip install --python .venv-hamer/bin/python ...`.
3. **chumpy needs pip importable at build time** (`--no-build-isolation`): seed with
   `uv pip install ... pip` first.
4. **Slow single-stream downloads everywhere**: HF and fbaipublicfiles throttle
   per-connection (~0.1–0.6 MB/s); the line itself does ~6 MB/s. Fix: `aria2c -x16 -s16 -c`
   (~7 MB/s). MPI's MANO server throttles server-side — parallelism doesn't help there.
5. **A stale NVIDIA PyIndex pip config** (`/etc/pip.conf` → `pypi.ngc.nvidia.com`)
   causes resolver timeouts; `UV_HTTP_TIMEOUT=600` works around it.
6. **First `demo.py` run looks hung** — it's silently downloading the 2.4 GB ViTDet
   checkpoint (gotcha 4 applies). Pre-fetch it with aria2c into
   `~/.torch/iopath_cache/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/`.
7. **numpy<2 pinned** — chumpy/mmcv 1.3.9 predate numpy 2.

## Running the demo

```bash
cd hamer && PYOPENGL_PLATFORM=egl ../.venv-hamer/bin/python demo.py \
    --img_folder example_data --out_folder demo_out \
    --batch_size=64 --side_view --save_mesh --full_frame
```

- `PYOPENGL_PLATFORM=egl` = headless GPU rendering for pyrender.
- `--batch_size` only batches hand crops; with few images the runtime is dominated
  by model loading + the one-time detector download, not GPU memory.

## Next steps (per `monocular_capture_pipeline_plan.md`)

1. HaMeR demo smoke test (in progress)
2. WiLoR setup in same env + its checkpoints; same MANO file
3. Head-to-head on real tabletop footage under contact occlusion → lock Stage-1 model
4. Stage 2+: intrinsics, metric depth anchor, table plane, Rerun panel
