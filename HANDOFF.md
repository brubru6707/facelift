# FaceLift Optimization Handoff

This document brings you fully up to speed on the FaceLift 3D reconstruction pipeline, what optimizations have already been done, and what to tackle next.

---

## What This Project Does

FaceLift takes a single face photo and produces a 3D Gaussian Splatting reconstruction (`.ply` file). The pipeline runs on a GPU node at Brown's Oscar HPC cluster and is exposed publicly via ngrok.

**Pipeline stages (in order):**
1. **Preprocessing** — background removal (`rembg`) + face crop/align (`MTCNN`)
2. **Multi-view diffusion** — generates 6 synthetic views of the face (70 diffusion steps)
3. **GSLRM reconstruction** — lifts the 6 views into 3D Gaussians
4. **Filter + save** — prunes low-quality Gaussians, writes `gaussians.ply`
5. **Turntable render** (`main` branch only) — renders 150-view video + composite PNG

---

## Repo Layout

```
facelift/
├── ngrok.py                        # Flask server + ngrok tunnel (entry point)
├── inference.py                    # Core pipeline: process_single_image()
├── utils_folder/
│   └── face_utils.py               # rembg + MTCNN preprocessing
├── mvdiffusion/                    # Multi-view diffusion model
├── gslrm/                          # Gaussian splatting LRM model
├── configs/gslrm.yaml              # GSLRM config
├── checkpoints/                    # Model weights (not relevant to edit)
├── diff-gaussian-rasterization/    # Git submodule — CUDA rasterizer
├── outputs/<job_id>/               # Per-job outputs (not committed)
├── job_statuses/<job_id>.json      # Per-job status files (not committed)
├── RUNNING.md                      # How to start the server
└── OPTIMIZATION_ANALYSIS.md       # Full timing data and analysis
```

**The three files you'll edit most:**
- [ngrok.py](ngrok.py) — server, job dispatch, step_2D (diffusion steps) hardcoded here
- [inference.py](inference.py) — pipeline logic, `split_data` flag, gc cleanup
- [utils_folder/face_utils.py](utils_folder/face_utils.py) — rembg model + provider config

---

## Environment

- **Cluster:** Brown Oscar HPC
- **GPU:** A100 (44 GiB VRAM), ~9.35 GiB used during turntable render
- **Python:** 3.9.21
- **PyTorch:** 2.8.0+cu126
- **xformers:** 0.0.32.post2 (was 0.0.27 — had to upgrade, see Gotchas)
- **onnxruntime-gpu:** 1.19.2 (replaces CPU-only onnxruntime — critical for rembg speed)

**To start the server:**
```bash
cd /oscar/home/brrodrig/facelift
source .venv/bin/activate
export CUDA_HOME=/oscar/rt/9.6/25/spack/x86_64_v3/cuda-12.9.0-cinrl2oeqemd3szbcakkugp2vtk2fh5t
export PATH=$CUDA_HOME/bin:$PATH
module load ffmpeg
python ngrok.py
```

---

## Branch Structure

| Branch | Description |
|---|---|
| `main` | Original pipeline + timing logs. Includes turntable video render. Uses `u2net` rembg model. `split_data=True`. |
| `optimize-rembg-gpu` | Optimized branch. `u2netp` model + GPU provider for rembg. No turntable. `split_data=False`. |

Both branches have **identical timing instrumentation** (`[TIMING]` logs) so results are directly comparable.

---

## What Was Done (Completed Optimizations)

### 1. Added granular timing logs
Every stage is now individually timed. Logs look like:
```
[TIMING]   rembg background removal: 1.38s
[TIMING]   MTCNN face detection: 0.25s
[TIMING]   crop/resize/paste: 0.00s
[TIMING] Preprocessing: 1.64s
[TIMING] Multi-view diffusion (70 steps): 7.40s
[TIMING] GSLRM reconstruction: 0.43s
[TIMING] apply_all_filters: 0.01s
[TIMING] save_ply: 1.03s
[TIMING] Total: 10.77s
[TIMING] Job a2654914: total wall time (POST→done): 10.97s
```

### 2. GPU-accelerated rembg (`optimize-rembg-gpu` branch)
**File:** [utils_folder/face_utils.py:44](utils_folder/face_utils.py#L44)

Changed from:
```python
REMBG_SESSION = new_session()  # u2net, CPU
```
To:
```python
REMBG_SESSION = new_session("u2netp", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
```

This required installing `onnxruntime-gpu` (replaces CPU-only `onnxruntime`):
```bash
pip uninstall onnxruntime -y && pip install onnxruntime-gpu
```

### 3. Fixed xformers crash
`xformers==0.0.27.post2` was incompatible with PyTorch 2.8 and crashed on import. Upgraded:
```bash
pip install -U xformers --index-url https://download.pytorch.org/whl/cu126
```

---

## Timing Results (All Runs, Same Image)

| Run | rembg | Diffusion | Total* | Notes |
|---|---|---|---|---|
| Original baseline | 13.98s | 7.41s | **23.88s** | CPU, u2net, no persistent session |
| + u2netp on CPU | 6.62s | 7.40s | **16.52s** | Lighter model, still CPU onnxruntime |
| + onnxruntime-gpu (branch) | 1.38s | 7.40s | **10.77s** | u2netp + CUDA |
| `main` with GPU | 2.44s | 7.30s | **13.47s** | u2net + CUDA, includes turntable (1.19s) |

\* `main` total includes turntable render (1.19s). Branch does not generate turntable.  
Fair comparison (no turntable): **main ~12.28s vs branch 10.77s**

---

## Current Bottleneck (`optimize-rembg-gpu` branch)

| Stage | Time | % of Total |
|---|---|---|
| Multi-view diffusion (70 steps) | 7.40s | **69%** |
| save_ply | 1.03s | 10% |
| Preprocessing | 1.64s | 15% |
| GSLRM reconstruction | 0.43s | 4% |

**Diffusion is now the dominant cost.** Everything else is fast.

---

## Next Optimization Targets

### Priority 1 — Reduce diffusion steps (highest leverage, ~3s savings)

**File:** [ngrok.py:134](ngrok.py#L134)  
`step_2D=70` is hardcoded. Change to 40:

```python
step_2D=40,  # was 70 — saves ~3s, needs quality check
```

At 10.23 it/s, step counts map to approximate times:
| Steps | Time | Savings vs 70 |
|---|---|---|
| 70 | 7.40s | baseline |
| 50 | 5.30s | −2.1s |
| 40 | 4.20s | −3.2s |
| 30 | 3.10s | −4.3s |

**Recommended:** Test 40 and 50 side-by-side for quality, pick the lowest that still looks good. This is the single biggest remaining win — it could bring total time from **10.77s → ~7.5s**.

### Priority 2 — Async `save_ply` (no quality trade-off, ~1s savings)

`save_ply` takes ~1s and blocks the pipeline even though the result is already computed. It could be written to disk in a background thread while the status is updated.

**Caution:** The `/download` endpoint reads from the same path — you'd need to ensure the file is fully written before marking status as `success`.

### Priority 3 — Evaluate `split_data` flag

`main` uses `split_data=True`, the branch uses `split_data=False`.  
- `split_data=False` skips an internal rendering pass → faster GSLRM (0.43s vs 0.71s)  
- Check whether this affects output quality vs `split_data=True`

---

## Gotchas / Things That Will Bite You

1. **`outputs/` and `job_statuses/` have no `.gitignore`** — every test run adds files to the working tree and they get swept into commits. Add a `.gitignore` before doing more commits.

2. **`onnxruntime-gpu` must be installed, not `onnxruntime`** — the CPU package silently falls back even if you pass `CUDAExecutionProvider`. Verify with:
   ```python
   import onnxruntime as ort
   print(ort.get_available_providers())  # must include 'CUDAExecutionProvider'
   ```

3. **xformers version must match PyTorch** — if you reinstall PyTorch, xformers will likely break again. Always re-run:
   ```bash
   pip install -U xformers --index-url https://download.pytorch.org/whl/cu126
   ```

4. **ngrok URL changes every session** — the public URL is regenerated each time `ngrok.py` starts. The auth token is hardcoded in `ngrok.py:26`.

5. **`diff-gaussian-rasterization` is a git submodule** pointing to `https://github.com/graphdeco-inria/diff-gaussian-rasterization.git`. Clone with `--recurse-submodules` or run `git submodule update --init` after cloning.

6. **The server uses a shared `_generator`** — concurrent requests will share random state. Fine for now (requests queue via threading), but worth noting if concurrency is ever added.

7. **`main` branch runs `split_data=True` which triggers an internal rendering pass** — this is why GSLRM takes 0.71s on main vs 0.43s on the branch. The branch avoids a 130+ TiB BinningState allocation by skipping this pass.
