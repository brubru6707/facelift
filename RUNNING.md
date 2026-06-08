# Running FaceLift on Oscar

FaceLift takes a single face photo and generates a 3D Gaussian Splatting reconstruction (`.ply` file). The endpoint is exposed publicly via ngrok.

---

## First-Time Setup

These steps only need to be done once.

### 1. Request a GPU node

```bash
interact -n 4 -t 10:00:00 -m 20g -g 1 -q gpu
```

Wait until your prompt changes to a GPU node (e.g. `gpu3004`).

### 2. Navigate to the repo and activate the virtual environment

```bash
cd /oscar/home/brrodrig/facelift
source .venv/bin/activate
```

### 3. Set CUDA environment variables

```bash
export CUDA_HOME=/oscar/rt/9.6/25/spack/x86_64_v3/cuda-12.9.0-cinrl2oeqemd3szbcakkugp2vtk2fh5t
export PATH=$CUDA_HOME/bin:$PATH
```

### 4. Install dependencies

```bash
pip install --upgrade pip

# Core ML stack
pip install packaging==24.2 typing-extensions==4.14.0
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
pip install transformers==4.44.2 diffusers[torch]==0.30.3 huggingface-hub==0.35.3 xformers==0.0.27.post2 accelerate==0.33.0
pip install Pillow==10.4.0 opencv-python==4.10.0.84 scikit-image==0.21.0 lpips==0.1.4
pip install facenet-pytorch --no-deps
pip install rembg onnxruntime
pip install numpy==1.26.4 matplotlib==3.7.5 scikit-learn==1.3.2 einops==0.8.0 jaxtyping==0.2.19 pytorch-msssim==1.0.0 rich
pip install easydict==1.13 pyyaml==6.0.2 wandb==0.19.1 termcolor==2.4.0 plyfile==1.0.3 tqdm
pip install videoio==0.3.0 ffmpeg-python==0.2.0
pip install flask pyngrok
pip install wheel ninja

# diff-gaussian-rasterization (requires CUDA to be set first — see step 3)
pip install --no-build-isolation /oscar/home/brrodrig/facelift/diff-gaussian-rasterization
```

**Expected warnings you can safely ignore:**
- `facenet-pytorch` version conflicts with torch/Pillow — harmless, it still works
- `gradio==5.49.1` not found — gradio is not used by the endpoint
- `onnxruntime` pthread affinity errors at runtime — harmless

---

## Every Time You Want to Run

### 1. Request a GPU node

```bash
interact -n 4 -t 10:00:00 -m 20g -g 1 -q gpu
```

### 2. Set up the environment

```bash
cd /oscar/home/brrodrig/facelift
source .venv/bin/activate

export CUDA_HOME=/oscar/rt/9.6/25/spack/x86_64_v3/cuda-12.9.0-cinrl2oeqemd3szbcakkugp2vtk2fh5t
export PATH=$CUDA_HOME/bin:$PATH

module load ffmpeg
```

### 3. Start the server

```bash
python ngrok.py
```

On the **first ever run**, model weights are downloaded automatically from HuggingFace (`wlyu/OpenFaceLift`) — this takes a few minutes. Subsequent runs start faster.

When ready, you will see:

```
ngrok tunnel URL: https://xxxx.ngrok-free.app
POST image to /process_image, then poll /status/<job_id> for results.
```

The URL changes every session — share the new URL each time.

---

## Using the Endpoint

### Submit an image

```bash
curl -X POST https://xxxx.ngrok-free.app/process_image \
  -F "image=@your_photo.jpg"
```

Response:
```json
{"status": "processing", "job_id": "abc-123-..."}
```

### Poll for completion

Processing takes a couple of minutes. Keep polling until status is no longer `processing`:

```bash
curl https://xxxx.ngrok-free.app/status/abc-123-...
```

Possible responses:
- `{"status": "processing"}` — still running
- `{"status": "success", ...}` — done, ready to download
- `{"status": "error", "message": "..."}` — something went wrong

### Download the result

```bash
curl -OJ https://xxxx.ngrok-free.app/download/abc-123-...
```

This downloads a `.ply` file (3D Gaussian Splatting) named after your original image.

---

## Troubleshooting

| Error | Fix |
|---|---|
| `nvcc not found` | Re-run the `export CUDA_HOME` and `export PATH` commands |
| `ffprobe not found` | Run `module load ffmpeg` |
| `ModuleNotFoundError: No module named 'onnxruntime'` | `pip install onnxruntime` |
| `error: invalid command 'bdist_wheel'` | `pip install wheel ninja` |
| `ERR_NGROK_334` | The previous ngrok session is still alive — wait ~30 seconds and try again |
| Job status returns `error` | Check the server terminal for the full traceback |
