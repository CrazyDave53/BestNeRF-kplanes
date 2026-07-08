# Kaggle Runbook

This is the working checklist for running the BestNeRF/K-Planes Kaggle setup, RGB baseline, and OpenSeg cache extraction.

## Expected Repo State

Kaggle should use our fork and branch, not upstream K-Planes:

```bash
cd /kaggle/working/BestNeRF/k-planes
git remote -v
git branch --show-current
```

Expected:

```text
origin https://github.com/CrazyDave53/BestNeRF-kplanes.git
bestnerf-kaggle-baseline
```

If Kaggle has the upstream repo instead:

```bash
cd /kaggle/working/BestNeRF
mv k-planes k-planes_upstream_old
git clone -b bestnerf-kaggle-baseline https://github.com/CrazyDave53/BestNeRF-kplanes.git k-planes
```

## Startup After Instance Restart

```bash
cd /kaggle/working/BestNeRF/k-planes
git pull
bash scripts/kaggle_setup_runtime.sh
```

The setup script installs normal Python deps, sets CUDA library paths inside the script, installs/reuses the `tinycudann` wheel, and verifies imports.

If `tinycudann` is installed but broken, force reinstall from the wheel cache:

```bash
FORCE_REINSTALL_TCNN=1 bash scripts/kaggle_setup_runtime.sh
```

For the current interactive shell, export these before running training/extraction:

```bash
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda-12.8/compat:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda-12.8/compat:$LIBRARY_PATH
export TCNN_CUDA_ARCHITECTURES=75
```

Quick check:

```bash
python - <<'PY'
import torch, tinycudann, tensorflow as tf
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.device_count())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("tensorflow", tf.__version__)
print("tinycudann ok")
PY
```

`nvidia-smi` may be missing on Kaggle. That is okay if `torch.cuda.is_available()` is `True`.

## tiny-cuda-nn Wheel Cache

Keep the compiled `tinycudann` wheel in `/kaggle/working/wheels` so a restarted instance can reinstall quickly.

Build the wheel if it is missing:

```bash
export TCNN_CUDA_ARCHITECTURES=75
export LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda-12.8/compat:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda-12.8/compat:$LD_LIBRARY_PATH

mkdir -p /kaggle/working/wheels
python -m pip wheel -v --no-build-isolation --no-deps \
  -w /kaggle/working/wheels \
  /kaggle/working/tiny-cuda-nn/bindings/torch
```

Install or reinstall from the newest cached wheel:

```bash
cd /kaggle/working/BestNeRF/k-planes
bash scripts/kaggle_install_tcnn_from_wheel.sh
```

Install a specific wheel:

```bash
TCNN_WHEEL="$(ls -t /kaggle/working/wheels/tinycudann*.whl | head -1)" \
  bash scripts/kaggle_install_tcnn_from_wheel.sh
```

## Coffee Martini Data Symlink

After recloning K-Planes, the data symlink may point at the wrong location.

Find the real videos:

```bash
find /kaggle/working -type f -name 'cam00.mp4' 2>/dev/null | head -20
find /kaggle/input -type f -name 'cam00.mp4' 2>/dev/null | head -20
```

If the data is in the old repo backup:

```bash
cd /kaggle/working/BestNeRF/k-planes

REAL_DATA=/kaggle/working/BestNeRF/k-planes_upstream_old/data/neu3d/coffee_martini

mkdir -p data/neu3d
rm -rf data/neu3d/coffee_martini
ln -s "$REAL_DATA" data/neu3d/coffee_martini

find -L data/neu3d/coffee_martini -maxdepth 1 -type f -name 'cam*.mp4' | head
```

Use `find -L` because `data/neu3d/coffee_martini` is usually a symlink.

## OpenSeg Model

After a restart, the OpenSeg SavedModel may be missing:

```bash
cd /kaggle/working/BestNeRF/opennerf

mkdir -p models
cd models
wget -c https://geometry.stanford.edu/projects/openseg/openseg_exported_clip.zip
unzip -q openseg_exported_clip.zip
```

Verify:

```bash
find /kaggle/working/BestNeRF/opennerf/models/openseg_exported_clip -maxdepth 2 -name 'saved_model.pb*'
```

## Resolution Choices

Coffee Martini raw videos are `2028 x 2704`.

Our intended first semantic setup:

```text
K-Planes data_downsample = 4:     507 x 676 RGB
OpenSeg downsample from RGB = 4:  126 x 169 features
Raw mp4 feature_downsample = 16
```

Useful size estimates:

```text
ds8, 300 frames, all cams:  ~660 GiB, too big for one Kaggle dataset
ds8, 64 frames, all cams:   ~141 GiB
ds16, 300 frames, 16 cams:  ~146 GiB
ds16, 64 frames, all cams:  ~35 GiB
ds32, 300 frames, all cams: ~40 GiB
```

First recommended cache: `feature_downsample=16`, `max_frames=64`.

Expected per-camera feature shape:

```text
[64, 126, 169, 768] float16
```

## Two-GPU OpenSeg Cache Extraction

Always run split creation from the K-Planes repo root:

```bash
cd /kaggle/working/BestNeRF/k-planes

rm -rf /kaggle/temp/openseg_split
mkdir -p /kaggle/temp/openseg_split/gpu0
mkdir -p /kaggle/temp/openseg_split/gpu1

python - <<'PY'
from pathlib import Path

src = Path("data/neu3d/coffee_martini").resolve()
print("src:", src)
print("exists:", src.exists())

gpu0 = Path("/kaggle/temp/openseg_split/gpu0")
gpu1 = Path("/kaggle/temp/openseg_split/gpu1")

cams = sorted(src.glob("cam*.mp4"))
for i, cam in enumerate(cams):
    dst_dir = gpu0 if i % 2 == 0 else gpu1
    dst = dst_dir / cam.name
    if dst.exists():
        dst.unlink()
    dst.symlink_to(cam)

print("gpu0:", [p.name for p in sorted(gpu0.glob("cam*.mp4"))])
print("gpu1:", [p.name for p in sorted(gpu1.glob("cam*.mp4"))])
PY
```

Launch both GPUs for ds16/64f:

```bash
cd /kaggle/working/BestNeRF/k-planes

export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda-12.8/compat:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda-12.8/compat:$LIBRARY_PATH
export TCNN_CUDA_ARCHITECTURES=75

rm -rf /kaggle/temp/openseg_cache_ds16_64f
rm -rf /kaggle/temp/openseg_logs
mkdir -p /kaggle/temp/openseg_cache_ds16_64f/neu3d/coffee_martini/openseg_camckpts
mkdir -p /kaggle/temp/openseg_logs

CUDA_VISIBLE_DEVICES=0 python scripts/extract_openseg_neu3d.py \
  --data-dir /kaggle/temp/openseg_split/gpu0 \
  --opennerf-root ../opennerf \
  --output-dir /kaggle/temp/openseg_cache_ds16_64f/neu3d/coffee_martini/openseg_camckpts \
  --max-frames 64 \
  --feature-downsample 16 \
  > /kaggle/temp/openseg_logs/gpu0.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python scripts/extract_openseg_neu3d.py \
  --data-dir /kaggle/temp/openseg_split/gpu1 \
  --opennerf-root ../opennerf \
  --output-dir /kaggle/temp/openseg_cache_ds16_64f/neu3d/coffee_martini/openseg_camckpts \
  --max-frames 64 \
  --feature-downsample 16 \
  > /kaggle/temp/openseg_logs/gpu1.log 2>&1 &

wait
```

Monitor logs in separate terminals:

```bash
tail -f /kaggle/temp/openseg_logs/gpu0.log
tail -f /kaggle/temp/openseg_logs/gpu1.log
```

Monitor output size:

```bash
du -sh /kaggle/temp/openseg_cache_ds16_64f/neu3d/coffee_martini
ls -lh /kaggle/temp/openseg_cache_ds16_64f/neu3d/coffee_martini/openseg_camckpts | tail
```

The extractor writes a `camXX.tmp.npy` memmap first, then renames it to `camXX.npy` when that camera finishes.

## Package Cache As Kaggle Dataset

Kaggle dataset limit is about `200 GB` per dataset, so ds16/64f should fit comfortably.

```bash
DATASET_DIR=/kaggle/temp/openseg_cache_ds16_64f/neu3d/coffee_martini
DATASET_ID=minhchau3c/coffee-martini-openseg-ds16-64f

cd "$DATASET_DIR"
kaggle datasets init -p "$DATASET_DIR"

cat > "$DATASET_DIR/dataset-metadata.json" <<'JSON'
{
  "title": "Coffee Martini OpenSeg Features DS16 64F",
  "id": "minhchau3c/coffee-martini-openseg-ds16-64f",
  "licenses": [
    {
      "name": "CC0-1.0"
    }
  ]
}
JSON

kaggle datasets create -p "$DATASET_DIR" -t -r tar
```

Check status:

```bash
kaggle datasets status "$DATASET_ID"
```

Create a new version later:

```bash
kaggle datasets version -p "$DATASET_DIR" -m "Update OpenSeg ds16 64f cache" -t -r tar
```

## Common Failures

### `SavedModel file does not exist`

OpenSeg model is missing. Re-download `openseg_exported_clip.zip` under `/kaggle/working/BestNeRF/opennerf/models`.

### GPU split prints empty lists

You ran from the wrong directory, or the data symlink is broken. Run the split command from:

```bash
/kaggle/working/BestNeRF/k-planes
```

Then check:

```bash
find -L data/neu3d/coffee_martini -maxdepth 1 -type f -name 'cam*.mp4' | head
```

### `Killed` during ds8 extraction

Likely disk or RAM pressure. ds8/300f all-cams is too large. Use ds16/64f first.

### VS Code SSH host key changed

Ngrok changed the tunnel host key:

```powershell
ssh-keygen -R "[0.tcp.ap.ngrok.io]:PORT"
```

Then reconnect and accept the new fingerprint.
