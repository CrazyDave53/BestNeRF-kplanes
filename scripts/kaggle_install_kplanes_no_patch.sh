#!/usr/bin/env bash
set -euo pipefail

TCNN_TORCH_DIR="${TCNN_TORCH_DIR:-/kaggle/working/tiny-cuda-nn/bindings/torch}"
TCNN_WHEEL_GLOB="${TCNN_WHEEL_GLOB:-/kaggle/working/wheels/tinycudann*.whl}"

python - <<'PY'
import sys
print("Python:", sys.version)
PY

# Keep Kaggle's CUDA-enabled torch installation intact. Do not install the
# upstream requirements.txt here, because it can replace torch with a bad wheel.
python -m pip install --upgrade pip setuptools wheel packaging ninja cmake

python -m pip install \
  tqdm \
  pillow \
  opencv-python \
  pandas \
  lpips \
  "imageio[pyav]" \
  torchmetrics \
  scikit-image \
  configargparse \
  tensorboard

export TCNN_CUDA_ARCHITECTURES="${TCNN_CUDA_ARCHITECTURES:-75}"
export LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/cuda-12.8/compat:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/cuda-12.8/compat:${LD_LIBRARY_PATH:-}"

if python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
import tinycudann
print("tinycudann already works; skipping rebuild.")
PY
then
  :
else
  shopt -s nullglob
  tcnn_wheels=( ${TCNN_WHEEL_GLOB} )
  shopt -u nullglob

  if (( ${#tcnn_wheels[@]} > 0 )); then
    echo "Installing tiny-cuda-nn wheel: ${tcnn_wheels[0]}"
    python -m pip install "${tcnn_wheels[0]}"
  else
    if [[ ! -f "${TCNN_TORCH_DIR}/setup.py" ]]; then
      echo "tiny-cuda-nn torch bindings not found at: ${TCNN_TORCH_DIR}" >&2
      echo "Clone and patch tiny-cuda-nn first, then rerun this installer." >&2
      exit 1
    fi

    echo "Installing tiny-cuda-nn from ${TCNN_TORCH_DIR}"
    python -m pip install -v --no-build-isolation "${TCNN_TORCH_DIR}"
  fi
fi

python - <<'PY'
import importlib
import torch

for name in ["torch", "tinycudann", "cv2", "imageio", "av", "lpips", "pandas", "torchmetrics", "skimage"]:
    mod = importlib.import_module(name)
    print(name, getattr(mod, "__version__", "unknown"))

print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("K-Planes Kaggle dependencies installed.")
PY
