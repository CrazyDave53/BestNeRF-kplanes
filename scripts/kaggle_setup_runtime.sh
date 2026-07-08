#!/usr/bin/env bash
set -euo pipefail

TCNN_ROOT="${TCNN_ROOT:-/kaggle/working/tiny-cuda-nn}"
TCNN_TORCH_DIR="${TCNN_TORCH_DIR:-${TCNN_ROOT}/bindings/torch}"
TCNN_WHEEL_DIR="${TCNN_WHEEL_DIR:-/kaggle/working/wheels}"
FORCE_REINSTALL_TCNN="${FORCE_REINSTALL_TCNN:-0}"

export TCNN_CUDA_ARCHITECTURES="${TCNN_CUDA_ARCHITECTURES:-75}"
export LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/cuda-12.8/compat:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/cuda-12.8/compat:${LD_LIBRARY_PATH:-}"

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
  tensorboard \
  tensorflow

mkdir -p "${TCNN_WHEEL_DIR}"

tcnn_works() {
  python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
import tinycudann
print("tinycudann import ok")
PY
}

install_latest_tcnn_wheel() {
  local wheel
  wheel="$(find "${TCNN_WHEEL_DIR}" -maxdepth 1 -type f -name 'tinycudann*.whl' -printf '%T@ %p\n' | sort -nr | awk 'NR==1 {print $2}')"
  if [[ -z "${wheel}" ]]; then
    return 1
  fi
  echo "Installing tiny-cuda-nn wheel: ${wheel}"
  python -m pip install --no-deps --force-reinstall "${wheel}"
}

build_tcnn_wheel() {
  if [[ ! -f "${TCNN_TORCH_DIR}/setup.py" ]]; then
    echo "tiny-cuda-nn torch bindings not found at: ${TCNN_TORCH_DIR}" >&2
    echo "Clone and patch tiny-cuda-nn first, then rerun this setup script." >&2
    exit 1
  fi

  echo "Building tiny-cuda-nn wheel from ${TCNN_TORCH_DIR}"
  python -m pip wheel -v --no-build-isolation --no-deps -w "${TCNN_WHEEL_DIR}" "${TCNN_TORCH_DIR}"
}

if [[ "${FORCE_REINSTALL_TCNN}" != "1" ]] && tcnn_works; then
  echo "tinycudann already works; keeping current install."
else
  if ! install_latest_tcnn_wheel; then
    build_tcnn_wheel
    install_latest_tcnn_wheel
  fi
fi

python - <<'PY'
import importlib
import torch

for name in ["torch", "tinycudann", "cv2", "imageio", "av", "lpips", "pandas", "torchmetrics", "skimage", "tensorflow"]:
    mod = importlib.import_module(name)
    print(name, getattr(mod, "__version__", "unknown"))

print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("K-Planes Kaggle runtime setup complete.")
PY
