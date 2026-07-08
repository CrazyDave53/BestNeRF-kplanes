#!/usr/bin/env bash
set -euo pipefail

TCNN_WHEEL_DIR="${TCNN_WHEEL_DIR:-/kaggle/working/wheels}"
TCNN_WHEEL="${TCNN_WHEEL:-}"

export TCNN_CUDA_ARCHITECTURES="${TCNN_CUDA_ARCHITECTURES:-75}"
export LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/cuda-12.8/compat:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/cuda-12.8/compat:${LD_LIBRARY_PATH:-}"

if [[ -z "${TCNN_WHEEL}" ]]; then
  TCNN_WHEEL="$(find "${TCNN_WHEEL_DIR}" -maxdepth 1 -type f -name 'tinycudann*.whl' -printf '%T@ %p\n' | sort -nr | awk 'NR==1 {print $2}')"
fi

if [[ -z "${TCNN_WHEEL}" || ! -f "${TCNN_WHEEL}" ]]; then
  echo "No tinycudann wheel found." >&2
  echo "Looked in: ${TCNN_WHEEL_DIR}" >&2
  echo "Or set TCNN_WHEEL=/path/to/tinycudann.whl" >&2
  exit 1
fi

echo "Installing tiny-cuda-nn wheel: ${TCNN_WHEEL}"
python -m pip install --no-deps --force-reinstall "${TCNN_WHEEL}"

python - <<'PY'
import torch
import tinycudann

print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("tinycudann ok")
PY
