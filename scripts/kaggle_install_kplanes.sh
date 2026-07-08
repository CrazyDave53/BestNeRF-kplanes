#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import sys
print("Python:", sys.version)
PY

python -m pip install --upgrade pip setuptools wheel ninja

# Keep Kaggle's CUDA-enabled torch installation intact. Installing torch from
# requirements.txt can accidentally replace it with an incompatible wheel.
python -m pip install \
  tqdm \
  pillow \
  opencv-python \
  pandas \
  lpips \
  "imageio[pyav]" \
  torchmetrics \
  scikit-image \
  tensorboard

python -m pip install "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"

echo "K-Planes baseline dependencies installed."

