#!/usr/bin/env bash
set -euo pipefail

echo "== GPU =="
nvidia-smi || true

echo
echo "== Python imports =="
python - <<'PY'
import importlib
import sys

print("Python:", sys.version)

mods = [
    "torch",
    "torchvision",
    "tinycudann",
    "cv2",
    "imageio",
    "av",
    "lpips",
    "pandas",
    "torchmetrics",
    "skimage",
]

for name in mods:
    mod = importlib.import_module(name)
    version = getattr(mod, "__version__", "unknown")
    print(f"{name}: {version}")

import torch
print("torch.cuda.is_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device:", torch.cuda.get_device_name(0))
PY

echo
echo "Environment check complete."

