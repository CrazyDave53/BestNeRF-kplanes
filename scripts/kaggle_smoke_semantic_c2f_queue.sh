#!/usr/bin/env bash
set -euo pipefail

cd "${KPLANES_ROOT:-/kaggle/working/BestNeRF/k-planes}"

export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/cuda-12.8/compat:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/cuda-12.8/compat:${LIBRARY_PATH:-}"
export TCNN_CUDA_ARCHITECTURES="${TCNN_CUDA_ARCHITECTURES:-75}"

COARSE_CONFIG="plenoxels/configs/local/dynerf_cm_semantic_64f_ds16_c2f_coarse.py"
FINE_CONFIG="plenoxels/configs/local/dynerf_cm_semantic_64f_ds16_c2f_fine.py"
LOG_ROOT="${LOG_ROOT:-logs/baseline}"

read -r -d '' C2F_ABLATIONS <<'EOF' || true
topk8|topk_weighted|8|0.00
topk16|topk_weighted|16|0.00
topk24|topk_weighted|24|0.00
topk16_smooth010|topk_weighted|16|0.10
EOF

run_one() {
  local name="$1"
  local render_mode="$2"
  local topk="$3"
  local smooth="$4"
  local smoke_name="smoke_${name}"
  local coarse_expname="cm_semantic_64f_ds16_c2f_${smoke_name}_coarse"

  echo "==== smoke c2f ${name}: render=${render_mode} topk=${topk} smooth=${smooth} ===="
  SEM_C2F_NAME="$smoke_name" \
  SEM_RENDER_MODE="$render_mode" \
  SEM_TOPK="$topk" \
  SEM_SMOOTH_L1_WEIGHT="$smooth" \
  SEM_C2F_COARSE_STEPS=20 \
  PYTHONPATH=. python plenoxels/main.py --config-path "$COARSE_CONFIG"

  SEM_C2F_NAME="$smoke_name" \
  SEM_RENDER_MODE="$render_mode" \
  SEM_TOPK="$topk" \
  SEM_SMOOTH_L1_WEIGHT="$smooth" \
  SEM_C2F_FINE_STEPS=40 \
  PYTHONPATH=. python plenoxels/main.py \
    --config-path "$FINE_CONFIG" \
    --log-dir "${LOG_ROOT}/${coarse_expname}"
}

while IFS='|' read -r name render_mode topk smooth; do
  [[ -z "${name}" ]] && continue
  run_one "$name" "$render_mode" "$topk" "$smooth"
done <<< "$C2F_ABLATIONS"

echo "All semantic coarse-to-fine smoke runs finished."
