#!/usr/bin/env bash
set -euo pipefail

cd "${KPLANES_ROOT:-/kaggle/working/BestNeRF/k-planes}"

export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/cuda-12.8/compat:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/cuda-12.8/compat:${LIBRARY_PATH:-}"
export TCNN_CUDA_ARCHITECTURES="${TCNN_CUDA_ARCHITECTURES:-75}"

CONFIG_PATH="plenoxels/configs/local/dynerf_cm_semantic_smoke_ablation.py"

read -r -d '' ABLATIONS <<'EOF' || true
smooth005|full_weighted|24|0.05
smooth010|full_weighted|24|0.10
smooth020|full_weighted|24|0.20
topk1|topk_weighted|1|0.00
topk8|topk_weighted|8|0.00
topk16|topk_weighted|16|0.00
topk24|topk_weighted|24|0.00
topk32|topk_weighted|32|0.00
topk48|topk_weighted|48|0.00
topk8_smooth010|topk_weighted|8|0.10
topk16_smooth010|topk_weighted|16|0.10
topk24_smooth010|topk_weighted|24|0.10
topk32_smooth010|topk_weighted|32|0.10
topk48_smooth010|topk_weighted|48|0.10
topk24_smooth005|topk_weighted|24|0.05
topk24_smooth020|topk_weighted|24|0.20
EOF

run_one() {
  local name="$1"
  local render_mode="$2"
  local topk="$3"
  local smooth="$4"

  echo "==== smoke ${name}: render=${render_mode} topk=${topk} smooth=${smooth} ===="
  SEM_ABLATION_NAME="$name" \
  SEM_RENDER_MODE="$render_mode" \
  SEM_TOPK="$topk" \
  SEM_SMOOTH_L1_WEIGHT="$smooth" \
  PYTHONPATH=. python plenoxels/main.py --config-path "$CONFIG_PATH"
}

while IFS='|' read -r name render_mode topk smooth; do
  [[ -z "${name}" ]] && continue
  run_one "$name" "$render_mode" "$topk" "$smooth"
done <<< "$ABLATIONS"

echo "All semantic ablation smoke runs finished."
