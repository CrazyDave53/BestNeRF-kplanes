#!/usr/bin/env bash
set -euo pipefail

cd "${KPLANES_ROOT:-/kaggle/working/BestNeRF/k-planes}"

export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/cuda-12.8/compat:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/cuda-12.8/compat:${LIBRARY_PATH:-}"
export TCNN_CUDA_ARCHITECTURES="${TCNN_CUDA_ARCHITECTURES:-75}"

CONFIG_PATH="plenoxels/configs/local/dynerf_cm_semantic_64f_ds16_ablation.py"
LOG_ROOT="${LOG_ROOT:-logs/baseline}"
PACKAGE_ROOT="${PACKAGE_ROOT:-/kaggle/temp/semantic_ablation_packages}"
DATASET_OWNER="${KAGGLE_DATASET_OWNER:-}"
DATASET_PREFIX="${KAGGLE_DATASET_PREFIX:-cm-sem-ablation}"
SKIP_TRAINED="${SKIP_TRAINED:-1}"
UPLOAD_DATASETS="${UPLOAD_DATASETS:-1}"

mkdir -p "$PACKAGE_ROOT"

if [[ -z "$DATASET_OWNER" ]]; then
  DATASET_OWNER="$(python - <<'PY'
import json
import os
from pathlib import Path

path = Path.home() / ".kaggle" / "kaggle.json"
if path.is_file():
    print(json.loads(path.read_text())["username"])
elif os.environ.get("KAGGLE_USERNAME"):
    print(os.environ["KAGGLE_USERNAME"])
else:
    raise SystemExit("Set KAGGLE_DATASET_OWNER or configure Kaggle credentials first.")
PY
)"
fi

slugify() {
  echo "$1" | tr '[:upper:]_' '[:lower:]-' | tr -cd 'a-z0-9-'
}

write_metadata() {
  local package_dir="$1"
  local dataset_id="$2"
  local title="$3"

  cat > "${package_dir}/dataset-metadata.json" <<JSON
{
  "title": "${title}",
  "id": "${dataset_id}",
  "licenses": [
    {
      "name": "CC0-1.0"
    }
  ]
}
JSON
}

verify_checkpoint() {
  local expname="$1"
  local ckpt="${LOG_ROOT}/${expname}/model.pth"
  test -s "$ckpt"
  python - <<PY
from pathlib import Path
p = Path("${ckpt}")
print("${expname}", p, p.stat().st_size)
PY
}

upload_dataset() {
  local name="$1"
  local expname="$2"
  local log_dir="${LOG_ROOT}/${expname}"
  local slug
  local dataset_id
  local package_dir
  local archive
  local title

  slug="$(slugify "$name")"
  dataset_id="${DATASET_OWNER}/${DATASET_PREFIX}-${slug}"
  title="CM Sem Ablation ${slug}"
  package_dir="${PACKAGE_ROOT}/${expname}_dataset"
  archive="${package_dir}/${expname}_logs.tar.gz"

  rm -rf "$package_dir"
  mkdir -p "$package_dir"
  tar -czf "$archive" -C "$LOG_ROOT" "$expname"
  write_metadata "$package_dir" "$dataset_id" "$title"

  if kaggle datasets files "$dataset_id" >/dev/null 2>&1; then
    kaggle datasets version -p "$package_dir" -m "Update ${expname}" -r zip
  else
    kaggle datasets create -p "$package_dir" -r zip
  fi

  rm -rf "$package_dir"
}

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
  local expname="cm_semantic_64f_ds16_${name}"

  echo "==== train ${name}: render=${render_mode} topk=${topk} smooth=${smooth} ===="
  if [[ "$SKIP_TRAINED" == "1" && -s "${LOG_ROOT}/${expname}/model.pth" ]]; then
    echo "Checkpoint already exists for ${expname}; skipping training."
  else
    SEM_ABLATION_NAME="$name" \
    SEM_RENDER_MODE="$render_mode" \
    SEM_TOPK="$topk" \
    SEM_SMOOTH_L1_WEIGHT="$smooth" \
    PYTHONPATH=. python plenoxels/main.py --config-path "$CONFIG_PATH"
  fi

  verify_checkpoint "$expname"
  if [[ "$UPLOAD_DATASETS" == "1" ]]; then
    upload_dataset "$name" "$expname"
  else
    echo "UPLOAD_DATASETS=0; not uploading ${expname}."
  fi
}

while IFS='|' read -r name render_mode topk smooth; do
  [[ -z "${name}" ]] && continue
  run_one "$name" "$render_mode" "$topk" "$smooth"
done <<< "$ABLATIONS"

echo "All semantic ablation full runs finished."
