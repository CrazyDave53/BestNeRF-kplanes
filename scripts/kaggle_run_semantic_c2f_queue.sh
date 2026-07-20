#!/usr/bin/env bash
set -euo pipefail

cd "${KPLANES_ROOT:-/kaggle/working/BestNeRF/k-planes}"

export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/cuda-12.8/compat:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/cuda-12.8/compat:${LIBRARY_PATH:-}"
export TCNN_CUDA_ARCHITECTURES="${TCNN_CUDA_ARCHITECTURES:-75}"

COARSE_CONFIG="plenoxels/configs/local/dynerf_cm_semantic_64f_ds16_c2f_coarse.py"
FINE_CONFIG="plenoxels/configs/local/dynerf_cm_semantic_64f_ds16_c2f_fine.py"
LOG_ROOT="${LOG_ROOT:-logs/baseline}"
PACKAGE_ROOT="${PACKAGE_ROOT:-/kaggle/temp/semantic_c2f_packages}"
DATASET_OWNER="${KAGGLE_DATASET_OWNER:-}"
DATASET_PREFIX="${KAGGLE_DATASET_PREFIX:-cm-sem-c2f}"
SKIP_TRAINED="${SKIP_TRAINED:-1}"
UPLOAD_DATASETS="${UPLOAD_DATASETS:-1}"

mkdir -p "$PACKAGE_ROOT"
mkdir -p "$LOG_ROOT"

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
  title="CM Sem C2F ${slug}"
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
  local coarse_expname="cm_semantic_64f_ds16_c2f_${name}_coarse"
  local fine_expname="cm_semantic_64f_ds16_c2f_${name}"

  echo "==== c2f coarse ${name}: render=${render_mode} topk=${topk} smooth=${smooth} ===="
  if [[ "$SKIP_TRAINED" == "1" && -s "${LOG_ROOT}/${coarse_expname}/model.pth" ]]; then
    echo "Checkpoint already exists for ${coarse_expname}; skipping coarse training."
  else
    SEM_C2F_NAME="$name" \
    SEM_RENDER_MODE="$render_mode" \
    SEM_TOPK="$topk" \
    SEM_SMOOTH_L1_WEIGHT="$smooth" \
    SEM_C2F_COARSE_STEPS=3000 \
    LOG_ROOT="$LOG_ROOT" \
    PYTHONPATH=. python plenoxels/main.py --config-path "$COARSE_CONFIG"
  fi
  verify_checkpoint "$coarse_expname"

  echo "==== c2f fine ${name}: render=${render_mode} topk=${topk} smooth=${smooth} ===="
  if [[ "$SKIP_TRAINED" == "1" && -s "${LOG_ROOT}/${fine_expname}/model.pth" ]]; then
    echo "Checkpoint already exists for ${fine_expname}; skipping fine training."
  else
    SEM_C2F_NAME="$name" \
    SEM_RENDER_MODE="$render_mode" \
    SEM_TOPK="$topk" \
    SEM_SMOOTH_L1_WEIGHT="$smooth" \
    SEM_C2F_FINE_STEPS=10000 \
    LOG_ROOT="$LOG_ROOT" \
    PYTHONPATH=. python plenoxels/main.py \
      --config-path "$FINE_CONFIG" \
      --log-dir "${LOG_ROOT}/${coarse_expname}"
  fi
  verify_checkpoint "$fine_expname"

  if [[ "$UPLOAD_DATASETS" == "1" ]]; then
    upload_dataset "$name" "$fine_expname"
  else
    echo "UPLOAD_DATASETS=0; not uploading ${fine_expname}."
  fi
}

while IFS='|' read -r name render_mode topk smooth; do
  [[ -z "${name}" ]] && continue
  run_one "$name" "$render_mode" "$topk" "$smooth"
done <<< "$C2F_ABLATIONS"

echo "All semantic coarse-to-fine full runs finished."
