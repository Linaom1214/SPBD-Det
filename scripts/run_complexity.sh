#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:?Usage: scripts/run_complexity.sh CONFIG [OUT_DIR]}
OUT_DIR=${2:-work_dirs/complexity}
PYTHON_CMD=${PYTHON_CMD:-${PYTHON:-python}}
read -r -a PYTHON_ARR <<< "${PYTHON_CMD}"
mkdir -p "${OUT_DIR}"

run_one() {
  local NAME=$1
  local DEPLOY_FLAG=$2
  shift 2
  local EXTRA=("$@")
  local ARGS=(tools/model_complexity.py --config "${CONFIG}" --out "${OUT_DIR}/${NAME}.json" "${EXTRA[@]}")
  if [[ "${DEPLOY_FLAG}" == "deploy" ]]; then
    ARGS+=(--deploy)
  fi
  "${PYTHON_ARR[@]}" "${ARGS[@]}"
}

run_one full train
run_one full_deploy deploy
run_one wo_aedm train --cfg-options model.sobel=false
run_one wo_token_loss train --cfg-options train.loss.token_contrast_weight=0.0
run_one wo_spd train --cfg-options model.without_spd=true
run_one wo_rep train --cfg-options model.without_rep=true

"${PYTHON_ARR[@]}" tools/summarize_complexity.py "${OUT_DIR}"/*.json | tee "${OUT_DIR}/summary.md"
