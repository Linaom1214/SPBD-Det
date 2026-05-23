#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:?Usage: scripts/run_ablation.sh CONFIG [DATA_ROOT] [SPLIT]}
DATA_ROOT=${2:-}
SPLIT=${3:-test}
PYTHON_CMD=${PYTHON_CMD:-${PYTHON:-python}}
TORCHRUN_CMD=${TORCHRUN_CMD:-${TORCHRUN:-torchrun}}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
SEED=${SEED:-42}
read -r -a PYTHON_ARR <<< "${PYTHON_CMD}"
read -r -a TORCHRUN_ARR <<< "${TORCHRUN_CMD}"
BASE=$(basename "${CONFIG}" .yaml)

run_one() {
  local NAME=$1
  shift
  local CFG_OPTIONS=("seed=${SEED}" "experiment_name=${NAME}" "$@")
  if [[ -n "${DATA_ROOT}" ]]; then
    CFG_OPTIONS+=("data.root=${DATA_ROOT}")
  fi

  if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
    "${TORCHRUN_ARR[@]}" --nproc_per_node="${NPROC_PER_NODE}" train.py --config "${CONFIG}" --work-dir "work_dirs/${NAME}" --cfg-options "${CFG_OPTIONS[@]}"
  else
    "${PYTHON_ARR[@]}" train.py --config "${CONFIG}" --work-dir "work_dirs/${NAME}" --cfg-options "${CFG_OPTIONS[@]}"
  fi

  "${PYTHON_ARR[@]}" test.py --config "${CONFIG}" --checkpoint "work_dirs/${NAME}/best.pth" --split "${SPLIT}" --work-dir "work_dirs/${NAME}/eval_${SPLIT}" --cfg-options "${CFG_OPTIONS[@]}"
}

run_one "${BASE}_full_seed${SEED}"
run_one "${BASE}_wo_aedm_seed${SEED}" "model.sobel=false"
run_one "${BASE}_wo_token_loss_seed${SEED}" "train.loss.token_contrast_weight=0.0"
run_one "${BASE}_wo_spd_seed${SEED}" "model.without_spd=true"
run_one "${BASE}_wo_rep_seed${SEED}" "model.without_rep=true"

"${PYTHON_ARR[@]}" tools/summarize_ablation.py work_dirs/${BASE}_*_seed${SEED}/eval_${SPLIT}/metrics.json
