#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:?Usage: scripts/run_multiseed.sh CONFIG [DATA_ROOT] [SPLIT]}
DATA_ROOT=${2:-}
SPLIT=${3:-test}
PYTHON_CMD=${PYTHON_CMD:-${PYTHON:-python}}
TORCHRUN_CMD=${TORCHRUN_CMD:-${TORCHRUN:-torchrun}}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
read -r -a PYTHON_ARR <<< "${PYTHON_CMD}"
read -r -a TORCHRUN_ARR <<< "${TORCHRUN_CMD}"
read -r -a SEED_LIST <<< "${SEEDS:-42 3407 2026}"

for SEED in "${SEED_LIST[@]}"; do
  NAME=$(basename "${CONFIG}" .yaml)_seed${SEED}
  CFG_OPTIONS=("seed=${SEED}" "experiment_name=${NAME}")
  if [[ -n "${DATA_ROOT}" ]]; then
    CFG_OPTIONS+=("data.root=${DATA_ROOT}")
  fi

  if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
    "${TORCHRUN_ARR[@]}" --nproc_per_node="${NPROC_PER_NODE}" train.py --config "${CONFIG}" --work-dir "work_dirs/${NAME}" --cfg-options "${CFG_OPTIONS[@]}"
  else
    "${PYTHON_ARR[@]}" train.py --config "${CONFIG}" --work-dir "work_dirs/${NAME}" --cfg-options "${CFG_OPTIONS[@]}"
  fi

  "${PYTHON_ARR[@]}" test.py --config "${CONFIG}" --checkpoint "work_dirs/${NAME}/best.pth" --split "${SPLIT}" --work-dir "work_dirs/${NAME}/eval_${SPLIT}" --cfg-options "${CFG_OPTIONS[@]}"
done

"${PYTHON_ARR[@]}" tools/summarize_multiseed.py work_dirs/$(basename "${CONFIG}" .yaml)_seed*/eval_${SPLIT}/metrics.json
