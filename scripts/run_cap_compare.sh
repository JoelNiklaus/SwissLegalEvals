#!/bin/bash
# Compare 16k vs 32k generation caps on rambling models (fixed task subset).
#
# Usage:
#   bash scripts/run_cap_compare.sh cap16k minimax-m3
#   bash scripts/run_cap_compare.sh cap32k kimi-k2.6
set -euo pipefail

PROFILE="${1:?profile cap16k or cap32k required}"
MODEL="${2:?model name required}"
MAX_SAMPLES="${MAX_SAMPLES:-5}"
CAP_SUFFIX="${PROFILE#cap}"
OUTPUT_DIR="${OUTPUT_DIR:-results_cap_compare_${CAP_SUFFIX}}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TASK_STRING='slds:de_de|0,slt-paragraph_level:de-fr|0,lexam_mcq_16_idk:en|0'

cd "${PROJECT_ROOT}"

echo "=== cap compare: model=${MODEL} profile=${PROFILE} max_samples=${MAX_SAMPLES} out=${OUTPUT_DIR} ==="
./.venv/bin/python -m swiss_legal_evals.run \
  --models "${MODEL}" \
  --tasks-config configs/tasks_cap_compare.yaml \
  --profile "${PROFILE}" \
  --task-string "${TASK_STRING}" \
  --max-samples "${MAX_SAMPLES}" \
  --output-dir "${OUTPUT_DIR}"

echo "=== DONE: ${MODEL} ${PROFILE} ==="
