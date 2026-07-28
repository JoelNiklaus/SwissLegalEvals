#!/bin/bash
# Submit one model with the same naming/resources as launch_all.sh.
#
# Usage:
#   scripts/launch_model.sh MODEL [MAX_SAMPLES] [OUTPUT_DIR]
#
# Examples:
#   scripts/launch_model.sh deepseek-v4-pro
#   scripts/launch_model.sh gemma-4-31b-it "" results
#   scripts/launch_model.sh gemma-4-31b-it 20
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/launch_model.sh MODEL [MAX_SAMPLES] [OUTPUT_DIR]" >&2
  exit 2
fi

MODEL="$1"
MAX_SAMPLES=""
OUTPUT_DIR="results"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ $# -ge 2 ]]; then
  MAX_SAMPLES="$2"
  if [[ $# -ge 3 ]]; then
    OUTPUT_DIR="$3"
  elif [[ -n "$2" ]]; then
    OUTPUT_DIR="results_smoke${2}"
  fi
fi

declare -A GPU_MODELS=(
  [apertus-v1.5-70b]=4
  [mistral-medium-3.5-128b]=8
  [gpt-oss-120b]=4
  [gemma-4-31b-it]=4
  [qwen3.5-35b-a3b]=4
  [olmo-3.1-32b-think]=4
  [lfm2.5-8b]=1
  [hy-mt2-30b]=4
)

cd "${PROJECT_ROOT}"
mkdir -p scripts/logs/slurm

export_vars="ALL,MODEL=${MODEL},OUTPUT_DIR=${OUTPUT_DIR}"
if [[ -n "${MAX_SAMPLES}" ]]; then
  export_vars+=",MAX_SAMPLES=${MAX_SAMPLES}"
fi

sbatch_args=(--parsable --job-name="swisslegal-${MODEL}" --export="${export_vars}")

if [[ -v GPU_MODELS[$MODEL] ]]; then
  gpus="${GPU_MODELS[$MODEL]}"
  cpus=$(( gpus * 11 ))
  sbatch_args+=(--partition=hopper-prod --gres="gpu:h100:${gpus}" --cpus-per-task="${cpus}")
else
  sbatch_args+=(--partition=hopper-cpu --cpus-per-task=24)
fi

# A bounded smoke run must fit the cluster's short-job scheduling window.
if [[ -n "${MAX_SAMPLES}" ]]; then
  sbatch_args+=(--time=01:00:00)
fi

jobid=$(sbatch "${sbatch_args[@]}" scripts/launch_eval.sh)
echo "submitted ${MODEL} -> job ${jobid} (max_samples=${MAX_SAMPLES:-all}, output=${OUTPUT_DIR})"
