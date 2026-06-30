#!/bin/bash
# Submit one Slurm job per model in configs/models.yaml via scripts/launch_eval.sh.
#
# Usage:
#   scripts/launch_all.sh [MAX_SAMPLES] [OUTPUT_DIR]
#
# Examples:
#   scripts/launch_all.sh                      # full uncapped run → results/
#   scripts/launch_all.sh 20                     # smoke → results_smoke20/
#   scripts/launch_all.sh 20 results_smoke20   # smoke with explicit output dir
#   scripts/launch_all.sh "" results             # full run (same as no args)
#
# Commented-out models in models.yaml are intentionally excluded here too.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default: full uncapped run. Do NOT use ${1:-20} — bash treats "" as null and
# would silently substitute 20, so `launch_all.sh "" results` became a smoke run.
MAX_SAMPLES=""
OUTPUT_DIR="results"

if [[ $# -ge 1 ]]; then
  MAX_SAMPLES="$1"
  if [[ $# -ge 2 ]]; then
    OUTPUT_DIR="$2"
  elif [[ -n "$1" ]]; then
    OUTPUT_DIR="results_smoke${1}"
  fi
fi

cd "${PROJECT_ROOT}"
mkdir -p scripts/logs/slurm

# API models (no GPU): served via HF inference providers / litellm endpoints.
API_MODELS=(
  deepseek-v4-pro
  deepseek-v4-flash
  kimi-k2.6
  glm-5.2
  minimax-m3
  # qwen3.5-397b
  nemotron-3-ultra-550b-a55b-nvfp4
)

# Local vLLM models: Slurm GPU count = data_parallel_size * tensor_parallel_size
# from configs/models.yaml (defaults: DP=1, TP=4 for large models; lfm2.5 uses 1 GPU).
declare -A GPU_MODELS=(
  [gpt-oss-120b]=4
  [gemma-4-31b-it]=4
  [qwen3.5-35b-a3b]=4
  [olmo-3.1-32b-think]=4
  [lfm2.5-8b]=1
  [hy-mt2-30b]=4
)

submit() {
  local model="$1"
  shift
  local jobid
  local export_vars="ALL,MODEL=${model},OUTPUT_DIR=${OUTPUT_DIR}"
  if [[ -n "${MAX_SAMPLES}" ]]; then
    export_vars+=",MAX_SAMPLES=${MAX_SAMPLES}"
  fi
  jobid=$(sbatch --parsable "$@" \
    --job-name="swisslegal-${model}" \
    --export="${export_vars}" \
    scripts/launch_eval.sh)
  echo "submitted ${model} -> job ${jobid} (max_samples=${MAX_SAMPLES:-all}, output=${OUTPUT_DIR})"
}

for model in "${API_MODELS[@]}"; do
  # No --gres: inference runs remotely via HF Inference Providers; job is CPU-only.
  submit "$model" --partition=hopper-cpu --cpus-per-task=24
done

for model in "${!GPU_MODELS[@]}"; do
  gpus="${GPU_MODELS[$model]}"
  cpus=$(( gpus * 11 ))
  submit "$model" --partition=hopper-prod --gres="gpu:h100:${gpus}" --cpus-per-task="${cpus}"
done
