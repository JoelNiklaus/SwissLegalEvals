#!/bin/bash
# Reusable Slurm launcher for a single model from configs/models.yaml.
#
# Prefer scripts/launch_model.sh MODEL for one-off submits — it sets --job-name
# and resources. launch_all.sh uses this script internally with the same flags.
#
# Manual sbatch must pass --job-name or Slurm defaults to "launch_eval.sh":
#
#   scripts/launch_model.sh deepseek-v4-pro
#
#   # Or manually:
#   sbatch --partition=hopper-cpu --job-name=swisslegal-deepseek-v4-flash --cpus-per-task=24 \
#     --export=ALL,MODEL=deepseek-v4-flash,MAX_SAMPLES=20,OUTPUT_DIR=results_smoke20 \
#     scripts/launch_eval.sh
#
#   # Local vLLM model (4 GPUs, TP4):
#   sbatch --partition=hopper-prod --job-name=swisslegal-gemma-4-31b-it --gres=gpu:h100:4 --cpus-per-task=44 \
#     --export=ALL,MODEL=gemma-4-31b-it,MAX_SAMPLES=20,OUTPUT_DIR=results_smoke20 \
#     scripts/launch_eval.sh
#
# Env vars (all read at runtime):
#   MODEL        required: model `name` from configs/models.yaml
#   MAX_SAMPLES  optional: cap per task config (omit for full runs)
#   OUTPUT_DIR   optional: lighteval output dir (default: results)
#   TASK_GROUPS  optional: space-separated task groups (e.g. slds); do not use GROUPS (bash builtin)
#SBATCH --partition=hopper-prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=7-00:00:00
# hopper-cpu is preemptible (PreemptMode=REQUEUE). Without --requeue, preemption
# kills the job (SIGUSR1 -> FAILED) instead of requeuing it; append keeps logs across requeues.
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=scripts/logs/slurm/eval-%x-%j.out
#SBATCH --error=scripts/logs/slurm/eval-%x-%j.err

set -euo pipefail

if [[ -z "${MODEL:-}" ]]; then
  echo "ERROR: MODEL env var is required (set via --export=ALL,MODEL=...)." >&2
  exit 2
fi

# Drop empty MAX_SAMPLES so a inherited env var cannot leak a smoke cap via --export=ALL.
if [[ -z "${MAX_SAMPLES:-}" ]]; then
  unset MAX_SAMPLES
fi

source "$HOME/.bashrc"
module use /admin/opt/modulefiles
# glibc/2.38 is needed for the patched venv python; cuda/12.9 for the vLLM wheels.
module load glibc/2.38 cuda/12.9

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${PROJECT_ROOT}"

# Expose the uv-managed python runtime libs to vLLM/Ray workers.
PYTHON_LIB_DIR="${UV_PYTHON_LIB_DIR:-$(./.venv/bin/python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")')}"
export LD_LIBRARY_PATH="${PYTHON_LIB_DIR}:${LD_LIBRARY_PATH:-}"
# Per-job temp dirs so concurrent Ray/Triton workers never clash.
export RAY_TMPDIR="${RAY_TMPDIR:-${TMPDIR:-/tmp}/swisslegal-ray/${SLURM_JOB_ID}}"
export TRITON_CACHE_DIR=${RAY_TMPDIR}/triton
mkdir -p "$RAY_TMPDIR" "$TRITON_CACHE_DIR"

# torch>=2.10 makes vLLM default to AOT compile, whose TritonBundler races across
# data-parallel workers and crashes with KeyError "Unknown key: 'cubin'"
# (vllm-project/vllm#32033). Disable AOT compile (NOT eager mode / old engine).
export VLLM_USE_AOT_COMPILE=0

ARGS=(--models "${MODEL}")
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  ARGS+=(--max-samples "${MAX_SAMPLES}")
fi
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  ARGS+=(--output-dir "${OUTPUT_DIR}")
fi
if [[ -n "${TASK_GROUPS:-}" ]]; then
  # shellcheck disable=SC2206
  ARGS+=(--groups ${TASK_GROUPS})
fi

echo "=== Launching: ${MODEL} (max_samples=${MAX_SAMPLES:-all}, output_dir=${OUTPUT_DIR:-results}) ==="
./.venv/bin/python -m swiss_legal_evals.run "${ARGS[@]}"
echo "=== DONE: ${MODEL} ==="
