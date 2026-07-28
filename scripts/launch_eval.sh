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
# cuda/12.9 is required for the vLLM wheels. glibc/2.38 is only needed when the
# venv Python still points at /admin/opt/glibc-2.38 (legacy Hopper image).
if [[ -d /admin/opt/modulefiles ]]; then
  module use /admin/opt/modulefiles
fi
if [[ -e /admin/opt/glibc-2.38/lib/ld-linux-x86-64.so.2 ]]; then
  module load glibc/2.38
fi
module load cuda/12.9

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${PROJECT_ROOT}"

# All local vLLM models use the Python 3.12 environment created by
# setup_vllm.sh. Its Swiss AI vLLM fork is built for the cluster's CUDA 12.9
# runtime; the normal project vLLM wheel currently resolves to a CUDA 13 binary.
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
case "${MODEL}" in
  apertus-v1.5-70b|mistral-medium-3.5-128b|gpt-oss-120b|gemma-4-31b-it|qwen3.5-35b-a3b|olmo-3.1-32b-think|lfm2.5-8b|hy-mt2-30b)
    PYTHON_BIN="${PROJECT_ROOT}/.venv-apertus/bin/python"
    ;;
esac
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: ${PYTHON_BIN} does not exist; run scripts/setup_vllm.sh first." >&2
  exit 1
fi
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

# Expose the uv-managed python runtime libs to vLLM/Ray workers.
PYTHON_LIB_DIR="${UV_PYTHON_LIB_DIR:-$("$PYTHON_BIN" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")')}"
SITE_PACKAGES="$("$PYTHON_BIN" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
# Prefer wheel-bundled NVIDIA libs (esp. cuDNN 9.17) over the CUDA module's
# older cuDNN on LD_LIBRARY_PATH, which otherwise breaks Apertus multimodal init.
NVIDIA_LIB_PATH=""
for nvidia_lib in "${SITE_PACKAGES}"/nvidia/*/lib; do
  if [[ -d "${nvidia_lib}" ]]; then
    NVIDIA_LIB_PATH="${NVIDIA_LIB_PATH:+${NVIDIA_LIB_PATH}:}${nvidia_lib}"
  fi
done
export LD_LIBRARY_PATH="${NVIDIA_LIB_PATH:+${NVIDIA_LIB_PATH}:}${PYTHON_LIB_DIR}:${LD_LIBRARY_PATH:-}"
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
"${PYTHON_BIN}" -m swiss_legal_evals.run "${ARGS[@]}"
echo "=== DONE: ${MODEL} ==="
