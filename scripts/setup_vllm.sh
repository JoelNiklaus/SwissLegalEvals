#!/bin/bash
#SBATCH --job-name=swisslegal-vllm-setup
#SBATCH --partition=hopper-prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=01:00:00
#SBATCH --output=scripts/logs/slurm/setup-vllm-%j.out
#SBATCH --error=scripts/logs/slurm/setup-vllm-%j.err

set -euo pipefail

# Cluster-specific vLLM setup for the Hopper environment.
#
# The Swiss AI revisions are required for Apertus 1.5. The matching upstream
# precompiled wheel keeps installation practical on Hopper while the forked
# Python sources add the Apertus architecture. The fork remains compatible
# with the other local models, including Mistral Medium 3.5.

source "$HOME/.bashrc"
# Legacy Hopper images ship glibc 2.38 under /admin/opt; newer images use the
# system loader (Ubuntu 22.04 / glibc 2.35) and only need the CUDA module.
if [[ -d /admin/opt/modulefiles ]]; then
  module use /admin/opt/modulefiles
fi
if [[ -e /admin/opt/glibc-2.38/lib/ld-linux-x86-64.so.2 ]]; then
  module load glibc/2.38
fi
module load cuda/12.9

echo "CUDA version: $(nvcc --version | grep release)"
echo "glibc: $(ldd --version | head -1)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${PROJECT_ROOT}"

# Keep uv's resolver/build tree on node-local storage. Weka can stall Python
# metadata builds when temporary files are created on the project filesystem.
export UV_CACHE_DIR="${TMPDIR:-/tmp}/swisslegal-uv/${SLURM_JOB_ID:-$$}"
export UV_LINK_MODE=copy
export UV_HTTP_TIMEOUT=500
export GIT_LFS_SKIP_SMUDGE=1
mkdir -p "${UV_CACHE_DIR}"

# All cluster-local vLLM jobs use this environment. The project 3.13
# environment remains available for API-based runs and development.
VLLM_ENV="${PROJECT_ROOT}/.venv-apertus"
VENV_PYTHON="${VLLM_ENV}/bin/python3.12"
if [[ ! -x "${VENV_PYTHON}" ]]; then
  uv venv --python 3.12 "${VLLM_ENV}"
fi
if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "ERROR: ${VENV_PYTHON} was not created." >&2
  exit 1
fi
echo "Apertus venv python: ${VENV_PYTHON}"

APERTUS_VLLM_COMMIT="a601a9d998ddeb488f0c17e8512874b116aa7658"
APERTUS_TRANSFORMERS_COMMIT="3797303dda74844e3d1f8977ff5518bb91f818b4"
VLLM_PRECOMPILED_COMMIT="ae6170f874a7c66512bd24aa714b3a02f8a61838"
CACHE_ROOT="${TMPDIR:-/tmp}/swiss-legal-evals"
VLLM_ROOT="${CACHE_ROOT}/vllm-${APERTUS_VLLM_COMMIT}"
TRANSFORMERS_ROOT="${CACHE_ROOT}/transformers-${APERTUS_TRANSFORMERS_COMMIT}"

mkdir -p "${CACHE_ROOT}"

if [[ ! -d "${VLLM_ROOT}/.git" ]]; then
  git clone --branch apertus-1-5 \
    https://github.com/swiss-ai/vllm.git "${VLLM_ROOT}"
fi
git -C "${VLLM_ROOT}" fetch --depth=1 origin "${APERTUS_VLLM_COMMIT}"
git -C "${VLLM_ROOT}" checkout --detach "${APERTUS_VLLM_COMMIT}"

if [[ ! -d "${TRANSFORMERS_ROOT}/.git" ]]; then
  git clone --branch feature/apertus_1p5_pipeline \
    https://github.com/swiss-ai/transformers.git "${TRANSFORMERS_ROOT}"
fi
git -C "${TRANSFORMERS_ROOT}" fetch --depth=1 origin "${APERTUS_TRANSFORMERS_COMMIT}"
git -C "${TRANSFORMERS_ROOT}" checkout --detach "${APERTUS_TRANSFORMERS_COMMIT}"

# The official Apertus image uses Python 3.12 because PyNvVideoCodec, required
# by the Swiss AI vLLM fork, does not publish a CPython 3.13 wheel.
uv pip install \
  --python "$VENV_PYTHON" \
  --torch-backend=cu129 \
  torch==2.11.0 \
  torchvision==0.26.0 \
  torchaudio==2.11.0

uv pip install \
  --python "$VENV_PYTHON" \
  "lighteval[multilingual,litellm] @ git+https://github.com/huggingface/lighteval@4d470292936b9ec5523cb495b4165cc4f77bcc77" \
  "pandas>=2.2.0" \
  "plotly>=5.24.0" \
  "python-dotenv>=1.0.0" \
  "pyyaml>=6.0.2" \
  "tqdm>=4.66.0" \
  "ray>=2.40.0" \
  "more-itertools>=10.5.0"

# The precompiled vLLM wheel still builds the Python package from the forked
# source, so its build-time toolchain must exist before either fork is built.
uv pip install \
  --python "$VENV_PYTHON" \
  "cmake>=3.26.1" \
  ninja \
  "packaging>=24.2" \
  "setuptools>=77.0.3,<81.0.0" \
  "setuptools-scm>=8" \
  "setuptools-rust>=1.9.0" \
  wheel \
  jinja2

# Install the Apertus-specific Transformers classes before vLLM imports them.
uv pip install \
  --python "$VENV_PYTHON" \
  --no-deps \
  --no-build-isolation \
  --reinstall \
  "$TRANSFORMERS_ROOT"

# Remove the old wheel so the custom vLLM source and matching extension are
# installed as one coherent version.
uv pip uninstall vllm --python "$VENV_PYTHON" 2>/dev/null || true

# The custom source tree supplies Apertus; this precompiled cu129 wheel supplies
# the matching native extension without rebuilding all CUDA kernels locally.
export VLLM_USE_PRECOMPILED=1
export VLLM_MAIN_CUDA_VERSION=12.9
export VLLM_VERSION_OVERRIDE=0.23.1
export VLLM_PRECOMPILED_WHEEL_VARIANT=cu129
export VLLM_PRECOMPILED_WHEEL_COMMIT="${VLLM_PRECOMPILED_COMMIT}"
# Let the fork resolve its pinned variant metadata, including the `.cu129`
# filename suffix, instead of constructing a default CUDA 13 wheel URL.
cd "${VLLM_ROOT}"
VLLM_USE_PRECOMPILED=1 uv pip install \
  --python "$VENV_PYTHON" \
  --reinstall-package vllm \
  --no-build-isolation "." \
  --torch-backend=auto

# Make Torch prefer the CUDA 12.9 nvJitLink library shipped in the venv over
# older system CUDA libraries that can appear earlier on the dynamic loader path.
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
SITE_PACKAGES="$("$VENV_PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

# The custom vLLM tokenizer for Mistral exposes a callable tokenizer without
# Hugging Face's eos_token attribute. Keep lighteval compatible with both APIs.
LIGHTEVAL_VLLM_FILE="${SITE_PACKAGES}/lighteval/models/vllm/vllm_model.py"
"$VENV_PYTHON" - "${LIGHTEVAL_VLLM_FILE}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
source = path.read_text()
old = '        tokenizer.pad_token = tokenizer.eos_token\n'
new = '        eos_token = getattr(tokenizer, "eos_token", None)\n        if eos_token is not None:\n            tokenizer.pad_token = eos_token\n'

if old in source:
    path.write_text(source.replace(old, new, 1))
elif new not in source:
    raise RuntimeError(f"Unexpected lighteval tokenizer implementation in {path}")
PY

# The Apertus Transformers branch nests `rope_parameters` per attention type
# (`sliding_attention` / `full_attention`), which OLMo 3 uses to apply YaRN only
# on full-attention layers. This vLLM revision still expects the flat dict.
OLMO2_FILE="${SITE_PACKAGES}/vllm/model_executor/models/olmo2.py"
"$VENV_PYTHON" - "${OLMO2_FILE}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
source = path.read_text()
old = """        if sliding_window is None:
            rope_parameters = self.config.rope_parameters
        else:
            rope_theta = self.config.rope_parameters["rope_theta"]
            rope_parameters = {"rope_type": "default", "rope_theta": rope_theta}
"""
new = """        rope_parameters = self.config.rope_parameters
        if "full_attention" in rope_parameters:
            attention_type = "sliding_attention" if sliding_window else "full_attention"
            rope_parameters = rope_parameters[attention_type]
        elif sliding_window is not None:
            rope_theta = rope_parameters["rope_theta"]
            rope_parameters = {"rope_type": "default", "rope_theta": rope_theta}
"""

if old in source:
    path.write_text(source.replace(old, new, 1))
elif new not in source:
    raise RuntimeError(f"Unexpected vLLM Olmo2 rope implementation in {path}")
PY

{
  echo "${PROJECT_ROOT}/src"
  echo "import swiss_legal_evals.cuda_preload"
} > "$SITE_PACKAGES/swiss_legal_evals_cuda_preload.pth"

VENV_BIN_DIR="$(dirname "$VENV_PYTHON")"
REAL_PYTHON="$(readlink -f "$VENV_PYTHON")"
LOCAL_PYTHON="$VENV_BIN_DIR/python3.12-local"

if [[ "$REAL_PYTHON" != "$LOCAL_PYTHON" ]]; then
  echo "Copying uv-managed Python into venv before glibc-fix"
  cp "$REAL_PYTHON" "$LOCAL_PYTHON"
  chmod +x "$LOCAL_PYTHON"
  ln -sf "$(basename "$LOCAL_PYTHON")" "$VENV_BIN_DIR/python3.12"
  ln -sf python3.12 "$VENV_BIN_DIR/python3"
  ln -sf python3.12 "$VENV_BIN_DIR/python"
  VENV_PYTHON="$LOCAL_PYTHON"
fi

# Patch the venv Python binary onto the loaded glibc 2.38 only when that
# runtime exists. On clusters without /admin/opt, keep the system loader.
if [[ -e /admin/opt/glibc-2.38/lib/ld-linux-x86-64.so.2 ]] && command -v glibc-fix >/dev/null 2>&1; then
  glibc-fix "$VENV_PYTHON"
else
  echo "Skipping glibc-fix; using system dynamic loader"
fi

echo "=== vllm version ==="
"$VENV_PYTHON" -c "import vllm; print(vllm.__version__)"
echo "=== DONE ==="
