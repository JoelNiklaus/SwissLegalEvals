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

# Cluster-specific vLLM patch for the Hopper environment.
# Most users should prefer: uv sync --extra local && uv pip install vllm
# This script mirrors open-dirac/serve/upgrade_vllm.sh and only exists because
# our cluster needs glibc/2.38, CUDA 12.9, a cu129 nightly wheel, and glibc-fix.

source "$HOME/.bashrc"
module use /admin/opt/modulefiles
module load glibc/2.38 cuda/12.9

echo "CUDA version: $(nvcc --version | grep release)"
echo "glibc: $(ldd --version | head -1)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python3.13"
echo "venv python: $VENV_PYTHON"

# Remove the broken editable install (its _C.abi3.so needs glibc 2.34)
uv pip uninstall vllm --python "$VENV_PYTHON" 2>/dev/null || true

# Install prebuilt vllm wheel using manylinux_2_34 platform tag
# (the glibc/2.38 module makes this compatible at runtime)
uv pip install -U vllm \
  --prerelease=allow \
  --python-platform x86_64-manylinux_2_34 \
  --torch-backend=cu129 \
  --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
  --python "$VENV_PYTHON"

# Make Torch prefer the CUDA 12.9 nvJitLink library shipped in the venv over
# older system CUDA libraries that can appear earlier on the dynamic loader path.
SITE_PACKAGES="$("$VENV_PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "import swiss_legal_evals.cuda_preload" > "$SITE_PACKAGES/swiss_legal_evals_cuda_preload.pth"

VENV_BIN_DIR="$(dirname "$VENV_PYTHON")"
REAL_PYTHON="$(readlink -f "$VENV_PYTHON")"
LOCAL_PYTHON="$VENV_BIN_DIR/python3.13-local"

if [[ "$REAL_PYTHON" != "$LOCAL_PYTHON" ]]; then
  echo "Copying uv-managed Python into venv before glibc-fix"
  cp "$REAL_PYTHON" "$LOCAL_PYTHON"
  chmod +x "$LOCAL_PYTHON"
  ln -sf "$(basename "$LOCAL_PYTHON")" "$VENV_BIN_DIR/python3.13"
  ln -sf python3.13 "$VENV_BIN_DIR/python3"
  ln -sf python3.13 "$VENV_BIN_DIR/python"
  VENV_PYTHON="$LOCAL_PYTHON"
fi

# Patch the venv Python binary to use the loaded glibc 2.38
# (must not be run through the same Python process - use bash directly)
glibc-fix "$VENV_PYTHON"

echo "=== vllm version ==="
"$VENV_PYTHON" -c "import vllm; print(vllm.__version__)"
echo "=== DONE ==="
