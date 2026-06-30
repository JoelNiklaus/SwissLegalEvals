"""Preload CUDA wheel libraries needed by the local vLLM stack."""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path


def _cuda_runtime_is_active() -> bool:
    """Return whether the shell environment points at a CUDA runtime."""
    if "SLURM_JOB_ID" not in os.environ:
        return False
    if "LD_LIBRARY_PATH" not in os.environ:
        return False
    return any("cuda" in path.lower() for path in os.environ["LD_LIBRARY_PATH"].split(":"))


def preload_nvjitlink() -> None:
    """Load wheel-provided nvJitLink when the cluster CUDA module is active."""
    if not _cuda_runtime_is_active():
        return

    for base in sys.path:
        candidate = Path(base) / "nvidia" / "nvjitlink" / "lib" / "libnvJitLink.so.12"
        if candidate.exists():
            ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
            return
    raise FileNotFoundError("CUDA runtime is active, but wheel-provided libnvJitLink.so.12 was not found.")


preload_nvjitlink()
