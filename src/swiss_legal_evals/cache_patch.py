"""Make lighteval persist its sample cache incrementally instead of once at the end.

lighteval's ``@cached`` decorator writes the parquet cache only after the whole
batch of uncached docs has been generated (``cache_management.py``, step 2). A run
that dies mid-batch therefore saves nothing: Kimi K3 lost 25 h of generation when
it hung on prompt 8863 of 8865, because that single ``greedy_until`` call never
returned.

We cannot move the write without forking lighteval, but we can make the batches
smaller. Calling the already-decorated ``greedy_until`` once per chunk turns each
chunk boundary into a cache checkpoint, so a crash costs at most one chunk and a
relaunch resumes from the cache.
"""

from __future__ import annotations

import functools
import logging
import os
from collections.abc import Callable
from typing import Any

from lighteval.models.endpoints.inference_providers_model import InferenceProvidersClient
from lighteval.models.endpoints.litellm_model import LiteLLMClient
from lighteval.models.model_output import ModelResponse
from lighteval.models.vllm.vllm_model import VLLMModel
from lighteval.tasks.requests import Doc
from lighteval.utils.imports import is_package_available
from lighteval.utils.utils import as_list

logger = logging.getLogger(__name__)

# Trades throughput against the size of the loss window: each boundary is a barrier
# that drains in-flight requests, and each write rewrites the task's parquet. 500 is
# minutes of work on a local vLLM run and up to ~1.5 h on the slowest hosted models.
DEFAULT_CACHE_CHUNK_SIZE = 500
CHUNK_SIZE_ENV_VAR = "SWISS_LEGAL_EVALS_CACHE_CHUNK_SIZE"

_PATCH_MARKER = "_swiss_legal_evals_chunked"


def _patch_targets() -> tuple[type, ...]:
    """Backends to wrap: every one we launch generates through a sync ``greedy_until``.

    Async backends (``AsyncVLLMModel``) would need an async wrapper and stay untouched.
    """
    targets: list[type] = [InferenceProvidersClient, LiteLLMClient]
    # Without vllm installed, as in the API-only venv, lighteval swaps VLLMModel for a
    # placeholder that raises ImportError on any attribute access.
    if is_package_available("vllm"):
        targets.append(VLLMModel)
    return tuple(targets)


def chunk_size_from_env() -> int:
    """Chunk size for cache checkpoints, overridable per run."""
    return int(os.environ.get(CHUNK_SIZE_ENV_VAR, DEFAULT_CACHE_CHUNK_SIZE))


def _chunked_generation(
    method: Callable[..., list[ModelResponse]], chunk_size: int
) -> Callable[..., list[ModelResponse]]:
    """Split one generation call into successive calls of at most ``chunk_size`` docs."""

    @functools.wraps(method)
    def wrapper(self: Any, docs: Doc | list[Doc], *args: Any, **kwargs: Any) -> list[ModelResponse]:
        docs = as_list(docs)
        if len(docs) <= chunk_size:
            return method(self, docs, *args, **kwargs)

        logger.info(
            "Generating %d samples in chunks of %d so the cache is written incrementally",
            len(docs),
            chunk_size,
        )
        responses: list[ModelResponse] = []
        for start in range(0, len(docs), chunk_size):
            responses.extend(method(self, docs[start : start + chunk_size], *args, **kwargs))
        return responses

    setattr(wrapper, _PATCH_MARKER, True)
    return wrapper


def enable_incremental_caching(chunk_size: int | None = None) -> None:
    """Wrap the cached ``greedy_until`` of every backend we use so it checkpoints.

    Idempotent, because lighteval may import the custom-tasks module more than once.

    Args:
        chunk_size: Docs per cache checkpoint. Defaults to the value from the
            environment, falling back to :data:`DEFAULT_CACHE_CHUNK_SIZE`.
    """
    chunk_size = chunk_size if chunk_size is not None else chunk_size_from_env()
    for model_cls in _patch_targets():
        method = model_cls.greedy_until
        if getattr(method, _PATCH_MARKER, False):
            continue
        model_cls.greedy_until = _chunked_generation(method, chunk_size)
        logger.debug("Enabled incremental caching for %s", model_cls.__name__)
