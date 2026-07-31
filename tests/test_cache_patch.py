"""Chunked generation so a crashed run keeps the samples it already produced."""

from __future__ import annotations

from typing import Any

import pytest
from lighteval.models.endpoints.inference_providers_model import InferenceProvidersClient
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc

from swiss_legal_evals.cache_patch import (
    CHUNK_SIZE_ENV_VAR,
    DEFAULT_CACHE_CHUNK_SIZE,
    _chunked_generation,
    chunk_size_from_env,
    enable_incremental_caching,
)


def _docs(count: int) -> list[Doc]:
    return [Doc(query=f"q{i}", choices=["a"], gold_index=0, task_name="t") for i in range(count)]


def _recording_method() -> Any:
    """Stand-in for the cached ``greedy_until``; records the size of every call."""
    calls: list[int] = []

    def method(self: Any, docs: list[Doc]) -> list[ModelResponse]:
        calls.append(len(docs))
        return [ModelResponse(text=[d.query]) for d in docs]

    method.calls = calls  # type: ignore[attr-defined]
    return method


def test_long_batch_is_split_into_chunks() -> None:
    method = _recording_method()
    wrapped = _chunked_generation(method, chunk_size=100)

    responses = wrapped(None, _docs(250))

    assert method.calls == [100, 100, 50]
    # Each call is a cache checkpoint, so a crash costs at most the last chunk.
    assert [r.text[0] for r in responses] == [f"q{i}" for i in range(250)]


def test_short_batch_is_passed_through_unchanged() -> None:
    method = _recording_method()
    wrapped = _chunked_generation(method, chunk_size=100)

    wrapped(None, _docs(100))

    assert method.calls == [100]


def test_single_doc_is_accepted() -> None:
    """lighteval calls generation with a bare Doc in places."""
    method = _recording_method()
    wrapped = _chunked_generation(method, chunk_size=2)

    responses = wrapped(None, _docs(1)[0])

    assert method.calls == [1]
    assert responses[0].text == ["q0"]


def test_enabling_twice_does_not_nest_wrappers() -> None:
    original = InferenceProvidersClient.greedy_until
    try:
        enable_incremental_caching(chunk_size=10)
        once = InferenceProvidersClient.greedy_until
        enable_incremental_caching(chunk_size=10)

        assert InferenceProvidersClient.greedy_until is once
    finally:
        InferenceProvidersClient.greedy_until = original


def test_chunk_size_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    assert chunk_size_from_env() == DEFAULT_CACHE_CHUNK_SIZE
    monkeypatch.setenv(CHUNK_SIZE_ENV_VAR, "42")
    assert chunk_size_from_env() == 42
