"""Tests for provider environment validation."""

from __future__ import annotations

import pytest

from swiss_legal_evals.providers import validate_provider_env


def test_openai_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(OSError, match="OPENAI_API_KEY"):
        validate_provider_env("openai")


def test_vllm_no_key_required() -> None:
    validate_provider_env("vllm")
