"""Provider → environment variable requirements."""

from __future__ import annotations

import os
from typing import Literal

Provider = Literal["openai", "openrouter", "hf-inference-providers", "vllm"]

_PROVIDER_ENV: dict[Provider, str | None] = {
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "hf-inference-providers": "HF_TOKEN",
    "vllm": None,
}


def validate_provider_env(provider: Provider) -> None:
    """Fail fast if a required API key is missing for the provider."""
    env_var = _PROVIDER_ENV[provider]
    if env_var is not None and env_var not in os.environ:
        raise OSError(
            f"{env_var} must be set for provider {provider!r} "
            "(put it in .env or export it before running evals)"
        )


def validate_judge_env(judges: dict[str, dict[str, str]]) -> None:
    """Validate env vars for all judges referenced in judges.yaml."""
    for cfg in judges.values():
        validate_provider_env(cfg["provider"])  # type: ignore[arg-type]


def hf_org_to_bill() -> str | None:
    """HF Inference Providers org slug for the X-HF-Bill-To header, if set."""
    value = os.environ["HF_ORG_TO_BILL"] if "HF_ORG_TO_BILL" in os.environ else None
    if value is not None and value == "":
        raise ValueError("HF_ORG_TO_BILL is set but empty")
    return value
