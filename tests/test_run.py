"""Tests for lighteval command construction."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from swiss_legal_evals.run import (
    DEFAULT_API_TIMEOUT_SECONDS,
    LIGHTEVAL_CMD,
    _required_judge_keys_for_task_string,
    _subprocess_env,
    build_lighteval_command,
    load_env_file,
)


def test_subprocess_env_exports_tasks_profile(tmp_path: Path) -> None:
    tasks_config = tmp_path / "tasks.yaml"
    judge_config = tmp_path / "judges.yaml"
    tasks_config.write_text("profiles: {}\n")
    judge_config.write_text("judges: {}\n")
    env = _subprocess_env(tasks_config, "default", judge_config)
    assert env["SWISSLEGALEVALS_TASKS_CONFIG"] == str(tasks_config.resolve())
    assert env["SWISSLEGALEVALS_TASKS_PROFILE"] == "default"
    assert env["SWISSLEGALEVALS_JUDGE_CONFIG"] == str(judge_config.resolve())


def test_litellm_command() -> None:
    entry = {"name": "mimo", "provider": "openrouter", "model": "xiaomi/mimo-v2.5-pro"}
    cmd = build_lighteval_command(
        entry=entry,
        task_string="slds:de_de|0",
        output_dir=Path("/tmp/results"),
        max_samples=None,
    )
    assert cmd[: len(LIGHTEVAL_CMD)] == LIGHTEVAL_CMD
    assert cmd[len(LIGHTEVAL_CMD) : len(LIGHTEVAL_CMD) + 3] == [
        "endpoint",
        "litellm",
        "model_name=openrouter/xiaomi/mimo-v2.5-pro",
    ]
    assert "--custom-tasks" in cmd
    assert "--save-details" in cmd
    assert "--reasoning-tags" in cmd


def test_inference_providers_command() -> None:
    entry = {
        "name": "ds",
        "provider": "hf-inference-providers",
        "model": "deepseek-ai/DeepSeek-V4-Pro",
        "hf_provider": "together",
    }
    cmd = build_lighteval_command(
        entry=entry,
        task_string="lexam_oq:en|0",
        output_dir=Path("/tmp/results"),
        max_samples=10,
    )
    assert "inference-providers" in cmd
    model_args = cmd[cmd.index("inference-providers") + 1]
    assert model_args.startswith("model_name=deepseek-ai/DeepSeek-V4-Pro,provider=together")
    # Without a timeout the client waits forever on a stream the provider abandons.
    assert f"timeout={DEFAULT_API_TIMEOUT_SECONDS}" in model_args
    assert "--max-samples" in cmd
    assert "10" in cmd


def test_inference_providers_command_passes_extra_client_args() -> None:
    entry = {
        "name": "kimi",
        "provider": "hf-inference-providers",
        "model": "moonshotai/Kimi-K2.6",
        "hf_provider": "deepinfra",
        "parallel_calls_count": 2,
        "timeout": 900,
    }
    cmd = build_lighteval_command(
        entry=entry,
        task_string="slds:de_de|0",
        output_dir=Path("/tmp/results"),
        max_samples=20,
    )
    model_args = cmd[cmd.index("inference-providers") + 1]
    assert "provider=deepinfra" in model_args
    assert "parallel_calls_count=2" in model_args
    assert "timeout=900" in model_args


def test_inference_providers_command_bills_org_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_ORG_TO_BILL", "huggingface")
    entry = {
        "name": "ds",
        "provider": "hf-inference-providers",
        "model": "deepseek-ai/DeepSeek-V4-Pro",
        "hf_provider": "together",
    }
    cmd = build_lighteval_command(
        entry=entry,
        task_string="lexam_oq:en|0",
        output_dir=Path("/tmp/results"),
        max_samples=10,
    )
    model_args = cmd[cmd.index("inference-providers") + 1]
    assert "org_to_bill=huggingface" in model_args


def test_vllm_command() -> None:
    entry = {"name": "gemma", "provider": "vllm", "model": "google/gemma-4-31B-it"}
    cmd = build_lighteval_command(
        entry=entry,
        task_string="slds:de_de|0",
        output_dir=Path("/tmp/results"),
        max_samples=None,
    )
    assert "-m" in cmd
    assert "lighteval" in cmd
    assert "vllm" in cmd
    assert "model_name=google/gemma-4-31B-it,data_parallel_size=8" in cmd


def test_vllm_command_respects_configured_data_parallel_size() -> None:
    entry = {
        "name": "gemma",
        "provider": "vllm",
        "model": "google/gemma-4-31B-it",
        "data_parallel_size": 4,
    }
    cmd = build_lighteval_command(
        entry=entry,
        task_string="slds:de_de|0",
        output_dir=Path("/tmp/results"),
        max_samples=None,
    )
    assert "model_name=google/gemma-4-31B-it,data_parallel_size=4" in cmd


def test_required_judge_keys_for_task_string() -> None:
    keys = _required_judge_keys_for_task_string(
        "slds:de_de|0,sdst-text_level:de-fr|0,lexam_mcq_4:en|0,lexam_oq:en|0"
    )
    assert keys == {"slds", "swiltrabench", "lexam_oq"}


def test_load_env_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("HF_TOKEN=from-dotenv\n")
    monkeypatch.delenv("HF_TOKEN", raising=False)

    load_env_file(env_file)

    assert os.environ["HF_TOKEN"] == "from-dotenv"


def test_load_env_file_does_not_override_existing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("HF_TOKEN=from-dotenv\n")
    monkeypatch.setenv("HF_TOKEN", "from-shell")

    load_env_file(env_file)

    assert os.environ["HF_TOKEN"] == "from-shell"
