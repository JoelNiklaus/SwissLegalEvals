"""Run lighteval evaluations for all models in configs/models.yaml."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from swiss_legal_evals.postprocess import lighteval_reasoning_tags_cli
from swiss_legal_evals.providers import Provider, hf_org_to_bill, validate_provider_env
from swiss_legal_evals.task_lists import _task_enabled, build_task_list, load_tasks_config
from swiss_legal_evals.tasks import load_judge_config

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path.cwd()
PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS_CONFIG = PROJECT_ROOT / "configs" / "models.yaml"
DEFAULT_TASKS_CONFIG = PROJECT_ROOT / "configs" / "tasks.yaml"
DEFAULT_JUDGE_CONFIG = PROJECT_ROOT / "configs" / "judges.yaml"
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
CUSTOM_TASKS_PATH = PACKAGE_ROOT / "tasks.py"
OUTPUT_DIR = PROJECT_ROOT / "results"
LIGHTEVAL_CMD = [sys.executable, "-m", "lighteval"]
DEFAULT_VLLM_DATA_PARALLEL_SIZE = 8


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )


def load_env_file(path: Path) -> None:
    """Load API keys and local configuration from a dotenv file if present."""
    if path.exists():
        load_dotenv(dotenv_path=path, override=False)
        logger.info("Loaded environment variables from %s", path)
    else:
        logger.debug("No dotenv file found at %s", path)


def load_models_config(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        raw = yaml.safe_load(f)
    return raw["models"]


def _litellm_model_args(provider: Provider, model: str) -> str:
    if provider == "openai":
        model_name = model if model.startswith("openai/") else f"openai/{model}"
    elif provider == "openrouter":
        model_name = model if model.startswith("openrouter/") else f"openrouter/{model}"
    else:
        raise ValueError(f"Not a LiteLLM provider: {provider}")
    return f"model_name={model_name}"


def _inference_providers_model_args(model: str, hf_provider: str, extra: dict[str, Any]) -> str:
    parts = [f"model_name={model}", f"provider={hf_provider}"]
    if "org_to_bill" not in extra:
        org = hf_org_to_bill()
        if org is not None:
            parts.append(f"org_to_bill={org}")
    for key, value in extra.items():
        if key in ("name", "provider", "model", "tasks", "hf_provider"):
            continue
        parts.append(f"{key}={value}")
    return ",".join(parts)


def _vllm_model_args(model: str, extra: dict[str, Any]) -> str:
    parts = [f"model_name={model}"]
    if "data_parallel_size" not in extra:
        parts.append(f"data_parallel_size={DEFAULT_VLLM_DATA_PARALLEL_SIZE}")
    for key, value in extra.items():
        if key in ("name", "provider", "model", "tasks", "hf_provider"):
            continue
        parts.append(f"{key}={value}")
    return ",".join(parts)


def build_lighteval_command(
    entry: dict[str, Any],
    task_string: str,
    output_dir: Path,
    max_samples: int | None,
) -> list[str]:
    """Build argv for a single model evaluation."""
    provider: Provider = entry["provider"]
    model = entry["model"]

    base_flags = [
        "--custom-tasks",
        str(CUSTOM_TASKS_PATH),
        "--output-dir",
        str(output_dir),
        # Persist per-sample model inputs/outputs/metrics so runs can be inspected.
        "--save-details",
        "--reasoning-tags",
        lighteval_reasoning_tags_cli(),
    ]
    if max_samples is not None:
        base_flags.extend(["--max-samples", str(max_samples)])

    if provider in ("openai", "openrouter"):
        model_args = _litellm_model_args(provider, model)
        return [
            *LIGHTEVAL_CMD,
            "endpoint",
            "litellm",
            model_args,
            task_string,
            *base_flags,
        ]

    if provider == "hf-inference-providers":
        hf_provider = entry["hf_provider"]
        model_args = _inference_providers_model_args(model, hf_provider, entry)
        return [
            *LIGHTEVAL_CMD,
            "endpoint",
            "inference-providers",
            model_args,
            task_string,
            *base_flags,
        ]

    if provider == "vllm":
        model_args = _vllm_model_args(model, entry)
        return [*LIGHTEVAL_CMD, "vllm", model_args, task_string, *base_flags]

    raise ValueError(f"Unknown provider: {provider}")


def _subprocess_env(tasks_config: Path, profile: str, judge_config: Path = DEFAULT_JUDGE_CONFIG) -> dict[str, str]:
    """Build an environment that exposes uv Python runtime libs to vLLM workers."""
    env = dict(os.environ)
    env["SWISSLEGALEVALS_TASKS_CONFIG"] = str(tasks_config.resolve())
    env["SWISSLEGALEVALS_TASKS_PROFILE"] = profile
    env["SWISSLEGALEVALS_JUDGE_CONFIG"] = str(judge_config.resolve())
    python_lib_dir = Path(sys.base_prefix) / "lib"
    if (python_lib_dir / f"libpython{sys.version_info.major}.{sys.version_info.minor}.so.1.0").exists():
        if "LD_LIBRARY_PATH" in env and env["LD_LIBRARY_PATH"]:
            env["LD_LIBRARY_PATH"] = f"{python_lib_dir}:{env['LD_LIBRARY_PATH']}"
        else:
            env["LD_LIBRARY_PATH"] = str(python_lib_dir)

    tcl_library = python_lib_dir / "tcl8.6"
    if tcl_library.exists():
        env["TCL_LIBRARY"] = str(tcl_library)

    tk_library = python_lib_dir / "tk8.6"
    if tk_library.exists():
        env["TK_LIBRARY"] = str(tk_library)
    return env


def resolve_task_groups(entry: dict[str, Any], default_groups: list[str]) -> list[str]:
    if "tasks" in entry:
        return list(entry["tasks"])
    return default_groups


def _required_judge_keys(groups: list[str], profile: str, tasks_config_path: Path) -> set[str]:
    """Return judge config keys needed by the selected task groups."""
    tasks_config = load_tasks_config(tasks_config_path)
    profile_cfg = tasks_config["profiles"][profile]

    keys: set[str] = set()
    for group in groups:
        if group == "slds" and _task_enabled(profile_cfg.get("slds")):
            keys.add("slds")
        elif group in ("swiltrabench", "swiltrabench_full"):
            keys.add("swiltrabench")
        elif group == "lexam" and profile_cfg["lexam"]["open_question"]:
            keys.add("lexam_oq")
    return keys


def _required_judge_keys_for_task_string(task_string: str) -> set[str]:
    """Return judge config keys needed by an explicit lighteval task string."""
    keys: set[str] = set()
    for task_name in task_string.split(","):
        if task_name.startswith("slds:"):
            keys.add("slds")
        elif task_name.startswith(("sdst-", "slt-", "sscprt-")):
            keys.add("swiltrabench")
        elif task_name.startswith("lexam_oq:"):
            keys.add("lexam_oq")
    return keys


def _validate_env_for_selection(
    models: list[dict[str, Any]],
    groups: list[str],
    profile: str,
    tasks_config_path: Path,
) -> None:
    """Validate only providers that this run can actually touch."""
    judges = load_judge_config()
    for judge_key in _required_judge_keys(groups, profile, tasks_config_path):
        validate_provider_env(judges[judge_key]["provider"])  # type: ignore[arg-type]
    for entry in models:
        validate_provider_env(entry["provider"])


def _validate_env_for_task_string(models: list[dict[str, Any]], task_string: str) -> None:
    judges = load_judge_config()
    for judge_key in _required_judge_keys_for_task_string(task_string):
        validate_provider_env(judges[judge_key]["provider"])  # type: ignore[arg-type]
    for entry in models:
        validate_provider_env(entry["provider"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run Swiss legal benchmark evaluations.")
    parser.add_argument(
        "--models-config",
        type=Path,
        default=DEFAULT_MODELS_CONFIG,
        help="Path to models.yaml",
    )
    parser.add_argument(
        "--tasks-config",
        type=Path,
        default=DEFAULT_TASKS_CONFIG,
        help="Path to tasks.yaml",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help="Path to .env file with provider API keys",
    )
    parser.add_argument(
        "--profile",
        default="default",
        help="Task profile from tasks.yaml (default: default)",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["slds", "swiltrabench", "lexam"],
        help="Task groups to evaluate",
    )
    parser.add_argument(
        "--task-string",
        default=None,
        help="Explicit comma-separated lighteval task string (overrides --groups; useful for smoke tests)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Subset of model `name` fields to run (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="lighteval output directory",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap per task (debug only)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Explicitly run all configured models (default behavior)",
    )
    parser.add_argument(
        "--skip-env-check",
        action="store_true",
        help="Do not verify API keys before running (not recommended)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)
    load_env_file(args.env_file)
    models = load_models_config(args.models_config)
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir

    if args.models:
        selected = {n for n in args.models}
        models = [m for m in models if m["name"] in selected]
        if not models:
            raise ValueError(f"No models matched --models {args.models}")

    if not args.dry_run and not args.skip_env_check:
        if args.task_string is None:
            all_groups = sorted({group for entry in models for group in resolve_task_groups(entry, args.groups)})
            _validate_env_for_selection(models, all_groups, args.profile, args.tasks_config)
        else:
            _validate_env_for_task_string(models, args.task_string)

    logger.info("Running %d model(s) with profile=%s groups=%s", len(models), args.profile, args.groups)

    for entry in tqdm(models, desc="models"):
        groups = resolve_task_groups(entry, args.groups)
        task_string = args.task_string
        if task_string is None:
            task_string = build_task_list(
                groups=groups,
                profile=args.profile,
                tasks_config_path=args.tasks_config,
            )
        cmd = build_lighteval_command(
            entry=entry,
            task_string=task_string,
            output_dir=output_dir,
            max_samples=args.max_samples,
        )
        logger.info("Model %s: %s", entry["name"], " ".join(cmd))
        if args.dry_run:
            continue
        work_dir = output_dir / "_work" / entry["name"]
        work_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            cmd,
            check=True,
            cwd=work_dir,
            env=_subprocess_env(args.tasks_config, args.profile),
        )


if __name__ == "__main__":
    main()
