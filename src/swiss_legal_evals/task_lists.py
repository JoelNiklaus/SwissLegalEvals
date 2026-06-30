"""Build comma-separated lighteval task strings from configs/tasks.yaml."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from lighteval.tasks.multilingual.tasks.swiss_legal.main import (
    LEXAM_LANGUAGES,
    SwissDecisionSummaryTranslations,
    SwissLandmarkDecisionHeadnotes,
    SwissLawTranslations,
    SwissSupremeCourtPressReleaseTranslations,
)

SWILTRA_SCOPED_LEVELS: dict[str, str] = {
    "sdst": "text_level",
    "slt": "paragraph_level",
    "sscprt": "press_release",
}

TRANSLATION_DATASETS = [
    SwissDecisionSummaryTranslations,
    SwissLawTranslations,
    SwissSupremeCourtPressReleaseTranslations,
]

logger = logging.getLogger(__name__)

DEFAULT_TASKS_CONFIG = Path.cwd() / "configs" / "tasks.yaml"
FEWSHOT_SEED_SUFFIX = "|0"
DEFAULT_LEXAM_MCQ_NUM_CHOICES = [4, 8, 16]

GenerationSizeKey = Literal["slds", "swiltrabench", "lexam"]
GENERATION_SIZE_KEYS: tuple[GenerationSizeKey, ...] = (
    "slds",
    "swiltrabench",
    "lexam",
)
DEFAULT_GENERATION_SIZES: dict[GenerationSizeKey, int] = {
    "slds": 32768,
    "swiltrabench": 32768,
    "lexam": 32768,
}


def _task_enabled(task_cfg: Any) -> bool:
    """Return whether a task group is enabled in the profile."""
    return task_cfg is not None and task_cfg is not False


def _task_generation_size(task_cfg: Any, default: int) -> int:
    """Read generation_size from a nested task block in tasks.yaml."""
    if isinstance(task_cfg, dict) and "generation_size" in task_cfg:
        return int(task_cfg["generation_size"])
    return default


def _swiltra_scope(profile_cfg: dict[str, Any]) -> str:
    """Return SwiLTra scope (`scoped` or `full`) from the profile."""
    swiltra_cfg = profile_cfg["swiltrabench"]
    if isinstance(swiltra_cfg, str):
        return swiltra_cfg
    return swiltra_cfg["scope"]


def _task_with_seed(name: str) -> str:
    return f"{name}{FEWSHOT_SEED_SUFFIX}"


def slds_task_names() -> list[str]:
    return [
        _task_with_seed(f"{SwissLandmarkDecisionHeadnotes.name}:{subset}")
        for subset in SwissLandmarkDecisionHeadnotes.subsets
    ]


def _swiltra_level_filter(group: str, profile_cfg: dict[str, Any]) -> dict[str, str] | None:
    """Return level filter for SwiLTra, or None for all levels."""
    if group == "swiltrabench_full":
        return None
    scope = _swiltra_scope(profile_cfg)
    if scope == "full":
        return None
    return SWILTRA_SCOPED_LEVELS


def _swiltra_task_names(level_filter: dict[str, str] | None) -> list[str]:
    names: list[str] = []
    for dataset in TRANSLATION_DATASETS:
        for subset in dataset.subsets:
            if level_filter is not None:
                if dataset.name not in level_filter:
                    continue
                if subset != level_filter[dataset.name]:
                    continue
            for source_lang, target_lang in dataset.translation_pairs:
                names.append(
                    _task_with_seed(
                        f"{dataset.name}-{subset}:{source_lang}-{target_lang}"
                    )
                )
    return names


def lexam_oq_task_names() -> list[str]:
    return [_task_with_seed(f"lexam_oq:{lang}") for lang in LEXAM_LANGUAGES]


def lexam_mcq_task_names(
    num_choices: list[int] | None = None,
    with_idk: list[bool] | None = None,
) -> list[str]:
    choices = num_choices if num_choices is not None else DEFAULT_LEXAM_MCQ_NUM_CHOICES
    idk_flags = with_idk if with_idk is not None else [True]
    names: list[str] = []
    for lang in LEXAM_LANGUAGES:
        for n in choices:
            for use_idk in idk_flags:
                suffix = "_idk" if use_idk else ""
                names.append(_task_with_seed(f"lexam_mcq_{n}{suffix}:{lang}"))
    return names


def tasks_config_path() -> Path:
    """Return the active tasks.yaml path (env override or project default)."""
    if "SWISSLEGALEVALS_TASKS_CONFIG" in os.environ:
        return Path(os.environ["SWISSLEGALEVALS_TASKS_CONFIG"])
    return DEFAULT_TASKS_CONFIG


def tasks_profile_name() -> str:
    """Return the active task profile name for generation-size overrides."""
    return os.environ.get("SWISSLEGALEVALS_TASKS_PROFILE", "default")


def load_tasks_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or tasks_config_path()
    logger.info("Loading tasks config from %s", config_path)
    with config_path.open() as f:
        return yaml.safe_load(f)


def load_generation_sizes(
    tasks_config_path: Path | None = None,
    profile: str | None = None,
) -> dict[GenerationSizeKey, int]:
    """Load per-benchmark generation caps from the active tasks.yaml profile."""
    config = load_tasks_config(tasks_config_path)
    profile_name = profile or tasks_profile_name()
    profiles = config["profiles"]
    if profile_name not in profiles:
        raise KeyError(f"Unknown profile {profile_name!r}; available: {list(profiles)}")

    profile_cfg = profiles[profile_name]
    return {
        key: _task_generation_size(profile_cfg.get(key), DEFAULT_GENERATION_SIZES[key])
        for key in GENERATION_SIZE_KEYS
    }


def build_task_list(
    groups: list[str],
    profile: str = "default",
    tasks_config_path: Path | None = None,
) -> str:
    """Return comma-separated task names for lighteval."""
    config = load_tasks_config(tasks_config_path)
    profiles = config["profiles"]
    if profile not in profiles:
        raise KeyError(f"Unknown profile {profile!r}; available: {list(profiles)}")

    profile_cfg = profiles[profile]
    task_names: list[str] = []

    for group in groups:
        if group == "slds":
            if _task_enabled(profile_cfg.get("slds")):
                task_names.extend(slds_task_names())
        elif group in ("swiltrabench", "swiltrabench_full"):
            level_filter = _swiltra_level_filter(group, profile_cfg)
            task_names.extend(_swiltra_task_names(level_filter))
        elif group == "lexam":
            lexam_cfg = profile_cfg["lexam"]
            if lexam_cfg["open_question"]:
                task_names.extend(lexam_oq_task_names())
            if lexam_cfg["mcq"]:
                task_names.extend(
                    lexam_mcq_task_names(
                        num_choices=lexam_cfg["mcq_num_choices"],
                        with_idk=lexam_cfg["mcq_with_idk"],
                    )
                )
        else:
            raise ValueError(f"Unknown task group: {group}")

    if not task_names:
        raise ValueError(f"No tasks selected for groups={groups} profile={profile}")

    return ",".join(task_names)
