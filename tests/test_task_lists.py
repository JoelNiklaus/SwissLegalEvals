"""Tests for task list construction."""

from __future__ import annotations

from pathlib import Path

import pytest

from swiss_legal_evals.task_lists import (
    DEFAULT_GENERATION_SIZES,
    build_task_list,
    lexam_mcq_task_names,
    load_generation_sizes,
    slds_task_names,
)

TASKS_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "tasks.yaml"


def test_slds_task_count() -> None:
    assert len(slds_task_names()) == 9


def test_scoped_swiltra_subset() -> None:
    tasks = build_task_list(
        groups=["swiltrabench"],
        profile="default",
        tasks_config_path=TASKS_CONFIG,
    )
    names = tasks.split(",")
    assert len(names) == 32
    assert all(
        "text_level" in n or "paragraph_level" in n or "press_release" in n for n in names
    )
    assert not any("bge_level" in n for n in names)


def test_swiltrabench_full_group_overrides_profile() -> None:
    """Group swiltrabench_full must include high-granularity levels even on default profile."""
    scoped = build_task_list(
        groups=["swiltrabench"],
        profile="default",
        tasks_config_path=TASKS_CONFIG,
    )
    full = build_task_list(
        groups=["swiltrabench_full"],
        profile="default",
        tasks_config_path=TASKS_CONFIG,
    )
    assert len(full.split(",")) > len(scoped.split(","))
    assert any("bge_level" in n for n in full.split(","))


def test_lexam_mcq_count() -> None:
    names = lexam_mcq_task_names()
    assert len(names) == 6
    assert all("_idk:" in name for name in names)
    assert not any("mcq_32" in name for name in names)


def test_unknown_profile_raises() -> None:
    with pytest.raises(KeyError):
        build_task_list(groups=["slds"], profile="nonexistent", tasks_config_path=TASKS_CONFIG)


def test_load_generation_sizes_from_profile() -> None:
    sizes = load_generation_sizes(tasks_config_path=TASKS_CONFIG, profile="default")
    assert sizes == DEFAULT_GENERATION_SIZES
    assert sizes["lexam"] == 32768
    assert sizes["swiltrabench"] == 32768


def test_load_generation_sizes_profile_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = tmp_path / "tasks.yaml"
    config.write_text(
        """
profiles:
  custom:
    slds:
      generation_size: 16384
    swiltrabench:
      scope: scoped
      generation_size: 16384
    lexam:
      open_question: true
      mcq: true
      generation_size: 16384
      mcq_num_choices: [4]
      mcq_with_idk: [true]
"""
    )
    sizes = load_generation_sizes(tasks_config_path=config, profile="custom")
    assert sizes["lexam"] == 16384
    assert sizes["slds"] == 16384


def test_tasks_table_uses_profile_generation_sizes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = tmp_path / "tasks.yaml"
    config.write_text(
        """
profiles:
  custom:
    slds:
      generation_size: 1234
    swiltrabench:
      scope: scoped
      generation_size: 5678
    lexam:
      open_question: true
      mcq: true
      generation_size: 9012
      mcq_num_choices: [4]
      mcq_with_idk: [true]
"""
    )
    monkeypatch.setenv("SWISSLEGALEVALS_TASKS_CONFIG", str(config))
    monkeypatch.setenv("SWISSLEGALEVALS_TASKS_PROFILE", "custom")

    import swiss_legal_evals.tasks as tasks_module

    tasks_module._TASKS_TABLE_CACHE = None
    table = tasks_module.build_tasks_table()

    slds_task = next(task for task in table if task.name.startswith("slds:"))
    assert slds_task.generation_size == 1234

    lexam_mcq_task = next(task for task in table if task.name.startswith("lexam_mcq_4:"))
    assert lexam_mcq_task.generation_size == 9012

    lexam_oq_task = next(task for task in table if task.name.startswith("lexam_oq:"))
    assert lexam_oq_task.generation_size == 9012

    swiltra_task = next(task for task in table if task.name.startswith("slt-paragraph_level:"))
    assert swiltra_task.generation_size >= 5678
