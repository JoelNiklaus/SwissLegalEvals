"""Tests for results aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from swiss_legal_evals.aggregate import family_mean_summary, parse_family, results_to_long_df


def test_parse_family() -> None:
    assert parse_family("slds:de_de|0") == "slds"
    assert parse_family("sdst-text_level:de-fr|0") == "sdst"
    assert parse_family("lexam_mcq_4_idk:en|0") == "lexam_mcq_4_idk"


def test_partial_rerun_supersedes_only_the_tasks_it_repeated(tmp_path: Path) -> None:
    """A re-run of one family must not drop the families it did not touch."""
    model_dir = tmp_path / "swiss-ai__Apertus-v1.5-70B"
    model_dir.mkdir(parents=True)
    (model_dir / "results_2026-07-28T00-00-00.json").write_text(
        json.dumps({"results": {"slds:de_de|0": {"judge": 70.0}, "lexam_mcq_4_idk:en|0": {"trad_score": 0.14}}})
    )
    (model_dir / "results_2026-07-31T00-00-00.json").write_text(
        json.dumps({"results": {"lexam_mcq_4_idk:en|0": {"trad_score": 0.58}}})
    )

    df = results_to_long_df(tmp_path)
    by_task = df.set_index("task")["value"]

    assert by_task["slds:de_de|0"] == 70.0
    assert by_task["lexam_mcq_4_idk:en|0"] == 0.58


def test_results_to_long_df(tmp_path: Path) -> None:
    model_dir = tmp_path / "deepseek-ai__DeepSeek-V4-Pro"
    model_dir.mkdir(parents=True)
    payload = {
        "results": {
            "slds:de_de|0": {"slds_judge_deepseek_v3.2": 72.5},
            "lexam_oq:en|0": {"lexam_oq_judge_deepseek_r1": 0.85},
        }
    }
    (model_dir / "results_2026.json").write_text(json.dumps(payload))

    df = results_to_long_df(tmp_path)
    assert len(df) == 2
    assert set(df["family"]) == {"slds", "lexam_oq"}


def test_results_rglob_nested(tmp_path: Path) -> None:
    nested = tmp_path / "org" / "my-model"
    nested.mkdir(parents=True)
    payload = {"results": {"slds:de_de|0": {"slds_judge_deepseek_v3.2": 1.0}}}
    (nested / "results_2026.json").write_text(json.dumps(payload))

    df = results_to_long_df(tmp_path)
    assert df.iloc[0]["model"] == "my-model"


def test_results_skip_lighteval_summary_rows(tmp_path: Path) -> None:
    model_dir = tmp_path / "org" / "my-model"
    model_dir.mkdir(parents=True)
    payload = {
        "results": {
            "slds:de_de|0": {"slds_judge_deepseek_v3.2": 10.0},
            "slds:_average|0": {"slds_judge_deepseek_v3.2": 10.0},
            "all": {"slds_judge_deepseek_v3.2": 10.0},
        }
    }
    (model_dir / "results_2026.json").write_text(json.dumps(payload))

    df = results_to_long_df(tmp_path)
    assert df["task"].tolist() == ["slds:de_de|0"]


def test_family_mean_summary() -> None:
    df = pd.DataFrame(
        [
            {"model": "a", "family": "slds", "task": "t1", "metric": "j", "value": 10.0},
            {"model": "a", "family": "slds", "task": "t2", "metric": "j", "value": 20.0},
        ]
    )
    summary = family_mean_summary(df)
    assert summary.iloc[0]["mean_value"] == pytest.approx(15.0)
