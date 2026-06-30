#!/usr/bin/env python3
"""Count evaluation samples per lighteval task in the default profile."""

from __future__ import annotations

import logging
from collections import defaultdict

from datasets import load_dataset
from lighteval.tasks.multilingual.tasks.swiss_legal.main import (
    LEXAM_LANGUAGES,
    LEXAM_REPO,
    SwissDecisionSummaryTranslations,
    SwissLandmarkDecisionHeadnotes,
    SwissLawTranslations,
    SwissSupremeCourtPressReleaseTranslations,
    get_slds_filter_fn,
)

from swiss_legal_evals.task_lists import (
    SWILTRA_SCOPED_LEVELS,
    build_task_list,
    lexam_mcq_task_names,
    lexam_oq_task_names,
    slds_task_names,
)

logger = logging.getLogger(__name__)


def _count_slds(subset: str) -> int:
    decision_lang, headnote_lang = subset.split("_")
    ds = load_dataset(
        SwissLandmarkDecisionHeadnotes.hf_repo,
        subset,
        split="test",
    )
    filt = get_slds_filter_fn(decision_lang, headnote_lang)
    return sum(1 for row in ds if filt(row))


def _count_translation(dataset, subset: str, source_lang: str, target_lang: str) -> int:
    src_col = f"{source_lang}_{dataset.subsets[subset].text_col_name}"
    ds = load_dataset(dataset.hf_repo, subset, split="test")
    return sum(1 for row in ds if row[src_col])


def _count_lexam_oq(language: str) -> int:
    ds = load_dataset(LEXAM_REPO, "open_question", split="test")
    return sum(1 for row in ds if row["language"] == language)


def _count_lexam_mcq(language: str, num_choices: int) -> int:
    ds = load_dataset(LEXAM_REPO, f"mcq_{num_choices}_choices", split="test")
    return sum(1 for row in ds if row["language"] == language)


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    task_counts: dict[str, int] = {}
    group_totals: dict[str, int] = defaultdict(int)

    for task in slds_task_names():
        subset = task.split(":")[1].split("|")[0]
        task_counts[task] = _count_slds(subset)
        group_totals["slds"] += task_counts[task]

    for dataset in (
        SwissDecisionSummaryTranslations,
        SwissLawTranslations,
        SwissSupremeCourtPressReleaseTranslations,
    ):
        for subset in dataset.subsets:
            if dataset.name not in SWILTRA_SCOPED_LEVELS:
                continue
            if subset != SWILTRA_SCOPED_LEVELS[dataset.name]:
                continue
            for source_lang, target_lang in dataset.translation_pairs:
                task = f"{dataset.name}-{subset}:{source_lang}-{target_lang}|0"
                task_counts[task] = _count_translation(dataset, subset, source_lang, target_lang)
                group_totals["swiltrabench_scoped"] += task_counts[task]

    for lang in LEXAM_LANGUAGES:
        task = f"lexam_oq:{lang}|0"
        task_counts[task] = _count_lexam_oq(lang)
        group_totals["lexam_oq"] += task_counts[task]

    for task in lexam_mcq_task_names(num_choices=[4, 8, 16], with_idk=[True]):
        base = task.split(":")[0]
        num_choices = int(base.removeprefix("lexam_mcq_").split("_")[0])
        lang = task.split(":")[1].split("|")[0]
        task_counts[task] = _count_lexam_mcq(lang, num_choices)
        group_totals["lexam_mcq"] += task_counts[task]

    default_tasks = build_task_list(
        groups=["slds", "swiltrabench", "lexam"],
        profile="default",
    ).split(",")

    print("=== Per-task sample counts (default profile) ===")
    for task in default_tasks:
        print(f"{task:45s} {task_counts[task]:>6,}")

    print("\n=== Group totals ===")
    for group, total in sorted(group_totals.items()):
        print(f"{group:25s} {total:>8,}")

    print(f"\n{'TOTAL (default profile)':25s} {sum(task_counts[t] for t in default_tasks):>8,}")

    # Full SwiLTra for reference
    full_swiltra = 0
    for dataset in (
        SwissDecisionSummaryTranslations,
        SwissLawTranslations,
        SwissSupremeCourtPressReleaseTranslations,
    ):
        for subset in dataset.subsets:
            for source_lang, target_lang in dataset.translation_pairs:
                full_swiltra += _count_translation(dataset, subset, source_lang, target_lang)
    print(f"{'swiltrabench_full (all levels)':25s} {full_swiltra:>8,}")


if __name__ == "__main__":
    main()
