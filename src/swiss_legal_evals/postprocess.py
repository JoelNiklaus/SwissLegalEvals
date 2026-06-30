"""Strip reasoning traces from model outputs before judge / extractive scoring."""

from __future__ import annotations

import re

from lighteval.models.model_output import ModelResponse
from lighteval.utils.utils import remove_reasoning_tags

# Tag pairs removed before metric-specific extraction (in addition to lighteval CLI tags).
DEFAULT_REASONING_TAG_PAIRS: list[tuple[str, str]] = [
    ("<reasoning>", "</reasoning>"),
]

# gpt-oss / harmony-style final channel markers.
HARMONY_FINAL_MARKERS: tuple[str, ...] = (
    "assistantfinal",
    "assistant_final",
    "<|final|>",
)

# MCQ conclusion patterns used when the model never emits `###X###`.
_MCQ_CONCLUSION_PATTERN = re.compile(
    r"(?:"
    r"final\s+answer\s*[:\-]?\s*"
    r"|correct\s+(?:answer|choice|option)\s*(?:is|:)?\s*"
    r"|corresponds\s+to\s*"
    r"|matches\s*"
    r"|therefore,?\s*(?:the\s+)?(?:answer|choice)\s+is\s*"
    r")\**\s*([A-Z])\b",
    re.IGNORECASE,
)


def strip_model_reasoning(text: str, target_lang: str | None = None) -> str:
    """Return the answer-bearing portion of a model output.

    Reasoning models often emit long chain-of-thought before the actual answer.
    Judges and letter extractors must see only the final answer text.
    """
    if not text:
        return text

    cleaned = remove_reasoning_tags(text, DEFAULT_REASONING_TAG_PAIRS)

    for marker in HARMONY_FINAL_MARKERS:
        if marker in cleaned:
            cleaned = cleaned.split(marker)[-1]

    if cleaned.startswith("analysis"):
        cleaned = cleaned[len("analysis") :].lstrip()

    if target_lang is not None:
        cleaned = _extract_translation_continuation(cleaned, target_lang)

    return cleaned.strip()


def _extract_translation_continuation(text: str, target_lang: str) -> str:
    """For ``DE: ...\\nFR:`` prompts, keep the target-language continuation."""
    marker = f"{target_lang.upper()}:"
    if marker in text:
        return text.split(marker)[-1].strip()
    return text


def prediction_from_response(
    response: ModelResponse,
    target_lang: str | None = None,
) -> str:
    """Post-processed prediction text for judge or extractive metrics."""
    return strip_model_reasoning(response.final_text[0], target_lang=target_lang)


def lighteval_reasoning_tags_cli() -> str:
    """CLI value for lighteval --reasoning-tags (must include lighteval defaults)."""
    pairs: list[tuple[str, str]] = [
        ("<think>", "</think>"),
        *DEFAULT_REASONING_TAG_PAIRS,
    ]
    return repr(pairs)


def extract_mcq_letter_fallback(text: str, valid_choices: list[str]) -> str | None:
    """Extract an MCQ letter from reasoning-heavy outputs when ``###X###`` is missing."""
    stripped = strip_model_reasoning(text)
    matches = _MCQ_CONCLUSION_PATTERN.findall(stripped)
    if matches:
        letter = matches[-1].upper()
        if letter in valid_choices:
            return letter
    return None
