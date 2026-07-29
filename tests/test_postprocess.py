"""Tests for model-output post-processing."""

from __future__ import annotations

from swiss_legal_evals.postprocess import extract_mcq_letter_fallback, strip_model_reasoning


def test_strip_harmony_assistantfinal() -> None:
    text = "analysisWe need to translate.\n\nassistantfinalLe gouvernement fédéral accorde des aides financières."
    assert strip_model_reasoning(text) == "Le gouvernement fédéral accorde des aides financières."


def test_strip_orphan_closing_think_tag() -> None:
    """Templates that pre-seed `<think>` leave only the closing tag in the output."""
    text = "Here's a thinking process:\n1. Analyze the request.\n</think>\n\n**Leitsatz**\n\nDie Haftung entfällt."
    assert strip_model_reasoning(text) == "**Leitsatz**\n\nDie Haftung entfällt."


def test_orphan_rule_skipped_when_opening_tag_present() -> None:
    """Paired traces are lighteval's job; the orphan rule must not touch them."""
    text = "<think>step one</think>Final answer"
    assert strip_model_reasoning(text) == text


def test_orphan_strip_keeps_plain_text_untouched() -> None:
    assert strip_model_reasoning("Ein Leitsatz ohne Reasoning.") == "Ein Leitsatz ohne Reasoning."


def test_extract_translation_after_lang_marker() -> None:
    text = "Here is my thinking.\nFR: Droits du fonctionnaire"
    assert strip_model_reasoning(text, target_lang="fr") == "Droits du fonctionnaire"


def test_mcq_letter_fallback_from_conclusion() -> None:
    text = "Long reasoning...\nThe correct answer is **B** because..."
    assert extract_mcq_letter_fallback(text, ["A", "B", "C", "D", "E"]) == "B"
