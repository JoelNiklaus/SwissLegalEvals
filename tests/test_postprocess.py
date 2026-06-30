"""Tests for model-output post-processing."""

from __future__ import annotations

from swiss_legal_evals.postprocess import extract_mcq_letter_fallback, strip_model_reasoning


def test_strip_harmony_assistantfinal() -> None:
    text = (
        "analysisWe need to translate.\n\n"
        "assistantfinalLe gouvernement fédéral accorde des aides financières."
    )
    assert strip_model_reasoning(text) == "Le gouvernement fédéral accorde des aides financières."


def test_extract_translation_after_lang_marker() -> None:
    text = "Here is my thinking.\nFR: Droits du fonctionnaire"
    assert strip_model_reasoning(text, target_lang="fr") == "Droits du fonctionnaire"


def test_mcq_letter_fallback_from_conclusion() -> None:
    text = "Long reasoning...\nThe correct answer is **B** because..."
    assert extract_mcq_letter_fallback(text, ["A", "B", "C", "D", "E"]) == "B"
