"""Tests for custom task metrics."""

from __future__ import annotations

import asyncio

import pytest
from huggingface_hub.inference._generated.types.chat_completion import (
    ChatCompletionStreamOutput,
    ChatCompletionStreamOutputChoice,
    ChatCompletionStreamOutputDelta,
)
from lighteval.metrics.utils.llm_as_judge import JudgeLM
from lighteval.tasks.requests import SamplingMethod
from lighteval.utils.cache_management import TaskID

from swiss_legal_evals.tasks import (
    _TRANSLATION_STOP_SEQUENCE_OVERRIDES,
    LEXAM_MCQ_GENERATION_SIZE,
    SLDS_GENERATION_SIZE,
    RobustJudgeSwissLandmarkDecisionSummarization,
    _apply_chat_completion_stream_chunk,
    _build_chat_completion_output_from_stream,
    _cache_task_name,
    _hf_choice_text,
    _select_existing_cache_hash,
)


def test_hf_choice_text_prefers_content_over_reasoning() -> None:
    content = "This is a complete answer that should be preferred over hidden reasoning."
    assert _hf_choice_text(content, "thought") == content


def test_hf_choice_text_keeps_reasoning_when_content_is_fragment() -> None:
    assert _hf_choice_text("frag", "full reasoning") == "full reasoning\nfrag"


def test_hf_choice_text_falls_back_to_reasoning() -> None:
    assert _hf_choice_text(None, "thought") == "thought"


def test_judge_inference_failure_becomes_empty_judgment() -> None:
    judge = object.__new__(JudgeLM)
    judge.API_MAX_RETRY = 0
    judge.model = "test/judge"
    judge.hf_provider = "novita"

    call = judge._JudgeLM__call_hf_inference

    assert asyncio.run(call([])) == ""


def test_existing_cache_hash_is_reused_for_equivalent_task() -> None:
    task_id = TaskID("lexam_oq:en|0", "legacy-hash", SamplingMethod.GENERATIVE)
    existing_indices = {task_id: list(range(10))}

    assert _cache_task_name("suite|lexam_oq:en|0") == "lexam_oq:en|0"
    assert _select_existing_cache_hash(existing_indices, "lexam_oq:en|0", "new-hash") == "legacy-hash"


def test_stream_chunk_accumulator_merges_content_and_reasoning() -> None:
    contents = [""]
    reasonings = [""]
    finish_reasons: list[str | None] = [None]
    chunk = ChatCompletionStreamOutput(
        id="cmpl-1",
        created=123,
        model="moonshotai/Kimi-K2.6",
        system_fingerprint="fp",
        choices=[
            ChatCompletionStreamOutputChoice(
                index=0,
                delta=ChatCompletionStreamOutputDelta(
                    role="assistant",
                    content="Hello",
                    reasoning="Think",
                ),
                finish_reason=None,
                logprobs=None,
            )
        ],
        usage=None,
    )
    _apply_chat_completion_stream_chunk(contents, reasonings, finish_reasons, chunk)
    chunk2 = ChatCompletionStreamOutput(
        id="cmpl-1",
        created=123,
        model="moonshotai/Kimi-K2.6",
        system_fingerprint="fp",
        choices=[
            ChatCompletionStreamOutputChoice(
                index=0,
                delta=ChatCompletionStreamOutputDelta(role="assistant", content=" world"),
                finish_reason="stop",
                logprobs=None,
            )
        ],
        usage=None,
    )
    _apply_chat_completion_stream_chunk(contents, reasonings, finish_reasons, chunk2)
    output = _build_chat_completion_output_from_stream(
        contents,
        reasonings,
        finish_reasons,
        model="moonshotai/Kimi-K2.6",
        response_id="cmpl-1",
        created=123,
    )
    assert output.choices[0].message.content == "Hello world"
    assert output.choices[0].message.reasoning == "Think"
    assert output.choices[0].finish_reason == "stop"
    assert (
        _hf_choice_text(
            output.choices[0].message.content,
            output.choices[0].message.reasoning,
        )
        == "Think\nHello world"
    )


def test_stream_chunk_accumulator_keeps_deepinfra_reasoning_content() -> None:
    contents = [""]
    reasonings = [""]
    finish_reasons: list[str | None] = [None]
    chunk = ChatCompletionStreamOutput.parse_obj_as_instance(
        {
            "id": "cmpl-1",
            "created": 123,
            "model": "stepfun-ai/Step-3.5-Flash",
            "system_fingerprint": "fp",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "Final Answer: ###A###",
                    },
                    "finish_reason": "stop",
                    "logprobs": None,
                }
            ],
            "usage": None,
        }
    )
    _apply_chat_completion_stream_chunk(contents, reasonings, finish_reasons, chunk)
    assert contents == [""]
    assert reasonings == ["Final Answer: ###A###"]
    assert finish_reasons == ["stop"]


def test_translation_stop_sequence_drops_bare_newline() -> None:
    assert _TRANSLATION_STOP_SEQUENCE_OVERRIDES["paragraph_level"] == ["</s>", "\n\n"]
    assert _TRANSLATION_STOP_SEQUENCE_OVERRIDES["text_level"] == ["</s>", "\n\n"]
    assert "\n" not in _TRANSLATION_STOP_SEQUENCE_OVERRIDES["paragraph_level"]


def test_lexam_mcq_generation_size() -> None:
    assert LEXAM_MCQ_GENERATION_SIZE == 32768


def test_slds_generation_size() -> None:
    assert SLDS_GENERATION_SIZE == 32768


def _strict_slds_judge() -> RobustJudgeSwissLandmarkDecisionSummarization:
    return object.__new__(RobustJudgeSwissLandmarkDecisionSummarization)


def test_strict_slds_judge_score_parser() -> None:
    response = "\n".join(
        [
            "ACCURACY_FAITHFULNESS_SCORE: 3",
            "COMPLETENESS_RELEVANCE_SCORE: 2",
            "CLARITY_COHERENCE_SCORE: 1",
            "ARTICLES_SCORE: 3",
            "CONSIDERATIONS_SCORE: 2",
        ]
    )
    assert _strict_slds_judge()._process_judge_response(response) == pytest.approx(0.6)


def test_strict_slds_judge_downgrades_out_of_range_score() -> None:
    # 453 is out of range and not a "[1-3]+garbage" pattern, so it is treated as
    # missing and downgraded to the lowest score (1) instead of aborting.
    response = "\n".join(
        [
            "ACCURACY_FAITHFULNESS_SCORE: 453",
            "COMPLETENESS_RELEVANCE_SCORE: 2",
            "CLARITY_COHERENCE_SCORE: 1",
            "ARTICLES_SCORE: 3",
            "CONSIDERATIONS_SCORE: 2",
        ]
    )
    # Scores become 1,2,1,3,2 -> (0+1+0+2+1)/10 = 0.4
    assert _strict_slds_judge()._process_judge_response(response) == pytest.approx(0.4)


def test_strict_slds_judge_accepts_long_numeric_garbage_after_valid_score() -> None:
    response = "\n".join(
        [
            "ACCURACY_FAITHFULNESS_SCORE: 3",
            "COMPLETENESS_RELEVANCE_SCORE:1694350242",
            "CLARITY_COHERENCE_SCORE: 1",
            "ARTICLES_SCORE: 3",
            "CONSIDERATIONS_SCORE: 2",
        ]
    )
    assert _strict_slds_judge()._process_judge_response(response) == pytest.approx(0.5)


def test_strict_slds_judge_accepts_compact_numeric_garbage_after_valid_score() -> None:
    response = "\n".join(
        [
            "ACCURACY_FAITHFULNESS_SCORE:191",
            "COMPLETENESS_RELEVANCE_SCORE:192",
            "CLARITY_COHERENCE_SCORE:191",
            "ARTICLES_SCORE:191",
            "CONSIDERATIONS_SCORE:191",
        ]
    )
    assert _strict_slds_judge()._process_judge_response(response) == pytest.approx(0.0)


def test_strict_slds_judge_accepts_valid_score_after_malformed_draft() -> None:
    response = "\n".join(
        [
            "ACCURACY_FAITHFULNESS_SCORE: 1",
            "COMPLETENESS_RELEVANCE_SCORE:'draft text",
            "COMPLETENESS_RELEVANCE_SCORE: 1",
            "CLARITY_COHERENCE_SCORE: 2",
            "ARTICLES_SCORE: 1",
            "CONSIDERATIONS_SCORE: 1",
        ]
    )
    assert _strict_slds_judge()._process_judge_response(response) == pytest.approx(0.1)


def test_strict_slds_judge_accepts_score_on_next_line() -> None:
    response = "\n".join(
        [
            "ACCURACY_FAITHFULNESS_SCORE: 1",
            "COMPLETENESS_RELEVANCE_SCORE: 1",
            "CLARITY_COHERENCE_SCORE: noisy draft",
            "1",
            "ARTICLES_SCORE: 1",
            "CONSIDERATIONS_SCORE: 1",
        ]
    )
    assert _strict_slds_judge()._process_judge_response(response) == pytest.approx(0.0)


def test_strict_slds_judge_assigns_lowest_score_for_malformed_candidate() -> None:
    response = "\n".join(
        [
            "ACCURACY_FAITHFULNESS_SCORE: บ",
            "COMPLETENESS_RELEVANCE_SCORE: 2",
            "CLARITY_COHERENCE_SCORE: Percentage: <start>0<end>",
            "ARTICLES_SCORE: 1",
            "CONSIDERATIONS_SCORE: 1",
        ]
    )
    assert _strict_slds_judge()._process_judge_response(response) == pytest.approx(0.1)


def test_strict_slds_judge_downgrades_missing_score() -> None:
    # A non-numeric score is unparseable, so that rubric is downgraded to 1.
    response = "\n".join(
        [
            "ACCURACY_FAITHFULNESS_SCORE: noise",
            "COMPLETENESS_RELEVANCE_SCORE: 2",
            "CLARITY_COHERENCE_SCORE: 1",
            "ARTICLES_SCORE: 3",
            "CONSIDERATIONS_SCORE: 2",
        ]
    )
    # Scores become 1,2,1,3,2 -> (0+1+0+2+1)/10 = 0.4
    assert _strict_slds_judge()._process_judge_response(response) == pytest.approx(0.4)


def test_strict_slds_judge_downgrades_corrupted_rubric_name() -> None:
    # The judge emitted a corrupted rubric name; the canonical rubric is missing
    # and downgraded to the lowest score rather than aborting the run.
    response = "\n".join(
        [
            "ACCURACY_aithfulness_SCORE: 3",
            "COMPLETENESS_RELEVANCE_SCORE: 2",
            "CLARITY_COHERENCE_SCORE: 1",
            "ARTICLES_SCORE: 3",
            "CONSIDERATIONS_SCORE: 2",
        ]
    )
    # ACCURACY_FAITHFULNESS_SCORE missing -> 1; scores 1,2,1,3,2 -> 0.4
    assert _strict_slds_judge()._process_judge_response(response) == pytest.approx(0.4)


def test_strict_slds_judge_accepts_terminal_score_text() -> None:
    response = "\n".join(
        [
            "ACCURACY_FAITHFULNESS_SCORE: more on 2",
            "COMPLETENESS_RELEVANCE_SCORE: 2",
            "CLARITY_COHERENCE_SCORE: 2",
            "ARTICLES_SCORE: 2",
            "CONSIDERATIONS_SCORE: 2",
        ]
    )
    assert _strict_slds_judge()._process_judge_response(response) == pytest.approx(0.5)
