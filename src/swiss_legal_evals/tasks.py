"""Custom lighteval tasks: judge-only metrics with configurable judge backends."""

from __future__ import annotations

import hashlib
import logging
import os
import random
import re
import statistics
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
import yaml
from lighteval.metrics.utils.metric_utils import SampleLevelMetricGrouping
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.tasks.swiss_legal.main import (
    LEXAM_LANGUAGES,
    LEXAM_MCQ_NUM_CHOICES,
    LEXAM_REPO,
    LEXAM_STOP_SEQUENCES,
    HeadnoteGenerationTask,
    LEXamMCQTask,
    SwissDecisionSummaryTranslations,
    SwissLandmarkDecisionHeadnotes,
    SwissLawTranslations,
    SwissSupremeCourtPressReleaseTranslations,
    _build_lexam_mcq_prompt_fn,
    _lexam_language_filter,
    create_translation_prompt_fn,
    lexam_oq_prompt_fn,
)
from lighteval.tasks.multilingual.tasks.swiss_legal.metrics import (
    LEXAM_OQ_JUDGE_INSTRUCTION,
    LEXAM_OQ_JUDGE_SYSTEM_PROMPT,
    LEXAM_OQ_JUDGE_USER_PROMPT,
    SWISS_LEGAL_TRANSLATION_JUDGE_FEW_SHOT_EXAMPLES,
    SWISS_LEGAL_TRANSLATION_JUDGE_INSTRUCTION,
    SWISS_LEGAL_TRANSLATION_JUDGE_SYSTEM_PROMPT,
    SWISS_LEGAL_TRANSLATION_JUDGE_USER_PROMPT,
    JudgeLEXamOQ,
    JudgeSwissLandmarkDecisionSummarization,
    JudgeSwissLegalTranslation,
    LEXamMCQExtractive,
    process_judge_response_freeform_gpt,
)
from lighteval.tasks.requests import Doc, SamplingMethod
from transformers import AutoTokenizer

from swiss_legal_evals.cache_patch import enable_incremental_caching
from swiss_legal_evals.postprocess import (
    extract_mcq_letter_fallback,
    prediction_from_response,
    strip_model_reasoning,
)
from swiss_legal_evals.providers import hf_org_to_bill
from swiss_legal_evals.task_lists import DEFAULT_GENERATION_SIZES, load_generation_sizes

logger = logging.getLogger(__name__)


def _enable_trust_remote_code_tokenizers() -> None:
    """Default ``AutoTokenizer.from_pretrained`` to ``trust_remote_code=True``.

    lighteval's inference-providers client loads each model's tokenizer without
    exposing ``trust_remote_code``. Some hosted models (e.g. Kimi-K2.6) ship a
    custom tokenizer, so the default ``trust_remote_code=False`` aborts the run.
    We trust the Hub repos we explicitly evaluate, so opt in globally here.

    """
    original_from_pretrained = AutoTokenizer.from_pretrained

    def from_pretrained(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("trust_remote_code", True)
        return original_from_pretrained(*args, **kwargs)

    AutoTokenizer.from_pretrained = from_pretrained  # type: ignore[method-assign]


_enable_trust_remote_code_tokenizers()


def _hf_choice_text(content: str | None, reasoning: str | None) -> str:
    """Return model text from HF inference-providers, including reasoning-only responses."""
    if content and reasoning and len(content.strip()) < 50:
        return f"{reasoning}\n{content}"
    if content:
        return content
    if reasoning:
        return reasoning
    return ""


INFERENCE_PROVIDERS_API_MAX_RETRY = 12


JUDGE_API_MAX_RETRY = 12


def _cache_task_name(full_task_name: str) -> str:
    """Return the full task identifier used by lighteval cache directories."""
    parts = full_task_name.split("|")
    if len(parts) == 3:
        return f"{parts[1]}|{parts[2]}"
    return full_task_name


def _select_existing_cache_hash(
    existing_indices: Mapping[Any, list[int]],
    task_name: str,
    current_hash: str,
) -> str:
    """Select a previously loaded cache hash when the current hash is absent."""
    candidates = [
        (task_id.task_hash, len(sample_ids))
        for task_id, sample_ids in existing_indices.items()
        if task_id.task_name == task_name
    ]
    if not candidates or any(task_hash == current_hash for task_hash, _ in candidates):
        return current_hash
    return max(candidates, key=lambda candidate: (candidate[1], candidate[0]))[0]


def _apply_chat_completion_stream_chunk(
    contents: list[str],
    reasonings: list[str],
    finish_reasons: list[str | None],
    chunk: Any,
) -> None:
    """Accumulate one ``ChatCompletionStreamOutput`` chunk into per-choice buffers."""
    for choice in chunk.choices:
        idx = choice.index
        if idx >= len(contents):
            raise ValueError(
                f"Stream choice index {idx} exceeds expected num_samples={len(contents)}"
            )
        if choice.delta.content:
            contents[idx] += choice.delta.content
        reasoning_delta = choice.delta.reasoning or getattr(choice.delta, "reasoning_content", None)
        if reasoning_delta:
            reasonings[idx] += reasoning_delta
        if choice.finish_reason:
            finish_reasons[idx] = choice.finish_reason


def _build_chat_completion_output_from_stream(
    contents: list[str],
    reasonings: list[str],
    finish_reasons: list[str | None],
    *,
    model: str,
    response_id: str,
    created: int,
) -> Any:
    """Build a non-streaming ``ChatCompletionOutput`` from accumulated stream deltas."""
    from huggingface_hub.inference._generated.types.chat_completion import (
        ChatCompletionOutput,
        ChatCompletionOutputComplete,
        ChatCompletionOutputMessage,
    )

    choices = []
    for idx, (content, reasoning, finish_reason) in enumerate(
        zip(contents, reasonings, finish_reasons, strict=True)
    ):
        choices.append(
            ChatCompletionOutputComplete(
                index=idx,
                message=ChatCompletionOutputMessage(
                    role="assistant",
                    content=content or None,
                    reasoning=reasoning or None,
                ),
                finish_reason=finish_reason or "stop",
            )
        )
    return ChatCompletionOutput(
        choices=choices,
        id=response_id or "stream",
        created=created,
        model=model,
        system_fingerprint=None,
        usage=None,
    )


async def _consume_chat_completion_stream(stream: Any, num_samples: int) -> Any:
    """Read an async chat-completion stream and return a full ``ChatCompletionOutput``."""
    contents = [""] * num_samples
    reasonings = [""] * num_samples
    finish_reasons: list[str | None] = [None] * num_samples
    model = ""
    response_id = ""
    created = 0
    async for chunk in stream:
        if chunk.model:
            model = chunk.model
        if chunk.id:
            response_id = chunk.id
        created = chunk.created
        _apply_chat_completion_stream_chunk(contents, reasonings, finish_reasons, chunk)
    return _build_chat_completion_output_from_stream(
        contents,
        reasonings,
        finish_reasons,
        model=model,
        response_id=response_id,
        created=created,
    )


def _patch_inference_providers_reasoning_field() -> None:
    """Harden HF inference-providers for slow reasoning models (e.g. Kimi).

    - Stream chat completions so the HF router connection stays alive on long CoT gens.
    - Copy ``message.reasoning`` into ``content`` when the hosted API leaves content empty.
    - Retry individual failed samples after the parallel batch instead of aborting the run.
    - Bump per-request retry budget (lighteval default is 5).

    Must patch name-mangled ``__call_api*`` hooks on the class; module-level replacements
    break double-underscore attribute lookup.
    """
    import asyncio

    from lighteval.models.endpoints.inference_providers_model import InferenceProvidersClient
    from tqdm.asyncio import tqdm as async_tqdm

    if getattr(InferenceProvidersClient, "_swisslegal_reasoning_patch", False):
        return

    from huggingface_hub import AsyncInferenceClient, ChatCompletionOutput
    from lighteval.models.endpoints.inference_providers_model import InferenceProvidersModelConfig
    from lighteval.tasks.prompt_manager import PromptManager
    from lighteval.utils.cache_management import SampleCache, cached

    _original_call_api = InferenceProvidersClient._InferenceProvidersClient__call_api

    async def __call_api_streaming(
        self: InferenceProvidersClient,
        prompt: list[dict[str, str]],
        num_samples: int,
        generation_size: int | None = None,
    ) -> ChatCompletionOutput | None:
        """Call HF inference-providers with streaming for single-sample generations."""
        if num_samples > 1:
            return await _original_call_api(self, prompt, num_samples)

        for attempt in range(self.API_MAX_RETRY):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model_name,
                    "messages": prompt,
                    "n": num_samples,
                    "stream": True,
                }
                kwargs.update(self.generation_parameters.to_inference_providers_dict())
                if generation_size is not None and "max_tokens" not in kwargs:
                    kwargs["max_tokens"] = generation_size
                stream = await self.client.chat.completions.create(**kwargs)
                response = await _consume_chat_completion_stream(stream, num_samples)
                if not any(
                    _hf_choice_text(choice.message.content, getattr(choice.message, "reasoning", None)).strip()
                    for choice in response.choices
                ):
                    raise ValueError("Streaming inference-providers response contained no text")
                return response
            except Exception as exc:
                wait_time = min(64, self.API_RETRY_SLEEP * (2**attempt))
                logger.warning(
                    "Error in streaming inference-providers API call: %s, waiting %ds before retry %d/%d",
                    exc,
                    wait_time,
                    attempt + 1,
                    self.API_MAX_RETRY,
                )
                await asyncio.sleep(wait_time)

        logger.error(
            "Streaming inference-providers API call failed after %d attempts, returning empty response.",
            self.API_MAX_RETRY,
        )
        return None

    def __init__(self: InferenceProvidersClient, config: InferenceProvidersModelConfig) -> None:
        """Mirror lighteval init but tolerate tokenizer load failures (e.g. Step-3.5-Flash)."""
        self.config = config
        self.model_name = config.model_name
        self.provider = config.provider
        self.generation_parameters = config.generation_parameters
        self.API_MAX_RETRY = INFERENCE_PROVIDERS_API_MAX_RETRY
        self.API_RETRY_SLEEP = 3
        self.API_RETRY_MULTIPLIER = 2
        self.pairwise_tokenization = False
        self.parallel_calls_count = config.parallel_calls_count
        self.client = AsyncInferenceClient(
            provider=self.provider,
            timeout=config.timeout,
            proxies=config.proxies,
            bill_to=config.org_to_bill,
        )
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        except Exception as exc:
            logger.warning(
                "Could not load model's tokenizer for %s (%s); continuing without tokenizer",
                self.model_name,
                exc,
            )
            self._tokenizer = None
        self.prompt_manager = PromptManager(
            use_chat_template=True,
            tokenizer=self.tokenizer,
            system_prompt=config.system_prompt,
        )
        self._cache = SampleCache(config)

    async def __call_api_parallel_with_reasoning(
        self: InferenceProvidersClient,
        prompts: list[list[dict[str, str]]],
        num_samples: int | list[int],
        generation_sizes: list[int | None] | None = None,
        docs: list[Doc] | None = None,
    ) -> list[Any]:
        semaphore = asyncio.Semaphore(self.parallel_calls_count)
        num_sampless = [num_samples for _ in prompts] if not isinstance(num_samples, list) else num_samples
        if len(prompts) != len(num_sampless):
            raise ValueError(
                f"Length of prompts and num_samples should match: {len(prompts)} vs {len(num_sampless)}"
            )
        generation_sizess = generation_sizes or [None for _ in prompts]
        if len(prompts) != len(generation_sizess):
            raise ValueError(
                f"Length of prompts and generation_sizes should match: {len(prompts)} vs {len(generation_sizess)}"
            )

        async def bounded_api_call(
            prompt: list[dict[str, str]],
            sample_count: int,
            generation_size: int | None,
        ) -> Any:
            async with semaphore:
                return await __call_api_streaming(self, prompt, sample_count, generation_size)

        def empty_response(sample_count: int) -> Any:
            """Return an OpenAI-compatible empty completion for irrecoverable provider failures."""
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content="", reasoning=None))
                    for _ in range(sample_count)
                ]
            )

        def doc_label(idx: int) -> str:
            if docs is None or idx >= len(docs):
                return "doc=<unknown>"
            doc = docs[idx]
            return f"task={doc.task_name} doc_id={doc.id}"

        results: list[Any] = list(
            await async_tqdm.gather(
                *[
                    bounded_api_call(prompt, sample_count, generation_size)
                    for prompt, sample_count, generation_size in zip(
                        prompts,
                        num_sampless,
                        generation_sizess,
                        strict=True,
                    )
                ]
            )
        )

        for idx, result in enumerate(results):
            if result is not None:
                continue
            sample_count = num_sampless[idx]
            logger.error(
                "Accepting empty generation after exhausted streaming retries for request %d/%d (%s).",
                idx + 1,
                len(prompts),
                doc_label(idx),
            )
            results[idx] = empty_response(sample_count)

        for response in results:
            for choice in response.choices:
                choice.message.content = _hf_choice_text(
                    choice.message.content,
                    getattr(choice.message, "reasoning", None),
                )
        return results

    def greedy_until_with_generation_size(
        self: InferenceProvidersClient,
        docs: list[Doc],
    ) -> list[ModelResponse]:
        """Generate with lighteval doc-level generation sizes on HF inference providers."""
        from lighteval.data import GenerativeTaskDataset
        from tqdm import tqdm

        dataset = GenerativeTaskDataset(requests=docs, num_dataset_splits=self.DATASET_SPLITS)
        results = []
        for split in tqdm(
            dataset.splits_iterator(),
            total=dataset.num_dataset_splits,
            desc="Splits",
            position=0,
            disable=False,
        ):
            contexts = [self.prompt_manager.prepare_prompt_api(doc) for doc in split]
            num_samples = split[0].num_samples
            generation_sizes = [doc.generation_size for doc in split]
            responses = asyncio.run(
                self._InferenceProvidersClient__call_api_parallel(
                    contexts,
                    num_samples,
                    generation_sizes,
                    split,
                )
            )
            for response, context in zip(responses, contexts, strict=True):
                result: list[str] = [choice.message.content for choice in response.choices]
                results.append(ModelResponse(text=result if result[0] else [""], input=context))
        return dataset.get_original_order(results)

    InferenceProvidersClient.__init__ = __init__  # type: ignore[method-assign]
    InferenceProvidersClient._InferenceProvidersClient__call_api = (  # type: ignore[attr-defined]
        __call_api_streaming
    )
    InferenceProvidersClient._InferenceProvidersClient__call_api_parallel = (  # type: ignore[attr-defined]
        __call_api_parallel_with_reasoning
    )
    InferenceProvidersClient.greedy_until = cached(SamplingMethod.GENERATIVE)(  # type: ignore[method-assign]
        greedy_until_with_generation_size
    )
    InferenceProvidersClient._swisslegal_reasoning_patch = True  # type: ignore[attr-defined]


def _patch_judge_hf_org_billing() -> None:
    """Bill HF inference-providers judges to an org when HF_ORG_TO_BILL is set."""
    from lighteval.metrics.utils.llm_as_judge import JudgeLM

    if getattr(JudgeLM, "_swisslegal_hf_org_billing_patch", False):
        return

    original_lazy = JudgeLM._JudgeLM__lazy_load_client

    def __lazy_load_client(self: Any) -> Any:  # noqa: N807
        org = hf_org_to_bill()
        if self.backend == "inference-providers" and org is not None:
            from huggingface_hub import AsyncInferenceClient

            self.client = AsyncInferenceClient(
                token=self.api_key,
                base_url=self.url,
                provider=self.hf_provider,
                bill_to=org,
            )
            return self.__call_hf_inference_async
        return original_lazy(self)

    JudgeLM._JudgeLM__lazy_load_client = __lazy_load_client  # type: ignore[method-assign]
    JudgeLM._swisslegal_hf_org_billing_patch = True  # type: ignore[attr-defined]


def _patch_judge_inference_provider_retries() -> None:
    """Keep one failed HF judge request from aborting the whole evaluation.

    The installed lighteval implementation raises after its retry budget is
    exhausted. Since judge metrics are batched, that exception discards all
    already-computed scores. Returning an empty judgment lets the metric parser
    record a zero/missing score for only that sample and continue.
    """
    from lighteval.metrics.utils.llm_as_judge import JudgeLM

    if getattr(JudgeLM, "_swisslegal_judge_retry_patch", False):
        return

    original_call = JudgeLM._JudgeLM__call_hf_inference

    async def __call_hf_inference_with_fallback(
        self: Any,
        prompt: list[dict[str, str]],
    ) -> str:
        try:
            response = await original_call(self, prompt)
        except Exception as exc:
            logger.error(
                "Judge inference failed after %d retries for model=%s provider=%s; "
                "recording a missing judgment: %s",
                self.API_MAX_RETRY,
                self.model,
                self.hf_provider,
                exc,
            )
            return ""
        if response is None:
            logger.error(
                "Judge inference returned no response for model=%s provider=%s; "
                "recording a missing judgment.",
                self.model,
                self.hf_provider,
            )
            return ""
        return response

    JudgeLM._JudgeLM__call_hf_inference = (  # type: ignore[attr-defined]
        __call_hf_inference_with_fallback
    )
    JudgeLM._swisslegal_judge_retry_patch = True  # type: ignore[attr-defined]


def _patch_lighteval_cache_hash_reuse() -> None:
    """Reuse loaded sample caches when lighteval task hashes change between runs.

    lighteval includes object representations from custom metrics in its task
    hash, so equivalent task definitions can receive different hashes in new
    Python processes. Existing cache files are already validated while loading;
    reuse the fullest prior hash and let lighteval generate only missing sample
    IDs.
    """
    from lighteval.utils.cache_management import SampleCache

    if getattr(SampleCache, "_swisslegal_cache_hash_patch", False):
        return

    original_get_task_hash = SampleCache._get_task_hash

    def _get_task_hash_with_reuse(self: Any, full_task_name: str) -> str:
        current_hash = original_get_task_hash(self, full_task_name)
        task_name = _cache_task_name(full_task_name)
        reusable_hash = _select_existing_cache_hash(
            self.existing_indices,
            task_name,
            current_hash,
        )
        logged_tasks = getattr(self, "_swisslegal_cache_reuse_logged", set())
        if reusable_hash != current_hash and task_name not in logged_tasks:
            logger.info(
                "[CACHING] Reusing existing hash %s for task %s instead of current hash %s.",
                reusable_hash,
                task_name,
                current_hash,
            )
            logged_tasks.add(task_name)
            self._swisslegal_cache_reuse_logged = logged_tasks
        return reusable_hash

    SampleCache._get_task_hash = _get_task_hash_with_reuse  # type: ignore[method-assign]
    SampleCache._swisslegal_cache_hash_patch = True  # type: ignore[attr-defined]


def _patch_deterministic_mcq_shuffle() -> None:
    """Derive each LEXam MCQ option order from the question, not from a shared RNG stream.

    lighteval shuffles the substantive choices with one module-level ``random.Random``,
    so the permutation a question receives depends on how many questions were shuffled
    before it in that process. Scoring cached generations in a fresh process therefore
    re-rolls the letters and marks correct answers wrong: it silently invalidated the
    MCQ scores of the two runs that resumed from a cache. Seeding from the choice texts
    makes the order identical in every process, whatever the task order or sample count.
    """
    from lighteval.tasks.multilingual.tasks.swiss_legal import main as swiss_legal_main

    if getattr(swiss_legal_main, "_swisslegal_deterministic_shuffle", False):
        return

    class _PerQuestionShuffler:
        """Stands in for the module RNG; the prompt function only calls ``shuffle``."""

        def shuffle(self, choices: list[str]) -> None:
            digest = hashlib.sha256("\x1f".join(map(str, choices)).encode()).hexdigest()
            random.Random(digest).shuffle(choices)

    swiss_legal_main._LEXAM_RNG = _PerQuestionShuffler()
    swiss_legal_main._swisslegal_deterministic_shuffle = True


_patch_inference_providers_reasoning_field()
_patch_judge_hf_org_billing()
_patch_judge_inference_provider_retries()
_patch_lighteval_cache_hash_reuse()
_patch_deterministic_mcq_shuffle()
# Must come last: chunking has to wrap the fully patched `greedy_until`, so every
# chunk still goes through the reasoning-field handling and the cache decorator.
enable_incremental_caching()

JudgeProvider = Literal["openai", "openrouter", "hf-inference-providers"]
JudgeBackend = Literal["litellm", "inference-providers"]

LEXAM_OQ_JUDGE_MAX_TOKENS = 32768
SLDS_JUDGE_MAX_TOKENS = 4096
SLDS_GENERATION_SIZE = DEFAULT_GENERATION_SIZES["slds"]
SWILTRABENCH_GENERATION_SIZE = DEFAULT_GENERATION_SIZES["swiltrabench"]
LEXAM_GENERATION_SIZE = DEFAULT_GENERATION_SIZES["lexam"]
# Backwards-compatible aliases used in tests and docs.
REASONING_MODEL_MIN_GENERATION_SIZE = SWILTRABENCH_GENERATION_SIZE
LEXAM_MCQ_GENERATION_SIZE = LEXAM_GENERATION_SIZE
LEXAM_OQ_GENERATION_SIZE = LEXAM_GENERATION_SIZE

# Upstream SwiLTra configs stop on a bare ``\n``, which truncates paragraph translations
# to a single line. Use paragraph/sentence boundaries instead.
_TRANSLATION_STOP_SEQUENCE_OVERRIDES: dict[str, list[str]] = {
    "paragraph_level": ["</s>", "\n\n"],
    "text_level": ["</s>", "\n\n"],
}

DEFAULT_JUDGE_CONFIG = Path.cwd() / "configs" / "judges.yaml"

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


class RobustLEXamMCQExtractive(LEXamMCQExtractive):
    """LEXam MCQ scorer that strips reasoning traces before letter extraction."""

    def _extract_letter(self, pred: str, doc: Doc) -> str | None:
        stripped = strip_model_reasoning(pred)
        letter = super()._extract_letter(stripped, doc)
        if letter is not None:
            return letter
        return extract_mcq_letter_fallback(stripped, doc.choices)


def get_robust_lexam_mcq_metric(with_idk: bool) -> SampleLevelMetricGrouping:
    """Build the LEXam MCQ metric with reasoning-aware letter extraction."""
    if with_idk:
        metric_names = ["trad_score", "idk_score", "idk_freq", "extract_fail"]
        higher_is_better = {
            "trad_score": True,
            "idk_score": True,
            "idk_freq": False,
            "extract_fail": False,
        }
    else:
        metric_names = ["acc", "extract_fail"]
        higher_is_better = {"acc": True, "extract_fail": False}

    return SampleLevelMetricGrouping(
        metric_name=metric_names,
        higher_is_better=higher_is_better,
        category=SamplingMethod.GENERATIVE,
        sample_level_fn=RobustLEXamMCQExtractive(with_idk=with_idk),
        corpus_level_fn=dict.fromkeys(metric_names, np.mean),
        batched_compute=False,
    )


class RobustJudgeSwissLandmarkDecisionSummarization(JudgeSwissLandmarkDecisionSummarization):
    """SLDS judge that tolerates malformed judge output without aborting the run.

    Unparseable or missing rubric scores are conservatively treated as the
    lowest score (1) with a warning instead of raising, so a single noisy judge
    response never kills a whole evaluation.
    """

    def _process_judge_response(self, response: str) -> float:
        """Parse the five rubric scores, defaulting unparseable ones to the lowest score.

        Judge models occasionally emit corrupted rubric names or non-numeric
        scores. Rather than aborting an entire evaluation, we conservatively
        assign the lowest rubric score (1) to any rubric we cannot parse and log
        a warning so the frequency can be audited (per-sample judge inputs are
        persisted via lighteval's ``--save-details``).
        """
        by_metric: dict[str, int] = {}
        lines = response.splitlines()
        for line_number, line in enumerate(lines):
            if "_SCORE:" not in line:
                continue
            metric_name, raw_score = line.split(":", maxsplit=1)
            metric_name = metric_name.strip()
            if metric_name not in self.RUBRIC_NAMES:
                logger.warning("Ignoring unexpected SLDS rubric name %r: %s", metric_name, line)
                continue
            if metric_name in by_metric:
                logger.warning("Ignoring duplicate SLDS rubric %r: %s", metric_name, line)
                continue

            score = self._parse_rubric_score(raw_score)
            if score is None and line_number + 1 < len(lines) and lines[line_number + 1].strip() in {"1", "2", "3"}:
                score = int(lines[line_number + 1].strip())
                logger.warning("SLDS score for %s parsed from the next line as %s: %s", metric_name, score, line)
            if score is None:
                logger.warning("Unparseable SLDS score for %s; will default to lowest: %s", metric_name, line)
                continue
            by_metric[metric_name] = score

        for rubric_name in self.RUBRIC_NAMES:
            if rubric_name not in by_metric:
                logger.warning("Assigning lowest SLDS score for missing/unparseable rubric %s.", rubric_name)
                by_metric[rubric_name] = 1

        aggregated_score = sum(score - 1 for score in by_metric.values())
        return aggregated_score / (len(self.RUBRIC_NAMES) * 2)

    def _parse_rubric_score(self, raw_score: str) -> int | None:
        """Return the rubric score in [1, 3], or None if it cannot be parsed."""
        score_match = re.search(r"(?<!\d)([1-3])(?!\d)\s*$", raw_score)
        if score_match is not None:
            return int(score_match.group(1))

        # Some judges append numeric garbage to a valid leading digit (e.g. "1694350242").
        garbage_score_match = re.fullmatch(r"([1-3])\d{2,}", raw_score.strip())
        if garbage_score_match is not None:
            logger.warning(
                "SLDS score had trailing numeric garbage; using leading digit %s: %s",
                garbage_score_match.group(1),
                raw_score,
            )
            return int(garbage_score_match.group(1))

        return None

    def compute(
        self,
        responses: list[ModelResponse],
        docs: list[Doc],
        **kwargs: Any,
    ) -> list[dict[str, float]]:
        """Judge post-processed generated headnotes rather than raw reasoning traces."""
        logger.info("Judging %d samples with %s...", len(docs), self.short_judge_name)
        not_considered = [None for _ in docs]
        original_headnotes = [doc.get_golds()[0] for doc in docs]
        generated_headnotes = [prediction_from_response(response) for response in responses]

        scores, _, _ = self.judge.evaluate_answer_batch(
            questions=not_considered,
            answers=generated_headnotes,
            options=not_considered,
            golds=original_headnotes,
        )
        return [
            {
                self.short_judge_name: score * 100,
            }
            for score in scores
        ]


class PostProcessedJudgeSwissLegalTranslation(JudgeSwissLegalTranslation):
    """SwiLTra judge that evaluates reasoning-stripped translations."""

    def compute(
        self,
        responses: list[ModelResponse],
        docs: list[Doc],
        **kwargs: Any,
    ) -> list[dict[str, float]]:
        """Judge post-processed translations rather than raw reasoning traces."""
        logger.info("Judging %d samples with %s...", len(docs), self.short_judge_name)
        questions = [doc.specific["source"] for doc in docs]
        options = [doc.choices for doc in docs]
        golds = [doc.get_golds()[0] for doc in docs]
        predictions = [
            prediction_from_response(response, target_lang=doc.specific["target_lang"])
            for response, doc in zip(responses, docs, strict=True)
        ]

        scores, _, _ = self.judge.evaluate_answer_batch(questions, predictions, options, golds)
        return [
            {
                self.short_judge_name: score * 100,
            }
            for score in scores
        ]


class PostProcessedJudgeLEXamOQ(JudgeLEXamOQ):
    """LEXam OQ judge that evaluates reasoning-stripped answers."""

    def compute(
        self,
        responses: list[ModelResponse],
        docs: list[Doc],
        **kwargs: Any,
    ) -> list[dict[str, float]]:
        logger.info("Judging %d samples with %s...", len(docs), self.short_judge_name)
        questions = [doc.specific["question"] for doc in docs]
        options = [doc.choices for doc in docs]
        golds = [doc.get_golds()[0] for doc in docs]
        predictions = [prediction_from_response(response) for response in responses]

        scores, _, _ = self.judge.evaluate_answer_batch(questions, predictions, options, golds)
        return [{self.short_judge_name: score * 100} for score in scores]


def _judge_config_path() -> Path:
    if "SWISSLEGALEVALS_JUDGE_CONFIG" in os.environ:
        return Path(os.environ["SWISSLEGALEVALS_JUDGE_CONFIG"])
    return DEFAULT_JUDGE_CONFIG


def load_judge_config(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Load per-benchmark judge settings from YAML."""
    config_path = path or _judge_config_path()
    logger.info("Loading judge config from %s", config_path)
    with config_path.open() as f:
        raw = yaml.safe_load(f)
    return raw["judges"]


def _judge_backend(provider: JudgeProvider) -> JudgeBackend:
    if provider in ("openai", "openrouter"):
        return "litellm"
    if provider == "hf-inference-providers":
        return "inference-providers"
    raise ValueError(f"Unsupported judge provider: {provider}")


def _litellm_judge_model(provider: JudgeProvider, model: str) -> str:
    if provider == "openai":
        return model if model.startswith("openai/") else f"openai/{model}"
    if provider == "openrouter":
        return model if model.startswith("openrouter/") else f"openrouter/{model}"
    raise ValueError(f"Provider {provider} is not a LiteLLM provider")


def _build_judge_model_name(provider: JudgeProvider, model: str) -> str:
    if provider == "hf-inference-providers":
        return model
    return _litellm_judge_model(provider, model)


def _build_judge(
    provider: JudgeProvider,
    model: str,
    hf_provider: str | None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Map YAML judge provider to JudgeLLM constructor kwargs."""
    backend = _judge_backend(provider)
    kwargs: dict[str, Any] = {
        "judge_model_name": _build_judge_model_name(provider, model),
        "judge_backend": backend,
    }
    if backend == "inference-providers":
        if hf_provider is None:
            raise ValueError(
                f"hf_provider is required for judge provider {provider} (model={model})"
            )
        kwargs["hf_provider"] = hf_provider
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return kwargs


def _slds_judge_metric(
    language: Literal["de", "fr", "it"],
    judge_cfg: dict[str, Any],
) -> SampleLevelMetricGrouping:
    provider = judge_cfg["provider"]
    short_name = judge_cfg["short_judge_name"]
    judge = RobustJudgeSwissLandmarkDecisionSummarization(
        language=language,
        short_judge_name=short_name,
        **_build_judge(
            provider,
            judge_cfg["model"],
            judge_cfg["hf_provider"] if "hf_provider" in judge_cfg else None,
            max_tokens=SLDS_JUDGE_MAX_TOKENS,
        ),
    )
    judge.judge.API_MAX_RETRY = 60
    return SampleLevelMetricGrouping(
        metric_name=[short_name],
        higher_is_better={short_name: True},
        category=SamplingMethod.GENERATIVE,
        sample_level_fn=judge,
        corpus_level_fn={short_name: statistics.mean},
        batched_compute=True,
    )


def _swiltra_judge_metric(judge_cfg: dict[str, Any]) -> SampleLevelMetricGrouping:
    provider = judge_cfg["provider"]
    short_name = judge_cfg["short_judge_name"]
    system_style = judge_cfg["system_style"]
    few_shot_style = judge_cfg["few_shot_style"]
    judgment_style = judge_cfg["judgment_style"]

    def template(question: str, options: list, answer: str, gold: str) -> list[dict[str, str]]:
        system_prompt = SWISS_LEGAL_TRANSLATION_JUDGE_SYSTEM_PROMPT[system_style]
        user = SWISS_LEGAL_TRANSLATION_JUDGE_USER_PROMPT[system_style]
        few_shot = SWISS_LEGAL_TRANSLATION_JUDGE_FEW_SHOT_EXAMPLES[
            f"{few_shot_style}_{judgment_style}"
        ]
        instruction = SWISS_LEGAL_TRANSLATION_JUDGE_INSTRUCTION.format(
            question=question, gold=gold, answer=answer
        )
        user_prompt = user + few_shot + instruction
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    judge = PostProcessedJudgeSwissLegalTranslation(
        template=template,
        process_judge_response=process_judge_response_freeform_gpt,
        short_judge_name=short_name,
        **_build_judge(
            provider,
            judge_cfg["model"],
            judge_cfg["hf_provider"] if "hf_provider" in judge_cfg else None,
        ),
    )
    judge.judge.API_MAX_RETRY = JUDGE_API_MAX_RETRY
    return SampleLevelMetricGrouping(
        metric_name=[short_name],
        higher_is_better={short_name: True},
        category=SamplingMethod.GENERATIVE,
        sample_level_fn=judge,
        corpus_level_fn={short_name: statistics.mean},
        batched_compute=True,
    )


def _lexam_oq_judge_metric(judge_cfg: dict[str, Any]) -> SampleLevelMetricGrouping:
    provider = judge_cfg["provider"]
    short_name = judge_cfg["short_judge_name"]

    def template(question: str, options: list, answer: str, gold: str) -> list[dict[str, str]]:
        instruction = LEXAM_OQ_JUDGE_INSTRUCTION.format(
            question=question, gold=gold, answer=answer
        )
        return [
            {"role": "system", "content": LEXAM_OQ_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": LEXAM_OQ_JUDGE_USER_PROMPT + instruction},
        ]

    judge = PostProcessedJudgeLEXamOQ(
        template=template,
        process_judge_response=process_judge_response_freeform_gpt,
        short_judge_name=short_name,
        **_build_judge(
            provider,
            judge_cfg["model"],
            judge_cfg["hf_provider"] if "hf_provider" in judge_cfg else None,
            max_tokens=LEXAM_OQ_JUDGE_MAX_TOKENS,
        ),
    )
    judge.judge.API_MAX_RETRY = JUDGE_API_MAX_RETRY
    return SampleLevelMetricGrouping(
        metric_name=[short_name],
        higher_is_better={short_name: True},
        category=SamplingMethod.GENERATIVE,
        sample_level_fn=judge,
        corpus_level_fn={short_name: statistics.mean},
        batched_compute=True,
    )


class JudgeOnlyHeadnoteTask(HeadnoteGenerationTask):
    """SLDS headnote generation with a single LLM-as-judge metric."""

    def __init__(
        self,
        level_name: str,
        judge_metric: SampleLevelMetricGrouping,
        generation_size: int = SLDS_GENERATION_SIZE,
    ) -> None:
        self._judge_metric = judge_metric
        super().__init__(SwissLandmarkDecisionHeadnotes, level_name)
        self.generation_size = max(self.generation_size, generation_size)

    def _get_metrics(self, headnote_language: Literal["de", "fr", "it"]) -> list[Any]:
        return [self._judge_metric]


class JudgeOnlyTranslationTask(LightevalTaskConfig):
    """SwiLTra translation task with a single LLM-as-judge metric."""

    def __init__(
        self,
        dataset_config: Any,
        level_name: str,
        source_lang: str,
        target_lang: str,
        judge_metric: SampleLevelMetricGrouping,
        min_generation_size: int = SWILTRABENCH_GENERATION_SIZE,
    ) -> None:
        level_config = dataset_config.subsets[level_name]
        LightevalTaskConfig.__init__(
            self,
            name=f"{dataset_config.name}-{level_name}:{source_lang}-{target_lang}",
            prompt_function=create_translation_prompt_fn(
                level_config, source_lang, target_lang
            ),
            hf_repo=dataset_config.hf_repo,
            hf_subset=level_name,
            hf_filter=None,
            hf_avail_splits=["train", "validation", "test"],
            evaluation_splits=["test"],
            few_shots_split="validation",
            few_shots_select="sequential",
            generation_size=max(
                level_config.generation_size,
                min_generation_size,
            ),
            metrics=[judge_metric],
            stop_sequence=_TRANSLATION_STOP_SEQUENCE_OVERRIDES.get(
                level_name,
                level_config.stop_sequence,
            ),
        )


class JudgeOnlyLEXamOpenQuestionTask(LightevalTaskConfig):
    """LEXam open questions with a single LLM-as-judge metric."""

    def __init__(
        self,
        language: Literal["en", "de"],
        judge_metric: SampleLevelMetricGrouping,
        generation_size: int = LEXAM_GENERATION_SIZE,
    ) -> None:
        super().__init__(
            name=f"lexam_oq:{language}",
            prompt_function=lexam_oq_prompt_fn,
            hf_repo=LEXAM_REPO,
            hf_subset="open_question",
            hf_filter=_lexam_language_filter(language),
            hf_avail_splits=["dev", "test"],
            evaluation_splits=["test"],
            few_shots_split="dev",
            few_shots_select="sequential",
            generation_size=generation_size,
            stop_sequence=LEXAM_STOP_SEQUENCES,
            metrics=[judge_metric],
        )


class CompactLEXamMCQTask(LEXamMCQTask):
    """LEXam MCQ task sized for reasoning models that chain-of-thought before answering."""

    def __init__(
        self,
        language: Literal["en", "de"],
        num_choices: int,
        with_idk: bool,
        generation_size: int = LEXAM_GENERATION_SIZE,
    ) -> None:
        super().__init__(language=language, num_choices=num_choices, with_idk=with_idk)
        self.generation_size = generation_size
        self.metrics = [get_robust_lexam_mcq_metric(with_idk=with_idk)]


class SmokeLEXamMCQTask(LightevalTaskConfig):
    """Tiny no-judge LEXam MCQ task for local vLLM infrastructure smoke tests."""

    def __init__(self) -> None:
        super().__init__(
            name="smoke_lexam_mcq_4:en",
            prompt_function=_build_lexam_mcq_prompt_fn(with_idk=False),
            hf_repo=LEXAM_REPO,
            hf_subset="mcq_4_choices",
            hf_filter=_lexam_language_filter("en"),
            hf_avail_splits=["test"],
            evaluation_splits=["test"],
            few_shots_split=None,
            few_shots_select=None,
            generation_size=16,
            stop_sequence=LEXAM_STOP_SEQUENCES,
            metrics=[get_robust_lexam_mcq_metric(with_idk=False)],
        )


def _translation_tasks(
    datasets: list[Any],
    level_filter: dict[str, str] | None,
    judge_metric: SampleLevelMetricGrouping,
    min_generation_size: int,
) -> list[LightevalTaskConfig]:
    tasks: list[LightevalTaskConfig] = []
    for dataset in datasets:
        for subset in dataset.subsets:
            if level_filter is not None:
                if dataset.name not in level_filter:
                    continue
                if subset != level_filter[dataset.name]:
                    continue
            for source_lang, target_lang in dataset.translation_pairs:
                tasks.append(
                    JudgeOnlyTranslationTask(
                        dataset_config=dataset,
                        level_name=subset,
                        source_lang=source_lang,
                        target_lang=target_lang,
                        judge_metric=judge_metric,
                        min_generation_size=min_generation_size,
                    )
                )
    return tasks


def build_tasks_table(judge_config_path: Path | None = None) -> list[LightevalTaskConfig]:
    """Assemble TASKS_TABLE from judge config."""
    judges = load_judge_config(judge_config_path)
    generation_sizes = load_generation_sizes()
    slds_judge = judges["slds"]
    swiltra_judge = judges["swiltrabench"]
    lexam_judge = judges["lexam_oq"]

    slds_tasks: list[LightevalTaskConfig] = []
    for subset in SwissLandmarkDecisionHeadnotes.subsets:
        lang = SwissLandmarkDecisionHeadnotes.subsets[subset].custom_attributes[
            "headnote_language"
        ]
        metric = _slds_judge_metric(lang, slds_judge)
        slds_tasks.append(
            JudgeOnlyHeadnoteTask(
                subset,
                metric,
                generation_size=generation_sizes["slds"],
            )
        )

    swiltra_metric = _swiltra_judge_metric(swiltra_judge)
    swiltra_tasks = _translation_tasks(
        TRANSLATION_DATASETS,
        None,
        swiltra_metric,
        min_generation_size=generation_sizes["swiltrabench"],
    )

    lexam_oq_metric = _lexam_oq_judge_metric(lexam_judge)
    lexam_generation_size = generation_sizes["lexam"]
    lexam_oq = [
        JudgeOnlyLEXamOpenQuestionTask(
            language=lang,
            judge_metric=lexam_oq_metric,
            generation_size=lexam_generation_size,
        )
        for lang in LEXAM_LANGUAGES
    ]
    lexam_mcq = [
        CompactLEXamMCQTask(
            language=lang,
            num_choices=num_choices,
            with_idk=with_idk,
            generation_size=lexam_generation_size,
        )
        for lang in LEXAM_LANGUAGES
        for num_choices in LEXAM_MCQ_NUM_CHOICES
        for with_idk in (False, True)
    ]

    # Task subset selection (scoped vs full SwiLTra) is done in run.py via configs/tasks.yaml.
    # Per-family generation caps come from the same profile via load_generation_sizes().
    return [*slds_tasks, *swiltra_tasks, *lexam_oq, *lexam_mcq, SmokeLEXamMCQTask()]


_TASKS_TABLE_CACHE: list[LightevalTaskConfig] | None = None


def __getattr__(name: str) -> list[LightevalTaskConfig]:
    """Lazy-build TASKS_TABLE so importing this module stays fast."""
    global _TASKS_TABLE_CACHE
    if name == "TASKS_TABLE":
        if _TASKS_TABLE_CACHE is None:
            _TASKS_TABLE_CACHE = build_tasks_table()
        return _TASKS_TABLE_CACHE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
