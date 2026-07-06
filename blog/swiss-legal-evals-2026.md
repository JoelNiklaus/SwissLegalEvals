# The state-of-the-art in open-source AI for Swiss legal tasks

*Joel Niklaus · July 6 2026*

If you need an open-weight LLM for Swiss legal work in German, French, or Italian, there is no single best model in 2026. We evaluated 12 open models on three Swiss legal benchmarks, 55,861 scored samples each, and the top five finish within 2.1 points on our composite score.

The model that wins overall is not the best translator. And the best translator is one of the weakest at multiple-choice law questions. So the right question is not "which open model is best" but "best at what."

This post is for anyone deciding which open model to run on Swiss legal text, and for anyone who wants the current numbers instead of vibes. All results come from a single full run: the raw outputs live in the public [Hugging Face bucket](https://huggingface.co/buckets/joelniklaus/SwissLegalEvals), and the [`run.py` entry point](https://github.com/JoelNiklaus/SwissLegalEvals/blob/main/src/swiss_legal_evals/run.py) reproduces every number below.

![Overall composite score across SLDS, SwiLTra-Bench, and LEXam for 11 comparable open models](figures/overall_ranking.png)

NVIDIA's Nemotron 3 Ultra 550B leads the composite at 59.0. But Kimi K2.6 (58.6), DeepSeek V4 Pro (58.3), DeepSeek V4 Flash (57.3), and MiniMax M3 (56.9) trail by only 0.4, 0.7, 1.7, and 2.1 points.

Treat the top of this chart as a five-way tie, not a clean ranking. A task-level bootstrap over the same result files (10,000 resamples within each benchmark family) gives 95% intervals with half-widths of roughly 3.2 to 3.6 points for the top five models, larger than the 2.1-point spread between first and fifth. That uncertainty is also consistent with recent LLM-as-judge work: [JudgeSense](https://arxiv.org/abs/2604.23478) finds that equivalent prompt rephrasings can flip judge decisions, and [Li et al.](https://arxiv.org/abs/2506.22316) show that score-based judges shift under rubric and reference-answer perturbations. The composite averages four task groups (SLDS summarization, LEXam open questions, translation, and MCQ), each on a 0-100 scale.

The gap only becomes decisive further down, with one exception. Gemma 4 31B reaches 54.8 overall, seventh place and within 0.7 of GLM 5.2 (55.5), which is striking for a 31B model sitting among the 500B-class frontier. Size alone does not explain the rest. Qwen3.5 35B (42.8) is a mixture-of-experts that activates only 3B parameters per token. OLMo 3.1 32B Think is dense and the same size as Gemma, yet scores 26.0, under half as well. And LFM2.5 8B (24.8), the only genuinely small model here, sits at the bottom. None of the three are ready for Swiss legal work yet.

## What we measure: three Swiss legal benchmarks in three languages

Swiss law is written in German, French, and Italian, and all three benchmarks cover the multilingual case.

We score generative tasks with LLM judges from the benchmarks' own papers, not BLEU or ROUGE. The papers show lexical overlap correlates poorly with human judgment on legal text, so a judge model reading the output is the more faithful metric.

- **SLDS** ([Rolshoven et al., 2025](https://arxiv.org/abs/2410.13456)): write a legal headnote for a Swiss Federal Supreme Court leading decision. Judged 0-100 by DeepSeek V4 Pro on a 5-rubric prompt.
- **SwiLTra-Bench** ([Niklaus et al., 2025](https://arxiv.org/abs/2503.01372)): legal translation across court decisions (`sdst`), laws (`slt`), and press releases (`sscprt`). Judged 0-100 by gpt-4o-mini against a codebook. We report the mean of the three.
- **LEXam** ([Fan et al., 2026](https://arxiv.org/abs/2505.12864)): law-exam open questions (judged 0-100 by DeepSeek R1) plus multiple-choice questions scored by accuracy at 4, 8, and 16 answer options.

## Where the frontier models differ: pick by task, not by rank

Averaging into one number hides where these models actually diverge. The per-group view is what you should use to choose.

![Per-group scores for the top seven models by composite](figures/task_group_profile.png)

Nemotron wins the composite by leading two groups outright: SLDS summarization at 55.1 (the only model above 53) and LEXam open questions at 66.5.

GLM 5.2 is the mirror image. It is the best translator at 66.2, but the weakest of this tier on LEXam open questions (58.8) and MCQ (44.6), which is why it drops to sixth overall despite topping translation.

If your workload is translation, run GLM 5.2. If it is summarization or open-ended legal reasoning, run Nemotron or a DeepSeek V4 variant. And if you need to run on your own hardware, Gemma 4 31B is the standout: it holds 54.8 overall and tops the field on MCQ, at a size that fits on local hardware while every model above it is a far larger MoE served in the cloud.

## A 3B-active specialist nearly matches the frontier in translation

The  models perform best in translation: nine of twelve score at least 57 out of 100. GLM 5.2 tops it at 66.2, ahead of Kimi K2.6 (65.3) and DeepSeek V4 Pro (64.2).


![SwiLTra-Bench translation average for all 12 models](figures/translation.png)


The efficiency surprise is Hunyuan MT2 30B, a translation specialist with only 3B active parameters (we ran it on SwiLTra-Bench only). It lands sixth at 61.9, within 4.3 points of the best open translator, GLM 5.2 (66.2). The five models ahead of it activate far more per token, from MiniMax M3's 23B up to Nemotron 3 Ultra's 55B, so a model with 8x to 18x fewer active parameters is nearly matching the frontier on translation. The takeaway cuts two ways: a specialist does not win on raw quality, but no model on the board delivers more per active parameter. If you are compute-bound, Hunyuan MT2 is the translation model to watch. If you only want the best output, the big MoEs still win.

## Not all legal text is equally hard to translate

SwiLTra-Bench splits into three document types, and they are not equally hard.

![Translation judge scores on the three SwiLTra-Bench subsets for all 12 models](figures/translation_subsets.png)

For the strong models the order is consistent: court-decision summaries are easiest, statutory law sits in the middle, and press releases are hardest. GLM 5.2 scores 71.0 on decision summaries but 60.5 on press releases, and every model in the top eight ranks the subsets in that same order. DeepSeek V4 Pro is the one that pulls ahead on press releases (61.2, the only score above 61 on that subset).

The pattern breaks for the weaker models, and the breakage is revealing. Gemma 4 31B stays near the frontier on decision summaries (63.5) and press releases (59.2) but drops to 48.9 on statutory law, a 14.6-point hole that its translation average hides. Qwen3.5 35B inverts the usual order entirely, with law its worst subset (30.6) and press releases its best (51.5). OLMo 3.1 32B all but fails press releases at 9.6. If your workload is one specific document type, these subset scores matter more than the translation average.

## Multiple-choice accuracy collapses as options grow

Every model degrades from 4 to 16 answer options, and the drop is steep.

![LEXam MCQ accuracy at 4, 8, and 16 answer options](figures/mcq_scaling.png)

Kimi K2.6 falls from 67.9% at four options to 44.0% at sixteen. Nemotron falls from 63.1% to 41.7%.

Model size does not protect against this. Gemma 4 31B posts the best MCQ average of the entire field (56.0), ahead of every 500B-class model. For these exam questions, a smaller dense model beats much larger MoEs.

The `trad_score` in the chart treats "I don't know" as wrong. The benchmark also reports a calibration score (`idk_score`) that rewards abstaining and penalizes confident wrong answers, and it turns negative for every model at 16 options, from -4.3 for the best-calibrated DeepSeek V4 Pro down to -61.3 for LFM2.5 8B. With a wide option set, open models still guess wrong more often than they admit uncertainty.

## Italian is the hardest of the three Swiss languages

Averaging every German, French, and Italian task a model produces (SLDS plus the three translation subsets, the benchmarks that cover all three languages) exposes a small but consistent gap.

![Mean judge score by Swiss language across SLDS and translation for the 11 comparable models](figures/language_comparison.png)

French comes out highest on average (51.7 across the 11 comparable models), German is close behind (50.7), and Italian is last (48.4). The gap is only about three points, but it holds up: Italian is the lowest of the three for 8 of the 11 models, and German and French are within a point of each other. The likely cause is data volume, since Italian is the smallest of the three languages in Swiss legal corpora and on the open web. LEXam is left out of this cut because it exists only in German and English, and including it would compare the languages on an unequal benchmark mix.

## Model sizes and the hardware to run them

Two numbers decide the cost of serving a model. Total parameters set how much memory the weights occupy, and active parameters set how much compute each token costs. The open field here runs from 8.5B to 1.6T total parameters, and native precision matters as much as parameter count.

| Model                 | Total | Active | Precision | Weights  | Runs on       |
|-----------------------|------:|-------:|-----------|---------:|---------------|
| DeepSeek V4 Pro       |  1.6T |    49B | FP4+FP8   | ~0.85 TB | 8x H200       |
| Kimi K2.6             |    1T |    32B | INT4      |  ~0.5 TB | 4x H200       |
| GLM 5.2               |  753B |    40B | BF16      |  ~1.5 TB | 11x H200      |
| Nemotron 3 Ultra 550B |  550B |    55B | NVFP4     |  ~275 GB | 4x H100       |
| MiniMax M3            |  428B |    23B | BF16      | ~0.85 TB | 7x H200       |
| DeepSeek V4 Flash     |  284B |    13B | FP4+FP8   |  ~150 GB | 2x H100       |
| gpt-oss 120B          |  117B |   5.1B | MXFP4     |   ~60 GB | 1x H100       |
| Qwen3.5 35B           |   36B |     3B | BF16      |   ~72 GB | 1x H100       |
| Gemma 4 31B           |   33B |    33B | BF16      |   ~65 GB | 1x H100       |
| OLMo 3.1 32B Think    |   32B |    32B | BF16      |   ~64 GB | 1x H100       |
| Hunyuan MT2 30B       |   30B |     3B | BF16      |   ~60 GB | 1x H100       |
| LFM2.5 8B             |  8.5B |     1B | BF16      |   ~17 GB | 1x 24 GB GPU  |

Weights are total parameters times bytes per parameter (BF16 = 2, FP8 = 1, INT4/FP4 = 0.5), before the KV cache that long context adds; GPU counts assume 80 GB (H100) or 141 GB (H200) cards. We ran the local vLLM models on 4 H100s each for 64k-context throughput, well above these weight-only minimums.

Precision shifts the picture as much as size. Kimi K2.6 has more total parameters than GLM 5.2 (1T versus 753B), but it ships as INT4, so its weights (~0.5 TB) are a third of GLM's BF16 footprint (~1.5 TB). Six of the twelve models need a multi-GPU server; the other six fit on one accelerator, and five of those run on a single 80 GB H100. That is the case for Gemma 4 31B: near-frontier quality (54.8 overall) from one GPU, while every model ranked above it needs several.

## Known limitations

Treat the composite as indicative rather than a normalized benchmark: it mixes 0-100 judge scores with MCQ accuracy rescaled from 0-1, so the four per-group charts are the ground truth and the overall bar is a convenience. A few other caveats apply. SLDS is judged by DeepSeek V4 Pro, so the DeepSeek models' summarization scores may carry a mild self-preference bias. SwiLTra-Bench is scoped to the lowest granularity per dataset here; the full profile adds finer levels and roughly 87k samples per model. And Hunyuan MT2 30B ran translation only, so it is absent from the composite, SLDS, LEXam, and profile charts.

## Conclusion

The open frontier on Swiss legal tasks is close and specialized. Five models finish within 2.1 composite points, no single one wins every group, and the right pick depends on the job: GLM 5.2 for translation, Nemotron 3 Ultra or a DeepSeek V4 variant for summarization and open-ended reasoning, and Gemma 4 31B when you need near-frontier quality on a single GPU. Absolute quality is not there yet. The strongest models score in the mid-60s on open legal exam questions and fall into the 40s on 16-option multiple choice, so a lawyer still has to check the output on real matters.

## Reproduce this

The three benchmarks are integrated directly into [lighteval](https://github.com/huggingface/lighteval), including the paper-grounded LLM judges, so scoring a model is one command rather than a pile of glue scripts. That integration is what makes the whole pipeline, from generation to judging to aggregation, run end to end on any model you point it at.

The full-run results, the four figures above, and the underlying tables live in the public bucket [`joelniklaus/SwissLegalEvals`](https://huggingface.co/buckets/joelniklaus/SwissLegalEvals). To regenerate the charts:

```bash
uv sync --extra dev   # figure export needs kaleido, in the dev group
hf buckets sync hf://buckets/joelniklaus/SwissLegalEvals/summary.csv results
uv run python blog/make_figures.py   # writes blog/figures/*.png
```

The evaluation code lives in the same [SwissLegalEvals repository](https://github.com/JoelNiklaus/SwissLegalEvals). To score a new model, add one entry to `configs/models.yaml`. For a model served through Hugging Face inference providers:

```yaml
# configs/models.yaml
- name: my-model
  provider: hf-inference-providers
  model: org/My-Model
  hf_provider: fireworks-ai
```

For a model you host locally with vLLM, set the parallelism instead:

```yaml
- name: my-model
  provider: vllm
  model: org/My-Model
  tensor_parallel_size: 4
  max_model_length: 65536
```

Put the provider key in `.env` (`HF_TOKEN`, `OPENAI_API_KEY`, or `OPENROUTER_API_KEY`), then run the model and rebuild the tables and charts:

```bash
uv run swiss-legal-evals --models my-model --max-samples 5   # quick smoke test; drop --max-samples for the full run
uv run swiss-legal-evals-aggregate
uv run swiss-legal-evals-plot
```

Add `--dry-run` to print the underlying lighteval commands without calling any API.
