# SwissLegalEvals

Evaluate open-source LLMs on Swiss legal benchmarks (**SLDS**, **SwiLTra-Bench**, **LEXam**) using [lighteval](https://github.com/huggingface/lighteval) with paper-grounded LLM-as-judge metrics.

## Install

Default install for API-based evaluation, aggregation, plotting, blog figures, and tests:

```bash
git clone https://github.com/JoelNiklaus/SwissLegalEvals.git
cd SwissLegalEvals
uv venv --python 3.13
GIT_LFS_SKIP_SMUDGE=1 uv sync --extra dev   # skip lighteval fixture blobs; install pytest, ruff, kaleido
```

Optional local vLLM support on a normal machine:

```bash
uv sync --extra local
uv pip install vllm
```

Cluster patch for our Hopper environment only:

```bash
uv sync --extra local
mkdir -p scripts/logs/slurm
sbatch scripts/setup_vllm.sh
```

`scripts/setup_vllm.sh` is not the general installation path. It is a cluster-specific setup for our Hopper nodes: it creates a Python 3.12 `.venv-apertus` because the Apertus dependency stack has no CPython 3.13 wheel, installs the pinned Swiss AI vLLM/Transformers revisions with the matching precompiled cu129 extension, optionally applies `glibc-fix` when `/admin/opt/glibc-2.38` is present (skipped on images that only have system glibc 2.35), and installs a tiny `nvJitLink` preload hook so the wheel does not accidentally pick up older system CUDA libraries. `scripts/launch_eval.sh` uses this environment for all local vLLM models; the project `.venv` remains for API-based runs and development. Other users should first try the normal `uv pip install vllm` path.

## Environment variables

Provider API keys are loaded automatically from `.env` in the project root. Shell exports still work and take precedence over `.env` values.

Example `.env`:

```bash
cp .env.example .env
# Then fill in the values:
HF_TOKEN=...
OPENAI_API_KEY=...
OPENROUTER_API_KEY=...
```

| Provider                 | Variable             |
|--------------------------|----------------------|
| `openai`                 | `OPENAI_API_KEY`     |
| `openrouter`             | `OPENROUTER_API_KEY` |
| `hf-inference-providers` | `HF_TOKEN`           |
| `vllm`                   | (local GPU)          |

Optional: set `HF_ORG_TO_BILL=<org-slug>` to bill HF Inference Providers usage to a Team/Enterprise organization instead of your personal account. Applies to eval models and HF-backed judges (SLDS, LEXam OQ).

Useful overrides:

| Variable                         | Purpose                                                  |
|----------------------------------|----------------------------------------------------------|
| `SWISSLEGALEVALS_JUDGE_CONFIG`   | Absolute path to an alternate `judges.yaml`.             |
| `SWISSLEGALEVALS_TASKS_CONFIG`   | Task config path passed to lighteval subprocesses.       |
| `SWISSLEGALEVALS_TASKS_PROFILE`  | Active task profile passed to lighteval subprocesses.    |

Use `--env-file /path/to/.env` to load a different dotenv file. Default HF judges use `novita` because the currently configured DeepSeek judge IDs are not served by `together` on HF inference providers.

## Usage

Dry-run all configured models (prints lighteval commands, no API keys required):

```bash
uv run swiss-legal-evals --dry-run
```

Run all models (`evaluate` is an alias for `swiss-legal-evals`):

```bash
uv run evaluate
```

Run one configured model by provider:

```bash
# HF inference providers (large MoE models)
uv run swiss-legal-evals --models deepseek-v4-pro

# Local vLLM on the cluster (after setup_vllm.sh)
scripts/launch_model.sh gemma-4-31b-it
scripts/launch_model.sh qwen3.5-35b-a3b
scripts/launch_model.sh apertus-v1.5-70b
scripts/launch_model.sh mistral-medium-3.5-128b

# SwiLTra-only translation specialist on the cluster
scripts/launch_model.sh hy-mt2-30b
```

OpenRouter is supported by the runner, but no OpenRouter models are part of the published 2026 run. Add one to `configs/models.yaml` before selecting it:

```yaml
- name: my-openrouter-model
  provider: openrouter
  model: provider/model-id
```

Local `vllm` runs use `tensor_parallel_size` and `data_parallel_size` from `configs/models.yaml`. Current local models use DP1 after Ray data-parallel deadlocks on this cluster: most use TP4 (4 H100s), Apertus 1.5 uses TP4, Mistral Medium 3.5 uses TP8, and `lfm2.5-8b` uses a single GPU. `scripts/launch_all.sh` requests `data_parallel_size * tensor_parallel_size` GPUs per vLLM job. HF Inference Provider models request no GPUs, only CPUs for orchestration and judging.

Full SwiLTra-Bench (all granularity levels):

```bash
uv run swiss-legal-evals --profile swiltrabench_full
# or keep default profile and add the group:
uv run swiss-legal-evals --groups slds swiltrabench_full lexam
```

Debug cap:

```bash
uv run swiss-legal-evals --models gemma-4-31b-it --max-samples 5
```

Single-task smoke test with an API model:

```bash
uv run swiss-legal-evals \
  --models deepseek-v4-flash \
  --task-string 'slds:de_de|0' \
  --max-samples 1 \
  --output-dir results_smoke/hf
```

Use an alternate model config for infrastructure smoke tests:

```bash
uv run swiss-legal-evals \
  --models-config configs/models_smoke.yaml \
  --models qwen2.5-0.5b-vllm \
  --max-samples 1 \
  --output-dir results_smoke/vllm
```

Other useful flags:

| Flag               | Purpose                                                               |
|--------------------|-----------------------------------------------------------------------|
| `--models-config`  | Use a different model list, such as `configs/models_smoke.yaml`.       |
| `--tasks-config`   | Use a different task file, such as `configs/tasks_cap_compare.yaml`.   |
| `--skip-env-check` | Skip API-key validation before launching jobs. Useful only for debug. |
| `--verbose`        | Enable more detailed runner logs.                                     |

## Development checks

Run the same checks as CI before opening a pull request or release:

```bash
uv run ruff check src tests scripts/analyze_cap_compare.py
uv run pytest tests -q
```

## Results, plots, and reports

Run the evaluation, aggregate raw JSON, then render Plotly family charts:

```bash
uv run swiss-legal-evals
uv run swiss-legal-evals-aggregate
uv run swiss-legal-evals-plot
```

Output files:

| Path                                      | Produced by                     | Purpose                                  |
|-------------------------------------------|---------------------------------|------------------------------------------|
| `results/results/<model>/results_*.json`  | `swiss-legal-evals` / lighteval | Raw model-level result JSON.             |
| `results/details/**/details_*.parquet`    | `swiss-legal-evals` / lighteval | Per-sample generations and scores.       |
| `results/summary_long.csv`                | `swiss-legal-evals-aggregate`   | Tidy task/metric table.                  |
| `results/summary_family_mean.csv`         | `swiss-legal-evals-aggregate`   | Model/family/metric means, used by plot. |
| `results/summary.csv`                     | `swiss-legal-evals-plot`        | Model x family primary-metric table.     |
| `plots/<family>.html`                     | `swiss-legal-evals-plot`        | Interactive Plotly family charts.        |

The completed 2026 full-run outputs are mirrored in the public Hugging Face bucket [`joelniklaus/SwissLegalEvals`](https://huggingface.co/buckets/joelniklaus/SwissLegalEvals):

```bash
# Download the published results into ./results
hf buckets sync hf://buckets/joelniklaus/SwissLegalEvals results

# Maintainers: mirror local ./results back to the bucket
hf buckets sync results hf://buckets/joelniklaus/SwissLegalEvals --delete
```

Local result directories (`results/`, `results_smoke*`, `results_cap_compare_*`) are gitignored to keep the repository lightweight.

### Blog post

[`blog/swiss-legal-evals-2026.md`](blog/swiss-legal-evals-2026.md) is a short write-up of the 2026 full run (state of the art of open models on Swiss legal tasks). Regenerate its static PNG charts from the published summaries with:

```bash
uv sync --extra dev
hf buckets sync hf://buckets/joelniklaus/SwissLegalEvals/summary.csv results
hf buckets sync hf://buckets/joelniklaus/SwissLegalEvals/summary_long.csv results
uv run python blog/make_figures.py   # writes blog/figures/*.png
```

`blog/make_figures.py` writes nine static PNGs and uses cached Hugging Face org avatars from `blog/logos/`. Two of them cover the LEXam IDK calibration, `mcq_idk_score.png` (penalty-adjusted score) and `mcq_abstention.png` (how often each model picks the reserved "I don't know" letter), and read the per-task `idk_score` / `idk_freq` values from `summary_long.csv`. The first run downloads missing logos; later runs use the checked-in cache.

### Task profiles and generation caps

`configs/tasks.yaml` selects which benchmark families to run and sets `generation_size` (max new tokens) under each task block. Example:

```yaml
profiles:
  default:
    slds:
      generation_size: 32768
    swiltrabench:
      scope: scoped
      generation_size: 32768
    lexam:
      generation_size: 32768
      open_question: true
      mcq: true
```

All task families use the same 32k generation cap for simplicity. SwiLTra still applies `max(level default, generation_size)` so paragraph-level tasks respect the configured floor. LEXam MCQ and open questions share `lexam.generation_size`.

`run.py` passes the active profile to lighteval via `SWISSLEGALEVALS_TASKS_CONFIG` and `SWISSLEGALEVALS_TASKS_PROFILE`.

Model entries can narrow the global groups with a `tasks:` field. The translation specialist `hy-mt2-30b` uses `tasks: [swiltrabench]`, so it never runs SLDS or LEXam.

Temporary cap-comparison profiles live in `configs/tasks_cap_compare.yaml` (`cap16k`, `cap32k`). Count default-profile samples with:

```bash
uv run python scripts/count_task_samples.py
```

## Maintainer and cluster scripts

The scripts in `scripts/` are mostly for the Hopper Slurm environment used for the published run. They are useful references, but they are not a portable cloud deployment layer.

| Script                           | Purpose                                                 |
|----------------------------------|---------------------------------------------------------|
| `scripts/launch_all.sh`          | Submit one Slurm job per configured benchmark model.    |
| `scripts/launch_model.sh`        | Submit a single model with the configured GPU count.    |
| `scripts/launch_eval.sh`         | Slurm worker that loads modules and runs the evaluator. |
| `scripts/setup_vllm.sh`          | Hopper-specific vLLM/CUDA/glibc setup patch.            |
| `scripts/count_task_samples.py`  | Print per-task and per-group sample counts.             |
| `scripts/run_cap_compare.sh`     | Run 16k vs 32k generation-cap experiments.              |
| `scripts/analyze_cap_compare.py` | Summarize cap-comparison outputs.                       |

## Implementation notes

These choices explain the shape of the repo if you are getting oriented:

1. **Generation is scored by paper-grounded judges.** SLDS, SwiLTra-Bench, and LEXam open questions use the judge prompts/models from their papers rather than BLEU, ROUGE, or embedding metrics. MCQ tasks use extractive scoring (`trad_score`, `idk_score`, `extract_fail`) instead of a judge.
2. **Judge configuration is separate from model configuration.** `configs/judges.yaml` controls the scorer models and providers; `configs/models.yaml` controls the models being evaluated. Both can use API providers, while local evaluation models use `vllm`.
3. **lighteval is pinned to a git revision.** `pyproject.toml` uses `huggingface/lighteval@4d470292936b9ec5523cb495b4165cc4f77bcc77` because these Swiss legal tasks were not in the PyPI release when this repo was built.
4. **The default SwiLTra-Bench profile is scoped.** The normal run uses one granularity per translation dataset (`text_level`, `paragraph_level`, `press_release`) to keep the full benchmark feasible. Use `swiltrabench_full` for all levels.
5. **LEXam MCQ with IDK stops at 16 choices.** `mcq_32` with IDK would need labels beyond `A-Z`, so it is disabled until the prompt/scorer supports wider labels.
6. **Reasoning-model outputs are normalized before scoring.** `postprocess.py` strips common chain-of-thought wrappers and final-answer markers before judges or MCQ extraction see the text.
7. **API generations stream.** HF inference-provider model calls stream chat completions so long generations keep the router connection alive instead of timing out.
8. **Transient provider failures are isolated per sample.** Model and judge calls retry independently; exhausted calls are logged as missing judgments or generations so one provider outage does not discard an entire batch.
9. **Existing sample caches are reused across restarts.** If lighteval assigns an equivalent task a different in-process hash, the fullest previously loaded cache is selected and only missing samples are regenerated.

## Layout

```
configs/
  judges.yaml              judge providers and model IDs
  models.yaml              published benchmark model list
  models_smoke.yaml        tiny infra smoke models
  tasks.yaml               default/full task groups and generation caps
  tasks_cap_compare.yaml   16k vs 32k cap profiles
src/swiss_legal_evals/
  aggregate.py             lighteval JSON -> summary CSVs
  cuda_preload.py          nvJitLink preload helper for Hopper vLLM setup
  plot.py                  Plotly family charts
  postprocess.py           reasoning-output cleanup and MCQ extraction helpers
  providers.py             provider/API-key validation
  run.py                   CLI entry point that builds and launches lighteval
  task_lists.py            YAML profiles -> lighteval task strings
  tasks.py                 custom Swiss legal TASKS_TABLE and judge metrics
scripts/
  launch_all.sh            Slurm submitter for the full model set
  launch_model.sh          Slurm submitter for one model
  launch_eval.sh           Slurm worker
  setup_vllm.sh            Hopper vLLM setup patch
  count_task_samples.py    sample-count utility
  run_cap_compare.sh       generation-cap comparison runner
  analyze_cap_compare.py   cap-comparison analysis
blog/
  swiss-legal-evals-2026.md
  make_figures.py
  figures/                 static PNGs used by the post
  logos/                   cached Hugging Face org avatars
tests/                     pytest coverage for runner, configs, tasks, aggregation, postprocessing
.github/workflows/ci.yml   ruff + pytest CI
```

## Citation

If you use SwissLegalEvals, cite the repository:

```bibtex
@misc{niklaus2026swisslegalevals,
  author       = {Joel Niklaus},
  title        = {The state-of-the-art in open-source AI for Swiss legal tasks},
  year         = {2026},
  howpublished = {\url{https://github.com/JoelNiklaus/SwissLegalEvals}},
  note         = {SLDS, SwiLTra-Bench, and LEXam evaluations with lighteval}
}
```

## License

MIT - see [LICENSE](LICENSE).
