# SwissLegalEvals

Evaluate open-source LLMs on Swiss legal benchmarks (**SLDS**, **SwiLTra-Bench**, **LEXam**) using [lighteval](https://github.com/huggingface/lighteval) with paper-grounded LLM-as-judge metrics.

## Design decisions

1. **Judge-only generative metrics** — SLDS, SwiLTra-Bench, and LEXam open questions are scored only with LLM judges. Lexical and embedding metrics (BLEU, ROUGE, BERTScore, etc.) correlate poorly with human judgments in the underlying papers; MCQ tasks use accuracy only.
2. **Paper-grounded judges** (defaults in `configs/judges.yaml`):
   - **SLDS**: [Rolshoven et al., EMNLP Findings 2025](https://arxiv.org/abs/2410.13456) — `deepseek-ai/DeepSeek-V4-Pro` via HF inference providers (5-rubric prompt from lighteval).
   - **SwiLTra-Bench**: [Niklaus et al., ACL 2025](https://arxiv.org/abs/2503.01372) — `gpt-4o-mini` via OpenAI with codebook + deduction + diverse few-shot.
   - **LEXam OQ**: [Fan et al., ICLR 2026](https://arxiv.org/abs/2505.12864) — `deepseek-ai/DeepSeek-R1-0528` via HF inference providers.
3. **lighteval from git `main`** — Swiss legal tasks are not in PyPI `lighteval` 0.13.0 yet.
4. **Configurable providers** — Judges and eval models support `openai`, `openrouter`, `hf-inference-providers`, and local `vllm`.
5. **Scoped SwiLTra-Bench** — Default profile runs lowest granularity only (`text_level`, `paragraph_level`, `press_release`); opt into all levels via profile `swiltrabench_full`.
6. **LEXam MCQ IDK only up to 16 choices** — `mcq_32` with IDK would require labels beyond `A-Z`; add a wide-label prompt/scorer before enabling it.
7. **Reasoning-model post-processing** — Judges and MCQ letter extraction strip chain-of-thought (harmony `assistantfinal`, `FR:` continuations, `###X###` / conclusion patterns). Translation tasks drop bare-`\n` stop sequences on paragraph/text levels; per-task generation caps live under each task block in `configs/tasks.yaml`.
8. **HF inference-providers streaming** — API eval models stream chat completions (`stream=True`) so long generations keep the HF router connection alive and avoid 504 gateway timeouts.

## Install

Default install for API-based evaluation, aggregation, plotting, and tests:

```bash
git clone https://github.com/JoelNiklaus/SwissLegalEvals.git
cd SwissLegalEvals
uv venv --python 3.13
GIT_LFS_SKIP_SMUDGE=1 uv sync   # skip LFS blobs in lighteval test fixtures
uv sync --extra dev             # pytest, ruff
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

`scripts/setup_vllm.sh` is not the general installation path. It is a cluster-specific patch for our Hopper nodes: it loads glibc 2.38 + CUDA 12.9, installs the cu129 manylinux_2_34 vLLM wheel into `.venv`, applies `glibc-fix`, and installs a tiny `nvJitLink` preload hook so the cu129 wheel does not accidentally pick up older system CUDA libraries. Other users should first try the normal `uv pip install vllm` path and only adapt this script if their cluster has the same kind of CUDA/glibc wheel issue.

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

Optional: set `HF_ORG_TO_BILL=<org-slug>` to bill HF Inference Providers usage to a Team/Enterprise organization instead of your personal account (the default). Applies to eval models and HF-backed judges (SLDS, LEXam OQ).

Override judge config: `SWISSLEGALEVALS_JUDGE_CONFIG=/path/to/judges.yaml`

Override tasks profile/config: `SWISSLEGALEVALS_TASKS_CONFIG=/path/to/tasks.yaml` and `SWISSLEGALEVALS_TASKS_PROFILE=default` (set automatically by `run.py`).

Use `--env-file /path/to/.env` to load a different dotenv file.

Default HF judges use `novita` because the currently configured DeepSeek judge IDs are not served by `together` on HF inference providers.

## Usage

Dry-run all models (prints lighteval commands, no API keys required):

```bash
uv run swiss-legal-evals --dry-run
```

Run all models (`evaluate` is an alias for `swiss-legal-evals`):

```bash
uv run evaluate
```

Run one model by provider:

```bash
# HF inference providers (large MoE models)
uv run swiss-legal-evals --models deepseek-v4-pro

# OpenRouter (MiMo, Mistral Large)
uv run swiss-legal-evals --models mimo-v2.5-pro mistral-large-2512

# Local vLLM (after setup_vllm.sh on a GPU node)
uv run swiss-legal-evals --models gemma-4-31b-it qwen3.5-35b-a3b

# SwiLTra-only translation specialist
uv run swiss-legal-evals --models hy-mt2-30b
```

Local `vllm` runs use `tensor_parallel_size` and `data_parallel_size` from `configs/models.yaml`. Current local models use TP4 with DP1 (4 H100s) after Ray data-parallel deadlocks on this cluster; `lfm2.5-8b` uses a single GPU. `scripts/launch_all.sh` requests `data_parallel_size * tensor_parallel_size` GPUs per vLLM job. HF Inference Provider models request **no GPUs** — only CPUs for orchestration and judging.

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

Single-task smoke test:

```bash
uv run swiss-legal-evals \
  --models gpt-oss-120b \
  --task-string 'slds:de_de|0' \
  --max-samples 1 \
  --output-dir results_smoke/hf
```

## Development checks

Run the same checks as CI before opening a pull request or release:

```bash
uv run ruff check src tests scripts/analyze_cap_compare.py
uv run pytest tests -q
```

Aggregate and plot:

```bash
uv run swiss-legal-evals-aggregate
uv run swiss-legal-evals-plot
```

Outputs: `results/results/<model>/results_*.json`, `results/summary_long.csv`, `results/summary.csv`, `plots/<family>.html`.

The completed 2026 full-run outputs are mirrored in the public Hugging Face bucket
[`joelniklaus/SwissLegalEvals`](https://huggingface.co/buckets/joelniklaus/SwissLegalEvals):

```bash
# Download the published results into ./results
hf buckets sync hf://buckets/joelniklaus/SwissLegalEvals results

# Maintainers: mirror local ./results back to the bucket
hf buckets sync results hf://buckets/joelniklaus/SwissLegalEvals --delete
```

Local result directories (`results/`, `results_smoke*`, `results_cap_compare_*`) are gitignored to keep the repository lightweight.

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

## Layout

```
configs/          models, judges, task profiles
src/swiss_legal_evals/
  tasks.py        custom TASKS_TABLE (judge-only)
  task_lists.py   build task strings from YAML
  run.py          orchestrate lighteval
  aggregate.py    JSON → pandas
  plot.py         Plotly charts
scripts/setup_vllm.sh
```

## License

MIT — see [LICENSE](LICENSE).
