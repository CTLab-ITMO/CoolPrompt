# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CoolPrompt is a Python framework for automatic prompt engineering and optimization for LLMs. It provides four optimization methods (HyPE, ReflectivePrompt, DistillPrompt, PE2), multiple evaluation metrics, and synthetic data generation. Published on PyPI as `coolprompt`.

## Common Commands

Always use `uv` to run Python in this project — never bare `python` or `pip`.

```bash
# Install from source
uv pip install -r requirements.txt

# Build package
uv run python -m build

# Run all tests
uv run python -m unittest discover -s test

# Run a single test file
uv run python -m unittest test.coolprompt.data_generator.test_generator

# Lint
uv run flake8 coolprompt/

# Syntax-check a file
uv run python -m py_compile <file>
```

## Code Style

- PEP8 + Google Style Guide
- Flake8: max-line-length=80, ignore W503/W504
- Google-style docstrings (Args/Returns with types)
- Python >=3.12

## Architecture

**Entry point:** `PromptTuner` class in `coolprompt/assistant.py` — orchestrates the full optimization pipeline. Exported as `coolprompt.PromptTuner`.

**Four optimization methods** (in `coolprompt/optimizer/`):
- **HyPE** (`hype/`) — Non-data-driven, template injection + single LLM call. Fast baseline, no evaluation needed.
- **ReflectivePrompt** (`reflective_prompt/`) — Data-driven, evolutionary. Iteratively refines prompts via evaluation feedback. Core logic in `evoluter.py`.
- **DistillPrompt** (`distill_prompt/`) — Data-driven, knowledge distillation. Core in `distiller.py` and `generate.py`.
- **PE2** (`pe2/`) — Data-driven, beam-search. Samples failure examples, analyzes errors, and proposes refined prompts. Core in `trainer.py` and `proposer.py`.

**Evaluation** (`coolprompt/evaluator/`):
- `evaluator.py` — runs model on dataset, computes metrics
- `metrics.py` — 13+ metrics: classification (Accuracy, F1), generation (BLEU, ROUGE, METEOR, BertScore), LLM-based (LLMAsJudge, GEval, ExactMatch). All extend `BaseMetric`.

**LLM integration** (`coolprompt/language_model/`):
- `llm.py` — `DefaultLLM` wraps HuggingFace models via LangChain. Default model: Qwen/Qwen3-4B-Instruct-2507.
- `tracker.py` — `OpenAITracker` singleton for cost/token tracking. `TrackedLLMWrapper` decorates models.
- `deepeval_model.py` — adapter for deepeval library integration.

**Key enums** (`coolprompt/utils/enums.py`): `Method` (HYPE, REFLECTIVE, DISTILL, PE2), `Task` (CLASSIFICATION, GENERATION).

**Prompt templates** in `coolprompt/utils/prompt_templates/` — task/method-specific templates selected via `TEMPLATE_MAP` in `assistant.py` keyed by `(Task, Method)` tuples.

**Supporting modules:**
- `data_generator/` — synthetic dataset generation using structured output (Pydantic schemas)
- `task_detector/` — auto-detects whether a prompt is classification or generation
- `prompt_assistant/` — generates human-readable optimization feedback
- `utils/correction/` — language/grammar correction
- `utils/var_validation.py` — input validation for `PromptTuner.run()`

## Benchmarks

`src/solutions/PE2/` contains a thesis benchmark suite that runs the four
optimization methods (pe2, pe2_sgr, ape, opro) across nine datasets. Loaders
live in `src/utils/` and return DataFrames with columns `input_data` / `target`.

### Datasets

| Name | Task | Metric | HuggingFace source id | Description |
|------|------|--------|-----------------------|-------------|
| ifeval | generation | ifeval | `google/IFEval` | Instruction following with verifiable constraints |
| gsm8k | generation | em | `openai/gsm8k` | Grade-school math word problems |
| svamp | generation | em | `ChilleD/SVAMP` | Arithmetic word problems |
| sst2 | classification | accuracy | `stanfordnlp/sst2` | Movie review sentiment |
| agnews | classification | accuracy | `fancyzhx/ag_news` | News topic classification |
| trec | classification | accuracy | `OxAISH-AL-LLM/trec6` | Question type classification |
| subj | classification | accuracy | `SetFit/subj` | Subjectivity classification |
| rucola | classification | accuracy | `RussianNLP/rucola` | Russian linguistic acceptability |

The `ifeval` metric (prompt-level strict accuracy over verifiable constraints)
is implemented in `coolprompt/evaluator/metrics.py` (`IFEvalMetric`) and
`coolprompt/evaluator/ifeval_checkers.py`. The loader filters IFEval to the
instruction-type subset that coolprompt's checkers can verify.

### Local model ladder

Models are served by LM Studio's OpenAI-compatible server at
`http://localhost:1234/v1` (override via `CP_LMSTUDIO_URL`).
`src/solutions/PE2/local_models.py` exposes `make_llm(name)` which accepts
a ladder key, an Anthropic model key, or a raw LM Studio model id.

| Ladder key | LM Studio model id | Role |
|------------|--------------------|------|
| weak | `qwen/qwen3-1.7b` | Small/fast model |
| mid | `qwen3-4b-instruct-2507-mlx` | Mid-size model (default for benchmarks) |
| strong | `qwen/qwen3-14b` | Large local model |
| cross | `openai/gpt-oss-20b` | Cross-family non-Qwen check |
| judge | `qwen3-30b-a3b-instruct-2507-mlx` | Local judge for llm_as_judge metrics |

An Anthropic cross-family slot (`claude-haiku`) is also wired up; set
`CP_ANTHROPIC_KEY` to use it.

### LM Studio setup

```bash
# Add lms to PATH if needed (one-time)
export PATH="$HOME/.lmstudio/bin:$PATH"

# Start the server and load a model
lms server start
lms load <model-id>   # e.g. lms load qwen3-4b-instruct-2507-mlx
```

### Running benchmarks

```bash
# Run all four methods on gsm8k, 50 samples, mid model
uv run python src/solutions/PE2/extra_benchmarks_test.py \
    --benchmark gsm8k --model mid --sample 50

# Run a single method
uv run python src/solutions/PE2/extra_benchmarks_test.py \
    --benchmark sst2 --model mid --method pe2_sgr --sample 50
```

Key flags:
- `--benchmark` — one of the dataset names above (required)
- `--model` — ladder key or raw model id (default: `mid`)
- `--method` — one of `pe2 pe2_sgr ape opro`; omit to run all four
- `--sample` — number of rows sampled per run (default: 50, seed 42)
- `--train-steps` — PE2 training steps (default: 3)
- `--workers` — parallel task workers (default: 1)
- `--out` — output JSON path (default: `logs/extra_benchmarks_results.json`)

## CI/CD

- `.github/workflows/workflow.yml` — publishes to PyPI on version tags
- `.github/workflows/test-publish.yaml` — publishes to TestPyPI on `test_v*.*.*` tags

## Git Submodule

`src/solutions/EvoPrompt` is a git submodule. Run `git submodule update --init` if needed.
