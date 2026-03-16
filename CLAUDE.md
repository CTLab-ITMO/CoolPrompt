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

## CI/CD

- `.github/workflows/workflow.yml` — publishes to PyPI on version tags
- `.github/workflows/test-publish.yaml` — publishes to TestPyPI on `test_v*.*.*` tags

## Git Submodule

`src/solutions/EvoPrompt` is a git submodule. Run `git submodule update --init` if needed.
