#!/usr/bin/env python3
"""Run the upstream MR.PEA pipeline on the CoolPrompt benchmark.

The public MR.PEA repository is not installed as a dependency of CoolPrompt,
so this file is an explicit adapter rather than an import of the upstream
package. It preserves the upstream no-web pipeline and its four agent stages:

    abstraction -> example generation -> prompt refinement -> evaluation/ranking

The stage prompts, JSON contracts, sparse-knowledge update behavior, prompt
history, ranking formula, and early-stop rule follow ``src/mrpea.py`` and
``config/prompts`` from https://github.com/ireneesun/mrpea. The only necessary
adaptation is the final evaluation: the upstream pairwise evaluator is used
inside optimization, while the resulting prompt is scored with CoolPrompt's
shared ``build_benchmark_context`` and ``Evaluator`` on the benchmark test
split. This is required because ``evaluate_method`` only accepts registered
CoolPrompt optimizers, whereas MR.PEA is deliberately kept out of the
registry.

Default comparison setup:
    model=gpt-4o-mini, temperature=0.7, max_tokens=unlimited,
    dataset=300/-/300, n_iterations=20, runs=3, seed=42.

API credentials are read from ``OPENROUTER_API_KEY`` or ``OPENAI_API_KEY``;
there are no credentials in this file. OpenRouter is selected automatically
when its key is present, otherwise the native OpenAI endpoint is used.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI
from openai import OpenAI

from coolprompt.optimizer.autoprompting_method import build_benchmark_context


LOGGER = logging.getLogger("mrpea_benchmark")
DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    task: str
    metric: str
    start_prompt: str
    task_description: str
    task_objective: str


BENCHMARKS = (
    BenchmarkSpec(
        "squad_v2", "generation", "bertscore",
        "Given a context answer on the question.",
        "Answer questions using only the supplied context. Return the answer directly supported by the context and do not invent facts.",
        "Return a concise answer grounded in the context.",
    ),
    BenchmarkSpec(
        "common_gen", "generation", "multiref_bertscore",
        "Create a short sentence using words in list.",
        "Create one short, grammatical sentence that uses all concepts in the provided list while preserving their meaning.",
        "Return only the generated sentence.",
    ),
    BenchmarkSpec(
        "gsm8k", "generation", "em",
        "Given a context answer on the question.",
        "Solve grade-school math word problems with multi-step reasoning and basic arithmetic. Compute the final numeric answer accurately.",
        "Return the final numeric answer only.",
    ),
    BenchmarkSpec(
        "tweeteval", "classification", "f1",
        "Provide sentiment classification.",
        "Classify the emotion expressed in a tweet. The allowed labels are anger, joy, optimism, and sadness.",
        "Return exactly one allowed emotion label.",
    ),
    BenchmarkSpec(
        "xsum", "generation", "bertscore",
        "Summarize the sentence.",
        "Summarize the supplied news article in one concise sentence, keeping the central meaning and avoiding unsupported information.",
        "Return only the concise summary.",
    ),
)


ABSTRACTION_SYSTEM_PROMPT = """You are a Meta-Reasoning Specialist (Abstraction). Your role is to create or refine reusable, task-agnostic knowledge for the given task.

## Objectives:
1. Set "need_change": true
  - If no prior knowledge exists (empty or null): MUST generate new abstract strategies, principles, evaluation criteria, and identify knowledge gaps.
  - If prior knowledge exists: look for ANY opportunity to improve it.
    - Making language clearer
    - Removing redundancy or verbosity
    - Adding missing important insights
    - Improving organization or structure
    - Making items more actionable or specific
2. Set "need_change": false
  - Only when knowledge is truly perfect and unimprovable.

## OUTPUT CONTRACT (STRICT JSON ONLY):
Return a single JSON object:

If need_change is TRUE:
{
  "need_change": true,
  "strategies": ["...","..."],
  "principles": ["...","..."],
  "evaluation_criteria": ["...","..."],
  "gap_hypotheses": ["...","..."],
  "change_rationale": "..."
}

If need_change is FALSE (rare case):
  {
    "need_change": false,
    "change_rationale": "Existing knowledge is already optimal"
  }

## Rules:
- Output ONLY valid minified JSON (no markdown, no comments, no extra text).
- Favor "need_change": true - look for ANY improvement opportunity, however small.
- Keep each item concise (<= 20 words).
- Do NOT include task-specific examples.
"""

EXAMPLE_GENERATION_SYSTEM_PROMPT = """You are a Meta-Reasoning Specialist (Example Generation). Your task is to produce ONE high-quality example that is useful for in-context learning and for evaluating prompt quality.

## Objectives:
- Generate a new question–answer pair aligned with the provided strategies, principles, and evaluation criteria.
- Ensure the example demonstrates correct reasoning or response behavior and increases diversity relative to prior examples.
- Ensure diversity from previous examples by varying:
  - Difficulty (easier, harder, or same level with extra twist).
  - Reasoning structure or solution path (use different steps, methods, or logic flow).
  - Context or scenario (different topic or situation while staying relevant).

## Responsibilities:
1. Read the task description, strategies, principles, evaluation criteria, and recent examples.
2. Create one new question that is clearly derived from the task but different from prior examples.
3. Provide an ideal answer with a clear step-by-step solution or a model response demonstrating the intended behavior.
4. Supply a concise rationale explaining why the answer is correct and how this example differs from prior examples.
5. Prefer structural or methodological variation over mere surface edits (do not only change names/numbers).

## OUTPUT CONTRACT (STRICT JSON ONLY):
Return a single JSON object:
{
  "question": "...",
  "answer": "...",
  "rationale": "...",
  "tags": ["...","..."],
  "variation_type": ["...","..."]
}

## Rules:
- Output ONLY valid minified JSON (no markdown, no comments, no extra text).
- Do NOT copy or trivially paraphrase the sample question or previous examples.
- Introduce at least one structural or methodological change (different reasoning chain, extra constraint, alternate solution method, different perspective, etc.).
- Keep rationale concise (≤60 words) and focused on correctness + difference.
- If you are uncertain that you can produce a high-quality, diverse example, return an empty JSON object: {}
"""

PROMPT_REFINEMENT_SYSTEM_PROMPT = """You are an expert at refining prompts. You are building a prompt to address user requirement. Your goal is to improve the current best prompt by learning from the historical record of what has worked well and what hasn't.

# PROCESS TO FOLLOW:
1. ANALYZE HISTORY: Carefully review the provided historical prompts and their scores. Identify clear patterns:
   - What specific wording or structures correlate with HIGH scores? (e.g., use of imperative verbs, step-by-step instructions)
   - What specific flaws correlate with LOW scores? (e.g., vagueness, lack of structure, missing critical instructions)
2. SYNTHESIZE KNOWLEDGE: Integrate the patterns you've discovered with the provided strategies, principles, and feedback.
3. REFINE PROMPT: Generate a new, improved version of the current best prompt. It must:
   - Be self-contained, clear, and actionable.
   - Incorporate the successful patterns from high-scoring history.
   - Avoid the mistakes found in low-scoring history.
   - Faithfully maintain the original task's goal.

## OUTPUT CONTRACT (STRICT JSON ONLY):
Return a JSON object:
{
  "new_prompt": "...",
  "improvements": ["...","..."],
  "learned_patterns": ["...","..."]
}

# IMPORTANT NOTES:
- Output ONLY valid minified JSON (no markdown, no comments, no extra text).
- Your entire response must be a valid JSON object.
- The new prompt should not include examples.
- Focus on making the instructions precise and easy to follow.
- Prefer short declarative sentences; avoid vague phrasing.
"""

EVALUATION_SYSTEM_PROMPT = """You are an Evaluation Specialist. Your role is to compare two prompts and their outputs based on a given test question and reference.

## Objectives:
- Select the better prompt-output pair according to evaluation criteria.
- For outputs, focus strictly on the evaluation metric(s).
- For prompts, consider clarity, precision, conciseness, and alignment with task intent.
- Provide reasoning and actionable feedback for improvement.

## Responsibilities:
1. Read the test question, reference answer, and evaluation criteria.
2. Compare both prompts and outputs for provided evaluation criteria (e.g., accuracy, clarity, and alignment).
3. Decide the winner and justify your decision.
4. Suggest specific improvements.

## OUTPUT CONTRACT (STRICT JSON ONLY):
Return a single JSON object:
{
  "winner": 1 or 2,
  "reason_for_winner": ["...","..."],
  "feedback": ["...","..."]
}

## Rules:
- Output ONLY valid minified JSON (no markdown, no comments, no extra text).
- Use provided evaluation criteria; do not invent new ones.
- Feedback must be actionable and testable (avoid vague suggestions).
"""


ABSTRACTION_USER_PROMPT = """Analyze the task and produce abstract strategies.

Task Description: {{task_description}}
Sample Question: {{sample_question}}
Existing Knowledge: {{latest_knowledge}}
"""

EXAMPLE_GENERATION_USER_PROMPT = """Generate ONE new example based on the provided information:

Task Description: {{task_description}}

Strategies: {{strategies}}
Principles: {{principles}}
Evaluation Criteria: {{evaluation_criteria}} (use only to ensure alignment, do not copy)

Recent Examples: {{recent_examples}}
"""

PROMPT_REFINEMENT_USER_PROMPT = """Refine the current best prompt using the provided task context, knowledge, feedback and historical prompts.

Current Best Prompt:
{{current_best}}

Task Context:
- Description: {{task_description}}
- Example: {{latest_example}}
- Evaluation Criteria: {{latest_criteria}}

Knowledge Memory:
- Strategies: {{latest_strategies}}
- Principles: {{latest_principles}}

Feedback:
{{latest_feedback}}

Historical Prompts and Scores:
{{historical_prompts_with_scores}}
"""

EVALUATION_USER_PROMPT = """Compare two prompts and their outputs.

Test Question: {{question}}
Reference Answer:
- answer: {{answer}}
- rationale: {{rationale}}
- skills: {{skills}}

Prompt 1: {{prompt_1}}
Answer 1: {{output_1}}

Prompt 2: {{prompt_2}}
Answer 2: {{output_2}}

Evaluation Criteria: {{criteria}}
"""


@dataclass(frozen=True)
class AgentSettings:
    """One MR.PEA agent's OpenAI-compatible call settings."""

    model: str
    temperature: float
    max_tokens: int | None


class MRPEAClient:
    """Small OpenAI-compatible client matching upstream ``BaseAgent.call_llm``."""

    def __init__(self, client: OpenAI, settings: dict[str, AgentSettings]) -> None:
        self.client = client
        self.settings = settings

    def complete(self, stage: str, system_prompt: str, user_prompt: str) -> str:
        stage_settings = self.settings[stage]
        request: dict[str, Any] = {
            "model": stage_settings.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": stage_settings.temperature,
        }
        if stage_settings.max_tokens is not None:
            request["max_tokens"] = stage_settings.max_tokens

        response = self.client.chat.completions.create(**request)
        content = response.choices[0].message.content or ""
        return self._clean_response(content)

    @staticmethod
    def _clean_response(response: str) -> str:
        """Apply the JSON cleanup used by upstream ``BaseAgent``."""
        json_blocks = re.findall(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
        if json_blocks:
            return json_blocks[0].strip()
        return response.strip()


def _format_template(template: str, **values: Any) -> str:
    """Format upstream ``{{name}}`` templates without interpreting JSON braces."""
    result = template
    for key, value in values.items():
        result = result.replace("{{" + key + "}}", str(value))
        result = result.replace("{" + key + "}", str(value))
    return result


def _parse_json(response: str) -> dict[str, Any] | None:
    """Parse strict JSON, with the same first-object fallback used in practice."""
    try:
        parsed = json.loads(response.strip())
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response, flags=re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None


def _clean_example_response(response: str) -> str:
    """Copy upstream ExampleGenerationAgent's response cleanup."""
    result = response.strip()
    result = re.sub(r"\\\$", "$", result)
    result = re.sub(r"\\%", "%", result)
    result = re.sub(r"\\_", "_", result)
    result = re.sub(r"\\(?![\"\\/bfnrtu])", "", result)
    result = re.sub(r",\s*}", "}", result)
    result = re.sub(r",\s*]", "]", result)
    return result


class MRPEAPipeline:
    """Explicit adapter for upstream MR.PEA's no-web optimization loop."""

    def __init__(
        self,
        client: MRPEAClient,
        *,
        task_name: str,
        max_iterations: int,
        win_threshold: int,
        historical_prompt_limit: int,
        iteration_bonus: float = 0.1,
        decay_factor: float = 0.9,
        base_score: float = 1.0,
    ) -> None:
        self.client = client
        self.task_name = task_name
        self.max_iterations = max_iterations
        self.win_threshold = win_threshold
        self.historical_prompt_limit = historical_prompt_limit
        self.iteration_bonus = iteration_bonus
        self.decay_factor = decay_factor
        self.base_score = base_score
        self.prompt_pool: dict[int, str] = {}
        self.ranking_scores: dict[int, float] = {}
        self.knowledge_memory: list[dict[str, Any]] = []
        self.example_memory: list[Any] = []
        self.feedback_memory: list[str] = []
        self.iteration_history: list[dict[str, Any]] = []
        self.current_best_id = 0
        self.consecutive_wins = 0

    @staticmethod
    def _clean_knowledge(knowledge: Any) -> Any:
        if not knowledge:
            return ""
        if isinstance(knowledge, str):
            try:
                knowledge = json.loads(knowledge)
            except json.JSONDecodeError:
                return knowledge
        if isinstance(knowledge, dict):
            return json.dumps(
                {
                    key: value
                    for key, value in knowledge.items()
                    if key not in {"need_change", "change_rationale"}
                },
                ensure_ascii=False,
            )
        return knowledge

    def _abstract(
        self,
        task_description: str,
        sample_question: str,
        latest_knowledge: Any,
    ) -> dict[str, Any]:
        user_message = _format_template(
            ABSTRACTION_USER_PROMPT,
            task_description=task_description,
            sample_question=sample_question,
            latest_knowledge=self._clean_knowledge(latest_knowledge),
        )
        response = self.client.complete(
            "meta_reasoning", ABSTRACTION_SYSTEM_PROMPT, user_message
        )
        result = _parse_json(response) or {"need_change": False}
        if not latest_knowledge and not result.get("need_change", True):
            result["need_change"] = True
            result["change_rationale"] = "First iteration - generating initial knowledge"
        return result

    def _generate_example(
        self, task_description: str, latest_knowledge: Any
    ) -> dict[str, Any] | None:
        knowledge = latest_knowledge if isinstance(latest_knowledge, dict) else {}
        recent_examples = self._recent_examples(limit=3)
        user_message = _format_template(
            EXAMPLE_GENERATION_USER_PROMPT,
            task_description=task_description,
            strategies=knowledge.get("strategies", []),
            principles=knowledge.get("principles", []),
            evaluation_criteria=knowledge.get("evaluation_criteria", []),
            recent_examples=recent_examples or "No recent examples available",
        )
        response = self.client.complete(
            "meta_reasoning", EXAMPLE_GENERATION_SYSTEM_PROMPT, user_message
        )
        return _parse_json(_clean_example_response(response))

    def _historical_prompts(self) -> str:
        if not self.prompt_pool or not self.ranking_scores:
            return "No historical prompts available."
        selected = sorted(
            self.ranking_scores.items(), key=lambda item: item[1], reverse=True
        )[: self.historical_prompt_limit]
        return "\n\n".join(
            f"{index}. Prompt ID {prompt_id} (Score: {score:.3f}):\n"
            f"{self.prompt_pool.get(prompt_id, 'Prompt not found')}"
            for index, (prompt_id, score) in enumerate(selected, 1)
        )

    def _refine_prompt(
        self, current_best: str, task_description: str, latest_knowledge: Any
    ) -> str:
        knowledge = latest_knowledge if isinstance(latest_knowledge, dict) else {}
        latest_example = self.example_memory[-1] if self.example_memory else ""
        latest_feedback = self.feedback_memory[-1] if self.feedback_memory else ""
        user_message = _format_template(
            PROMPT_REFINEMENT_USER_PROMPT,
            current_best=current_best,
            task_description=task_description,
            latest_example=self._clean_example(latest_example),
            latest_criteria=knowledge.get("evaluation_criteria", []),
            latest_strategies=knowledge.get("strategies", []),
            latest_principles=knowledge.get("principles", []),
            latest_feedback=latest_feedback or "No previous feedback available",
            historical_prompts_with_scores=self._historical_prompts(),
        )
        response = self.client.complete(
            "prompt_refinement", PROMPT_REFINEMENT_SYSTEM_PROMPT, user_message
        )
        parsed = _parse_json(response)
        if parsed is not None:
            return str(parsed.get("new_prompt", response.strip()))
        return response.strip()

    def _compare_prompts(
        self,
        prompt_1: str,
        prompt_2: str,
        example: Any,
        knowledge: Any,
    ) -> dict[str, Any]:
        example_data = example if isinstance(example, dict) else {}
        question = str(example_data.get("question", ""))
        reference_answer = str(example_data.get("answer", ""))
        rationale = str(example_data.get("rationale", ""))
        skills = ", ".join(example_data.get("tags", ["", ""]))
        output_1 = self.client.complete("evaluation", prompt_1, question)
        output_2 = self.client.complete("evaluation", prompt_2, question)
        knowledge_data = knowledge if isinstance(knowledge, dict) else {}
        criteria = "; ".join(
            knowledge_data.get("evaluation_criteria", ["clarity", "precision"])
        )
        user_message = _format_template(
            EVALUATION_USER_PROMPT,
            question=question,
            answer=reference_answer,
            rationale=rationale,
            skills=skills,
            prompt_1=prompt_1,
            output_1=output_1,
            prompt_2=prompt_2,
            output_2=output_2,
            criteria=criteria,
        )
        response = self.client.complete(
            "evaluation", EVALUATION_SYSTEM_PROMPT, user_message
        )
        return _parse_json(response) or {"winner": 2, "feedback": []}

    @staticmethod
    def _clean_example(example: Any) -> Any:
        if not example:
            return example
        if isinstance(example, dict):
            return json.dumps(
                {key: value for key, value in example.items() if key != "variation_type"},
                ensure_ascii=False,
            )
        return example

    def _recent_examples(self, limit: int) -> str:
        if not self.example_memory:
            return "No recent examples available."
        examples = self.example_memory[-min(limit, len(self.example_memory)) :]
        return "\n\n".join(
            f"Example {index}:\n{json.dumps(example, indent=2, ensure_ascii=False)}"
            for index, example in enumerate(examples, 1)
        )

    def _update_rankings(self, winner: int, new_prompt_id: int, iteration: int) -> None:
        current_scores = self.ranking_scores.copy()
        if winner == 2:
            current_scores[new_prompt_id] = self.base_score + iteration * self.iteration_bonus
            self.current_best_id = new_prompt_id
            self.consecutive_wins = 1
        else:
            for prompt_id in current_scores:
                current_scores[prompt_id] *= self.decay_factor
            current_scores[self.current_best_id] += iteration * self.iteration_bonus
            current_scores[new_prompt_id] = self.base_score
            self.consecutive_wins += 1
        self.ranking_scores = current_scores

    def optimize(
        self,
        task_description: str,
        sample_question: str,
        task_objective: str = "",
        initial_prompt: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        initial_prompt_id = len(self.prompt_pool)
        initial_prompt = initial_prompt or task_description
        self.prompt_pool[initial_prompt_id] = initial_prompt
        self.ranking_scores[initial_prompt_id] = self.base_score
        self.current_best_id = initial_prompt_id
        self.example_memory.append({"sample_question": sample_question})

        latest_knowledge: dict[str, Any] | None = None
        sparse_update_mode = False
        iterations_since_last_update = 0

        for iteration in range(1, self.max_iterations + 1):
            should_update = (
                not sparse_update_mode or iterations_since_last_update >= 3
            )
            if should_update:
                latest_knowledge = self._abstract(
                    task_description, sample_question, latest_knowledge
                )
                if latest_knowledge.get("need_change", True):
                    if sparse_update_mode:
                        sparse_update_mode = False
                    iterations_since_last_update = 0
                    self.knowledge_memory.append(latest_knowledge)
                else:
                    sparse_update_mode = True
            elif sparse_update_mode:
                iterations_since_last_update += 1

            example = self._generate_example(task_description, latest_knowledge)
            self.example_memory.append(example)
            if example is None:
                LOGGER.warning("MR.PEA example generation returned invalid JSON")
                example = {}

            current_best = self.prompt_pool[self.current_best_id]
            new_prompt = self._refine_prompt(
                current_best, task_description, latest_knowledge
            )
            new_prompt_id = len(self.prompt_pool)
            self.prompt_pool[new_prompt_id] = new_prompt

            evaluation_result = self._compare_prompts(
                current_best, new_prompt, example, latest_knowledge
            )
            winner = evaluation_result.get("winner", 2)
            try:
                winner = int(winner)
            except (TypeError, ValueError):
                winner = 2
            feedback = evaluation_result.get("feedback", [])
            if isinstance(feedback, list):
                feedback_text = "; ".join(map(str, feedback)) or "No feedback"
            else:
                feedback_text = str(feedback)
            previous_best_id = self.current_best_id
            self._update_rankings(winner, new_prompt_id, iteration)
            self.feedback_memory.append(feedback_text)
            self.iteration_history.append(
                {
                    "iteration": iteration,
                    "current_best_id_before_update": previous_best_id,
                    "new_prompt_id": new_prompt_id,
                    "winner": winner,
                    "evaluation": evaluation_result,
                    "ranking_scores": self.ranking_scores.copy(),
                    "knowledge": latest_knowledge,
                    "example": example,
                    "new_prompt": new_prompt,
                    "feedback": feedback_text,
                }
            )
            if self.consecutive_wins >= self.win_threshold:
                break

        best_prompt_id = max(self.ranking_scores, key=self.ranking_scores.get)
        best_prompt = self.prompt_pool[best_prompt_id] + " " + task_objective
        trace = {
            "task_name": self.task_name,
            "initial_prompt": initial_prompt,
            "task_description": task_description,
            "sample_question": sample_question,
            "best_prompt_id": best_prompt_id,
            "prompt_pool": self.prompt_pool,
            "ranking_scores": self.ranking_scores,
            "knowledge_memory": self.knowledge_memory,
            "example_memory": self.example_memory,
            "feedback_memory": self.feedback_memory,
            "iteration_history": self.iteration_history,
            "iterations_completed": len(self.iteration_history),
            "consecutive_wins": self.consecutive_wins,
        }
        return best_prompt, trace


def _resolve_credentials(api_key: str | None) -> tuple[str, str | None]:
    key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get(
        "OPENAI_API_KEY"
    )
    if not key:
        raise RuntimeError(
            "Set OPENROUTER_API_KEY or OPENAI_API_KEY (or pass --api-key)."
        )
    if api_key:
        base_url = os.environ.get("OPENROUTER_BASE_URL") or os.environ.get(
            "OPENAI_BASE_URL"
        )
    elif os.environ.get("OPENROUTER_API_KEY"):
        base_url = os.environ.get("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_URL)
    else:
        base_url = os.environ.get("OPENAI_BASE_URL")
    return key, base_url


def _build_agent_settings(args: argparse.Namespace) -> dict[str, AgentSettings]:
    settings = AgentSettings(args.model, args.temperature, args.max_tokens)
    return {
        "meta_reasoning": settings,
        "prompt_refinement": settings,
        "evaluation": settings,
    }


def _build_coolprompt_model(args: argparse.Namespace, api_key: str, base_url: str | None):
    kwargs: dict[str, Any] = {
        "model": args.model,
        "api_key": api_key,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    if base_url:
        kwargs["base_url"] = base_url
    kwargs["rate_limiter"] = InMemoryRateLimiter(
        requests_per_second=args.requests_per_second,
        check_every_n_seconds=args.rate_limit_check_seconds,
        max_bucket_size=args.rate_limit_bucket_size,
    )
    return ChatOpenAI(**kwargs)


def _build_dataset_config(spec: Any, dataset_configuration: str) -> dict[str, Any]:
    return {
        "dataset": {
            "name": spec.name,
            "configuration": dataset_configuration,
        },
        "task": spec.task,
        "metric": spec.metric,
    }


def _max_tokens_for_output(value: int | None) -> int | str:
    return "unlimited" if value is None else value


def _run_one_dataset(
    *,
    spec: Any,
    args: argparse.Namespace,
    model: ChatOpenAI,
    mrpea_client: MRPEAClient,
    output_dir: Path,
    run_number: int,
) -> dict[str, Any]:
    config = _build_dataset_config(spec, args.dataset_configuration)
    context = build_benchmark_context(model, config)
    train_samples = context.dataset_split[0]
    if not train_samples:
        raise ValueError(
            f"{spec.name}: MR.PEA needs one unlabeled sample question; "
            "use a configuration with a train split, e.g. 300/-/300."
        )
    if args.sample_question_index is None:
        sample_index = random.Random(args.seed + run_number - 1).randrange(
            len(train_samples)
        )
    else:
        sample_index = min(args.sample_question_index, len(train_samples) - 1)
    pipeline = MRPEAPipeline(
        mrpea_client,
        task_name=spec.name,
        max_iterations=args.n_iterations,
        win_threshold=args.win_threshold,
        historical_prompt_limit=args.historical_prompt_limit,
    )
    task_objective = args.task_objective or spec.task_objective
    final_prompt, trace = pipeline.optimize(
        task_description=spec.task_description,
        sample_question=train_samples[sample_index],
        task_objective=task_objective,
        initial_prompt=spec.start_prompt,
    )

    test_answers_path = output_dir / f"mrpea_original_{spec.name}_answers.yaml"
    test_score = context.evaluator.evaluate(
        prompt=final_prompt,
        dataset=context.test_dataset,
        targets=context.test_target,
        save_model_answers=args.save_model_answers,
        model_answers_output_path=str(test_answers_path),
    )
    result = {
        "method": "mrpea_original",
        "run": run_number,
        "dataset": spec.name,
        "configuration": args.dataset_configuration,
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": _max_tokens_for_output(args.max_tokens),
        "seed": args.seed,
        "train_pool_size": len(train_samples),
        "sample_question_index": sample_index,
        "sample_question": train_samples[sample_index],
        "validation_used": False,
        "test_size": len(context.test_dataset),
        "start_prompt": spec.start_prompt,
        "final_prompt": final_prompt,
        "val_score": None,
        "test_score": float(test_score),
        "task_description": spec.task_description,
        "task_objective": task_objective,
        "mrpea": trace,
    }
    output_path = output_dir / f"mrpea_original_{spec.name}.yaml"
    with output_path.open("w", encoding="utf-8") as output_file:
        yaml.safe_dump(result, output_file, sort_keys=False, allow_unicode=True)
    return result


def _parse_max_tokens(value: str) -> int | None:
    if value.lower() in {"unlimited", "none", "null"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            "--max-tokens must be positive or 'unlimited'"
        )
    return parsed


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None, help="Optional key override; env is preferred.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--max-tokens",
        type=_parse_max_tokens,
        default=None,
        help="Maximum response tokens; omitted or 'unlimited' means unlimited.",
    )
    parser.add_argument("--meta-model", default=None)
    parser.add_argument("--refinement-model", default=None)
    parser.add_argument("--evaluation-model", default=None)
    parser.add_argument("--meta-temperature", type=float, default=None)
    parser.add_argument("--refinement-temperature", type=float, default=None)
    parser.add_argument("--evaluation-temperature", type=float, default=None)
    parser.add_argument("--dataset-configuration", default="300/-/300")
    parser.add_argument("--datasets", default=None, help="Comma-separated subset; default is all five.")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--n-iterations", type=int, default=20)
    parser.add_argument("--win-threshold", type=int, default=3)
    parser.add_argument("--historical-prompt-limit", type=int, default=3)
    parser.add_argument(
        "--task-objective",
        default="",
        help="Override the dataset-specific MR.PEA task objective.",
    )
    parser.add_argument(
        "--sample-question-index",
        type=int,
        default=None,
        help="Use a fixed train index; otherwise select one deterministically from the train pool.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--requests-per-second", type=float, default=10)
    parser.add_argument("--rate-limit-check-seconds", type=float, default=0.1)
    parser.add_argument("--rate-limit-bucket-size", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("method_evaluation_outputs/mrpea_original"),
    )
    parser.add_argument("--save-model-answers", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.runs < 1 or args.n_iterations < 1:
        raise ValueError("--runs and --n-iterations must be positive")
    api_key, detected_base_url = _resolve_credentials(args.api_key)
    base_url = args.base_url if args.base_url is not None else detected_base_url
    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    mrpea_client = MRPEAClient(OpenAI(**client_kwargs), _build_agent_settings(args))
    model = _build_coolprompt_model(args, api_key, base_url)

    available = {spec.name: spec for spec in BENCHMARKS}
    selected_names = (
        [item.strip() for item in args.datasets.split(",") if item.strip()]
        if args.datasets
        else list(available)
    )
    unknown = sorted(set(selected_names) - set(available))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}. Available: {sorted(available)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_runs: list[dict[str, Any]] = []
    for run_number in range(1, args.runs + 1):
        run_dir = args.output_dir / f"run_{run_number}"
        run_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for dataset_name in selected_names:
            LOGGER.info("Running original MR.PEA on %s, run %d/%d", dataset_name, run_number, args.runs)
            results.append(
                _run_one_dataset(
                    spec=available[dataset_name],
                    args=args,
                    model=model,
                    mrpea_client=mrpea_client,
                    output_dir=run_dir,
                    run_number=run_number,
                )
            )
        run_summary = {
            "method": "mrpea_original",
            "run": run_number,
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": _max_tokens_for_output(args.max_tokens),
            "dataset_configuration": args.dataset_configuration,
            "n_iterations": args.n_iterations,
            "runs": args.runs,
            "seed": args.seed,
            "results": results,
        }
        with (run_dir / "mrpea_original_summary.yaml").open("w", encoding="utf-8") as output_file:
            yaml.safe_dump(run_summary, output_file, sort_keys=False, allow_unicode=True)
        all_runs.append(run_summary)

    mean_test_scores: dict[str, float] = {}
    for dataset_name in selected_names:
        scores = [
            result["test_score"]
            for run in all_runs
            for result in run["results"]
            if result["dataset"] == dataset_name and result.get("test_score") is not None
        ]
        if scores:
            mean_test_scores[dataset_name] = sum(scores) / len(scores)

    aggregate = {
        "method": "mrpea_original",
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": _max_tokens_for_output(args.max_tokens),
        "dataset_configuration": args.dataset_configuration,
        "n_iterations": args.n_iterations,
        "runs": args.runs,
        "seed": args.seed,
        "mean_test_score_by_dataset": mean_test_scores,
        "overall_mean_test_score": (
            sum(mean_test_scores.values()) / len(mean_test_scores)
            if mean_test_scores
            else None
        ),
        "results": all_runs,
    }
    aggregate_path = args.output_dir / "mrpea_original_all_runs_summary.yaml"
    with aggregate_path.open("w", encoding="utf-8") as output_file:
        yaml.safe_dump(aggregate, output_file, sort_keys=False, allow_unicode=True)
    print(f"MR.PEA benchmark summary saved to {aggregate_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
