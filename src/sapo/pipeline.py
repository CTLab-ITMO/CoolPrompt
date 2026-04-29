"""Legacy SAPO optimization pipeline.

This module keeps the original candidate -> critique -> improve loop,
while providing clearer naming, typing, and documentation.
"""

from __future__ import annotations

import ast
import json
from typing import Any

import numpy as np
from bert_score import score as bert_score

from .llm import get_openrouter_llm
from .prompt import (
    BASE_META_PROMPT,
    CANDIDATE_GEN_TEMPLATE,
    IMPROVEMENT_TEMPLATE,
    WEAKNESS_ANALYSIS_TEMPLATE,
)

Example = dict[str, str]


class LegacySAPOPipeline:
    """Iterative prompt optimizer based on candidate generation and refinement.

    The algorithm follows the original implementation:
    1) generate candidates from the current prompt,
    2) evaluate candidates with BERTScore,
    3) analyze weak examples for each candidate,
    4) improve each candidate using the recommendation,
    5) keep the best improved candidate if it beats current score.
    """

    def __init__(
        self,
        model_name: str,
        dataset: list[Example],
        n_iterations: int = 10,
        n_candidates: int = 5,
        early_stopping_rounds: int = 3,
        bertscore_device: str = "auto",
    ) -> None:
        """Initialize the optimizer.

        Args:
            model_name: OpenRouter model id.
            dataset: Fixed evaluation dataset with ``input`` and ``reference`` keys.
            n_iterations: Maximum optimization iterations.
            n_candidates: Number of generated candidates per iteration.
            early_stopping_rounds: Stop after this many non-improving rounds.
        """
        self.llm_base = get_openrouter_llm(model_name, response_type="base")
        self.llm_reasoning = get_openrouter_llm(
            model_name,
            response_type="reasoning",
            temperature=0.5,
        )
        self.llm_candgen = get_openrouter_llm(
            model_name,
            response_type="cand_gen",
            temperature=0.5,
        )

        self.dataset = dataset
        self.n_iterations = n_iterations
        self.n_candidates = n_candidates
        self.early_stopping_rounds = early_stopping_rounds
        self.bertscore_device = bertscore_device

        self.meta_prompt_template = BASE_META_PROMPT
        self.candidate_generation_template = CANDIDATE_GEN_TEMPLATE
        self.weakness_analysis_template = WEAKNESS_ANALYSIS_TEMPLATE
        self.improvement_template = IMPROVEMENT_TEMPLATE

        self.history: list[dict[str, Any]] = []
        self._resolved_bertscore_device: str | None = None

    @staticmethod
    def _format_prompt(prompt: str, input_text: str) -> str:
        """Inject input safely and auto-add ``{input}`` placeholder when missing."""
        normalized_prompt = LegacySAPOPipeline._ensure_input_placeholder(prompt)
        return normalized_prompt.replace("{input}", input_text)

    @staticmethod
    def _ensure_input_placeholder(prompt: str) -> str:
        normalized = (prompt or "").strip()
        if "{input}" in normalized:
            return normalized
        if not normalized:
            normalized = "Provide a summary of the following text."
        return f"{normalized}\n\nInput:\n{{input}}"

    def _resolve_bertscore_device(self) -> str:
        """Resolve BERTScore device from user preference with safe fallback."""
        preferred = (self.bertscore_device or "auto").strip().lower()

        try:
            import torch  # type: ignore

            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            cuda_available = False

        if preferred == "cuda":
            if cuda_available:
                return "cuda"
            print("[BERTScore] CUDA requested, but not available; falling back to CPU.")
            return "cpu"

        if preferred == "cpu":
            return "cpu"

        return "cuda" if cuda_available else "cpu"

    def evaluate(self, prompt: str) -> tuple[float, list[float], list[str]]:
        """Evaluate a prompt on the whole dataset using BERTScore F1.

        Returns:
            Tuple of ``(average_f1, per_example_f1, model_responses)``.
        """
        hypotheses: list[str] = []
        references = [example["reference"] for example in self.dataset]
        inputs = [example["input"] for example in self.dataset]

        for input_text in inputs:
            full_prompt = self._format_prompt(prompt, input_text)
            response = self.llm_base.invoke(full_prompt).response
            hypotheses.append(response)

        if self._resolved_bertscore_device is None:
            self._resolved_bertscore_device = self._resolve_bertscore_device()
            print(f"[BERTScore] Using device: {self._resolved_bertscore_device}")

        _, _, f1_scores = bert_score(
            hypotheses,
            references,
            lang="en",
            verbose=False,
            device=self._resolved_bertscore_device,
        )
        return f1_scores.mean().item(), f1_scores.tolist(), hypotheses

    @staticmethod
    def _format_bad_examples(bad_examples: list[tuple[str, str, str]]) -> str:
        """Render low-scoring examples into a readable multiline block."""
        lines: list[str] = []
        for index, (inp, ref, out) in enumerate(bad_examples, start=1):
            lines.append(
                f"Example {index}:\n"
                f"Input: {inp}\n"
                f"Reference: {ref}\n"
                f"Model response: {out}\n"
            )
        return "\n".join(lines)

    @staticmethod
    def _parse_candidates(raw_response: Any, fallback_prompt: str, n: int) -> list[str]:
        """Parse candidate prompts from LLM output while keeping backward compatibility."""
        if isinstance(raw_response, list):
            return [str(item) for item in raw_response]

        if isinstance(raw_response, str):
            cleaned = raw_response.strip()
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(cleaned)
                    if isinstance(parsed, list):
                        return [str(item) for item in parsed]
                except Exception:
                    continue

        return [fallback_prompt] * n

    def generate_candidates(self, base_prompt: str, n: int) -> list[str]:
        """Generate ``n`` candidate prompts from ``base_prompt``."""
        prompt = self.candidate_generation_template.format(base_prompt=base_prompt, n=n)

        try:
            response = self.llm_candgen.invoke(prompt).response
        except Exception:
            response = self.llm_reasoning.invoke(prompt).response

        return self._parse_candidates(response, fallback_prompt=base_prompt, n=n)

    def analyze_weaknesses(
        self,
        prompt: str,
        scores: list[float],
        responses: list[str],
    ) -> str:
        """Analyze worst examples and return a textual improvement recommendation."""
        worst_indices = np.argsort(scores)[:5]
        bad_examples = [
            (self.dataset[i]["input"], self.dataset[i]["reference"], responses[i])
            for i in worst_indices
        ]
        bad_examples_text = self._format_bad_examples(bad_examples)

        meta_prompt = self.weakness_analysis_template.format(
            prompt=prompt,
            bad_examples_str=bad_examples_text,
        )
        return self.llm_reasoning.invoke(meta_prompt).response.strip()

    def improve_prompt(self, prompt: str, recommendation: str) -> str:
        """Apply a recommendation and return an improved prompt text."""
        meta_prompt = self.improvement_template.format(
            prompt=prompt,
            recommendation=recommendation,
        )
        return self.llm_reasoning.invoke(meta_prompt).response.strip()

    def run(self, initial_prompt: str) -> tuple[str, float]:
        """Run iterative optimization and return ``(best_prompt, best_score)``."""
        current_prompt = self._ensure_input_placeholder(initial_prompt)
        if current_prompt != initial_prompt:
            print("Initial prompt had no {input}; auto-inserted input placeholder.")
        current_score, _, _ = self.evaluate(current_prompt)
        best_prompt = current_prompt
        best_score = current_score

        self.history.append(
            {
                "iteration": 0,
                "prompt": current_prompt,
                "score": current_score,
                "candidates": [],
            }
        )

        print(f"Iteration 0 (initial prompt): BERTScore = {current_score:.4f}")
        no_improve_count = 0

        for iteration in range(1, self.n_iterations + 1):
            print(f"\n=== Iteration {iteration} ===")

            candidates = self.generate_candidates(current_prompt, self.n_candidates)
            print(f"Generated {len(candidates)} candidates.")

            candidate_data: list[tuple[str, float, list[float], list[str]]] = []
            for idx, candidate_prompt in enumerate(candidates, start=1):
                print(f"  Evaluating candidate {idx}...")
                avg_score, scores, responses = self.evaluate(candidate_prompt)
                candidate_data.append((candidate_prompt, avg_score, scores, responses))
                print(f"    avg BERTScore = {avg_score:.4f}")

            recommendations: list[str] = []
            for idx, (candidate_prompt, _, scores, responses) in enumerate(candidate_data, start=1):
                print(f"  Analyzing candidate {idx}...")
                recommendation = self.analyze_weaknesses(candidate_prompt, scores, responses)
                recommendations.append(recommendation)

            improved_candidates: list[str] = []
            for idx, (candidate_prompt, recommendation) in enumerate(zip(candidates, recommendations), start=1):
                print(f"  Improving candidate {idx}...")
                improved_candidates.append(self.improve_prompt(candidate_prompt, recommendation))

            improved_data: list[tuple[str, float, list[float], list[str]]] = []
            for idx, improved_prompt in enumerate(improved_candidates, start=1):
                print(f"  Evaluating improved candidate {idx}...")
                avg_score, scores, responses = self.evaluate(improved_prompt)
                improved_data.append((improved_prompt, avg_score, scores, responses))
                print(f"    avg BERTScore = {avg_score:.4f}")

            new_prompt, new_score, _, _ = max(improved_data, key=lambda item: item[1])
            print(f"Best improved candidate: BERTScore = {new_score:.4f}")

            self.history.append(
                {
                    "iteration": iteration,
                    "current_prompt": current_prompt,
                    "current_score": current_score,
                    "candidates": candidates,
                    "candidate_scores": [item[1] for item in candidate_data],
                    "recommendations": recommendations,
                    "improved_candidates": improved_candidates,
                    "improved_scores": [item[1] for item in improved_data],
                    "best_improved": new_prompt,
                    "best_improved_score": new_score,
                }
            )

            if new_score > current_score:
                print(f"  -> Improvement! (was {current_score:.4f})")
                current_prompt = new_prompt
                current_score = new_score
                no_improve_count = 0
            else:
                print(f"  -> No improvement (current best {current_score:.4f})")
                no_improve_count += 1

            if current_score > best_score:
                best_prompt = current_prompt
                best_score = current_score

            if no_improve_count >= self.early_stopping_rounds:
                print(
                    "Early stopping after "
                    f"{iteration} iterations (no improvements for "
                    f"{self.early_stopping_rounds} rounds)."
                )
                break

        print(f"\nOptimization finished. Best BERTScore: {best_score:.4f}")
        return best_prompt, best_score


# Backward-compatible alias.
AP = LegacySAPOPipeline
