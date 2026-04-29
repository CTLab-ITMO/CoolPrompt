"""Segment-based Automatic Prompt Optimization (SAPO) pipeline.

This module contains the structured SAPO variant that:
1) segments the prompt,
2) analyzes strengths/weaknesses from best and worst examples,
3) generates improved candidates,
4) keeps improvements based on BERTScore.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from bert_score import score as bert_score
from pydantic import BaseModel, Field

from .llm import get_openrouter_llm

Example = dict[str, str]


class PromptSegments(BaseModel):
    """Structured decomposition of a prompt into core segments."""

    role: str = Field(default="", description="Role the model should assume")
    context: str = Field(default="", description="Task context including placeholders")
    tasks: str = Field(default="", description="Concrete actions the model must perform")
    output_format: str = Field(default="", description="Output format requirements")


class WeaknessAnalysis(BaseModel):
    """Strength/weakness analysis for prompt segments."""

    weak_segments: list[str] = Field(
        description="Segment names that require improvement"
    )
    strong_segments: list[str] = Field(
        description="Segment names that should be preserved"
    )
    recommendations: dict[str, str] = Field(
        description="Actionable recommendations for each weak segment"
    )


class CandidatesList(BaseModel):
    """Container for generated prompt candidates."""

    prompts: list[str] = Field(description="List of generated prompt candidates")


class SAPOPipeline:
    """Segment-based automatic prompt optimizer."""

    def __init__(
        self,
        model_name: str,
        dataset: list[Example],
        n_iterations: int = 10,
        n_candidates: int = 5,
        early_stopping_rounds: int = 3,
    ) -> None:
        """Initialize SAPO pipeline.

        Args:
            model_name: OpenRouter model id.
            dataset: Evaluation dataset with ``input`` and ``reference`` keys.
            n_iterations: Maximum optimization iterations.
            n_candidates: Number of generated candidates per optimization step.
            early_stopping_rounds: Early-stop patience for non-improving rounds.
        """
        self.llm_base = get_openrouter_llm(model_name, response_type="base")
        self.llm_reasoning = get_openrouter_llm(model_name, response_type="reasoning")

        self.dataset = dataset
        self.n_iterations = n_iterations
        self.n_candidates = n_candidates
        self.early_stopping_rounds = early_stopping_rounds

        self.structured_llm = self.llm_reasoning.with_structured_output

        self.segmentation_prompt = (
            "You are an expert in analyzing prompts. Given the following prompt, decompose it into four segments:\n"
            "1. Role: what role the model should assume (e.g., 'You are an experienced analyst').\n"
            "2. Context: the background information or domain context, including any variables in {{}} (e.g., 'Given the text: {{input}}').\n"
            "3. Tasks: the specific instructions or tasks the model must perform.\n"
            "4. Output format: any requirements about the format of the response.\n"
            "If a segment is missing, output an empty string for it.\n\n"
            "Prompt:\n\"\"\"\n{prompt}\n\"\"\"\n\n"
            "Output a JSON object with keys: role, context, tasks, output_format."
        )

        self.weakness_analysis_prompt = (
            "You are an expert prompt engineer. We have a prompt that was used to generate responses for a dataset. "
            "Below are the 5 best-scoring examples (highest BERTScore) and the 5 worst-scoring examples (lowest BERTScore). "
            "For each example, we provide the input, reference answer, model response, and the BERTScore F1.\n\n"
            "Prompt:\n\"\"\"\n{prompt}\n\"\"\"\n\n"
            "Here are the best examples (high similarity to reference):\n{best_examples}\n\n"
            "Here are the worst examples (low similarity):\n{worst_examples}\n\n"
            "Based on this, analyze the prompt's strengths and weaknesses in terms of its four segments: "
            "role, context, tasks, output_format. Identify which segments are strong (working well) and which are weak "
            "(causing poor performance). For each weak segment, provide a specific, actionable recommendation on how to "
            "improve it. Return your analysis as a JSON object with keys:\n"
            "- weak_segments: list of segment names that need improvement (e.g., [\"tasks\", \"output_format\"])\n"
            "- strong_segments: list of segment names that work well\n"
            "- recommendations: dictionary mapping each weak segment to a text recommendation"
        )

        self.candidate_generation_prompt = (
            "You are an expert prompt engineer. You have a current prompt and its segment analysis. "
            "The prompt is:\n\"\"\"\n{current_prompt}\n\"\"\"\n\n"
            "Its segments are:\n"
            "Role: {role}\n"
            "Context: {context}\n"
            "Tasks: {tasks}\n"
            "Output format: {output_format}\n\n"
            "Weak segments that need improvement: {weak_segments}\n"
            "Recommendations for improvement:\n{recommendations_text}\n\n"
            "Strong segments to preserve: {strong_segments}\n\n"
            "Generate {n_candidates} diverse improved versions of the prompt. Each version should incorporate the recommendations "
            "for the weak segments while keeping the strengths. The new prompts should be standalone and suitable for the "
            "same task. Output a JSON object with a single key 'prompts' containing a list of strings."
        )

        self.history: list[dict[str, Any]] = []

    @staticmethod
    def _format_prompt(prompt: str, input_text: str) -> str:
        """Inject ``input_text`` into prompt template via ``{input}`` placeholder."""
        return prompt.format(input=input_text)

    @staticmethod
    def _dump_model(model: BaseModel) -> dict[str, Any]:
        """Return model as dict for both Pydantic v1 and v2."""
        if hasattr(model, "model_dump"):
            return model.model_dump()
        return model.dict()

    @staticmethod
    def _normalize_segment_name(segment: str) -> str:
        """Normalize incoming segment labels to supported canonical names."""
        mapping = {
            "role": "role",
            "context": "context",
            "tasks": "tasks",
            "task": "tasks",
            "output_format": "output_format",
            "output format": "output_format",
            "format": "output_format",
        }
        return mapping.get((segment or "").strip().lower(), "")

    def _sanitize_segment_analysis(
        self,
        segments: PromptSegments,
        analysis: WeaknessAnalysis,
    ) -> WeaknessAnalysis:
        """Sanitize strong/weak segment labels and enforce non-empty strong segments.

        Rules:
        - Keep only known segment names.
        - Remove duplicates.
        - Remove intersections (weak has priority).
        - Do not allow empty segments to be marked as strong.
        - If recommendation keys are unknown, drop them.
        """
        allowed = {"role", "context", "tasks", "output_format"}
        non_empty = {
            name
            for name in allowed
            if (getattr(segments, name, "") or "").strip()
        }

        weak = []
        for item in analysis.weak_segments:
            key = self._normalize_segment_name(item)
            if key and key in allowed and key not in weak:
                weak.append(key)

        strong = []
        for item in analysis.strong_segments:
            key = self._normalize_segment_name(item)
            if key and key in allowed and key in non_empty and key not in strong:
                strong.append(key)

        strong = [item for item in strong if item not in set(weak)]

        cleaned_recommendations: dict[str, str] = {}
        for seg, rec in analysis.recommendations.items():
            key = self._normalize_segment_name(seg)
            if key and key in allowed:
                cleaned_recommendations[key] = str(rec)

        return WeaknessAnalysis(
            weak_segments=weak,
            strong_segments=strong,
            recommendations=cleaned_recommendations,
        )

    def evaluate(self, prompt: str) -> tuple[float, list[float], list[str]]:
        """Evaluate prompt quality on the full dataset using BERTScore F1."""
        hypotheses: list[str] = []
        references = [example["reference"] for example in self.dataset]
        inputs = [example["input"] for example in self.dataset]

        for input_text in inputs:
            full_prompt = self._format_prompt(prompt, input_text)
            try:
                response = self.llm_base.invoke(full_prompt).response
            except Exception as exc:
                print(f"Generation failed for input: {input_text}\n{exc}")
                response = ""
            hypotheses.append(response)

        device = "cpu"
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                device = "cuda"
        except Exception:
            device = "cpu"

        _, _, f1_scores = bert_score(
            hypotheses,
            references,
            lang="en",
            verbose=False,
            device=device,
        )
        return f1_scores.mean().item(), f1_scores.tolist(), hypotheses

    def _extract_segments(self, prompt: str) -> PromptSegments:
        """Split a prompt into ``role/context/tasks/output_format`` segments."""
        structured = self.structured_llm(PromptSegments)
        return structured.invoke(self.segmentation_prompt.format(prompt=prompt))

    def _analyze_weaknesses(
        self,
        prompt: str,
        segments: PromptSegments,
        scores: list[float],
        responses: list[str],
        best_n: int = 5,
        worst_n: int = 5,
    ) -> WeaknessAnalysis:
        """Analyze strengths and weaknesses using best/worst scoring examples."""
        ranked_examples = self._rank_examples(scores, responses, best_n=best_n, worst_n=worst_n)
        best_examples = self._format_ranked_examples(ranked_examples["best_examples"])
        worst_examples = self._format_ranked_examples(ranked_examples["worst_examples"])

        structured = self.structured_llm(WeaknessAnalysis)
        raw_analysis = structured.invoke(
            self.weakness_analysis_prompt.format(
                prompt=prompt,
                best_examples=best_examples,
                worst_examples=worst_examples,
            )
        )
        return self._sanitize_segment_analysis(segments, raw_analysis)

    def _rank_examples(
        self,
        scores: list[float],
        responses: list[str],
        best_n: int = 5,
        worst_n: int = 5,
    ) -> dict[str, list[dict[str, Any]]]:
        """Build top/bottom example packs with metric scores for logging and analysis."""
        sorted_indices = np.argsort(scores)[::-1]
        best_indices = sorted_indices[:best_n]
        worst_indices = sorted_indices[-worst_n:]

        def to_records(indices: np.ndarray) -> list[dict[str, Any]]:
            records: list[dict[str, Any]] = []
            for index in indices:
                example = self.dataset[index]
                records.append(
                    {
                        "input": example["input"],
                        "reference": example["reference"],
                        "response": responses[index],
                        "score": float(scores[index]),
                    }
                )
            return records

        return {
            "best_examples": to_records(best_indices),
            "worst_examples": to_records(worst_indices),
        }

    @staticmethod
    def _format_ranked_examples(examples: list[dict[str, Any]]) -> str:
        """Render ranked examples for LLM analysis prompt."""
        lines: list[str] = []
        for i, item in enumerate(examples, start=1):
            lines.append(
                f"Example {i}:\n"
                f"Input: {item['input']}\n"
                f"Reference: {item['reference']}\n"
                f"Model response: {item['response']}\n"
                f"BERTScore F1: {item['score']:.4f}\n"
            )
        return "\n".join(lines)

    @staticmethod
    def _log_ranked_examples(
        label: str,
        examples: list[dict[str, Any]],
        max_input_len: int = 140,
        max_text_len: int = 120,
    ) -> None:
        """Print concise ranked examples with metric values."""
        print(label)
        for idx, item in enumerate(examples, start=1):
            input_preview = (item["input"] or "")[:max_input_len].replace("\n", " ")
            ref_preview = (item["reference"] or "")[:max_text_len].replace("\n", " ")
            resp_preview = (item["response"] or "")[:max_text_len].replace("\n", " ")
            print(
                f"  {idx}. score={item['score']:.4f} | "
                f"input='{input_preview}' | ref='{ref_preview}' | resp='{resp_preview}'"
            )

    def _generate_candidates(
        self,
        current_prompt: str,
        segments: PromptSegments,
        analysis: WeaknessAnalysis,
    ) -> list[str]:
        """Generate five improved candidates based on segment-level analysis."""
        recommendations_text = "\n".join(
            f"- {segment}: {recommendation}"
            for segment, recommendation in analysis.recommendations.items()
        )

        weak_segments = ", ".join(analysis.weak_segments) if analysis.weak_segments else "none"
        strong_segments = ", ".join(analysis.strong_segments) if analysis.strong_segments else "none"

        prompt_text = self.candidate_generation_prompt.format(
            current_prompt=current_prompt,
            role=segments.role,
            context=segments.context,
            tasks=segments.tasks,
            output_format=segments.output_format,
            weak_segments=weak_segments,
            recommendations_text=recommendations_text,
            strong_segments=strong_segments,
            n_candidates=self.n_candidates,
        )

        structured = self.structured_llm(CandidatesList)
        response = structured.invoke(prompt_text)
        prompts = response.prompts
        if len(prompts) > self.n_candidates:
            return prompts[: self.n_candidates]
        return prompts

    def run(self, initial_prompt: str) -> tuple[str, float]:
        """Run SAPO optimization and return ``(best_prompt, best_score)``."""
        current_prompt = initial_prompt
        current_score, current_scores_list, current_responses = self.evaluate(current_prompt)
        current_segments = self._extract_segments(current_prompt)
        print(f"Initial prompt score (BERTScore): {current_score:.4f}")

        best_prompt = current_prompt
        best_score = current_score

        self.history.append(
            {
                "iteration": 0,
                "prompt": current_prompt,
                "segments": self._dump_model(current_segments),
                "score": current_score,
                "scores_list": current_scores_list,
                "responses": current_responses,
            }
        )
        print(f"Iteration 0 (initial prompt): BERTScore = {current_score:.4f}")

        no_improve_count = 0

        for iteration in range(1, self.n_iterations + 1):
            print(f"\n=== Iteration {iteration} ===")

            analysis = self._analyze_weaknesses(
                current_prompt,
                current_segments,
                current_scores_list,
                current_responses,
            )
            current_ranked = self._rank_examples(
                current_scores_list,
                current_responses,
                best_n=5,
                worst_n=5,
            )
            print(
                "Current segments: "
                f"role={current_segments.role!r}, "
                f"context={current_segments.context!r}, "
                f"tasks={current_segments.tasks!r}, "
                f"output_format={current_segments.output_format!r}"
            )
            print(f"Weak segments: {analysis.weak_segments}")
            print(f"Strong segments: {analysis.strong_segments}")
            print(f"Recommendations: {analysis.recommendations}")
            self._log_ranked_examples("Top-5 good examples (current prompt):", current_ranked["best_examples"])
            self._log_ranked_examples("Top-5 bad examples (current prompt):", current_ranked["worst_examples"])

            candidates = self._generate_candidates(current_prompt, current_segments, analysis)
            print(f"Generated {len(candidates)} candidates.")

            candidate_results: list[tuple[str, float, list[float], list[str]]] = []
            candidate_segments: list[dict[str, Any]] = []
            candidate_analyses: list[dict[str, Any]] = []
            candidate_best_examples: list[list[dict[str, Any]]] = []
            candidate_worst_examples: list[list[dict[str, Any]]] = []
            for idx, candidate_prompt in enumerate(candidates, start=1):
                print(f"  Evaluating candidate {idx}...")
                avg_score, scores, responses = self.evaluate(candidate_prompt)
                candidate_results.append((candidate_prompt, avg_score, scores, responses))
                print(f"    avg BERTScore = {avg_score:.4f}")

                candidate_segment = self._extract_segments(candidate_prompt)
                segment_dump = self._dump_model(candidate_segment)
                candidate_segments.append(segment_dump)
                print(
                    "    candidate segments: "
                    f"role={segment_dump.get('role', '')!r}, "
                    f"context={segment_dump.get('context', '')!r}, "
                    f"tasks={segment_dump.get('tasks', '')!r}, "
                    f"output_format={segment_dump.get('output_format', '')!r}"
                )

                candidate_analysis = self._analyze_weaknesses(
                    candidate_prompt,
                    candidate_segment,
                    scores,
                    responses,
                )
                analysis_dump = self._dump_model(candidate_analysis)
                candidate_analyses.append(analysis_dump)
                print(f"    candidate weak segments: {analysis_dump.get('weak_segments', [])}")
                print(f"    candidate strong segments: {analysis_dump.get('strong_segments', [])}")

                candidate_ranked = self._rank_examples(scores, responses, best_n=5, worst_n=5)
                candidate_best_examples.append(candidate_ranked["best_examples"])
                candidate_worst_examples.append(candidate_ranked["worst_examples"])
                self._log_ranked_examples(
                    f"    candidate {idx} top-5 good examples:",
                    candidate_ranked["best_examples"],
                )
                self._log_ranked_examples(
                    f"    candidate {idx} top-5 bad examples:",
                    candidate_ranked["worst_examples"],
                )

            new_prompt, new_score, new_scores_list, new_responses = max(
                candidate_results,
                key=lambda item: item[1],
            )
            print(f"Best candidate: BERTScore = {new_score:.4f}")

            previous_score = current_score
            improved = new_score > previous_score

            if improved:
                print(f"  -> Improvement! (was {previous_score:.4f})")
                current_prompt = new_prompt
                current_score = new_score
                current_scores_list = new_scores_list
                current_responses = new_responses
                current_segments = self._extract_segments(current_prompt)
                no_improve_count = 0
            else:
                print(f"  -> No improvement (current best {current_score:.4f})")
                no_improve_count += 1

            if current_score > best_score:
                best_prompt = current_prompt
                best_score = current_score

            self.history.append(
                {
                    "iteration": iteration,
                    "current_prompt": current_prompt,
                    "current_segments": self._dump_model(current_segments),
                    "current_score": current_score,
                    "analysis": self._dump_model(analysis),
                    "current_best_examples": current_ranked["best_examples"],
                    "current_worst_examples": current_ranked["worst_examples"],
                    "candidates": candidates,
                    "candidate_scores": [result[1] for result in candidate_results],
                    "candidate_segments": candidate_segments,
                    "candidate_analyses": candidate_analyses,
                    "candidate_best_examples": candidate_best_examples,
                    "candidate_worst_examples": candidate_worst_examples,
                    "best_candidate": new_prompt,
                    "best_candidate_score": new_score,
                    "improved": improved,
                }
            )

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
AP = SAPOPipeline
