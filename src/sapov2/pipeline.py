"""SAPO v2: Segment-aware bandit prompt optimizer."""

from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from bert_score import score as bert_score

from .llm import get_openrouter_llm
from .prompt import (
    CANDIDATE_LIST_PROMPT,
    FULL_REWRITE_PROMPT,
    PATCH_SEGMENT_PROMPT,
    SEGMENTATION_PROMPT,
    WEAKNESS_ANALYSIS_PROMPT,
)
from .schema import (
    CandidateGenerationResponse,
    PromptSegments,
    ScoreBreakdown,
    WeaknessAnalysis,
)

Example = dict[str, str]
SEGMENTS = ("role", "context", "tasks", "output_format")


@dataclass
class CandidateRecord:
    """Container for candidate-level optimization traces."""

    prompt: str
    arm: str
    mode: str
    val_breakdown: ScoreBreakdown
    val_scores: list[float]
    val_responses: list[str]


class SAPOV2Pipeline:
    """SAPO v2 optimizer with bandit search and multi-objective selection.

    Design goals:
    - Bandit-over-segment-edits instead of greedy best-of-k.
    - Multi-objective ranking (quality + format + length + cost).
    - Strict train/val split usage: optimize with train diagnostics, select on val.
    - Failure memory and retrieval for targeted fixes.
    - Segment confidence calibration and adaptive candidate budget.
    - Hybrid full-rewrite + field-level patch mode (LLPO-style).
    """

    def __init__(
        self,
        model_name: str,
        train_dataset: list[Example],
        val_dataset: list[Example],
        n_iterations: int = 10,
        n_candidates: int = 5,
        min_candidates: int = 3,
        max_candidates: int = 12,
        early_stopping_rounds: int = 3,
        patch_mode_probability: float = 0.5,
        objective_weights: dict[str, float] | None = None,
        bertscore_device: str = "auto",
        random_seed: int = 42,
    ) -> None:
        self.llm_base = get_openrouter_llm(model_name, response_type="base")
        self.llm_reasoning = get_openrouter_llm(
            model_name,
            response_type="reasoning",
            temperature=0.3,
        )
        self.llm_candidate = get_openrouter_llm(
            model_name,
            response_type="candidate_generation",
            temperature=0.5,
        )

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.n_iterations = n_iterations
        self.base_candidates = n_candidates
        self.min_candidates = min_candidates
        self.max_candidates = max_candidates
        self.early_stopping_rounds = early_stopping_rounds
        self.patch_mode_probability = patch_mode_probability
        self.bertscore_device = bertscore_device

        self.objective_weights = objective_weights or {
            "bertscore": 0.70,
            "format_compliance": 0.20,
            "length_penalty": 0.07,
            "estimated_cost": 0.03,
        }

        self.rng = random.Random(random_seed)

        # Thompson sampling stats for segment arms.
        self.bandit_alpha: dict[str, float] = defaultdict(lambda: 1.0)
        self.bandit_beta: dict[str, float] = defaultdict(lambda: 1.0)

        # Segment confidence in [0, 1].
        self.segment_confidence: dict[str, float] = {segment: 0.5 for segment in SEGMENTS}

        # Failure memory entries: prompt, segment, score, input, reference, response.
        self.failure_memory: list[dict[str, Any]] = []

        self.history: list[dict[str, Any]] = []
        self._resolved_bertscore_device: str | None = None

    @staticmethod
    def _dump_model(model: Any) -> dict[str, Any]:
        if hasattr(model, "model_dump"):
            return model.model_dump()
        if hasattr(model, "dict"):
            return model.dict()
        return dict(model)

    @staticmethod
    def _format_prompt(prompt: str, input_text: str) -> str:
        return prompt.format(input=input_text)

    @staticmethod
    def _safe_segment_name(name: str) -> str:
        mapping = {
            "role": "role",
            "context": "context",
            "tasks": "tasks",
            "task": "tasks",
            "output_format": "output_format",
            "output format": "output_format",
            "format": "output_format",
        }
        return mapping.get((name or "").strip().lower(), "")

    @staticmethod
    def _compose_prompt_from_segments(segments: PromptSegments) -> str:
        parts = [segment for segment in [segments.role, segments.context, segments.tasks, segments.output_format] if segment.strip()]
        return "\n\n".join(parts).strip()

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

    def _extract_segments(self, prompt: str) -> PromptSegments:
        structured = self.llm_reasoning.with_structured_output(PromptSegments)
        return structured.invoke(SEGMENTATION_PROMPT.format(prompt=prompt))

    def _sanitize_analysis(self, segments: PromptSegments, analysis: WeaknessAnalysis) -> WeaknessAnalysis:
        allowed = set(SEGMENTS)
        non_empty = {s for s in SEGMENTS if getattr(segments, s, "").strip()}

        weak: list[str] = []
        for seg in analysis.weak_segments:
            key = self._safe_segment_name(seg)
            if key and key in allowed and key not in weak:
                weak.append(key)

        strong: list[str] = []
        for seg in analysis.strong_segments:
            key = self._safe_segment_name(seg)
            if key and key in allowed and key in non_empty and key not in strong:
                strong.append(key)

        strong = [s for s in strong if s not in set(weak)]

        recommendations: dict[str, str] = {}
        for seg, rec in analysis.recommendations.items():
            key = self._safe_segment_name(seg)
            if key in allowed:
                recommendations[key] = str(rec)

        return WeaknessAnalysis(
            weak_segments=weak,
            strong_segments=strong,
            recommendations=recommendations,
        )

    def _evaluate_dataset(self, prompt: str, dataset: list[Example]) -> tuple[float, list[float], list[str]]:
        responses: list[str] = []
        references = [example["reference"] for example in dataset]

        for example in dataset:
            full_prompt = self._format_prompt(prompt, example["input"])
            try:
                model_response = self.llm_base.invoke(full_prompt).response
            except Exception:
                model_response = ""
            responses.append(model_response)

        if self._resolved_bertscore_device is None:
            self._resolved_bertscore_device = self._resolve_bertscore_device()
            print(f"[BERTScore] Using device: {self._resolved_bertscore_device}")

        _, _, f1 = bert_score(
            responses,
            references,
            lang="en",
            verbose=False,
            device=self._resolved_bertscore_device,
        )
        return float(f1.mean().item()), [float(x) for x in f1.tolist()], responses

    def _format_compliance_score(self, prompt: str, responses: list[str]) -> float:
        if not responses:
            return 0.0

        lower_prompt = prompt.lower()

        if "one sentence" in lower_prompt or "single sentence" in lower_prompt:
            checks = []
            for resp in responses:
                sentence_like = [s for s in re.split(r"[.!?]", resp) if s.strip()]
                checks.append(1.0 if len(sentence_like) <= 1 else 0.0)
            return float(np.mean(checks))

        if "bullet" in lower_prompt:
            checks = [1.0 if resp.strip().startswith(("-", "*", "1.")) else 0.0 for resp in responses]
            return float(np.mean(checks))

        return 0.8

    @staticmethod
    def _length_penalty_score(responses: list[str], references: list[str]) -> float:
        if not responses or not references:
            return 0.0

        response_lens = [max(1, len(resp.split())) for resp in responses]
        reference_lens = [max(1, len(ref.split())) for ref in references]

        ratios = []
        for resp_len, ref_len in zip(response_lens, reference_lens):
            ratio = abs(resp_len - ref_len) / max(ref_len, 1)
            ratios.append(math.exp(-ratio))
        return float(np.mean(ratios))

    def _estimated_cost(self, prompt: str, dataset: list[Example], responses: list[str]) -> float:
        if not dataset:
            return 0.0

        prompt_tokens = len(prompt.split())
        input_tokens = np.mean([len(example["input"].split()) for example in dataset])
        output_tokens = np.mean([len(resp.split()) for resp in responses]) if responses else 0.0

        raw = prompt_tokens + input_tokens + output_tokens
        return float(min(1.0, raw / 1200.0))

    def _score_breakdown(self, prompt: str, dataset: list[Example]) -> tuple[ScoreBreakdown, list[float], list[str]]:
        bert_avg, scores, responses = self._evaluate_dataset(prompt, dataset)
        format_score = self._format_compliance_score(prompt, responses)
        length_score = self._length_penalty_score(responses, [example["reference"] for example in dataset])
        cost = self._estimated_cost(prompt, dataset, responses)

        w = self.objective_weights
        objective = (
            w["bertscore"] * bert_avg
            + w["format_compliance"] * format_score
            + w["length_penalty"] * length_score
            - w["estimated_cost"] * cost
        )

        return (
            ScoreBreakdown(
                bertscore=bert_avg,
                format_compliance=format_score,
                length_penalty=length_score,
                estimated_cost=cost,
                objective=float(objective),
            ),
            scores,
            responses,
        )

    def _rank_examples(self, dataset: list[Example], scores: list[float], responses: list[str], top_n: int = 5) -> dict[str, list[dict[str, Any]]]:
        order = np.argsort(scores)[::-1]
        best_idx = order[:top_n]
        worst_idx = order[-top_n:]

        def collect(indices: np.ndarray) -> list[dict[str, Any]]:
            rows = []
            for idx in indices:
                rows.append(
                    {
                        "input": dataset[idx]["input"],
                        "reference": dataset[idx]["reference"],
                        "response": responses[idx],
                        "score": float(scores[idx]),
                    }
                )
            return rows

        return {"best": collect(best_idx), "worst": collect(worst_idx)}

    @staticmethod
    def _format_examples_for_prompt(examples: list[dict[str, Any]]) -> str:
        lines = []
        for i, ex in enumerate(examples, start=1):
            lines.append(
                f"Example {i}:\n"
                f"Input: {ex['input']}\n"
                f"Reference: {ex['reference']}\n"
                f"Model response: {ex['response']}\n"
                f"Score: {ex['score']:.4f}\n"
            )
        return "\n".join(lines)

    @staticmethod
    def _log_examples(label: str, examples: list[dict[str, Any]]) -> None:
        print(label)
        for i, ex in enumerate(examples, start=1):
            inp = ex["input"].replace("\n", " ")[:120]
            ref = ex["reference"].replace("\n", " ")[:100]
            out = ex["response"].replace("\n", " ")[:100]
            print(f"  {i}. score={ex['score']:.4f} | input='{inp}' | ref='{ref}' | resp='{out}'")

    def _update_failure_memory(self, prompt: str, worst_examples: list[dict[str, Any]]) -> None:
        for ex in worst_examples:
            self.failure_memory.append(
                {
                    "prompt": prompt,
                    "input": ex["input"],
                    "reference": ex["reference"],
                    "response": ex["response"],
                    "score": ex["score"],
                }
            )
        # Keep memory bounded.
        self.failure_memory = sorted(self.failure_memory, key=lambda x: x["score"])[:200]

    @staticmethod
    def _token_set(text: str) -> set[str]:
        return set(re.findall(r"\w+", (text or "").lower()))

    def _retrieve_similar_failures(self, query_examples: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
        if not self.failure_memory:
            return []

        query_tokens = set()
        for ex in query_examples:
            query_tokens.update(self._token_set(ex["input"]))

        scored = []
        for entry in self.failure_memory:
            entry_tokens = self._token_set(entry["input"])
            inter = len(query_tokens & entry_tokens)
            union = len(query_tokens | entry_tokens) or 1
            sim = inter / union
            scored.append((sim, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for sim, entry in scored[:top_k] if sim > 0]

    def _summarize_failure_memory(self, failures: list[dict[str, Any]]) -> str:
        if not failures:
            return "None"
        lines = []
        for i, item in enumerate(failures, start=1):
            inp = item["input"].replace("\n", " ")[:140]
            resp = item["response"].replace("\n", " ")[:100]
            lines.append(f"{i}. score={item['score']:.4f} | input='{inp}' | bad_response='{resp}'")
        return "\n".join(lines)

    def _analyze_weaknesses(
        self,
        prompt: str,
        segments: PromptSegments,
        best_examples: list[dict[str, Any]],
        worst_examples: list[dict[str, Any]],
    ) -> WeaknessAnalysis:
        retrieved = self._retrieve_similar_failures(worst_examples, top_k=5)
        prompt_text = WEAKNESS_ANALYSIS_PROMPT.format(
            prompt=prompt,
            best_examples=self._format_examples_for_prompt(best_examples),
            worst_examples=self._format_examples_for_prompt(worst_examples),
            retrieved_failures=self._summarize_failure_memory(retrieved),
        )

        structured = self.llm_reasoning.with_structured_output(WeaknessAnalysis)
        raw = structured.invoke(prompt_text)
        return self._sanitize_analysis(segments, raw)

    def _thompson_sample_arm(self, weak_segments: list[str]) -> str:
        candidates = weak_segments if weak_segments else list(SEGMENTS)
        best_arm = candidates[0]
        best_value = -1.0
        for arm in candidates:
            a = self.bandit_alpha[arm]
            b = self.bandit_beta[arm]
            sample = self.rng.betavariate(a, b)
            if sample > best_value:
                best_value = sample
                best_arm = arm
        return best_arm

    def _adaptive_candidate_budget(self, no_improve_count: int, last_gain: float) -> int:
        k = self.base_candidates
        if no_improve_count > 0:
            k += min(4, no_improve_count + 1)
        if last_gain > 0.02:
            k -= 1
        return max(self.min_candidates, min(self.max_candidates, k))

    def _build_candidate_via_full_rewrite(
        self,
        current_prompt: str,
        segments: PromptSegments,
        analysis: WeaknessAnalysis,
        selected_arm: str,
    ) -> str:
        retrieved = self._retrieve_similar_failures([], top_k=5)
        recommendations_text = "\n".join(f"- {k}: {v}" for k, v in analysis.recommendations.items()) or "None"
        prompt_text = FULL_REWRITE_PROMPT.format(
            current_prompt=current_prompt,
            role=segments.role,
            context=segments.context,
            tasks=segments.tasks,
            output_format=segments.output_format,
            selected_arm=selected_arm,
            segment_confidence=f"{self.segment_confidence[selected_arm]:.3f}",
            weak_segments=", ".join(analysis.weak_segments) or "none",
            strong_segments=", ".join(analysis.strong_segments) or "none",
            recommendations_text=recommendations_text,
            failure_memory=self._summarize_failure_memory(retrieved),
        )
        return self.llm_reasoning.invoke(prompt_text).response.strip()

    def _build_candidate_via_patch(
        self,
        segments: PromptSegments,
        selected_arm: str,
        analysis: WeaknessAnalysis,
    ) -> str:
        recommendation = analysis.recommendations.get(selected_arm, "Make this segment clearer and more task-specific.")
        segment_value = getattr(segments, selected_arm)

        patch_prompt = PATCH_SEGMENT_PROMPT.format(
            selected_arm=selected_arm,
            segment_value=segment_value,
            recommendation=recommendation,
            segment_confidence=f"{self.segment_confidence[selected_arm]:.3f}",
        )
        patched = self.llm_reasoning.invoke(patch_prompt).response.strip()

        new_segments = PromptSegments(**self._dump_model(segments))
        setattr(new_segments, selected_arm, patched)
        return self._compose_prompt_from_segments(new_segments)

    def _generate_candidates(
        self,
        current_prompt: str,
        segments: PromptSegments,
        analysis: WeaknessAnalysis,
        n_candidates: int,
    ) -> list[tuple[str, str, str]]:
        candidates: list[tuple[str, str, str]] = []

        # First candidate from list-based generation for diversity.
        recommendations_text = "\n".join(f"- {k}: {v}" for k, v in analysis.recommendations.items()) or "None"
        list_prompt = CANDIDATE_LIST_PROMPT.format(
            current_prompt=current_prompt,
            weak_segments=", ".join(analysis.weak_segments) or "none",
            strong_segments=", ".join(analysis.strong_segments) or "none",
            recommendations_text=recommendations_text,
            n_candidates=max(1, min(3, n_candidates)),
        )
        try:
            list_response = self.llm_candidate.invoke(list_prompt)
            for prompt in list_response.prompts[: max(1, min(3, n_candidates))]:
                arm = self._thompson_sample_arm(analysis.weak_segments)
                candidates.append((prompt.strip(), arm, "list"))
        except Exception:
            pass

        while len(candidates) < n_candidates:
            arm = self._thompson_sample_arm(analysis.weak_segments)
            use_patch = self.rng.random() < self.patch_mode_probability

            if use_patch:
                prompt = self._build_candidate_via_patch(segments, arm, analysis)
                mode = "patch"
            else:
                prompt = self._build_candidate_via_full_rewrite(current_prompt, segments, analysis, arm)
                mode = "full"

            if prompt:
                candidates.append((prompt, arm, mode))

        # Deduplicate while preserving order.
        seen = set()
        unique: list[tuple[str, str, str]] = []
        for prompt, arm, mode in candidates:
            key = prompt.strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique.append((prompt.strip(), arm, mode))

        return unique[:n_candidates]

    def _update_segment_confidence(self, segment: str, gain: float) -> None:
        old = self.segment_confidence[segment]
        normalized = max(0.0, min(1.0, 0.5 + 8.0 * gain))
        self.segment_confidence[segment] = 0.8 * old + 0.2 * normalized

    def _update_bandit(self, arm: str, improved: bool) -> None:
        if improved:
            self.bandit_alpha[arm] += 1.0
        else:
            self.bandit_beta[arm] += 1.0

    def run(self, initial_prompt: str) -> tuple[str, ScoreBreakdown]:
        """Run SAPO v2 optimization and return best prompt + val score breakdown."""
        current_prompt = initial_prompt
        current_train_breakdown, train_scores, train_responses = self._score_breakdown(current_prompt, self.train_dataset)
        current_val_breakdown, _, _ = self._score_breakdown(current_prompt, self.val_dataset)
        current_segments = self._extract_segments(current_prompt)

        best_prompt = current_prompt
        best_val = current_val_breakdown
        no_improve_count = 0
        last_gain = 0.0

        print(f"Initial train objective={current_train_breakdown.objective:.4f}, val objective={current_val_breakdown.objective:.4f}")

        self.history.append(
            {
                "iteration": 0,
                "prompt": current_prompt,
                "train_breakdown": self._dump_model(current_train_breakdown),
                "val_breakdown": self._dump_model(current_val_breakdown),
                "segments": self._dump_model(current_segments),
            }
        )

        for iteration in range(1, self.n_iterations + 1):
            print(f"\n=== SAPO v2 Iteration {iteration} ===")

            ranked = self._rank_examples(self.train_dataset, train_scores, train_responses, top_n=5)
            self._log_examples("Top-5 good examples (train):", ranked["best"])
            self._log_examples("Top-5 bad examples (train):", ranked["worst"])

            analysis = self._analyze_weaknesses(
                current_prompt,
                current_segments,
                ranked["best"],
                ranked["worst"],
            )

            self._update_failure_memory(current_prompt, ranked["worst"])

            candidate_budget = self._adaptive_candidate_budget(no_improve_count, last_gain)
            candidates = self._generate_candidates(
                current_prompt,
                current_segments,
                analysis,
                n_candidates=candidate_budget,
            )
            print(f"Candidate budget={candidate_budget}, generated={len(candidates)}")

            candidate_records: list[CandidateRecord] = []
            for idx, (candidate_prompt, arm, mode) in enumerate(candidates, start=1):
                val_breakdown, val_scores, val_responses = self._score_breakdown(candidate_prompt, self.val_dataset)
                candidate_records.append(
                    CandidateRecord(
                        prompt=candidate_prompt,
                        arm=arm,
                        mode=mode,
                        val_breakdown=val_breakdown,
                        val_scores=val_scores,
                        val_responses=val_responses,
                    )
                )
                print(
                    f"  candidate {idx}: arm={arm}, mode={mode}, "
                    f"val_objective={val_breakdown.objective:.4f}, "
                    f"bert={val_breakdown.bertscore:.4f}, format={val_breakdown.format_compliance:.4f}"
                )

            if not candidate_records:
                print("No valid candidates produced; stopping.")
                break

            winner = max(candidate_records, key=lambda record: record.val_breakdown.objective)
            gain = winner.val_breakdown.objective - current_val_breakdown.objective
            improved = gain > 0

            for record in candidate_records:
                self._update_bandit(record.arm, record.val_breakdown.objective > current_val_breakdown.objective)
                self._update_segment_confidence(record.arm, record.val_breakdown.objective - current_val_breakdown.objective)

            iteration_snapshot = {
                "iteration": iteration,
                "current_prompt": current_prompt,
                "analysis": self._dump_model(analysis),
                "candidate_budget": candidate_budget,
                "candidates": [
                    {
                        "prompt": rec.prompt,
                        "arm": rec.arm,
                        "mode": rec.mode,
                        "val_breakdown": self._dump_model(rec.val_breakdown),
                    }
                    for rec in candidate_records
                ],
                "winner": {
                    "prompt": winner.prompt,
                    "arm": winner.arm,
                    "mode": winner.mode,
                    "val_breakdown": self._dump_model(winner.val_breakdown),
                },
                "improved": improved,
                "gain": gain,
                "segment_confidence": dict(self.segment_confidence),
                "bandit_alpha": dict(self.bandit_alpha),
                "bandit_beta": dict(self.bandit_beta),
            }

            if improved:
                current_prompt = winner.prompt
                current_val_breakdown = winner.val_breakdown
                current_train_breakdown, train_scores, train_responses = self._score_breakdown(current_prompt, self.train_dataset)
                current_segments = self._extract_segments(current_prompt)

                no_improve_count = 0
                last_gain = gain
                print(f"  -> improved by {gain:.4f}; new val objective={current_val_breakdown.objective:.4f}")

                if current_val_breakdown.objective > best_val.objective:
                    best_val = current_val_breakdown
                    best_prompt = current_prompt
            else:
                no_improve_count += 1
                last_gain = 0.0
                print(f"  -> no improvement; current val objective={current_val_breakdown.objective:.4f}")

            self.history.append(iteration_snapshot)

            if no_improve_count >= self.early_stopping_rounds:
                print(
                    f"Early stopping: {no_improve_count} non-improving rounds "
                    f"(patience={self.early_stopping_rounds})."
                )
                break

        print(f"\nSAPO v2 finished. Best val objective={best_val.objective:.4f}")
        return best_prompt, best_val
