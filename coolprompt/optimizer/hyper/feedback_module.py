"""Feedback generation: section-targeted recommendations from failed evaluations."""

import json
import logging
import random as _random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from coolprompt.evaluator.evaluator import FailedExampleDetailed
from coolprompt.utils.parsing import extract_json, get_model_answer_extracted
from coolprompt.utils.prompt_templates.hyper_templates import (
    CONTRASTIVE_FEEDBACK_PROMPT,
    DROP_INSTANCE_LEAK_PROMPT,
    FEEDBACK_PROMPT_TEMPLATE,
    GENERAL_SECTION,
    PromptSectionSpec,
    RECOMMENDATIONS_GROUP_PROMPT,
    Recommendation,
    SECTION_GROUPS_FILTER_PROMPT,
)
from coolprompt.utils.structured_schemas.optimizer.hyper import (
    InstanceLeakAuditResponse,
    RecommendationGroupsResponse,
    SectionRecommendationResponse,
    SynthesizedRecommendationsResponse,
)

logger = logging.getLogger(__name__)


@dataclass
class ContrastiveCandidate:
    """Another prompt that scored higher on the same task as the failing prompt.

    Attributes:
        prompt: Competing candidate prompt text.
        score: Scalar score on the shared mini-batch index (task-dependent).
        raw_answer: Model output for that candidate on the task instance.
        parsed_answer: Optional parsed form from the task metric.
    """

    prompt: str
    score: float
    raw_answer: str
    parsed_answer: Optional[str] = None


class FeedbackModule:
    """Build textual recommendations for prompt sections."""

    def __init__(
        self,
        model: Any,
        section_specs: Optional[List[PromptSectionSpec]] = None,
        contrastive_probability: float = 0.5,
        contrastive_max_answer_chars: int = 500,
        feedback_answer_head_chars: int = 500,
        feedback_answer_tail_chars: int = 500,
        use_structured_output: bool = False,
        **kwargs: Any,
    ) -> None:
        """Configure the feedback LLM client and truncation budgets.

        Args:
            model: Chat model passed to :func:`~coolprompt.utils.parsing.get_model_answer_extracted`.
            section_specs: Allowed target sections; empty uses only ``general``.
            contrastive_probability: Bernoulli probability to attempt contrastive prompts.
            contrastive_max_answer_chars: Max chars (head+tail) for winning answers.
            feedback_answer_head_chars: Head budget for failing answers in prompts.
            feedback_answer_tail_chars: Tail budget for failing answers in prompts.
            **kwargs: Rejected; present only to catch typos early.

        Raises:
            TypeError: If unexpected keyword arguments are supplied.
            ValueError: If numeric parameters are out of range.
        """
        if kwargs:
            raise TypeError(
                f"FeedbackModule.__init__() got unexpected keyword argument(s): {sorted(kwargs)!r}"
            )

        if not 0.0 <= contrastive_probability <= 1.0:
            raise ValueError(
                f"contrastive_probability must be in [0, 1], got {contrastive_probability}"
            )
        if contrastive_max_answer_chars < 0:
            raise ValueError("contrastive_max_answer_chars must be >= 0")
        if feedback_answer_head_chars < 0 or feedback_answer_tail_chars < 0:
            raise ValueError(
                "feedback_answer_head_chars and feedback_answer_tail_chars must be >= 0"
            )

        self.model = model
        self.section_specs: List[PromptSectionSpec] = section_specs or []
        self._valid_sections: set[str] = (
            {spec.name for spec in self.section_specs} | {GENERAL_SECTION}
        )
        self.contrastive_probability = contrastive_probability
        self.contrastive_max_answer_chars = contrastive_max_answer_chars
        self.feedback_answer_head_chars = feedback_answer_head_chars
        self.feedback_answer_tail_chars = feedback_answer_tail_chars
        self.use_structured_output = use_structured_output
        self.last_audit_trace: List[Dict[str, Any]] = []

    def _build_section_descriptions(self) -> str:
        """Format configured sections as a bullet list for LLM instructions.

        Returns:
            Markdown-style lines ``- [name]: description`` or a default general line.
        """
        if not self.section_specs:
            return f"- [{GENERAL_SECTION}]: target the prompt as a whole"
        return "\n".join(
            f"- [{spec.name}]: {spec.description}" for spec in self.section_specs
        )

    def generate_recommendation(
        self,
        prompt: str,
        instance: str,
        model_answer: str,
        model_answer_parsed: Optional[str] = None,
        metric_value: float | int = 0.0,
        ground_truth: str | int = "",
        contrastive_candidates: Optional[List[ContrastiveCandidate]] = None,
    ) -> Recommendation:
        """Produce one recommendation for a single failed example.

        Args:
            prompt: Prompt that produced the failure.
            instance: Task input / instance text shown to the model.
            model_answer: Raw assistant output on the failure.
            model_answer_parsed: Parsed output if available.
            metric_value: Scalar score assigned to the failure.
            ground_truth: Reference target for the task.
            contrastive_candidates: Optional list of better-scoring alternatives.

        Returns:
            Parsed :class:`Recommendation` (section + text).
        """
        try_contrastive = (
            contrastive_candidates is not None
            and _random.random() < self.contrastive_probability
        )

        if try_contrastive:
            best = self._pick_best_contrastive(contrastive_candidates, metric_value)
            if best is not None:
                return self._generate_contrastive(
                    failed_prompt=prompt,
                    failed_answer_raw=model_answer,
                    failed_answer_parsed=model_answer_parsed,
                    failed_score=metric_value,
                    instance=instance,
                    ground_truth=ground_truth,
                    success=best,
                )
        model_answer_for_feedback = self._truncate_head_tail(
            model_answer,
            self.feedback_answer_head_chars,
            self.feedback_answer_tail_chars,
        )
        formatted_prompt = FEEDBACK_PROMPT_TEMPLATE.format(
            prompt=prompt,
            instance=instance,
            model_answer=model_answer_for_feedback,
            model_answer_parsed=model_answer_parsed or "",
            metric_value=metric_value,
            ground_truth=ground_truth,
            section_descriptions=self._build_section_descriptions(),
        )
        if self.use_structured_output:
            return self._invoke_structured_recommendation(formatted_prompt)
        result = get_model_answer_extracted(self.model, formatted_prompt)
        return self._parse_recommendation(result)

    def _invoke_structured_recommendation(self, formatted_prompt: str) -> Recommendation:
        """Run a structured LLM call returning a SectionRecommendationResponse.

        Validates the returned section against ``self._valid_sections`` and
        maps unknown / blank sections to the ``general`` section, mirroring
        the behavior of :meth:`_try_parse` for the text-mode path.
        """
        structured = self.model.with_structured_output(
            SectionRecommendationResponse, method="json_schema"
        )
        try:
            response = structured.invoke(formatted_prompt)
        except Exception as exc:
            logger.debug(f"[Feedback] structured recommendation call failed: {exc}")
            return Recommendation(section=GENERAL_SECTION, text="")
        section = (response.section or "").strip()
        text = (response.text or "").strip()
        if not text:
            return Recommendation(section=GENERAL_SECTION, text="")
        if section not in self._valid_sections:
            if section:
                logger.debug(
                    f"[Feedback] Unknown section '{section}' from model -> general"
                )
            section = GENERAL_SECTION
        return Recommendation(section=section, text=text)

    @staticmethod
    def _pick_best_contrastive(
        candidates: List[ContrastiveCandidate], failing_score: float | int
    ) -> Optional[ContrastiveCandidate]:
        """Pick the highest-scoring contrastive candidate strictly above ``failing_score``.

        Args:
            candidates: Non-empty list of alternative prompts with scores.
            failing_score: Score of the prompt that failed on the same instance.

        Returns:
            The winning :class:`ContrastiveCandidate`, or ``None`` if none qualify.
        """
        best = None
        for c in candidates:
            if c.score > failing_score and (best is None or c.score > best.score):
                best = c
        return best

    @staticmethod
    def _truncate_head_tail(text: str, head_chars: int, tail_chars: int) -> str:
        """Truncate long text to head and tail segments for compact prompts.

        Args:
            text: Original model output (may be empty).
            head_chars: Characters to keep from the start (may be zero).
            tail_chars: Characters to keep from the end (may be zero).

        Returns:
            Original string if within budget, otherwise a stitched head/tail excerpt.
        """
        text = text or ""
        max_chars = head_chars + tail_chars
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        head = text[:head_chars] if head_chars else ""
        tail = text[-tail_chars:] if tail_chars else ""
        return (
            f"{head}\n"
            f"...[truncated middle: kept first {head_chars} chars and "
            f"last {tail_chars} chars of {len(text)} total chars]...\n"
            f"{tail}"
        )

    def _generate_contrastive(
        self,
        *,
        failed_prompt: str,
        failed_answer_raw: str,
        failed_answer_parsed: Optional[str],
        failed_score: float | int,
        instance: str,
        ground_truth: str | int,
        success: ContrastiveCandidate,
    ) -> Recommendation:
        """Build and run the contrastive feedback prompt for one failure.

        Args:
            failed_prompt: Prompt associated with the failure.
            failed_answer_raw: Raw assistant output for the failure.
            failed_answer_parsed: Parsed assistant output if available.
            failed_score: Scalar score for the failing run.
            instance: Shared task instance text.
            ground_truth: Reference answer for the task.
            success: Best contrastive candidate beating ``failed_score``.

        Returns:
            Parsed recommendation from the contrastive LLM response.
        """
        failed_answer_raw = self._truncate_head_tail(
            failed_answer_raw,
            self.feedback_answer_head_chars,
            self.feedback_answer_tail_chars,
        )
        succeeded_answer_raw = success.raw_answer or ""
        truncation_note = ""
        max_chars = self.contrastive_max_answer_chars
        if max_chars > 0 and len(succeeded_answer_raw) > max_chars:
            head = max_chars // 2
            tail = max_chars - head
            succeeded_answer_raw = self._truncate_head_tail(
                succeeded_answer_raw, head, tail
            )
            truncation_note = (
                f", truncated to {max_chars} chars head/tail - possibly incomplete "
                f"but the visible part is still useful"
            )

        formatted_prompt = CONTRASTIVE_FEEDBACK_PROMPT.format(
            failed_prompt=failed_prompt,
            failed_answer_raw=failed_answer_raw,
            failed_answer_parsed=failed_answer_parsed or "",
            failed_score=failed_score,
            succeeded_prompt=success.prompt,
            succeeded_answer_raw=succeeded_answer_raw,
            truncation_note=truncation_note,
            succeeded_answer_parsed=success.parsed_answer or "",
            succeeded_score=success.score,
            instance=instance,
            ground_truth=ground_truth,
            section_descriptions=self._build_section_descriptions(),
        )
        if self.use_structured_output:
            return self._invoke_structured_recommendation(formatted_prompt)
        result = get_model_answer_extracted(self.model, formatted_prompt)
        return self._parse_recommendation(result)

    def _try_parse(self, raw_str: str) -> Tuple[str, str, Optional[str]]:
        """Parse a single JSON object ``{"section": ..., "text": ...}`` from model text.

        Args:
            raw_str: Raw model output.

        Returns:
            Tuple ``(section, text, error_kind)`` where ``error_kind`` is ``None`` on
            success, ``"json_error"`` for malformed JSON, or ``"invalid_section"`` when
            the section is not whitelisted (text still returned, mapped to ``general`` upstream by callers).
        """
        try:
            data = extract_json(raw_str)
            if not isinstance(data, dict) or "section" not in data or "text" not in data:
                return GENERAL_SECTION, raw_str.strip(), "json_error"
            section = str(data["section"]).strip()
            text = str(data["text"]).strip()
            if not text:
                return GENERAL_SECTION, raw_str.strip(), "json_error"
            if section not in self._valid_sections:
                logger.debug(f"[Feedback] Unknown section '{section}' from model -> general")
                return GENERAL_SECTION, text, "invalid_section"
            return section, text, None
        except Exception as exc:
            logger.debug(f"[Feedback] JSON parse failed: {exc}")
            return GENERAL_SECTION, raw_str.strip(), "json_error"

    def _parse_recommendation(self, raw: Any) -> Recommendation:
        """Convert arbitrary model output into a structured :class:`Recommendation`.

        Args:
            raw: Model return value (string or coercible).

        Returns:
            Recommendation with section/text from :meth:`_try_parse` (errors map
            to the ``general`` section with best-effort text).
        """
        raw_str = raw if isinstance(raw, str) else str(raw)
        section, text, _error_kind = self._try_parse(raw_str)

        return Recommendation(section=section, text=text)

    def generate_recommendations(
        self,
        prompt: str,
        failed_examples: List[FailedExampleDetailed],
        contrastive_candidates_per_failure: Optional[List[List[ContrastiveCandidate]]] = None,
    ) -> List[Recommendation]:
        """Call :meth:`generate_recommendation` for each failed example.

        Args:
            prompt: Prompt shared across failures.
            failed_examples: Detailed failure rows from the evaluator.
            contrastive_candidates_per_failure: Optional parallel list of contrastive
                rows per failure index (same length as ``failed_examples``).

        Returns:
            List of recommendations, one per failure (order preserved).

        Raises:
            ValueError: If contrastive list length mismatches ``failed_examples``.
        """
        cc = contrastive_candidates_per_failure
        if cc is not None and len(cc) != len(failed_examples):
            raise ValueError(
                "contrastive_candidates_per_failure must have the same length as failed_examples"
            )
        out: List[Recommendation] = []
        for i, fe in enumerate(failed_examples):
            contrastive = cc[i] if cc is not None else None
            out.append(
                self.generate_recommendation(
                    prompt=prompt,
                    instance=fe.instance,
                    model_answer=fe.assistant_answer,
                    model_answer_parsed=fe.model_answer_parsed,
                    metric_value=fe.metric_value,
                    ground_truth=fe.ground_truth,
                    contrastive_candidates=contrastive,
                )
            )
        return out

    def filter_recommendations(
        self, recommendations: List[Recommendation]
    ) -> List[Recommendation]:
        """Partition by section, merge semantically similar recs, then synthesize.

        Args:
            recommendations: Possibly redundant cross-section recommendations.

        Returns:
            Filtered list sorted by descending weight within each section.
        """
        if not recommendations:
            return []

        by_section: Dict[str, List[Recommendation]] = {}
        for rec in recommendations:
            by_section.setdefault(rec.section, []).append(rec)

        result: List[Recommendation] = []
        for section_name, section_recs in by_section.items():
            section_out = self._filter_section(section_name, section_recs)
            section_out.sort(key=lambda r: r.weight, reverse=True)
            result.extend(section_out)
        return result

    def drop_instance_leaks(
        self,
        recs: List[Recommendation],
        problem_description: str,
    ) -> List[Recommendation]:
        """LLM audit: KEEP / REWRITE / DROP recommendations that leak instance details.

        Args:
            recs: Recommendations after grouping/filtering.
            problem_description: Task definition text for leakage judgment.

        Returns:
            Surviving recommendations (rewritten where applicable). On parse errors,
            returns ``recs`` unchanged and stores a fallback trace in
            :attr:`last_audit_trace`.
        """
        if not recs or not problem_description:
            self.last_audit_trace = []
            return recs

        payload = [{"section": r.section, "text": r.text} for r in recs]
        prompt = DROP_INSTANCE_LEAK_PROMPT.format(
            problem_description=problem_description,
            recommendations_json=json.dumps(payload, ensure_ascii=False, indent=2),
        )

        raw_str = ""
        try:
            if self.use_structured_output:
                structured = self.model.with_structured_output(
                    InstanceLeakAuditResponse, method="json_schema"
                )
                response = structured.invoke(prompt)
                verdicts: List[Any] = [
                    {
                        "verdict": (v.verdict or "").strip(),
                        "text": (v.text or "").strip(),
                    }
                    for v in (response.verdicts or [])
                ]
                if len(verdicts) != len(recs):
                    raise ValueError("verdicts count mismatch")
            else:
                raw = get_model_answer_extracted(self.model, prompt)
                raw_str = raw if isinstance(raw, str) else str(raw)
                data = extract_json(raw_str)
                if not isinstance(data, dict) or "verdicts" not in data:
                    raise ValueError("missing 'verdicts' key")
                verdicts = data["verdicts"]
                if not isinstance(verdicts, list) or len(verdicts) != len(recs):
                    raise ValueError("verdicts count mismatch")
            kept: List[Recommendation] = []
            trace: List[Dict[str, Any]] = []
            for r, v in zip(recs, verdicts):
                verdict = str(v.get("verdict", "")).strip().upper() if isinstance(v, dict) else ""
                rewritten = str(v.get("text", "")).strip() if isinstance(v, dict) else ""
                final_text: Optional[str] = None
                kept_flag = False
                if verdict == "KEEP":
                    kept.append(r)
                    final_text = r.text
                    kept_flag = True
                elif verdict == "REWRITE":
                    if rewritten:
                        kept.append(
                            Recommendation(
                                section=r.section,
                                text=rewritten,
                                weight=r.weight,
                            )
                        )
                        final_text = rewritten
                        kept_flag = True
                    else:
                        verdict = "DROP_EMPTY_REWRITE"
                elif verdict == "DROP":
                    pass
                else:
                    kept.append(r)
                    final_text = r.text
                    kept_flag = True
                    verdict = "UNKNOWN_KEEP"
                trace.append(
                    {
                        "section": r.section,
                        "original_text": r.text,
                        "weight": r.weight,
                        "verdict": verdict,
                        "rewritten_text": rewritten,
                        "kept": kept_flag,
                        "final_text": final_text,
                    }
                )
            self.last_audit_trace = trace
            return kept
        except Exception as exc:
            logger.debug(f"[Feedback] drop_instance_leaks parse failed: {exc}")
            self.last_audit_trace = [
                {
                    "section": r.section,
                    "original_text": r.text,
                    "weight": r.weight,
                    "verdict": "FALLBACK_KEEP",
                    "rewritten_text": "",
                    "kept": True,
                    "final_text": r.text,
                    "error": str(exc),
                    "raw_output": raw_str,
                }
                for r in recs
            ]
            return recs

    def _filter_section(
        self,
        section_name: str,
        recs: List[Recommendation],
    ) -> List[Recommendation]:
        """Cluster, synthesize, and deduplicate recommendations for one section name.

        Args:
            section_name: Target section identifier.
            recs: Non-empty list of recommendations for that section.

        Returns:
            Synthesized recommendations after LLM filtering (or singleton fallback).
        """
        if len(recs) <= 1:
            return list(recs)

        texts = [r.text for r in recs]
        groups = self._partition_texts_into_groups(texts)

        group_payload = [
            {"weight": len(g), "members": [texts[i] for i in g]}
            for g in groups
        ]
        prompt = SECTION_GROUPS_FILTER_PROMPT.format(
            section_name=section_name,
            groups_json=json.dumps(group_payload, ensure_ascii=False, indent=2),
        )

        synthesized: Optional[List[Tuple[str, int]]] = None
        if self.use_structured_output:
            try:
                structured = self.model.with_structured_output(
                    SynthesizedRecommendationsResponse, method="json_schema"
                )
                response = structured.invoke(prompt)
                parsed: List[Tuple[str, int]] = []
                for item in response.synthesized or []:
                    text = (item.text or "").strip()
                    try:
                        weight = max(1, int(item.weight))
                    except (TypeError, ValueError):
                        weight = 1
                    if text:
                        parsed.append((text, weight))
                synthesized = parsed or None
            except Exception as exc:
                logger.debug(
                    f"[Feedback] structured section filter failed: {exc}"
                )
                synthesized = None
        else:
            raw = get_model_answer_extracted(self.model, prompt)
            synthesized = self._parse_synthesized_filter_response(raw)

        if synthesized is None:
            logger.warning(
                f"[Feedback] Section '{section_name}': group filter parse "
                f"failed; falling back to first-member representative per group."
            )
            return [
                Recommendation(
                    section=section_name,
                    text=texts[g[0]],
                    weight=len(g),
                )
                for g in groups
            ]

        return [
            Recommendation(section=section_name, text=t, weight=w)
            for t, w in synthesized
        ]

    def _partition_texts_into_groups(self, texts: List[str]) -> List[List[int]]:
        """Partition ``texts`` indices into semantic groups (delegates to LLM).

        Args:
            texts: Parallel texts to cluster.

        Returns:
            List of index groups covering every input index at least once.
        """
        return self._llm_partition_into_groups(texts)

    def _llm_partition_into_groups(self, texts: List[str]) -> List[List[int]]:
        """Call the grouping LLM and normalize the JSON list-of-id-lists response.

        Args:
            texts: Same-length list as the recommendation bodies being grouped.

        Returns:
            Partition as list of index lists; falls back to singleton groups on error.
        """
        payload = [{"id": i, "text": t} for i, t in enumerate(texts)]
        prompt = RECOMMENDATIONS_GROUP_PROMPT.format(
            items_json=json.dumps(payload, ensure_ascii=False, indent=2)
        )

        if self.use_structured_output:
            try:
                structured = self.model.with_structured_output(
                    RecommendationGroupsResponse, method="json_schema"
                )
                response = structured.invoke(prompt)
                raw_groups = response.groups or []
                groups: List[List[int]] = []
                for g in raw_groups:
                    if not isinstance(g, list):
                        continue
                    ids = [int(x) for x in g if isinstance(x, (int, float, str))]
                    ids = [i for i in ids if 0 <= i < len(texts)]
                    if ids:
                        groups.append(ids)
                seen = {i for grp in groups for i in grp}
                for i in range(len(texts)):
                    if i not in seen:
                        groups.append([i])
                if groups:
                    return groups
            except Exception as exc:
                logger.debug(
                    f"[Feedback] structured group partition failed: {exc}"
                )
            return [[i] for i in range(len(texts))]

        raw = get_model_answer_extracted(self.model, prompt)
        try:
            data = extract_json(raw if isinstance(raw, str) else str(raw))
            if isinstance(data, list):
                groups: List[List[int]] = []
                for g in data:
                    if not isinstance(g, list):
                        continue
                    ids = [int(x) for x in g if isinstance(x, (int, float, str))]
                    ids = [i for i in ids if 0 <= i < len(texts)]
                    if ids:
                        groups.append(ids)
                # Ensure every text appears at least once: any missing index becomes its own group
                seen = {i for grp in groups for i in grp}
                for i in range(len(texts)):
                    if i not in seen:
                        groups.append([i])
                if groups:
                    return groups
        except Exception as exc:
            logger.debug(f"[Feedback] LLM group partition parse failed: {exc}")

        # Fallback: each item in its own group
        return [[i] for i in range(len(texts))]

    def _parse_synthesized_filter_response(
        self, raw: Any
    ) -> Optional[List[Tuple[str, int]]]:
        """Parse synthesis JSON ``{"synthesized": [{"text", "weight"}, ...]}``.

        Args:
            raw: Raw model output for the section filter call.

        Returns:
            List of ``(text, weight)`` tuples, or ``None`` if parsing fails.
        """
        raw_str = raw if isinstance(raw, str) else str(raw)
        try:
            data = extract_json(raw_str)
            if not isinstance(data, dict) or "synthesized" not in data:
                return None
            items = data["synthesized"]
            if not isinstance(items, list):
                return None
            out: List[Tuple[str, int]] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text", "")).strip()
                weight_raw = item.get("weight", 1)
                try:
                    weight = max(1, int(weight_raw))
                except (TypeError, ValueError):
                    weight = 1
                if text:
                    out.append((text, weight))
            return out or None
        except Exception as exc:
            logger.debug(f"[Feedback] group filter JSON parse failed: {exc}")
            return None
