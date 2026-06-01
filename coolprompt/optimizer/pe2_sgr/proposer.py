"""PE2+SGR v4 proposer: adaptive free-form → structured.

Always starts with free-form reasoning for radical
exploration.  Tracks improvement between beam search
steps.  When improvement slows below ``min_improvement``,
locks into structured SGR for precision refinement.
One-way switch per task.
"""

import json
from pathlib import Path
from typing import Optional

from langchain_core.language_models.base import (
    BaseLanguageModel,
)

from coolprompt.optimizer.pe2.node import Node
from coolprompt.optimizer.pe2_sgr.schemas import (
    FullDiagnosis,
)
from coolprompt.utils.logging_config import logger
from coolprompt.utils.parsing import (
    extract_answer,
    get_model_answer_extracted,
)
from coolprompt.utils.prompt_templates.pe2_sgr_templates import (
    PE2_SGR_FREEFORM_DIAGNOSIS_TEMPLATE,
    PE2_SGR_FULL_DIAGNOSIS_TEMPLATE,
    PE2_SGR_GEN_INCREMENTAL,
    PE2_SGR_GEN_REIMAGINE,
    PE2_SGR_GEN_STRUCTURAL,
)

_GEN_TEMPLATES = {
    "incremental_edit": PE2_SGR_GEN_INCREMENTAL,
    "structural_rewrite": PE2_SGR_GEN_STRUCTURAL,
    "complete_reimagine": PE2_SGR_GEN_REIMAGINE,
}


class SGRProposer:
    """Adaptive proposer: starts free-form, switches to
    structured SGR when improvement slows.

    Args:
        model (BaseLanguageModel): LLM that supports
            structured output (tool use / function calling).
        prompt_max_tokens (int): Max token budget hint
            for the new prompt.
        log_path (str | Path | None): Optional path to a
            JSONL file for persisting diagnoses.
        min_improvement (float): Minimum improvement in
            best_val_score between beam steps to stay
            in free-form mode.  Below this threshold
            the proposer locks into structured SGR.
    """

    def __init__(
        self,
        model: BaseLanguageModel,
        prompt_max_tokens: int = 300,
        log_path: Optional[str | Path] = None,
        min_improvement: float = 0.02,
    ) -> None:
        self.model = model
        self.prompt_max_tokens = prompt_max_tokens
        self.log_path = (
            Path(log_path) if log_path else None
        )
        self.min_improvement = min_improvement
        self._prev_best: Optional[float] = None
        self._locked_structured: bool = False

    def propose(
        self,
        node: Node,
        examples_str: str,
        full_template: str,
        batch_size: int,
        best_val_score: float = 0.0,
        constraint_feedback: Optional[str] = None,
    ) -> tuple[str, str]:
        """Proposes a refined prompt via dual-path reasoning.

        Args:
            node (Node): Current beam node being refined.
            examples_str (str): Formatted failure examples.
            full_template (str): The full prompt template.
            batch_size (int): Number of failure examples.
            best_val_score (float): Best validation score
                among current beam candidates.
            constraint_feedback (Optional[str]): Per-
                constraint failure rates from the trainer.
                When None the diagnosis is unchanged.

        Returns:
            tuple[str, str]: (new_prompt, reasoning).
        """
        if self._prev_best is not None:
            if best_val_score != self._prev_best:
                delta = best_val_score - self._prev_best
                if (
                    delta < self.min_improvement
                    and not self._locked_structured
                ):
                    logger.debug(
                        "SGR switching to structured "
                        f"(improvement {delta:.4f} "
                        f"< {self.min_improvement})"
                    )
                    self._locked_structured = True

        if (
            self._prev_best is None
            or best_val_score != self._prev_best
        ):
            self._prev_best = best_val_score

        cf_block = (
            f"Per-constraint failure rates:\n"
            f"{constraint_feedback}"
            if constraint_feedback else ""
        )

        if self._locked_structured:
            return self._structured_path(
                node, examples_str,
                full_template, batch_size,
                best_val_score,
                constraint_feedback=cf_block,
            )
        else:
            return self._freeform_path(
                node, examples_str,
                full_template, batch_size,
                best_val_score,
                constraint_feedback=cf_block,
            )

    def _freeform_path(
        self,
        node: Node,
        examples_str: str,
        full_template: str,
        batch_size: int,
        best_val_score: float,
        constraint_feedback: str = "",
    ) -> tuple[str, str]:
        """Free-form reasoning → always reimagine."""
        logger.debug(
            "SGR free-form path "
            f"(best_val={best_val_score:.4f} "
            f"< {self.min_improvement})"
        )

        reasoning = self._freeform_diagnose(
            node, examples_str,
            full_template, batch_size,
            constraint_feedback=constraint_feedback,
        )

        logger.debug(
            "SGR Phase 2 strategy: complete_reimagine "
            "(free-form path)"
        )
        gen_prompt = PE2_SGR_GEN_REIMAGINE.format(
            prompt=node.prompt,
            formatted_diagnosis=reasoning,
            max_tokens=self.prompt_max_tokens,
        )
        result = get_model_answer_extracted(
            self.model, gen_prompt
        )
        new_prompt = self._extract_prompt(
            result, node.prompt
        )

        return new_prompt.strip(), reasoning[:200]

    def _freeform_diagnose(
        self,
        node: Node,
        examples_str: str,
        full_template: str,
        batch_size: int,
        constraint_feedback: str = "",
    ) -> str:
        """Free-form Phase 1: enhanced PE2-style reasoning."""
        prompt = PE2_SGR_FREEFORM_DIAGNOSIS_TEMPLATE.format(
            prompt=node.prompt,
            full_template=full_template,
            batch_size=batch_size,
            examples=examples_str,
            constraint_feedback=constraint_feedback,
        )
        return get_model_answer_extracted(
            self.model, prompt
        )

    def _structured_path(
        self,
        node: Node,
        examples_str: str,
        full_template: str,
        batch_size: int,
        best_val_score: float,
        constraint_feedback: str = "",
    ) -> tuple[str, str]:
        """Structured diagnosis → strategy override → gen."""
        logger.debug(
            "SGR structured path "
            f"(best_val={best_val_score:.4f} "
            f">= {self.min_improvement})"
        )

        diagnosis = self._structured_diagnose(
            node, examples_str,
            full_template, batch_size,
            constraint_feedback=constraint_feedback,
        )

        strategy = self._override_strategy(diagnosis)
        formatted = self._format_diagnosis(diagnosis)

        logger.debug(
            f"SGR Phase 2 strategy: {strategy}"
        )
        gen_template = _GEN_TEMPLATES[strategy]
        gen_prompt = gen_template.format(
            prompt=node.prompt,
            formatted_diagnosis=formatted,
            max_tokens=self.prompt_max_tokens,
        )
        result = get_model_answer_extracted(
            self.model, gen_prompt
        )
        new_prompt = self._extract_prompt(
            result, node.prompt
        )

        self._log_result(diagnosis, strategy)
        self._save_jsonl(diagnosis)

        reasoning = (
            diagnosis.rewrite_strategy.justification
        )
        return new_prompt.strip(), reasoning

    def _structured_diagnose(
        self,
        node: Node,
        examples_str: str,
        full_template: str,
        batch_size: int,
        constraint_feedback: str = "",
    ) -> FullDiagnosis:
        """Structured Phase 1: FullDiagnosis."""
        prompt = PE2_SGR_FULL_DIAGNOSIS_TEMPLATE.format(
            prompt=node.prompt,
            full_template=full_template,
            batch_size=batch_size,
            examples=examples_str,
            constraint_feedback=constraint_feedback,
        )
        structured_model = self.model.with_structured_output(
            FullDiagnosis,
        )
        return structured_model.invoke(prompt)

    @staticmethod
    def _extract_prompt(
        result: str, fallback: str
    ) -> str:
        """Extracts prompt from <prompt> tags with
        robust fallback for empty original prompts."""
        extracted = extract_answer(
            result,
            ("<prompt>", "</prompt>"),
            format_mismatch_label=fallback,
        )
        extracted_str = str(extracted).strip()
        if not extracted_str and result.strip():
            logger.debug(
                "SGR extraction empty, using raw "
                "result as prompt"
            )
            return result.strip()
        return extracted_str

    @staticmethod
    def _override_strategy(
        diagnosis: FullDiagnosis,
    ) -> str:
        """Overrides model's strategy choice based on its
        own severity + homogeneity signals.

        Fixes the conservatism bias where the model
        correctly diagnoses fundamental problems but
        conservatively picks incremental strategies.
        """
        sev = diagnosis.pattern_synthesis.pattern_severity
        hom = diagnosis.pattern_synthesis.error_homogeneity
        original = diagnosis.rewrite_strategy.approach

        if sev == "fundamental" and hom == "high":
            forced = "complete_reimagine"
        elif sev == "fundamental" and hom == "medium":
            forced = "structural_rewrite"
        elif sev == "structural" and hom == "high":
            forced = "structural_rewrite"
        else:
            forced = original

        if forced != original:
            logger.debug(
                f"SGR strategy override: "
                f"{original} -> {forced} "
                f"(severity={sev}, "
                f"homogeneity={hom})"
            )

        return forced

    @staticmethod
    def _format_diagnosis(
        diagnosis: FullDiagnosis,
    ) -> str:
        """Formats a FullDiagnosis into readable text
        for Phase 2 input."""
        parts = []

        parts.append("### Per-Example Analysis")
        for i, ea in enumerate(
            diagnosis.error_analyses, 1
        ):
            parts.append(
                f"Example {i}: "
                f"{ea.root_cause.value} — "
                f"{ea.root_cause_explanation}"
            )
        parts.append("")

        ed = diagnosis.edit_decision
        parts.append(
            f"### Edit Decision\n"
            f"Necessary: {ed.editing_necessary} "
            f"(confidence: {ed.confidence})\n"
            f"Justification: {ed.justification}"
        )
        parts.append("")

        ps = diagnosis.pattern_synthesis
        parts.append(
            f"### Cross-Example Pattern\n"
            f"Common failure pattern: "
            f"{ps.common_failure_pattern}\n"
            f"Severity: {ps.pattern_severity}\n"
            f"Error homogeneity: {ps.error_homogeneity}"
        )
        parts.append("")

        pa = diagnosis.prompt_analysis
        parts.append(
            f"### Prompt Analysis\n"
            f"Describes task correctly: "
            f"{pa.describes_task_correctly}"
        )
        if pa.missing_elements:
            parts.append(
                f"Missing: "
                f"{', '.join(pa.missing_elements)}"
            )
        if pa.misleading_elements:
            parts.append(
                f"Misleading: "
                f"{', '.join(pa.misleading_elements)}"
            )
        parts.append("")

        rs = diagnosis.rewrite_strategy
        parts.append(
            f"### Strategy: {rs.approach}\n"
            f"Key insight: {rs.key_insight}\n"
            f"Justification: {rs.justification}"
        )

        return "\n".join(parts)

    def _log_result(
        self,
        diagnosis: FullDiagnosis,
        final_strategy: str,
    ) -> None:
        """Logs diagnosis at DEBUG level."""
        ps = diagnosis.pattern_synthesis
        logger.debug(
            f"SGR pattern: "
            f"severity={ps.pattern_severity}, "
            f"homogeneity={ps.error_homogeneity}, "
            f"pattern={ps.common_failure_pattern!r}"
        )

        rs = diagnosis.rewrite_strategy
        logger.debug(
            f"SGR strategy: "
            f"model={rs.approach}, "
            f"final={final_strategy}, "
            f"key_insight={rs.key_insight!r}"
        )

        for i, ea in enumerate(
            diagnosis.error_analyses
        ):
            logger.debug(
                f"SGR error[{i}]: "
                f"cause={ea.root_cause.value}, "
                f"explanation="
                f"{ea.root_cause_explanation!r}"
            )

    def _save_jsonl(
        self,
        diagnosis: FullDiagnosis,
    ) -> None:
        """Appends diagnosis as JSON line to log_path."""
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(
            parents=True, exist_ok=True
        )
        with open(
            self.log_path, "a", encoding="utf-8"
        ) as f:
            f.write(
                json.dumps(diagnosis.model_dump()) + "\n"
            )
