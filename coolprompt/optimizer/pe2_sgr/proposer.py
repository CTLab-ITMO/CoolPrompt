"""PE2+SGR v2 proposer: two-phase structured reasoning.

Phase 1 — structured diagnosis via constrained decoding.
Phase 2 — free-form prompt generation informed by the
diagnosis.
"""

import json
from pathlib import Path
from typing import Optional, Union

from langchain_core.language_models.base import (
    BaseLanguageModel,
)

from coolprompt.optimizer.pe2.node import Node
from coolprompt.optimizer.pe2_sgr.schemas import (
    EditDecision,
    ErrorAnalysis,
    FullDiagnosis,
    LightDiagnosis,
    PatternSynthesis,
    PromptAnalysis,
    RewriteStrategy,
)
from coolprompt.utils.logging_config import logger
from coolprompt.utils.parsing import (
    extract_answer,
    get_model_answer_extracted,
)
from coolprompt.utils.prompt_templates.pe2_sgr_templates import (
    PE2_SGR_FULL_DIAGNOSIS_TEMPLATE,
    PE2_SGR_GEN_INCREMENTAL,
    PE2_SGR_GEN_REIMAGINE,
    PE2_SGR_GEN_STRUCTURAL,
    PE2_SGR_LIGHT_DIAGNOSIS_TEMPLATE,
)

_GEN_TEMPLATES = {
    "incremental_edit": PE2_SGR_GEN_INCREMENTAL,
    "structural_rewrite": PE2_SGR_GEN_STRUCTURAL,
    "complete_reimagine": PE2_SGR_GEN_REIMAGINE,
}


class SGRProposer:
    """Two-phase proposer: structured diagnosis then
    free-form generation.

    Args:
        model (BaseLanguageModel): LLM that supports
            structured output (tool use / function calling).
        prompt_max_tokens (int): Max token budget hint
            for the new prompt.
        log_path (str | Path | None): Optional path to a
            JSONL file for persisting diagnoses.
        score_threshold (float): Best validation score
            below which LightDiagnosis is used (exploration
            mode).  At or above this threshold,
            FullDiagnosis is used (precision mode).
    """

    def __init__(
        self,
        model: BaseLanguageModel,
        prompt_max_tokens: int = 300,
        log_path: Optional[str | Path] = None,
        score_threshold: float = 0.5,
    ) -> None:
        self.model = model
        self.prompt_max_tokens = prompt_max_tokens
        self.log_path = (
            Path(log_path) if log_path else None
        )
        self.score_threshold = score_threshold

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def propose(
        self,
        node: Node,
        examples_str: str,
        full_template: str,
        batch_size: int,
        best_val_score: float = 0.0,
    ) -> tuple[str, str]:
        """Proposes a refined prompt via two-phase reasoning.

        Args:
            node (Node): Current beam node being refined.
            examples_str (str): Formatted failure examples.
            full_template (str): The full prompt template.
            batch_size (int): Number of failure examples.
            best_val_score (float): Best validation score
                among current beam candidates.  Controls
                whether light or full diagnosis is used.

        Returns:
            tuple[str, str]: (new_prompt, reasoning).
        """
        # Phase 1: structured diagnosis
        diagnosis = self._diagnose(
            node=node,
            examples_str=examples_str,
            full_template=full_template,
            batch_size=batch_size,
            best_val_score=best_val_score,
        )

        # Phase 2: free-form generation
        new_prompt = self._generate(
            node=node,
            diagnosis=diagnosis,
        )

        self._log_result(diagnosis)
        self._save_jsonl(diagnosis)

        reasoning = (
            diagnosis.rewrite_strategy.justification
        )
        return new_prompt.strip(), reasoning

    # ----------------------------------------------------------
    # Phase 1 — Structured diagnosis
    # ----------------------------------------------------------

    def _diagnose(
        self,
        node: Node,
        examples_str: str,
        full_template: str,
        batch_size: int,
        best_val_score: float,
    ) -> Union[LightDiagnosis, FullDiagnosis]:
        """Runs Phase 1: structured diagnosis."""
        if best_val_score < self.score_threshold:
            schema = LightDiagnosis
            template = PE2_SGR_LIGHT_DIAGNOSIS_TEMPLATE
            logger.debug(
                "SGR using LightDiagnosis "
                f"(best_val={best_val_score:.4f} "
                f"< {self.score_threshold})"
            )
        else:
            schema = FullDiagnosis
            template = PE2_SGR_FULL_DIAGNOSIS_TEMPLATE
            logger.debug(
                "SGR using FullDiagnosis "
                f"(best_val={best_val_score:.4f} "
                f">= {self.score_threshold})"
            )

        prompt = template.format(
            prompt=node.prompt,
            full_template=full_template,
            batch_size=batch_size,
            examples=examples_str,
        )

        structured_model = self.model.with_structured_output(
            schema,
        )
        return structured_model.invoke(prompt)

    # ----------------------------------------------------------
    # Phase 2 — Free-form generation
    # ----------------------------------------------------------

    def _generate(
        self,
        node: Node,
        diagnosis: Union[LightDiagnosis, FullDiagnosis],
    ) -> str:
        """Runs Phase 2: free-form prompt generation."""
        strategy = diagnosis.rewrite_strategy.approach
        gen_template = _GEN_TEMPLATES[strategy]
        formatted = self._format_diagnosis(diagnosis)

        logger.debug(
            f"SGR Phase 2 strategy: {strategy}"
        )

        gen_prompt = gen_template.format(
            prompt=node.prompt,
            formatted_diagnosis=formatted,
            max_tokens=self.prompt_max_tokens,
        )

        result = get_model_answer_extracted(
            self.model, gen_prompt
        )

        new_prompt = extract_answer(
            result,
            ("<prompt>", "</prompt>"),
            format_mismatch_label=node.prompt,
        )
        return new_prompt

    # ----------------------------------------------------------
    # Formatting helpers
    # ----------------------------------------------------------

    @staticmethod
    def _format_diagnosis(
        diagnosis: Union[LightDiagnosis, FullDiagnosis],
    ) -> str:
        """Formats a diagnosis object into readable text
        for Phase 2 input."""
        parts = []

        # Per-example analyses (FullDiagnosis only)
        if isinstance(diagnosis, FullDiagnosis):
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

        # Pattern synthesis (always present)
        ps = diagnosis.pattern_synthesis
        parts.append(
            f"### Cross-Example Pattern\n"
            f"Common failure pattern: "
            f"{ps.common_failure_pattern}\n"
            f"Severity: {ps.pattern_severity}\n"
            f"Error homogeneity: {ps.error_homogeneity}"
        )
        parts.append("")

        # Prompt analysis (always present)
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

        # Rewrite strategy (always present)
        rs = diagnosis.rewrite_strategy
        parts.append(
            f"### Strategy: {rs.approach}\n"
            f"Key insight: {rs.key_insight}\n"
            f"Justification: {rs.justification}"
        )

        return "\n".join(parts)

    # ----------------------------------------------------------
    # Logging
    # ----------------------------------------------------------

    def _log_result(
        self,
        diagnosis: Union[LightDiagnosis, FullDiagnosis],
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
            f"approach={rs.approach}, "
            f"key_insight={rs.key_insight!r}"
        )

        if isinstance(diagnosis, FullDiagnosis):
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
        diagnosis: Union[LightDiagnosis, FullDiagnosis],
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
