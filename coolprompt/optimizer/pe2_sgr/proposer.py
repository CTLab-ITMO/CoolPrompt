"""PE2+SGR v3 proposer: dual-path structured reasoning.

Below score threshold: free-form reasoning (no structured
output) with always-reimagine generation.

Above score threshold: structured FullDiagnosis with
strategy override from severity/homogeneity signals,
then free-form generation.
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
    """Dual-path proposer: free-form exploration below
    threshold, structured precision above threshold.

    Args:
        model (BaseLanguageModel): LLM that supports
            structured output (tool use / function calling).
        prompt_max_tokens (int): Max token budget hint
            for the new prompt.
        log_path (str | Path | None): Optional path to a
            JSONL file for persisting diagnoses.
        score_threshold (float): Best validation score
            below which free-form reasoning is used
            (exploration mode).  At or above this
            threshold, structured FullDiagnosis with
            strategy override is used (precision mode).
    """

    def __init__(
        self,
        model: BaseLanguageModel,
        prompt_max_tokens: int = 300,
        log_path: Optional[str | Path] = None,
        score_threshold: float = 0.8,
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
        """Proposes a refined prompt via dual-path reasoning.

        Args:
            node (Node): Current beam node being refined.
            examples_str (str): Formatted failure examples.
            full_template (str): The full prompt template.
            batch_size (int): Number of failure examples.
            best_val_score (float): Best validation score
                among current beam candidates.

        Returns:
            tuple[str, str]: (new_prompt, reasoning).
        """
        if best_val_score < self.score_threshold:
            return self._freeform_path(
                node, examples_str,
                full_template, batch_size,
                best_val_score,
            )
        else:
            return self._structured_path(
                node, examples_str,
                full_template, batch_size,
                best_val_score,
            )

    # ----------------------------------------------------------
    # Free-form path (exploration, below threshold)
    # ----------------------------------------------------------

    def _freeform_path(
        self,
        node: Node,
        examples_str: str,
        full_template: str,
        batch_size: int,
        best_val_score: float,
    ) -> tuple[str, str]:
        """Free-form reasoning → always reimagine."""
        logger.debug(
            "SGR free-form path "
            f"(best_val={best_val_score:.4f} "
            f"< {self.score_threshold})"
        )

        # Phase 1: free-form reasoning
        reasoning = self._freeform_diagnose(
            node, examples_str,
            full_template, batch_size,
        )

        # Phase 2: always reimagine
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
        new_prompt = extract_answer(
            result,
            ("<prompt>", "</prompt>"),
            format_mismatch_label=node.prompt,
        )

        return new_prompt.strip(), reasoning[:200]

    def _freeform_diagnose(
        self,
        node: Node,
        examples_str: str,
        full_template: str,
        batch_size: int,
    ) -> str:
        """Free-form Phase 1: enhanced PE2-style reasoning."""
        prompt = PE2_SGR_FREEFORM_DIAGNOSIS_TEMPLATE.format(
            prompt=node.prompt,
            full_template=full_template,
            batch_size=batch_size,
            examples=examples_str,
        )
        return get_model_answer_extracted(
            self.model, prompt
        )

    # ----------------------------------------------------------
    # Structured path (precision, above threshold)
    # ----------------------------------------------------------

    def _structured_path(
        self,
        node: Node,
        examples_str: str,
        full_template: str,
        batch_size: int,
        best_val_score: float,
    ) -> tuple[str, str]:
        """Structured diagnosis → strategy override → gen."""
        logger.debug(
            "SGR structured path "
            f"(best_val={best_val_score:.4f} "
            f">= {self.score_threshold})"
        )

        # Phase 1: structured diagnosis
        diagnosis = self._structured_diagnose(
            node, examples_str,
            full_template, batch_size,
        )

        # Override strategy from signals
        strategy = self._override_strategy(diagnosis)
        formatted = self._format_diagnosis(diagnosis)

        # Phase 2: free-form generation
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
        new_prompt = extract_answer(
            result,
            ("<prompt>", "</prompt>"),
            format_mismatch_label=node.prompt,
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
    ) -> FullDiagnosis:
        """Structured Phase 1: FullDiagnosis."""
        prompt = PE2_SGR_FULL_DIAGNOSIS_TEMPLATE.format(
            prompt=node.prompt,
            full_template=full_template,
            batch_size=batch_size,
            examples=examples_str,
        )
        structured_model = self.model.with_structured_output(
            FullDiagnosis,
        )
        return structured_model.invoke(prompt)

    # ----------------------------------------------------------
    # Strategy override (Fix 1)
    # ----------------------------------------------------------

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

    # ----------------------------------------------------------
    # Formatting helpers
    # ----------------------------------------------------------

    @staticmethod
    def _format_diagnosis(
        diagnosis: FullDiagnosis,
    ) -> str:
        """Formats a FullDiagnosis into readable text
        for Phase 2 input."""
        parts = []

        # Per-example analyses
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

        # Pattern synthesis
        ps = diagnosis.pattern_synthesis
        parts.append(
            f"### Cross-Example Pattern\n"
            f"Common failure pattern: "
            f"{ps.common_failure_pattern}\n"
            f"Severity: {ps.pattern_severity}\n"
            f"Error homogeneity: {ps.error_homogeneity}"
        )
        parts.append("")

        # Prompt analysis
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

        # Rewrite strategy
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
