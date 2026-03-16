"""PE2+SGR proposer: structured prompt refinement."""

import json
from pathlib import Path
from typing import Optional

from langchain_core.language_models.base import BaseLanguageModel

from coolprompt.optimizer.pe2.node import Node
from coolprompt.optimizer.pe2_sgr.schemas import PE2SGROutput
from coolprompt.utils.prompt_templates.pe2_sgr_templates import (
    PE2_SGR_TEMPLATE,
)
from coolprompt.utils.logging_config import logger


class SGRProposer:
    """Generates refined prompts using structured output.

    Uses Pydantic schemas with LangChain's
    with_structured_output to enforce step-by-step
    reasoning via constrained decoding.

    Args:
        model (BaseLanguageModel): LLM that supports
            structured output (tool use / function calling).
        prompt_max_tokens (int): Max token budget hint
            for the new prompt.
        log_path (str | Path | None): Optional path to a
            JSONL file. When set, each propose() call
            appends the full PE2SGROutput as a JSON line.
    """

    def __init__(
        self,
        model: BaseLanguageModel,
        prompt_max_tokens: int = 300,
        log_path: Optional[str | Path] = None,
    ) -> None:
        self.model = model
        self.prompt_max_tokens = prompt_max_tokens
        self.structured_model = model.with_structured_output(
            PE2SGROutput,
        )
        self.log_path = Path(log_path) if log_path else None

    def _log_result(self, result: PE2SGROutput) -> None:
        """Logs full structured output at DEBUG level."""
        for i, ea in enumerate(result.error_analyses):
            logger.debug(
                f"SGR error_analysis[{i}]: "
                f"input={ea.input_summary!r}, "
                f"expected_vs_actual={ea.expected_vs_actual!r}, "
                f"root_cause={ea.root_cause.value}, "
                f"explanation={ea.root_cause_explanation!r}"
            )

        pa = result.prompt_analysis
        logger.debug(
            f"SGR prompt_analysis: "
            f"correct={pa.describes_task_correctly}, "
            f"missing={pa.missing_elements}, "
            f"misleading={pa.misleading_elements}"
        )

        ed = result.edit_decision
        logger.debug(
            f"SGR edit_decision: "
            f"necessary={ed.editing_necessary}, "
            f"confidence={ed.confidence}, "
            f"justification={ed.justification!r}"
        )

        for i, ch in enumerate(result.specific_changes):
            logger.debug(
                f"SGR change[{i}]: "
                f"type={ch.change_type}, "
                f"location={ch.location!r}, "
                f"rationale={ch.rationale!r}"
            )

        logger.debug(
            f"SGR summary: {result.improvement_summary}"
        )
        logger.debug(
            f"SGR improved_prompt:\n{result.improved_prompt}"
        )

    def _save_jsonl(self, result: PE2SGROutput) -> None:
        """Appends full result as a JSON line to log_path."""
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result.model_dump()) + "\n")

    def propose(
        self,
        node: Node,
        examples_str: str,
        full_template: str,
        batch_size: int,
    ) -> tuple[str, str]:
        """Proposes a refined prompt via structured reasoning.

        Args:
            node (Node): Current beam node being refined.
            examples_str (str): Formatted failure examples.
            full_template (str): The full prompt template.
            batch_size (int): Number of failure examples.

        Returns:
            tuple[str, str]: (new_prompt, reasoning) where
                reasoning is the JSON-serialized analysis.
        """
        prompt = PE2_SGR_TEMPLATE.format(
            prompt=node.prompt,
            full_template=full_template,
            batch_size=batch_size,
            examples=examples_str,
            max_tokens=self.prompt_max_tokens,
        )

        result: PE2SGROutput = self.structured_model.invoke(
            prompt
        )

        self._log_result(result)
        self._save_jsonl(result)

        reasoning = result.improvement_summary
        return result.improved_prompt.strip(), reasoning
