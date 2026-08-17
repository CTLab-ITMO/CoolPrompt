"""Iterative HyPER Light with MR.PEA-style playbook refinement."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional, override

from coolprompt.evaluator.evaluator import EvalResultDetailed
from coolprompt.optimizer.hyper.iterative_playbook import (
    HyPERLightPlaybookIterativeMethod,
)
from coolprompt.optimizer.hyper.pea_playbook import (
    PEA_PLAYBOOK_SYSTEM_PROMPT,
    PEA_PLAYBOOK_USER_PROMPT,
)
from coolprompt.utils.parsing import extract_json, get_model_answer_extracted

logger = logging.getLogger(__name__)


class HyPERLightPEAPlaybookIterativeMethod(
    HyPERLightPlaybookIterativeMethod
):
    """Iterative HyPER Light using the flat MR.PEA knowledge structure."""

    @staticmethod
    def _pea_request(
        *,
        task_description: str,
        sample_question: str,
        latest_knowledge: Any,
    ) -> str:
        return (
            PEA_PLAYBOOK_SYSTEM_PROMPT
            + "\n\n"
            + PEA_PLAYBOOK_USER_PROMPT.format(
                task_description=task_description,
                sample_question=sample_question,
                latest_knowledge=json.dumps(
                    latest_knowledge, ensure_ascii=False, indent=2
                ),
            )
        )

    def _generate_playbook(self, model: Any, initial_prompt: str) -> dict[str, Any]:
        """Generate the initial MR.PEA knowledge object."""
        request = self._pea_request(
            task_description=initial_prompt,
            sample_question="",
            latest_knowledge={},
        )
        raw_result = get_model_answer_extracted(model, request)
        parsed = extract_json(raw_result)
        if isinstance(parsed, dict):
            return parsed
        return {"raw_playbook": raw_result}

    def _update_playbook(
        self,
        model: Any,
        *,
        task_description: str,
        current_prompt: str,
        current_playbook: dict[str, Any],
        current_score: float,
        previous_score: float,
        result: EvalResultDetailed,
        max_failures: int,
        max_answer_chars: int,
        update_prompt: Optional[str],
    ) -> dict[str, Any]:
        """Refine MR.PEA knowledge using the latest evaluation evidence."""
        failures = self._format_failures(
            result, max_failures=max_failures, max_answer_chars=max_answer_chars
        )
        sample_question = (
            "Improve the existing abstract knowledge using this evidence.\n\n"
            f"Current prompt:\n{current_prompt}\n\n"
            f"Current score: {current_score}\n"
            f"Previous score: {previous_score}\n"
            f"Low-scoring evaluation pairs:\n{failures}"
        )

        if update_prompt is not None:
            request = update_prompt.format(
                task_description=task_description,
                current_prompt=current_prompt,
                current_playbook=json.dumps(
                    current_playbook, ensure_ascii=False, indent=2
                ),
                current_score=current_score,
                previous_score=previous_score,
                failures=failures,
            )
        else:
            request = self._pea_request(
                task_description=task_description,
                sample_question=sample_question,
                latest_knowledge=current_playbook,
            )

        raw_result = get_model_answer_extracted(model, request)
        parsed = extract_json(raw_result)
        if isinstance(parsed, dict):
            return parsed
        logger.warning(
            "PEA playbook update did not return JSON; keeping current playbook"
        )
        return current_playbook

    @property
    @override
    def name(self) -> str:
        return "hyper_light_pea_playbook_iterative"
