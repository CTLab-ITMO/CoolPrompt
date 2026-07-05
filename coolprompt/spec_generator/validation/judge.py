from __future__ import annotations

import json

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages.ai import AIMessage
from pydantic import BaseModel, Field

from coolprompt.spec_generator.schema import TaskSpec
from coolprompt.spec_generator.utils.model_utils import resolve_chat_model
from coolprompt.spec_generator.utils.retry_config import ValidationConfig
from coolprompt.spec_generator.utils.retry_utils import (
    RetryConfig,
    invoke_with_retry,
)
from coolprompt.spec_generator.validation.example_models import ExampleBase
from coolprompt.utils.logging_config import logger
from coolprompt.utils.parsing import extract_json
from coolprompt.utils.prompt_templates.judge_templates import JUDGE_TEMPLATE


def _format_list(items: list[str] | None, empty: str = "None.") -> str:
    return "; ".join(items) if items else empty


def _format_label_set(label_set: list[str] | None) -> str:
    if not label_set:
        return "None. Do not require a fixed output label."

    return ", ".join(label_set)


def _format_corner_cases(spec: TaskSpec) -> str:
    if not spec.corner_cases:
        return "No explicit corner-case patterns provided."

    return "\n".join(
        f"- {case.name}: {case.description}"
        for case in spec.corner_cases
    )


class JudgeVerdict(BaseModel):
    index: int = Field(ge=0, description="Zero-based index matching the candidate data.")
    is_valid: bool
    quality_score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)


class JudgeVerdictBatch(BaseModel):
    verdicts: list[JudgeVerdict]


class JudgeResponseError(ValueError):
    pass


class LLMJudge:
    def __init__(
            self,
            model: BaseLanguageModel,
            config: ValidationConfig | None = None,
            retry_config: RetryConfig | None = None,
    ) -> None:
        self._model = model
        self._config = (config if config is not None else ValidationConfig())
        self._retry_config = (retry_config if retry_config is not None else RetryConfig())

    def filter(
            self,
            examples: list[ExampleBase],
            spec: TaskSpec,
            *,
            is_corner: bool = False,
    ) -> tuple[list[ExampleBase], list[ExampleBase]]:
        logger.info(
            "LLM judge: enabled=%s, examples=%d, batch_size=%d, "
            "threshold=%.2f, is_corner=%s",
            self._config.judge_enabled,
            len(examples),
            self._config.judge_batch_size,
            self._config.judge_quality_threshold,
            is_corner,
        )

        if not self._config.judge_enabled or not examples:
            return list(examples), []

        accepted: list[ExampleBase] = []
        rejected: list[ExampleBase] = []
        batch_size = self._config.judge_batch_size

        for start in range(0, len(examples), batch_size):
            chunk = examples[start:start + batch_size]

            try:
                verdicts = self._judge_chunk(
                    chunk=chunk,
                    spec=spec,
                    is_corner=is_corner,
                )
            except Exception as exc:
                logger.warning(
                    "Judge failed for %s chunk of %d examples: %s. "
                    "Rejecting the whole chunk.",
                    "corner" if is_corner else "regular",
                    len(chunk),
                    exc,
                )
                rejected.extend(chunk)
                continue

            for example, verdict in zip(chunk, verdicts):
                if (
                        verdict.is_valid
                        and verdict.quality_score
                        >= self._config.judge_quality_threshold
                ):
                    accepted.append(example)
                    continue

                logger.info(
                    "Rejected example (judge): %s | "
                    "score=%.2f | reason=%s",
                    example.input,
                    verdict.quality_score,
                    verdict.reason,
                )
                rejected.append(example)

        return accepted, rejected

    def _judge_chunk(
            self,
            chunk: list[ExampleBase],
            spec: TaskSpec,
            is_corner: bool,
    ) -> list[JudgeVerdict]:
        pairs = [
            {
                "index": index,
                "input": example.input,
                "output": example.output,
            }
            for index, example in enumerate(chunk)
        ]

        if is_corner:
            dataset_kind = "corner-case examples in a synthetic dataset"
            corner_section = (
                "Expected corner-case patterns:\n"
                f"{_format_corner_cases(spec)}"
            )
            corner_rules = (
                "11. The example genuinely demonstrates at least one "
                "expected corner-case pattern.\n"
                "12. A correct example without a corner-case pattern "
                "must be marked invalid."
            )
        else:
            dataset_kind = "a synthetic dataset"
            corner_section = ""
            corner_rules = ""

        request = JUDGE_TEMPLATE.format(
            dataset_kind=dataset_kind,
            task_summary=spec.task_summary,
            language=spec.language or "English",
            input_description=spec.io_format.input_description,
            input_constraints=_format_list(
                spec.io_format.input_constraints
            ),
            output_description=spec.io_format.output_description,
            output_constraints=_format_list(
                spec.io_format.output_constraints
            ),
            label_set=_format_label_set(spec.label_set),
            constraints=_format_list(spec.constraints),
            typical_errors=_format_list(
                spec.typical_errors,
                empty="None documented.",
            ),
            corner_section=corner_section,
            corner_rules=corner_rules,
            pairs=json.dumps(
                pairs,
                ensure_ascii=False,
                indent=2,
            ),
        )

        result = invoke_with_retry(
            lambda: self._invoke(request),
            self._retry_config,
        )

        expected = list(range(len(chunk)))
        received = [
            verdict.index
            for verdict in result.verdicts
        ]

        if len(received) != len(set(received)):
            raise JudgeResponseError(f"Judge returned duplicate verdict indexes: {received}")

        if sorted(received) != expected:
            raise JudgeResponseError(
                f"Judge verdict indexes must be {expected}, "
                f"got {sorted(received)}"
            )

        by_index = {
            verdict.index: verdict
            for verdict in result.verdicts
        }

        return [
            by_index[index]
            for index in expected
        ]

    def _invoke(self, request: str) -> JudgeVerdictBatch:
        chat_model = resolve_chat_model(self._model)

        if chat_model is None:
            raw = self._model.invoke(request)
            content = (
                raw.content
                if isinstance(raw, AIMessage)
                else str(raw)
            )

            return JudgeVerdictBatch.model_validate(
                extract_json(content)
            )

        output = (chat_model.with_structured_output(schema=JudgeVerdictBatch, method="json_schema").invoke(request))

        if isinstance(output, JudgeVerdictBatch):
            return output

        if isinstance(output, AIMessage):
            return JudgeVerdictBatch.model_validate(
                extract_json(output.content)
            )

        if isinstance(output, dict):
            return JudgeVerdictBatch.model_validate(output)

        raise TypeError(f"Unexpected structured output type: {type(output)!r}")
