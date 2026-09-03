"""LLM-based quality filtering for generated examples."""

from __future__ import annotations

import json

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages.ai import AIMessage
from pydantic import BaseModel, Field, ValidationError

from coolprompt.spec_generator.models import Example, GenerationContext
from coolprompt.spec_generator.utils.model_utils import resolve_chat_model
from coolprompt.spec_generator.utils.retry import RetryConfig, invoke_with_retry
from coolprompt.utils.logging_config import logger
from coolprompt.utils.parsing import extract_json


def _bullets(items: tuple[str, ...], *, empty: str = "None") -> str:
    return "\n".join(f"- {item}" for item in items) if items else empty


class JudgeVerdict(BaseModel):
    """Verdict for one candidate example."""

    index: int = Field(ge=0)
    is_valid: bool
    quality_score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)


class JudgeVerdictBatch(BaseModel):
    """Verdicts returned for one model call."""

    verdicts: list[JudgeVerdict]


class JudgeResponseError(ValueError):
    """Raised when a judge response cannot be used safely."""


class LLMJudge:
    """Filter examples using an LLM quality rubric."""

    def __init__(
            self,
            model: BaseLanguageModel,
            *,
            quality_threshold: float = 0.7,
            batch_size: int = 15,
            retry_config: RetryConfig | None = None,
    ) -> None:
        if not 0.0 <= quality_threshold <= 1.0:
            raise ValueError("quality_threshold must be between 0 and 1")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        self._model = model
        self._quality_threshold = quality_threshold
        self._batch_size = batch_size
        self._retry_config = retry_config or RetryConfig()

    def filter(
            self,
            examples: list[Example],
            context: GenerationContext,
            *,
            is_corner: bool = False,
    ) -> tuple[list[Example], list[Example]]:
        accepted: list[Example] = []
        rejected: list[Example] = []

        for start in range(0, len(examples), self._batch_size):
            chunk = examples[start: start + self._batch_size]
            verdicts = self._judge_chunk(chunk, context, is_corner)

            for example, verdict in zip(chunk, verdicts, strict=True):
                if verdict.is_valid and verdict.quality_score >= self._quality_threshold:
                    accepted.append(example)
                else:
                    logger.info(
                        "Rejected by judge: %s | score=%.2f | reason=%s",
                        example.input,
                        verdict.quality_score,
                        verdict.reason,
                    )
                    rejected.append(example)

        return accepted, rejected

    def _judge_chunk(
            self,
            chunk: list[Example],
            context: GenerationContext,
            is_corner: bool,
    ) -> list[JudgeVerdict]:
        return invoke_with_retry(
            lambda: self._judge_chunk_once(chunk, context, is_corner),
            self._retry_config,
            extra_retry_exceptions=(JudgeResponseError,),
        )

    def _judge_chunk_once(
            self,
            chunk: list[Example],
            context: GenerationContext,
            is_corner: bool,
    ) -> list[JudgeVerdict]:
        spec = context.spec
        pairs = [
            {"index": index, "input": item.input, "output": item.output}
            for index, item in enumerate(chunk)
        ]

        corner_rule = ""
        if is_corner:
            corner_rule = (
                "\nFor each example, also require a clear match to at least one "
                "listed corner case.\nCorner cases:\n"
                f"{_bullets(spec.corner_cases)}\n"
            )

        request = f"""You are a strict evaluator of synthetic examples.

Task: {spec.description}
Input format: {spec.input_format}
Output format: {spec.output_format}
Requirements:
{_bullets(spec.requirements)}
Valid labels:
{_bullets(spec.labels or ())}
Language: {spec.language}
{corner_rule}
Evaluate every indexed pair for correctness, format compliance, clarity, and realism.
A classification output must be exactly one valid label.
For open-ended generation tasks, many different phrasings can be equally correct:
judge on whether the output satisfies the input constraints (e.g. uses all required
concepts), is fluent, and matches the requirements — do not penalize an output for
differing in wording or structure from any single "canonical" phrasing.
Reject ambiguous, unsupported, malformed, or low-quality examples.

Pairs:
{json.dumps(pairs, ensure_ascii=False, indent=2)}

Return one verdict per index using the provided schema.
"""
        result = self._invoke(request)

        expected = list(range(len(chunk)))
        received = [verdict.index for verdict in result.verdicts]
        if len(received) != len(set(received)):
            raise JudgeResponseError(f"Duplicate verdict indexes: {received}")
        if sorted(received) != expected:
            raise JudgeResponseError(f"Expected verdict indexes {expected}, received {sorted(received)}")

        by_index = {verdict.index: verdict for verdict in result.verdicts}
        return [by_index[index] for index in expected]

    def _invoke(self, request: str) -> JudgeVerdictBatch:
        try:
            chat_model = resolve_chat_model(self._model)
            if chat_model is None:
                raw = self._model.invoke(request)
                content = raw.content if isinstance(raw, AIMessage) else str(raw)
                return JudgeVerdictBatch.model_validate(extract_json(content))

            output = chat_model.with_structured_output(
                schema=JudgeVerdictBatch,
                method="json_schema",
            ).invoke(request)

            if isinstance(output, JudgeVerdictBatch):
                return output
            if isinstance(output, dict):
                return JudgeVerdictBatch.model_validate(output)
            if isinstance(output, AIMessage):
                return JudgeVerdictBatch.model_validate(extract_json(output.content))
            raise JudgeResponseError(f"Unexpected output type: {type(output)!r}")
        except ValidationError as exc:
            raise JudgeResponseError("Judge response failed validation") from exc
        except (TypeError, ValueError) as exc:
            raise JudgeResponseError("Judge response could not be parsed") from exc
