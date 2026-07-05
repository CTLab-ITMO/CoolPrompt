from __future__ import annotations

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages.ai import AIMessage

from coolprompt.spec_generator import DataSpec
from coolprompt.spec_generator.utils.model_utils import resolve_chat_model
from coolprompt.spec_generator.schema import (
    DATASET_LABEL_SETS,
    TASK_AREA_TO_DATASET,
    TaskSpec,
)
from coolprompt.spec_generator.utils.retry_utils import invoke_with_retry, RetryConfig
from coolprompt.task_detector.detector import TaskDetector
from coolprompt.utils.logging_config import logger
from coolprompt.utils.parsing import extract_json
from coolprompt.utils.prompt_templates.spec_generator_templates import (
    SPEC_FROM_PROMPT_AND_EXAMPLES_TEMPLATE,
    SPEC_FROM_PROMPT_TEMPLATE,
)


class SpecBuilder:
    def __init__(
            self,
            model: BaseLanguageModel,
            detector_confidence_threshold: float = 0.7,
            retry_config: RetryConfig | None = None,
    ) -> None:
        self._model = model
        self._retry_config = (
            retry_config
            if retry_config is not None
            else RetryConfig()
        )

        self._detector = TaskDetector(model, confidence_threshold=detector_confidence_threshold)

    def build(
            self,
            prompt: str,
            examples: list[tuple[str, str]] | None = None,
            user_spec: DataSpec | None = None,
            detect_dataset: bool = False,
    ) -> TaskSpec:
        has_user_spec = user_spec is not None and not user_spec.is_empty()

        logger.info(
            "Building TaskSpec from prompt%s%s.",
            f" + {len(examples)} examples" if examples else " only",
            " + user spec" if has_user_spec else "",
        )

        spec_prompt = (
            f"{prompt}\n\n{user_spec.to_prompt_block()}"
            if has_user_spec
            else prompt
        )

        spec = self._invoke(self._build_request(spec_prompt, examples))

        if detect_dataset:
            matched_dataset = self._detect_dataset(spec_prompt)
            spec = spec.model_copy(update={"matched_dataset": matched_dataset})

        if spec.matched_dataset and spec.label_set:
            expected_labels = DATASET_LABEL_SETS.get(spec.matched_dataset)

            if expected_labels and set(spec.label_set) != expected_labels:
                logger.info(
                    "Ignoring dataset %r: label_set=%r is incompatible with expected labels=%r.",
                    spec.matched_dataset,
                    spec.label_set,
                    sorted(expected_labels),
                )
                spec = spec.model_copy(update={"matched_dataset": None})

        logger.info(
            "TaskSpec ready: domain=%r, task_type=%r, skills=%d, "
            "corner_cases=%d, matched_dataset=%r",
            spec.domain,
            spec.task_type,
            len(spec.key_skills),
            len(spec.corner_cases),
            spec.matched_dataset,
        )

        return spec

    def _build_request(self, prompt: str, examples: list[tuple[str, str]] | None) -> str:
        if examples:
            examples_str = "\n\n".join(
                f"Input: {inp}\nOutput: {out}" for inp, out in examples
            )
            return SPEC_FROM_PROMPT_AND_EXAMPLES_TEMPLATE.format(
                prompt=prompt,
                examples=examples_str,
            )

        return SPEC_FROM_PROMPT_TEMPLATE.format(prompt=prompt)

    def _invoke(self, request: str) -> TaskSpec:
        chat_model = resolve_chat_model(self._model)

        if chat_model is None:
            raw = invoke_with_retry(
                lambda: self._model.invoke(request),
                self._retry_config,
            )

            content = (
                raw.content
                if isinstance(raw, AIMessage)
                else str(raw)
            )

            return TaskSpec.model_validate(
                extract_json(content)
            )

        output = invoke_with_retry(
            lambda: (
                chat_model
                .with_structured_output(
                    schema=TaskSpec,
                    method="json_schema",
                )
                .invoke(request)
            ),
            self._retry_config,
        )

        if isinstance(output, TaskSpec):
            return output

        if isinstance(output, dict):
            return TaskSpec.model_validate(output)

        if isinstance(output, AIMessage):
            return TaskSpec.model_validate(
                extract_json(output.content)
            )

        raise TypeError(f"Unexpected structured output type: {type(output)!r}")

    def _detect_dataset(self, prompt: str) -> str | None:
        try:
            detection = self._detector.detect_task_area(prompt)

            if detection.task_area is None:
                return None

            dataset = TASK_AREA_TO_DATASET.get(detection.task_area)

            if dataset is None:
                logger.info("Task area detected but no dataset mapping found: area=%r", detection.task_area)
                return None

            logger.info(
                "Dataset detected: area=%r -> dataset=%r (confidence=%.2f)",
                detection.task_area,
                dataset,
                detection.confidence,
            )

            return dataset

        except Exception as exc:
            logger.warning("Dataset detection failed, skipping: %s", exc)
            return None
