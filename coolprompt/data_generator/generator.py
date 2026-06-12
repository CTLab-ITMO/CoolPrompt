from typing import Optional, List, Tuple, Any
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages.ai import AIMessage
from pydantic import BaseModel

from coolprompt.utils.structured_schemas.data_generator import (
    ClassificationTaskExample,
    ClassificationTaskResponse,
    GenerationTaskExample,
    GenerationTaskResponse,
    ProblemDescriptionResponse,
)
from coolprompt.utils.prompt_templates.data_generator_templates import (
    PROBLEM_DESCRIPTION_TEMPLATE,
    CLASSIFICATION_DATA_GENERATING_TEMPLATE,
    GENERATION_DATA_GENERATING_TEMPLATE,
    PROBLEM_DESCRIPTION_BASED_ON_EXAMPLES_TEMPLATE,
    GENERATION_CORNER_CASE_GENERATING_TEMPLATE,
    CLASSIFICATION_CORNER_CASE_GENERATING_TEMPLATE,
)
from coolprompt.utils.enums import Task
from coolprompt.utils.logging_config import logger
from coolprompt.utils.parsing import extract_json


class SyntheticDataGenerator:
    """Synthetic Data Generator.

    Generates synthetic datasets for prompt optimization based on a
    given initial prompt and an optional problem description.

    Attributes:
        model: ``langchain.BaseLanguageModel`` instance to use for LLM calls.
        use_structured_output: When ``True``, every LLM call routes through
            ``model.with_structured_output(schema, method="json_schema")``
            using the Pydantic schemas defined in
            :mod:`coolprompt.utils.structured_schemas.data_generator`.
            When ``False`` (default), the generator falls back to a plain
            ``model.invoke`` call followed by JSON extraction from the
            raw model response — the same convention used by
            :mod:`coolprompt.optimizer` submodules.
    """

    def __init__(
        self,
        model: BaseLanguageModel,
        use_structured_output: bool = False,
    ) -> None:
        self.model = model
        self.use_structured_output = use_structured_output

    def _generate(
            self, request: str, schema: type[BaseModel], field_name: str
    ) -> Any:
        """Generates model output using either structured-output via
        LangChain or a raw JSON parsing fallback.

        Args:
            request (str): request to send to the LLM.
            schema (type[BaseModel]): Pydantic schema describing the
                expected structured output. Only used when
                ``self.use_structured_output`` is ``True``.
            field_name (str): top-level field name to extract from the
                model output.

        Returns:
            Any: extracted value of ``field_name`` from the model output.
        """
        if not self.use_structured_output:
            output = self.model.invoke(request)
            if isinstance(output, AIMessage):
                output = output.content
            return extract_json(output)[field_name]

        structured_model = self.model.with_structured_output(
            schema=schema, method="json_schema"
        )
        output = structured_model.invoke(request)
        if isinstance(output, AIMessage):
            output = output.content

        try:
            output = getattr(output, field_name)
        except Exception:
            output = output[field_name]
        return output

    def _examples_to_str(self, examples: List[Tuple[str, str]]) -> str:
        """Converts list of examples into string format.

        Args:
            examples (List[Tuple[str, str]]): list of examples.

        Returns:
            str: string representation of the provided examples.
        """
        return "\n\n".join(
            [f"Input: {inp}\nOutput: {out}" for (inp, out) in examples]
        )

    def _generate_problem_description(
            self, prompt: str, examples: Optional[List[Tuple[str, str]]] = None
    ) -> str:
        """Generates problem description based on given user prompt.

        Args:
            prompt (str): initial user prompt.
            examples (Optional[List[Tuple[str, str]]]): optional list of
                ``(input, output)`` examples drawn from the task dataset
                to ground the description.

        Returns:
            str: generated problem description.
        """
        if examples:
            request = PROBLEM_DESCRIPTION_BASED_ON_EXAMPLES_TEMPLATE.format(
                prompt=prompt, examples=self._examples_to_str(examples)
            )
        else:
            request = PROBLEM_DESCRIPTION_TEMPLATE.format(prompt=prompt)

        return self._generate(
            request,
            ProblemDescriptionResponse,
            "problem_description",
        )

    def _convert_dataset(
            self,
            examples: List[
                dict | ClassificationTaskExample | GenerationTaskExample
                ],
    ) -> Tuple[List[str], List[str]]:
        """Converts outputs to the dataset format.

        Args:
            examples (
                List[
                    dict |
                    ClassificationTaskExample |
                    GenerationTaskExample
                ]
            ): outputs of the model.

        Returns:
            Tuple[List[str], List[str]]:
                converted dataset and target.
        """
        dataset = []
        targets = []

        for example in examples:
            if isinstance(example, GenerationTaskExample) or isinstance(
                    example, ClassificationTaskExample
            ):
                dataset.append(example.input)
                targets.append(example.output)
            else:
                dataset.append(example["input"])
                targets.append(example["output"])
        return dataset, targets

    def generate(
            self,
            prompt: str,
            task: Task,
            problem_description: Optional[str] = None,
            num_samples: int = 8,
            corner_ratio: float = 0.4,
    ) -> Tuple[List[str], List[str], str]:
        """Generates synthetic dataset based on the given user prompt,
        optimization task and optionally provided problem description.

        If problem description isn't provided it will be generated
        automatically.

        Args:
            prompt (str): initial user prompt.
            task (Task): optimization task — either classification or
                generation.
            problem_description (Optional[str]): problem description
                provided by user. Will be generated if absent. Defaults
                to ``None``.
            num_samples (int): number of samples in dataset to generate.
                Defaults to ``8``.

        Returns:
            Tuple[List[str], List[str], str]:
                generated dataset, target and problem description.
        """
        if problem_description is None:
            logger.info(
                "Problem description was not provided, "
                + "so it will be generated automatically"
            )
            problem_description = self._generate_problem_description(prompt)
            logger.info(f"Generated problem description: {problem_description}")

        if task == Task.CLASSIFICATION:
            regular_template = CLASSIFICATION_DATA_GENERATING_TEMPLATE
            corner_template = CLASSIFICATION_CORNER_CASE_GENERATING_TEMPLATE
            schema = ClassificationTaskResponse
        else:
            regular_template = GENERATION_DATA_GENERATING_TEMPLATE
            corner_template = GENERATION_CORNER_CASE_GENERATING_TEMPLATE
            schema = GenerationTaskResponse

        if corner_template is None:
            request = regular_template.format(
                problem_description=problem_description, num_samples=num_samples
            )
            examples = self._generate(request, schema, "examples")
        else:
            n_corner = int(num_samples * corner_ratio)
            n_regular = num_samples - n_corner

            regular_request = regular_template.format(
                problem_description=problem_description,
                num_samples=n_regular,
            )

            corner_request = corner_template.format(
                problem_description=problem_description,
                num_samples=n_corner,
            )

            regular_examples = self._generate(
                regular_request, schema, "examples"
            )

            corner_examples = self._generate(
                corner_request, schema, "examples"
            )

            examples = list(regular_examples) + list(corner_examples)
        dataset, targets = self._convert_dataset(examples)

        return dataset, targets, problem_description
