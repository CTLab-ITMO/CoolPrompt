import json
from typing import Optional, List, Tuple, Any

import dirtyjson
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.ai import AIMessage
from pydantic import BaseModel

from coolprompt.data_generator.pydantic_formatters import (
    ProblemDescriptionStructuredOutputSchema,
    ClassificationTaskStructuredOutputSchema,
    ClassificationTaskExample,
    GenerationTaskExample,
    GenerationTaskStructuredOutputSchema,
)
from coolprompt.utils.prompt_templates.data_generator_templates import (
    PROBLEM_DESCRIPTION_TEMPLATE,
    CLASSIFICATION_DATA_GENERATING_TEMPLATE,
    GENERATION_DATA_GENERATING_TEMPLATE,
)
from coolprompt.utils.enums import Task
from coolprompt.utils.logging_config import logger
from coolprompt.utils.parsing import extract_json


class SyntheticDataGenerator:
    """Synthetic Data Generator
    Generates synthetic dataset for prompt optimization
    based on given initial prompt and optional problem description

    Attributes:
        model: langchain.BaseLanguageModel class of model to use.
    """

    def __init__(self, model: BaseLanguageModel) -> None:
        self.model = model

    def _generate(
        self, request: str, schema: BaseModel, field_name: str
    ) -> Any:
        """Generates model output
        either using structured output from langchain
        or just strict json output format for LLM

        Args:
            request (str): request to LLM
                when langchain structured output is used
            schema (BaseModel): Pydantic output format
            field_name (str): field name to select from output

        Returns:
            Any: generated data
        """
        if not isinstance(self.model, BaseChatModel):
            output = self.model.invoke(request)
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

    def _generate_problem_description(self, prompt: str) -> str:
        """Generates problem description based on given user prompt

        Args:
            prompt (str): initial user prompt

        Returns:
            str: generated problem description
        """
        request = PROBLEM_DESCRIPTION_TEMPLATE.format(prompt=prompt)

        return self._generate(
            request,
            ProblemDescriptionStructuredOutputSchema,
            "problem_description",
        )

    def _convert_dataset(
        self,
        examples: List[
            dict | ClassificationTaskExample | GenerationTaskExample
        ],
    ) -> Tuple[List[str], List[str]]:
        """Converts outputs to the dataset format

        Args:
            examples (
                List[
                    dict |
                    ClassificationTaskExample |
                    GenerationTaskExample
                ]
            ): outputs of the model

        Returns:
            Tuple[List[str], List[str]]:
                converted dataset and target
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
    ) -> Tuple[List[str], List[str], str]:
        """Generates synthetic dataset
        based on given user prompt, optimization task
        and optionally provided problem description

        If problem description isn't provided -
            it will be generated automatically

        Args:
            prompt (str): initial user prompt
            task (Task): optimization task
                Either classification or generation
            problem_description (Optional[str]):
                problem description provided by user
                Will be generated if absent
                Defaults to None
            num_samples (int):
                number of samples in dataset to generate
                Defaults to 8

        Returns:
            Tuple[List[str], List[str], str]:
                generated dataset, target and problem description
        """
        if problem_description is None:
            logger.info(
                "Problem description was not provided, "
                + "so it will be generated automatically"
            )
            problem_description = self._generate_problem_description(prompt)
            logger.info(
                f"Generated problem description: {problem_description}"
            )

        if task == Task.CLASSIFICATION:
            request = CLASSIFICATION_DATA_GENERATING_TEMPLATE
            schema = ClassificationTaskStructuredOutputSchema
        else:
            request = GENERATION_DATA_GENERATING_TEMPLATE
            schema = GenerationTaskStructuredOutputSchema

        request = request.format(
            problem_description=problem_description, num_samples=num_samples
        )

        examples = self._generate(request, schema, "examples")
        dataset, targets = self._convert_dataset(examples)

        return dataset, targets, problem_description
