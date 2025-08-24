import json
from typing import Optional, List, Tuple, Dict, Any

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

from coolprompt.data_generator.pydantic_formatters import (
    ProblemDescriptionStructuredOutputSchema,
    ClassificationTaskStructuredOutputSchema,
    GenerationTaskStructuredOutputSchema
)
from coolprompt.utils.prompt_templates.data_generator_templates import (
    PROBLEM_DESCRIPTION_TEMPLATE,
    CLASSIFICATION_DATA_GENERATING_TEMPLATE,
    GENERATION_DATA_GENERATING_TEMPLATE
)
from coolprompt.utils.enums import Task


class SyntheticDataGenerator:
    """Synthetic Data Generator
    Generates synthetic dataset for prompt optimization
    based on given initial prompt and optional problem description

    Attributes:
        model: langchain.BaseLanguageModel class of model to use.
    """

    def __init__(
        self,
        model: BaseLanguageModel
    ) -> None:
        self.model = model

    def _generate(
        self,
        request: str,
        schema: BaseModel,
        field_name: Optional[str] = None
    ) -> Dict[Any, Any]:
        """Generates model output
        either using structured output from langchain
        or just strict json output format for LLM

        Args:
            request_struct (str): request to LLM
                when langchain structured output is used
            request_json (str): request to LLM
                that contains strict JSON format for output

        Returns:
            Dict[Any, Any]: generated data (parsed from json)
        """
        if not isinstance(self.model, BaseChatModel):
            output = self.model.invoke(request)
            return json.loads(output)[field_name]

        structured_model = self.model.with_structured_output(
            schema=ProblemDescriptionStructuredOutputSchema,
            method="json_schema"
        )
        output = structured_model.invoke(request)
        return getattr(output, field_name)

    def _generate_problem_description(self, prompt: str) -> str:
        """Generates problem description based on given user prompt

        Args:
            prompt (str): initial user prompt

        Returns:
            str: generated problem description
        """
        request = PROBLEM_DESCRIPTION_TEMPLATE.format(
            prompt=prompt
        )

        return self._generate(
            request,
            ProblemDescriptionStructuredOutputSchema,
            "problem_description"
        )

    def generate(
        self,
        prompt: str,
        task: Task,
        problem_description: Optional[str] = None,
        num_samples: int = 20
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
                Defaults to 20

        Returns:
            Tuple[List[str], List[str], str]:
                generated dataset, target and problem description
        """
        if problem_description is None:
            problem_description = self._generate_problem_description(prompt)

        if task == Task.CLASSIFICATION:
            request = CLASSIFICATION_DATA_GENERATING_TEMPLATE
            schema = ClassificationTaskStructuredOutputSchema
        else:
            request = GENERATION_DATA_GENERATING_TEMPLATE
            schema = GenerationTaskStructuredOutputSchema

        request = request.format(
            problem_description=problem_description,
            num_samples=num_samples
        )

        examples = self._generate(request, schema, "examples")
        dataset = [example['input'] for example in examples]
        targets = [example['output'] for example in examples]

        return dataset, targets, problem_description
