import json
from typing import Optional, List, Tuple, Dict, Any

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.language_models.chat_models import BaseChatModel

from coolprompt.utils.prompt_templates.data_generator_templates import (
    PROBLEM_DESCRIPTION_JSON_OUTPUT_TEMPLATE,
    PROBLEM_DESCRIPTION_STRUCTURED_OUTPUT_TEMPLATE,
    CLASSIFICATION_DATA_GENERATING_STRUCTURED_OUTPUT_TEMPLATE,
    CLASSIFICATION_DATA_GENERATING_JSON_OUTPUT_TEMPLATE,
    GENERATION_DATA_GENERATING_JSON_OUTPUT_TEMPLATE,
    GENERATION_DATA_GENERATING_STRUCTURED_OUTPUT_TEMPLATE
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

        self._problem_description_structured_output_template = (
            PROBLEM_DESCRIPTION_STRUCTURED_OUTPUT_TEMPLATE
        )
        self._problem_description_json_output_template = (
            PROBLEM_DESCRIPTION_JSON_OUTPUT_TEMPLATE
        )

        self._cls_data_generating_structured_output_template = (
            CLASSIFICATION_DATA_GENERATING_STRUCTURED_OUTPUT_TEMPLATE
        )
        self._cls_data_generating_json_output_template = (
            CLASSIFICATION_DATA_GENERATING_JSON_OUTPUT_TEMPLATE
        )
        self._gen_data_generating_structured_output_template = (
            GENERATION_DATA_GENERATING_STRUCTURED_OUTPUT_TEMPLATE
        )
        self._gen_data_generating_json_output_template = (
            GENERATION_DATA_GENERATING_JSON_OUTPUT_TEMPLATE
        )

    def _generate(
        self,
        request_struct: str,
        request_json: str
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
            output = self.model.invoke(request_json)
            return json.loads(output)
        return self.model.invoke(request_struct)

    def _generate_problem_description(self, prompt: str) -> str:
        """Generates problem description based on given user prompt

        Args:
            prompt (str): initial user prompt

        Returns:
            str: generated problem description
        """
        request_struct = self._problem_description_structured_output_template
        request_struct = request_struct.format(prompt=prompt)

        request_json = self._problem_description_json_output_template
        request_json = request_json.format(prompt=prompt)

        return self._generate(
            request_struct,
            request_json
        )['problem_description']

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
            request_struct = (
                self._cls_data_generating_structured_output_template
            )
            request_json = self._cls_data_generating_json_output_template
        else:
            request_struct = (
                self._gen_data_generating_structured_output_template
            )
            request_json = self._gen_data_generating_json_output_template

        request_struct = request_struct.format(
            problem_description=problem_description,
            num_samples=num_samples
        )
        request_json = request_json.format(
            problem_description=problem_description,
            num_samples=num_samples
        )

        examples = self._generate(request_struct, request_json)['examples']
        dataset = [e['input'] for e in examples]
        targets = [e['output'] for e in examples]

        return dataset, targets, problem_description
